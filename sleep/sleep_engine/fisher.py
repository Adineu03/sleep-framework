"""
Fisher information computation for EWC regularisation in the SLEEP system.

Implements diagonal Fisher information estimation and EWC loss computation
as described in Q3.4 of SLEEP_Formalization.md.

The diagonal Fisher is used to weight how much each W_cons parameter is
allowed to change during sleep — parameters that are important for previously
consolidated knowledge receive stronger regularisation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from sleep.utils.logging import get_logger, metrics

if TYPE_CHECKING:
    from peft import PeftModel

logger = get_logger("sleep.sleep_engine.fisher")


# ---------------------------------------------------------------------------
# Diagonal Fisher computation
# ---------------------------------------------------------------------------

def compute_fisher_diagonal(
    model: PeftModel,
    calibration_data: list[torch.Tensor],
    adapter_name: str = "w_cons",
    max_samples: int = 200,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Compute the diagonal Fisher information matrix for adapter parameters.

    The diagonal Fisher approximation is:

    .. math::

        F_i = \\frac{1}{N} \\sum_{n=1}^{N}
              \\left(\\frac{\\partial \\log p(y_n \\mid x_n, \\theta)}
                          {\\partial \\theta_i}\\right)^2

    Only adapter parameters (those whose name contains *adapter_name*) are
    tracked.  Base-model parameters are excluded.

    Args:
        model:            The model in ``target_inference`` mode (W_slow + W_cons).
                          Must have gradients enabled for adapter params.
        calibration_data: List of 1-D ``LongTensor`` token sequences used to
                          estimate the Fisher.  A mix of general-knowledge and
                          consolidated-knowledge samples is recommended.
        adapter_name:     Name substring identifying the adapter whose params
                          we compute Fisher for (default ``"w_cons"``).
        max_samples:      Maximum number of calibration samples to use.
        device:           Device for forward/backward passes.

    Returns:
        A dict mapping parameter names to their diagonal Fisher tensors,
        all detached and on *device*.
    """
    # ---- Identify adapter parameters ----
    adapter_params: dict[str, torch.nn.Parameter] = {}
    for name, param in model.named_parameters():
        if adapter_name in name:
            adapter_params[name] = param

    if not adapter_params:
        logger.warning(
            "No parameters found matching adapter_name=%r — returning empty Fisher",
            adapter_name,
        )
        return {}

    # ---- Initialise Fisher accumulators ----
    fisher_accum: dict[str, torch.Tensor] = {
        name: torch.zeros_like(param, device=device)
        for name, param in adapter_params.items()
    }

    # ---- Temporarily enable gradients for adapter params ----
    original_grad_flags: dict[str, bool] = {}
    for name, param in adapter_params.items():
        original_grad_flags[name] = param.requires_grad
        param.requires_grad = True

    n_samples: int = min(len(calibration_data), max_samples)

    model.train()  # enable dropout etc. for a more faithful Fisher estimate

    for i in range(n_samples):
        tokens = calibration_data[i]
        if tokens.numel() < 2:
            continue

        input_ids = tokens.unsqueeze(0).to(device)  # (1, seq_len)

        # Labels = input_ids (causal LM — model shifts internally)
        labels = input_ids.clone()

        # Forward
        outputs = model(input_ids=input_ids, labels=labels)
        loss: torch.Tensor = outputs.loss

        # Backward — accumulate squared gradients
        model.zero_grad()
        loss.backward()

        for name, param in adapter_params.items():
            if param.grad is not None:
                fisher_accum[name] += param.grad.detach().pow(2)

    # ---- Average over samples ----
    if n_samples > 0:
        for name in fisher_accum:
            fisher_accum[name] /= n_samples

    # ---- Restore original grad flags ----
    for name, param in adapter_params.items():
        param.requires_grad = original_grad_flags[name]

    # Zero out leftover gradients
    model.zero_grad()

    logger.info(
        "Fisher diagonal computed | adapter=%s | n_samples=%d | n_params=%d",
        adapter_name, n_samples, len(fisher_accum),
    )
    metrics.log({
        "fisher/n_samples": n_samples,
        "fisher/n_params": len(fisher_accum),
        "fisher/mean_fisher": sum(
            f.mean().item() for f in fisher_accum.values()
        ) / max(len(fisher_accum), 1),
    })

    return fisher_accum


# ---------------------------------------------------------------------------
# EWC loss
# ---------------------------------------------------------------------------

def compute_ewc_loss(
    model: PeftModel,
    fisher_diag: dict[str, torch.Tensor],
    checkpoint: dict[str, torch.Tensor],
    lambda_ewc: float = 100.0,
    adapter_name: str = "w_cons",
) -> torch.Tensor:
    """Compute the Elastic Weight Consolidation penalty.

    .. math::

        L_{\\text{EWC}} = \\frac{\\lambda}{2}
            \\sum_i F_i \\, (\\theta_i - \\theta_i^*)^2

    where :math:`\\theta^*` is the checkpoint (pre-sleep adapter state) and
    :math:`F_i` is the diagonal Fisher information.

    The result is a **differentiable** scalar so that gradients flow back
    through the current adapter parameters.

    Args:
        model:        The model whose adapter parameters are being trained.
        fisher_diag:  Dict of ``{param_name: fisher_tensor}`` from
                      :func:`compute_fisher_diagonal`.
        checkpoint:   Dict of ``{param_name: tensor}`` — the saved adapter
                      state at the start of the sleep cycle.
        lambda_ewc:   EWC penalty strength (default 100.0).
        adapter_name: Adapter name substring for parameter matching.

    Returns:
        A scalar ``torch.Tensor`` (differentiable) representing the EWC
        penalty.  Returns ``tensor(0.0)`` if no matching parameters are found.
    """
    penalty = torch.tensor(0.0, device=next(model.parameters()).device)

    for name, param in model.named_parameters():
        if adapter_name not in name:
            continue
        if name not in fisher_diag or name not in checkpoint:
            continue

        fisher = fisher_diag[name]
        theta_star = checkpoint[name]

        # Move to the same device as param if needed
        if fisher.device != param.device:
            fisher = fisher.to(param.device)
        if theta_star.device != param.device:
            theta_star = theta_star.to(param.device)

        diff = param - theta_star
        penalty = penalty + (fisher * diff.pow(2)).sum()

    penalty = (lambda_ewc / 2.0) * penalty
    return penalty
