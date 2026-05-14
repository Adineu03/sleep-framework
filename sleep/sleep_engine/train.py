"""
Sleep training loop for the SLEEP system.

Implements Q4.4 (training procedure) from SLEEP_Formalization.md.

Runs gradient descent on W_cons during sleep, interleaving new replay samples
with old-knowledge samples according to a curriculum schedule.  EWC
regularisation anchors W_cons to its pre-sleep checkpoint so that previously
consolidated knowledge is not overwritten.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from sleep.config import SleepConfig, WeightsConfig
from sleep.utils.logging import get_logger, metrics

if TYPE_CHECKING:
    from sleep.weights import DualWeightSystem
    from sleep.sleep_engine.replay import ReplaySample

logger = get_logger("sleep.sleep_engine.train")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pad_and_collate(
    token_ids_list: list[torch.Tensor],
    pad_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length sequences and create an attention mask.

    Args:
        token_ids_list: List of 1-D ``LongTensor`` sequences.
        pad_id:         Token id used for padding (usually 0 or
                        ``tokenizer.pad_token_id``).

    Returns:
        input_ids:      ``(batch, max_len)`` padded token ids.
        attention_mask:  ``(batch, max_len)`` binary mask (1 = real, 0 = pad).
    """
    lengths = [t.shape[0] for t in token_ids_list]
    max_len = max(lengths)

    input_ids = torch.full(
        (len(token_ids_list), max_len),
        fill_value=pad_id,
        dtype=torch.long,
    )
    attention_mask = torch.zeros(
        len(token_ids_list), max_len, dtype=torch.long,
    )

    for i, (t, length) in enumerate(zip(token_ids_list, lengths)):
        input_ids[i, :length] = t
        attention_mask[i, :length] = 1

    return input_ids, attention_mask


def _get_curriculum_eta(
    step: int,
    n_steps: int,
    config: SleepConfig,
) -> int:
    """Return the old-to-new ratio eta for the current curriculum phase.

    Phase 1 (warmup,   first  ``curriculum_warmup``  fraction):  eta = 9
    Phase 2 (consolidate, middle ``curriculum_consolidate``):     eta = eta_default (4)
    Phase 3 (stabilize, final fraction):                          eta = 9
    """
    frac = step / max(n_steps, 1)
    if frac < config.curriculum_warmup:
        return 9
    elif frac < config.curriculum_warmup + config.curriculum_consolidate:
        return config.eta_default
    else:
        return 9


def _sample_batch(
    replay_dataset: list[ReplaySample],
    batch_size: int,
    eta: int,
) -> list[torch.Tensor]:
    """Sample a batch of token sequences from the replay dataset.

    Selects *batch_size* samples with replacement, weighting by PRP score.
    The old/new interleaving ratio ``eta`` is informational here — the
    replay dataset is assumed to already contain the appropriate mix produced
    by the orchestrator.  We weight by ``prp_score`` so higher-priority
    memories are seen more often.

    Returns:
        A list of 1-D ``LongTensor`` token sequences.
    """
    if not replay_dataset:
        return []

    # Build sampling weights from PRP scores (softmax-style normalisation)
    scores = torch.tensor(
        [max(s.prp_score, 1e-8) for s in replay_dataset],
        dtype=torch.float32,
    )
    probs = scores / scores.sum()

    indices = torch.multinomial(probs, num_samples=batch_size, replacement=True)
    return [replay_dataset[idx].text_ids for idx in indices]


def _cosine_warmup_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create a cosine-decay LR schedule with linear warmup."""

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return max(float(current_step) / max(warmup_steps, 1), 0.0)
        progress = (current_step - warmup_steps) / max(
            total_steps - warmup_steps, 1
        )
        return max(0.5 * (1.0 + math.cos(math.pi * progress)), 0.0)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def sleep_train(
    dual_weights: DualWeightSystem,
    replay_dataset: list[ReplaySample],
    tokenizer,
    config: SleepConfig,
    weights_config: WeightsConfig,
    fisher_diag: dict[str, torch.Tensor] | None = None,
    device: str = "cpu",
) -> dict:
    """Execute the sleep training loop.

    Steps:
        1. Set model to ``sleep_training`` mode.
        2. Save W_cons checkpoint for rollback.
        3. Set up AdamW optimizer on W_cons params only.
        4. For each step:
           a. Sample a weighted batch from the replay dataset.
           b. Compute causal LM cross-entropy loss.
           c. Compute EWC penalty (if *fisher_diag* is provided).
           d. Backward, gradient clip, plasticity scaling, optimizer step.
           e. Hard-clip weight bounds.
        5. Return training statistics.

    Args:
        dual_weights:   The :class:`DualWeightSystem` managing adapters.
        replay_dataset: List of :class:`ReplaySample` (new + old interleaved).
        tokenizer:      HuggingFace tokenizer (for pad token id).
        config:         :class:`SleepConfig` with training hyper-parameters.
        weights_config: :class:`WeightsConfig` with LR, EWC lambda, etc.
        fisher_diag:    Optional diagonal Fisher information dict.  When
                        provided, an EWC penalty anchors W_cons to its
                        pre-sleep state.
        device:         Device string (``"cpu"``, ``"cuda:0"``, ...).

    Returns:
        A dict with keys ``n_steps``, ``final_loss``, ``mean_loss``, and
        ``cons_checkpoint`` (the pre-training snapshot for rollback).
    """
    from sleep.sleep_engine.fisher import compute_ewc_loss  # local to avoid circular

    if not replay_dataset:
        logger.warning("sleep_train called with empty replay_dataset — nothing to do")
        return {
            "n_steps": 0,
            "final_loss": 0.0,
            "mean_loss": 0.0,
            "cons_checkpoint": dual_weights.save_cons_checkpoint(),
        }

    # ---- 1. Mode switch ----
    dual_weights.set_mode("sleep_training")
    model = dual_weights.model

    # ---- 2. Save checkpoint for rollback / EWC anchor ----
    cons_checkpoint: dict[str, torch.Tensor] = dual_weights.save_cons_checkpoint()

    # ---- 3. Optimizer + scheduler ----
    trainable_params: list[nn.Parameter] = dual_weights.get_cons_trainable_params()
    if not trainable_params:
        logger.error("No trainable parameters found — is the model in sleep_training mode?")
        return {
            "n_steps": 0,
            "final_loss": 0.0,
            "mean_loss": 0.0,
            "cons_checkpoint": cons_checkpoint,
        }

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=weights_config.alpha_slow,
        weight_decay=config.sleep_weight_decay,
    )

    n_steps: int = max(100, len(replay_dataset) * config.steps_per_memory)
    warmup_steps: int = int(0.1 * n_steps)
    scheduler = _cosine_warmup_schedule(optimizer, warmup_steps, n_steps)

    pad_id: int = getattr(tokenizer, "pad_token_id", None) or 0

    logger.info(
        "Starting sleep training | n_steps=%d | n_replay=%d | lr=%.2e | device=%s",
        n_steps, len(replay_dataset), weights_config.alpha_slow, device,
    )

    # ---- 4. Training loop ----
    loss_accum: float = 0.0
    final_loss: float = 0.0

    for step in range(1, n_steps + 1):
        # 4a. Curriculum-aware batch sampling
        eta: int = _get_curriculum_eta(step, n_steps, config)
        batch_seqs: list[torch.Tensor] = _sample_batch(
            replay_dataset, config.batch_size, eta,
        )
        if not batch_seqs:
            continue

        # 4b. Pad and move to device
        input_ids, attention_mask = pad_and_collate(batch_seqs, pad_id=pad_id)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Labels = input_ids shifted; mask out padding with -100
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        # 4c. Forward pass — causal LM loss
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        lm_loss: torch.Tensor = outputs.loss

        # 4d. EWC penalty (optional)
        total_loss: torch.Tensor = lm_loss
        if fisher_diag is not None:
            ewc_loss = compute_ewc_loss(
                model=model,
                fisher_diag=fisher_diag,
                checkpoint=cons_checkpoint,
                lambda_ewc=weights_config.lambda_ewc,
                adapter_name="w_cons",
            )
            total_loss = lm_loss + ewc_loss

        # 4e. Backward
        optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(trainable_params, config.grad_clip_norm)

        # Plasticity scaling (must be after backward, before optimizer.step)
        dual_weights.apply_plasticity_scaling()

        # Optimizer + scheduler step
        optimizer.step()
        scheduler.step()

        # 4f. Hard-clip weight bounds
        dual_weights.enforce_weight_bounds(cons_checkpoint)

        # ---- Tracking ----
        step_loss: float = total_loss.item()
        loss_accum += step_loss
        final_loss = step_loss

        if step % max(n_steps // 10, 1) == 0 or step == n_steps:
            mean_so_far = loss_accum / step
            logger.info(
                "  step %d/%d | loss=%.4f | mean_loss=%.4f | eta=%d | lr=%.2e",
                step, n_steps, step_loss, mean_so_far, eta,
                scheduler.get_last_lr()[0],
            )
            metrics.log({
                "sleep_train/step": step,
                "sleep_train/loss": step_loss,
                "sleep_train/mean_loss": mean_so_far,
                "sleep_train/eta": eta,
                "sleep_train/lr": scheduler.get_last_lr()[0],
            })

    mean_loss: float = loss_accum / max(n_steps, 1)

    logger.info(
        "Sleep training complete | n_steps=%d | final_loss=%.4f | mean_loss=%.4f",
        n_steps, final_loss, mean_loss,
    )
    metrics.log_summary({
        "sleep_train/final_loss": final_loss,
        "sleep_train/mean_loss": mean_loss,
        "sleep_train/n_steps": n_steps,
    })

    return {
        "n_steps": n_steps,
        "final_loss": final_loss,
        "mean_loss": mean_loss,
        "cons_checkpoint": cons_checkpoint,
    }
