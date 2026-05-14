"""
W_fast gradient update for surprising spans during wake phase.

Implements the "hippocampal encoding" step (Q3.3): when a surprising span is
detected (mean z-score > kappa_wfast), we perform a single SGD step on the
W_fast LoRA adapter parameters using the language-modeling loss over the span.

Key design choices from the formalization:
  - SGD with momentum (not Adam) -- prevents deep convergence, lower memory
  - Learning rate scaled by surprise magnitude up to a cap
  - Only W_fast parameters receive gradients; W_slow_base and W_cons are frozen
  - No weight decay (decay handled by tag/PRP system)

Reference: SLEEP_Formalization.md Q3.3
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F
from peft import PeftModel

from sleep.config import WeightsConfig
from sleep.weights.composition import get_trainable_params, set_wfast_training_mode

logger = logging.getLogger(__name__)

# Reference prediction error for learning rate scaling (from Q3.3).
_E_REF: float = 2.0


class FastWeightUpdater:
    """Performs gradient updates on W_fast adapter parameters for surprising spans.

    Usage::

        updater = FastWeightUpdater(model, config)

        # When a surprising span is detected during wake:
        loss = updater.update(model, token_ids, span_start, span_end, E_span)
    """

    def __init__(self, model: PeftModel, config: WeightsConfig) -> None:
        """Set up the SGD optimizer targeting W_fast adapter parameters only.

        Args:
            model: PeftModel with at least a ``w_fast`` adapter attached.
            config: ``WeightsConfig`` providing ``alpha_fast`` and ``momentum_fast``.
        """
        self._alpha_base: float = config.alpha_fast
        self._momentum: float = config.momentum_fast

        # Put model in W_fast training mode so that only W_fast params have
        # requires_grad=True, then collect those parameters for the optimizer.
        set_wfast_training_mode(model)
        trainable_params: list[torch.nn.Parameter] = get_trainable_params(model)

        if not trainable_params:
            raise RuntimeError(
                "No trainable parameters found after setting W_fast training mode. "
                "Ensure the model has a 'w_fast' adapter."
            )

        self._optimizer: torch.optim.SGD = torch.optim.SGD(
            trainable_params,
            lr=self._alpha_base,
            momentum=self._momentum,
            weight_decay=0.0,
        )

        logger.info(
            "FastWeightUpdater initialized: %d parameter groups, "
            "alpha_base=%.2e, momentum=%.2f",
            len(trainable_params),
            self._alpha_base,
            self._momentum,
        )

    def update(
        self,
        model: PeftModel,
        token_ids: torch.Tensor,
        span_start: int,
        span_end: int,
        E_span: float,
        device: str = "cpu",
    ) -> float:
        """Perform one gradient step on W_fast for the given surprising span.

        Steps:
            1. Set model to W_fast training mode (W_fast trainable, rest frozen).
            2. Scale learning rate by surprise: alpha_eff = alpha_base * min(1, E_span / E_ref).
            3. Forward pass on span tokens; compute cross-entropy LM loss on the span.
            4. Backward pass.
            5. Clip gradients (max_norm=1.0).
            6. Optimizer step and zero gradients.

        Args:
            model: The PeftModel (must be the same model used at init).
            token_ids: 1-D tensor of token IDs for the full context surrounding
                the span.  Shape ``(seq_len,)``.
            span_start: Start index of the surprising span (inclusive) within
                ``token_ids``.
            span_end: End index of the surprising span (exclusive) within
                ``token_ids``.
            E_span: Mean prediction error (surprise) of the span, used to scale
                the learning rate.
            device: Device string (``"cpu"``, ``"cuda"``, etc.).

        Returns:
            The scalar loss value on the span (for logging).

        Raises:
            ValueError: If span indices are invalid.
        """
        if span_start < 0 or span_end > len(token_ids) or span_start >= span_end:
            raise ValueError(
                f"Invalid span [{span_start}, {span_end}) for sequence of "
                f"length {len(token_ids)}"
            )

        # --- 1. Ensure correct training mode ---
        set_wfast_training_mode(model)

        # --- 2. Scale learning rate by surprise ---
        alpha_effective: float = self._alpha_base * min(1.0, E_span / _E_REF)
        for pg in self._optimizer.param_groups:
            pg["lr"] = alpha_effective

        # --- 3. Forward pass ---
        # Feed the full token sequence up to (and including) span_end.
        # span_end is INCLUSIVE in our Tag convention.
        # We compute loss only on the span tokens [span_start : span_end+1].
        seq_end: int = span_end + 1  # exclusive end for slicing
        input_ids: torch.Tensor = token_ids[:seq_end].unsqueeze(0).to(device)

        outputs = model(input_ids=input_ids)
        logits: torch.Tensor = outputs.logits  # (1, seq_len, vocab_size)

        # Standard causal LM loss: logits[t] predicts token[t+1].
        # For span tokens [span_start : span_end+1], the predictions come from
        # logits at positions [span_start-1 : span_end] (shifted by 1).
        # If span_start == 0, we skip the first span token (no preceding context).
        pred_start: int = max(span_start - 1, 0)
        pred_end: int = span_end  # logits position that predicts token at span_end
        target_start: int = max(span_start, 1)  # skip token 0 if span starts there
        target_end: int = span_end + 1

        pred_logits: torch.Tensor = logits[0, pred_start:pred_end, :]    # (span_len, vocab)
        target_tokens: torch.Tensor = token_ids[target_start:target_end].to(device)  # (span_len,)

        # Ensure shapes match (they can differ by 1 if span_start==0)
        min_len: int = min(pred_logits.size(0), target_tokens.size(0))
        pred_logits = pred_logits[:min_len]
        target_tokens = target_tokens[:min_len]

        if min_len == 0:
            # Span is too short to compute loss (e.g., single token at position 0)
            self._optimizer.zero_grad()
            return 0.0

        loss: torch.Tensor = F.cross_entropy(pred_logits, target_tokens)

        # --- 4. Backward pass ---
        loss.backward()

        # --- 5. Gradient clipping ---
        trainable_params: list[torch.nn.Parameter] = get_trainable_params(model)
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

        # --- 6. Optimizer step ---
        self._optimizer.step()
        self._optimizer.zero_grad()

        loss_value: float = loss.item()
        logger.debug(
            "W_fast update: span=[%d,%d), E_span=%.3f, alpha_eff=%.2e, loss=%.4f",
            span_start,
            span_end,
            E_span,
            alpha_effective,
            loss_value,
        )
        return loss_value
