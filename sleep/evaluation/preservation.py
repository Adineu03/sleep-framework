"""
Base Capability Preservation (BCP) for the SLEEP evaluation suite.

Implements Metric 3 from Q5.4 (SLEEP_Formalization.md):
    BCP(n) = PPL_benchmark(W_slow after n cycles) / PPL_benchmark(W_slow original)

BCP should stay close to 1.0. BCP > 1.05 (5% degradation) after any number of
sleep cycles is considered a failure.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from sleep.utils.logging import get_logger, metrics

logger = get_logger("sleep.evaluation.preservation")


# ---------------------------------------------------------------------------
# Perplexity evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_perplexity(
    model: Any,
    token_sequences: list[Tensor],
    device: str = "cpu",
) -> float:
    """Compute mean perplexity across token sequences.

    For each sequence, computes the cross-entropy loss over next-token
    predictions, then returns exp(mean_loss) across all sequences.

    Args:
        model:           A HuggingFace-style causal LM that returns a
                         ``loss`` when given ``labels``.
        token_sequences: List of 1-D tensors, each containing token IDs for
                         one evaluation sequence.
        device:          Device string ("cpu", "cuda", etc.).

    Returns:
        Mean perplexity (PPL = exp(mean cross-entropy loss)).

    Raises:
        ValueError: If token_sequences is empty.
    """
    if not token_sequences:
        raise ValueError("token_sequences must be non-empty")

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for seq in token_sequences:
        input_ids = seq.unsqueeze(0).to(device)  # (1, seq_len)

        # Labels = input_ids shifted internally by HF models
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss

        # Weight by number of predicted tokens (seq_len - 1)
        n_tokens = max(input_ids.shape[1] - 1, 1)
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    mean_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    ppl = math.exp(mean_loss)

    logger.info(
        "Perplexity evaluation: PPL=%.4f (mean_loss=%.4f, %d tokens)",
        ppl, mean_loss, total_tokens,
    )

    return ppl


# ---------------------------------------------------------------------------
# Base Capability Preservation
# ---------------------------------------------------------------------------

def compute_bcp(
    ppl_current: float,
    ppl_original: float,
) -> float:
    """Compute Base Capability Preservation ratio.

    BCP = ppl_current / ppl_original

    Interpretation:
        - BCP = 1.0 means no degradation.
        - BCP > 1.0 means degradation (higher PPL = worse language modeling).
        - BCP < 1.0 means improvement (rare but possible).

    Args:
        ppl_current:  Perplexity of the model after sleep cycles.
        ppl_original: Perplexity of the original (pre-SLEEP) model.

    Returns:
        BCP ratio (float).

    Raises:
        ValueError: If ppl_original is zero or negative.
    """
    if ppl_original <= 0:
        raise ValueError(f"ppl_original must be positive, got {ppl_original}")

    bcp = ppl_current / ppl_original

    logger.info(
        "BCP: %.4f (current_ppl=%.4f, original_ppl=%.4f)",
        bcp, ppl_current, ppl_original,
    )
    metrics.log({
        "evaluation/bcp": bcp,
        "evaluation/ppl_current": ppl_current,
        "evaluation/ppl_original": ppl_original,
    })

    return bcp


# ---------------------------------------------------------------------------
# Degradation check
# ---------------------------------------------------------------------------

def check_degradation(
    bcp: float,
    threshold: float = 1.05,
) -> tuple[bool, str]:
    """Check if base capabilities have degraded beyond the threshold.

    From Q5.4: BCP > 1.05 after any number of cycles is a failure.

    Args:
        bcp:       Base Capability Preservation ratio from :func:`compute_bcp`.
        threshold: Maximum acceptable BCP (default 1.05 = 5% degradation).

    Returns:
        Tuple of ``(degraded, message)`` where ``degraded`` is True if
        BCP exceeds the threshold.
    """
    degraded = bcp > threshold

    if degraded:
        message = (
            f"DEGRADATION DETECTED: BCP={bcp:.4f} exceeds threshold "
            f"{threshold:.4f}. Base capabilities have degraded by "
            f"{(bcp - 1.0) * 100:.1f}%."
        )
        logger.warning(message)
    else:
        pct = (bcp - 1.0) * 100
        if bcp >= 1.0:
            message = (
                f"BCP={bcp:.4f} is within threshold {threshold:.4f}. "
                f"Degradation: {pct:.1f}%."
            )
        else:
            message = (
                f"BCP={bcp:.4f} indicates improvement. "
                f"PPL decreased by {abs(pct):.1f}%."
            )
        logger.info(message)

    return degraded, message
