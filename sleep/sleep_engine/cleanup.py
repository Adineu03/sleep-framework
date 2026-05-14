"""
Post-consolidation validation and cleanup for the SLEEP system.

Implements Q4.6 (post-consolidation cleanup) from SLEEP_Formalization.md.

After sleep training, each consolidated tag is validated: did W_cons actually
learn the memory?  Tags that pass are removed from the buffer; tags that fail
are penalised and re-queued.  Tags that fail three times are permanently removed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from sleep.config import SleepConfig
from sleep.utils.logging import get_logger, metrics

if TYPE_CHECKING:
    from sleep.tagging.tags import Tag

logger = get_logger("sleep.sleep_engine.cleanup")


# ---------------------------------------------------------------------------
# Span surprise computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_span_surprise(
    model,
    token_ids: torch.Tensor,
    span_start: int,
    span_end: int,
    device: str = "cpu",
) -> float:
    """Compute mean per-token surprise (negative log-likelihood) on a span.

    Surprise for token *t* is ``-log p(x_t | x_{<t})``.  We average over all
    tokens in ``[span_start, span_end)`` using the full preceding context.

    Args:
        model:      A causal LM (in eval mode, no adapters need to be toggled
                    here — the caller is responsible for mode).
        token_ids:  1-D ``LongTensor`` of the full token sequence.
        span_start: Start index of the span to evaluate (inclusive).
        span_end:   End index of the span to evaluate (exclusive).
        device:     Device to run the forward pass on.

    Returns:
        Mean per-token surprise (float, non-negative).
    """
    if span_end <= span_start:
        return 0.0

    # We need context from at least position 0 up to span_end so that every
    # token in the span has the correct causal context.
    input_ids = token_ids[:span_end].unsqueeze(0).to(device)  # (1, span_end)

    outputs = model(input_ids=input_ids)
    logits = outputs.logits  # (1, span_end, vocab)

    # Shift: logits[t] predicts token[t+1].
    # For a token at position *p*, its prediction comes from logits at *p-1*.
    # We want surprises for positions [span_start, span_end).
    # logits[:, span_start-1 : span_end-1, :] predicts tokens at [span_start, span_end).
    pred_start = max(span_start - 1, 0)
    pred_end = span_end - 1

    if pred_start >= pred_end:
        # Only one token in the span and it's the first token — no prediction
        return 0.0

    log_probs = F.log_softmax(logits[:, pred_start:pred_end, :], dim=-1)  # (1, L, V)
    target_ids = token_ids[span_start:span_end].to(device)

    # If span_start == 0, skip the first target since there's no preceding logit
    if span_start == 0:
        target_ids = target_ids[1:]
        if target_ids.numel() == 0:
            return 0.0

    # Align lengths (log_probs and target_ids must match)
    n_tokens = min(log_probs.shape[1], target_ids.shape[0])
    log_probs = log_probs[:, :n_tokens, :]
    target_ids = target_ids[:n_tokens]

    # Gather the log-prob of each actual next token
    token_log_probs = log_probs[0].gather(
        dim=-1, index=target_ids.unsqueeze(-1),
    ).squeeze(-1)  # (n_tokens,)

    # Surprise = -log p(x_t | x_{<t})
    mean_surprise: float = (-token_log_probs).mean().item()
    return mean_surprise


# ---------------------------------------------------------------------------
# Consolidation validation
# ---------------------------------------------------------------------------

def validate_consolidation(
    tag: Tag,
    model_new,
    model_old_surprise: float,
    original_tokens: torch.Tensor | None,
    config: SleepConfig,
    device: str = "cpu",
) -> bool:
    """Check whether W_cons has learned a memory well enough.

    The check: W_target's surprise on the original span has decreased by at
    least ``epsilon_learn`` (default 10%).

    .. math::

        \\text{surprise\\_new} = \\text{mean\\_surprisal}(W_{\\text{target\\_new}},
                                 \\text{span})

        \\text{PASS if } \\text{surprise\\_new} < \\text{surprise\\_old}
                         \\times (1 - \\varepsilon_{\\text{learn}})

    Args:
        tag:                The tag being validated.
        model_new:          The model after sleep training (W_slow + W_cons_new),
                            already in target-inference mode.
        model_old_surprise: Pre-sleep mean surprise of the old model on this
                            tag's span.
        original_tokens:    Full original token sequence the tag references.
                            If ``None``, validation automatically fails.
        config:             :class:`SleepConfig` with ``epsilon_learn``.
        device:             Device for the forward pass.

    Returns:
        ``True`` if the memory passes validation, ``False`` otherwise.
    """
    if original_tokens is None:
        logger.debug(
            "Validation FAIL for tag %d: original_tokens unavailable", id(tag),
        )
        return False

    span_start, span_end, source_id = tag.ctx

    # Bounds check
    if span_start < 0 or span_end <= span_start or span_end > original_tokens.shape[0]:
        logger.debug(
            "Validation FAIL for tag %d: invalid span [%d, %d)",
            id(tag), span_start, span_end,
        )
        return False

    surprise_new: float = compute_span_surprise(
        model_new, original_tokens, span_start, span_end, device=device,
    )

    threshold: float = model_old_surprise * (1.0 - config.epsilon_learn)
    passed: bool = surprise_new < threshold

    logger.debug(
        "Validation %s for tag %d (source=%s): "
        "surprise_new=%.4f, surprise_old=%.4f, threshold=%.4f (eps=%.2f)",
        "PASS" if passed else "FAIL",
        id(tag), source_id,
        surprise_new, model_old_surprise, threshold, config.epsilon_learn,
    )

    return passed


# ---------------------------------------------------------------------------
# Tag cleanup
# ---------------------------------------------------------------------------

def cleanup_tags(
    tags: list[Tag],
    validation_results: dict[int, bool],
    config: SleepConfig | None = None,
) -> dict[str, list[Tag]]:
    """Process validation results and sort tags into outcome buckets.

    Actions:
        - **Passed** tags are marked for removal from the tag buffer
          (their knowledge is now in W_cons).
        - **Failed** tags stay in the buffer with penalised strength:
          ``p = 0``, ``s *= 0.5``, ``fail_count += 1``.
        - Tags with ``fail_count >= max_failures`` (default 3) are
          permanently removed.

    Args:
        tags:               The tags that were candidates for consolidation.
        validation_results: Mapping from ``id(tag)`` to pass/fail boolean.
        config:             Optional :class:`SleepConfig`; uses
                            ``max_failures`` (default 3) for the permanent
                            removal threshold.

    Returns:
        A dict with three lists::

            {
                "passed":              [Tag, ...],  # consolidated, to remove
                "failed":              [Tag, ...],  # stay in buffer, penalised
                "permanently_removed": [Tag, ...],  # removed after 3+ failures
            }
    """
    max_failures: int = config.max_failures if config is not None else 3

    passed: list[Tag] = []
    failed: list[Tag] = []
    permanently_removed: list[Tag] = []

    for tag in tags:
        tag_id = id(tag)
        result = validation_results.get(tag_id)

        if result is None:
            # Tag was not validated (e.g. missing original tokens) — treat as fail
            logger.debug("Tag %d not in validation_results — treating as fail", tag_id)
            result = False

        if result:
            # ---- Passed: consolidated successfully ----
            passed.append(tag)
        else:
            # ---- Failed: penalise and check failure count ----
            tag.p = 0
            tag.s *= 0.5
            tag.fail_count += 1

            if tag.fail_count >= max_failures:
                permanently_removed.append(tag)
            else:
                failed.append(tag)

    logger.info(
        "Cleanup complete | passed=%d | failed=%d | permanently_removed=%d",
        len(passed), len(failed), len(permanently_removed),
    )
    metrics.log({
        "cleanup/passed": len(passed),
        "cleanup/failed": len(failed),
        "cleanup/permanently_removed": len(permanently_removed),
    })

    return {
        "passed": passed,
        "failed": failed,
        "permanently_removed": permanently_removed,
    }
