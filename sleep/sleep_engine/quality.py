"""
Replay quality control for the SLEEP sleep engine.

Implements the Q4.1 quality check from SLEEP_Formalization.md (lines 1872-1903).
Validates that a generated replay sample is good enough for consolidation training
by checking:
    1. Semantic similarity — cosine similarity of projected keys must exceed theta_quality.
    2. Surprise level — W_slow's per-token surprisal must exceed its calibration baseline,
       ensuring the replay carries new information.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from sleep.config import SleepConfig
from sleep.tagging.tags import Tag, TagKeyProjection
from sleep.utils.logging import get_logger

logger = get_logger("sleep.sleep_engine.quality")


# ---------------------------------------------------------------------------
# Quality check  (Q4.1)
# ---------------------------------------------------------------------------

@torch.no_grad()
def quality_check(
    replay_ids: Tensor,
    tag: Tag,
    model: nn.Module,
    key_projection: TagKeyProjection,
    config: SleepConfig,
    mu_surprise: float,
    device: str = "cpu",
) -> tuple[bool, str]:
    """Check if a replay sample is good enough for consolidation.

    Runs two sequential checks (short-circuits on first failure):

    1. **Semantic similarity** — The mean-pooled hidden state of the replay is
       projected to key space and compared to the source tag's key vector via
       cosine similarity.  Must meet ``config.theta_quality`` (default 0.5).

    2. **Surprise level** — Mean per-token surprisal of the model on the replay.
       Must exceed ``mu_surprise`` (W_slow's calibration baseline) so the replay
       carries information W_slow has not yet absorbed.

    Args:
        replay_ids:      Generated replay token IDs, shape ``(seq_len,)``.
        tag:             Source tag whose ``.k`` field is the reference key vector.
        model:           Language model in target-inference mode (W_slow + W_cons).
                         Must return an object with ``.logits`` and
                         ``.hidden_states`` (``output_hidden_states=True``).
        key_projection:  Projection from hidden-state space to tag key space.
        config:          :class:`SleepConfig` providing ``theta_quality``.
        mu_surprise:     Baseline mean per-token surprisal of W_slow on
                         calibration data (nats).
        device:          Device string (e.g. ``"cpu"``, ``"cuda:0"``).

    Returns:
        ``(accepted, reason)`` where *reason* is ``""`` on acceptance or a
        human-readable rejection explanation.
    """
    replay_ids = replay_ids.to(device)

    # Ensure batch dimension: (1, seq_len)
    if replay_ids.dim() == 1:
        replay_ids = replay_ids.unsqueeze(0)

    # --- Forward pass to get hidden states and logits -----------------------
    outputs = model(replay_ids, output_hidden_states=True)
    # Last layer hidden states: (1, seq_len, d_model)
    hidden_states: Tensor = outputs.hidden_states[-1]

    # -----------------------------------------------------------------------
    # Check 1: Semantic similarity
    # -----------------------------------------------------------------------
    # Mean-pool over the sequence dimension → (1, d_model) → (d_model,)
    h_mean: Tensor = hidden_states.mean(dim=1).squeeze(0)
    k_replay: Tensor = key_projection(h_mean)  # (d_tag,)

    tag_k: Tensor = tag.k.to(device)
    sim: float = F.cosine_similarity(
        k_replay.unsqueeze(0),
        tag_k.unsqueeze(0),
        dim=1,
    ).item()

    if sim < config.theta_quality:
        reason = (
            f"Replay drifted too far from original experience "
            f"(cosine similarity {sim:.4f} < theta_quality {config.theta_quality})"
        )
        logger.info("Quality check REJECTED (similarity): %s", reason)
        return False, reason

    # -----------------------------------------------------------------------
    # Check 2: Surprise level
    # -----------------------------------------------------------------------
    logits: Tensor = outputs.logits  # (1, seq_len, vocab_size)

    # Compute per-token negative log-likelihood for tokens x_1 … x_{T-1}
    # using the conditional distribution p(x_t | x_{<t}).
    # Shift: logits[:, :-1] predict targets[:, 1:]
    shift_logits: Tensor = logits[:, :-1, :].contiguous()
    shift_targets: Tensor = replay_ids[:, 1:].contiguous()

    # Log-softmax over vocabulary → (1, seq_len-1, vocab_size)
    log_probs: Tensor = F.log_softmax(shift_logits, dim=-1)

    # Gather the log-prob of each actual next token → (1, seq_len-1)
    token_log_probs: Tensor = log_probs.gather(
        dim=-1,
        index=shift_targets.unsqueeze(-1),
    ).squeeze(-1)

    # Surprisal = -log p(x_t | x_{<t}) in nats
    surprises: Tensor = -token_log_probs  # (1, seq_len-1)
    mean_surprise: float = surprises.mean().item()

    if mean_surprise < mu_surprise:
        reason = (
            f"Replay contains no new information for W_slow "
            f"(mean surprise {mean_surprise:.4f} nats < mu_surprise {mu_surprise:.4f} nats)"
        )
        logger.info("Quality check REJECTED (surprise): %s", reason)
        return False, reason

    # -----------------------------------------------------------------------
    # Accepted
    # -----------------------------------------------------------------------
    logger.debug(
        "Quality check ACCEPTED: sim=%.4f, mean_surprise=%.4f nats",
        sim,
        mean_surprise,
    )
    return True, ""


# ---------------------------------------------------------------------------
# Baseline surprise computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_baseline_surprise(
    model: nn.Module,
    calibration_ids: list[Tensor],
    device: str = "cpu",
) -> float:
    """Compute mean per-token surprisal of the model on calibration data.

    This value is used as the ``mu_surprise`` threshold for
    :func:`quality_check`. It represents W_slow's baseline surprise on
    general text — any useful replay should be *more* surprising than this.

    Typical value for a 7B model: ~2.5--3.5 nats.

    Args:
        model:           Language model (W_slow, in eval mode).
        calibration_ids: List of 1-D token-ID tensors from calibration data.
                         Each tensor is an independent sequence.
        device:          Device string.

    Returns:
        Mean per-token surprisal in nats (float).

    Raises:
        ValueError: If ``calibration_ids`` is empty or all sequences are
            too short (< 2 tokens).
    """
    if not calibration_ids:
        raise ValueError("calibration_ids must be a non-empty list of token tensors")

    total_surprise: float = 0.0
    total_tokens: int = 0

    for seq_ids in calibration_ids:
        seq_ids = seq_ids.to(device)
        if seq_ids.dim() == 1:
            seq_ids = seq_ids.unsqueeze(0)  # (1, seq_len)

        seq_len: int = seq_ids.size(1)
        if seq_len < 2:
            # Need at least 2 tokens for a single next-token prediction
            continue

        outputs = model(seq_ids)
        logits: Tensor = outputs.logits  # (1, seq_len, vocab_size)

        shift_logits: Tensor = logits[:, :-1, :].contiguous()
        shift_targets: Tensor = seq_ids[:, 1:].contiguous()

        log_probs: Tensor = F.log_softmax(shift_logits, dim=-1)
        token_log_probs: Tensor = log_probs.gather(
            dim=-1,
            index=shift_targets.unsqueeze(-1),
        ).squeeze(-1)

        surprises: Tensor = -token_log_probs  # (1, seq_len-1)

        total_surprise += surprises.sum().item()
        total_tokens += surprises.numel()

    if total_tokens == 0:
        raise ValueError(
            "No valid calibration sequences (all shorter than 2 tokens)"
        )

    mu_surprise: float = total_surprise / total_tokens
    logger.info(
        "Baseline surprise computed: mu_surprise=%.4f nats over %d tokens",
        mu_surprise,
        total_tokens,
    )
    return mu_surprise
