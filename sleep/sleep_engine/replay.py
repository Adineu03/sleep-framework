"""
Replay generation for the SLEEP sleep engine.

Implements Q4.1 (generation model) from SLEEP_Formalization.md.

During sleep, generates synthetic replay samples from PRP-allocated tags.
The generator is the model itself (W_slow_base + W_cons + W_fast) used
autoregressively. A short seed from the original experience sets the topic,
and the model generates a compressed continuation that captures the gist.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from peft import PeftModel
from transformers import PreTrainedTokenizer

from sleep.config import SleepConfig
from sleep.tagging.tags import Tag
from sleep.utils.logging import get_logger

logger = get_logger("sleep.sleep_engine.replay")

# The replay prefix prepended to every seed (Section Q4.1).
REPLAY_PREFIX: str = "Recall: "


# ---------------------------------------------------------------------------
# ReplaySample dataclass
# ---------------------------------------------------------------------------

@dataclass
class ReplaySample:
    """A single synthetic replay generated from a PRP-allocated tag.

    Attributes:
        text_ids:        Full token IDs (seed + generated continuation).
        tag_id:          ``id()`` of the source :class:`Tag` (for tracking).
        prp_score:       The tag's PRP composite score (for weighted sampling).
        original_length: Token length of the original span the tag referenced.
        seed_length:     Number of tokens in ``text_ids`` that came from the
                         original experience (the rest are model-generated).
    """

    text_ids: torch.Tensor       # (seq_len,)
    tag_id: int                  # id() of the source tag
    prp_score: float             # tag's PRP score
    original_length: int         # span_end - span_start
    seed_length: int             # tokens taken from original


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_replay(
    tag: Tag,
    model: PeftModel,
    tokenizer: PreTrainedTokenizer,
    original_tokens: torch.Tensor | None,
    config: SleepConfig,
    device: str = "cpu",
    seed_from_episode_start: bool = True,
) -> ReplaySample | None:
    """Generate a replay sample for a single PRP-allocated tag.

    The generation follows the Q4.1 protocol:
      1. Extract a short seed from the original experience.
      2. Prepend the ``"Recall: "`` prefix.
      3. Auto-regressively sample a compressed continuation.
      4. Package seed + continuation as a :class:`ReplaySample`.

    Args:
        tag:             The tag to generate replay for.
        model:           The model in sleep-generation mode (W_slow + W_cons +
                         W_fast adapters active).
        tokenizer:       Tokenizer matching *model*, used to encode the
                         ``"Recall: "`` prefix.
        original_tokens: The full original token sequence this tag came from.
                         If ``None``, generation is impossible and we return
                         ``None``.
        config:          :class:`SleepConfig` with ``replay_temperature``,
                         ``replay_top_p``, ``seed_length_max``, etc.
        device:          Device to run generation on (e.g. ``"cpu"``,
                         ``"cuda:0"``).

    Returns:
        A :class:`ReplaySample` if generation succeeds, or ``None`` when
        the original tokens are unavailable or the tag's span is invalid.
    """

    # ---- Edge case: no original tokens available ----
    if original_tokens is None:
        logger.debug(
            "Skipping replay for tag %d: original_tokens is None", id(tag)
        )
        return None

    # ---- Unpack context ----
    span_start: int
    span_end: int
    source_id: str
    span_start, span_end, source_id = tag.ctx

    # ---- Validate span bounds ----
    if span_start < 0 or span_end <= span_start:
        logger.debug(
            "Skipping replay for tag %d: invalid span [%d, %d)",
            id(tag), span_start, span_end,
        )
        return None

    if span_end > original_tokens.shape[0]:
        logger.debug(
            "Skipping replay for tag %d: span_end %d exceeds token length %d",
            id(tag), span_end, original_tokens.shape[0],
        )
        return None

    # ---- Step 1: Construct seed ----
    # Two seeding strategies:
    #
    #   seed_from_episode_start=True (default, episode-storage architecture):
    #     The seed is the FIRST tokens of the original sequence. The tag is
    #     a pointer telling us "this episode is worth replaying"; the seed
    #     sets up generation from the natural start of the episode, so
    #     attention to the stored full-episode K/V can drive coherent
    #     continuation.
    #
    #   seed_from_episode_start=False (legacy span-storage architecture):
    #     The seed is the first few tokens of the tagged span itself. This
    #     is what the original Q4.1 protocol specified, and it matches the
    #     semantics where each stored entry is the tagged span only.
    original_length: int = span_end - span_start
    full_length: int = int(original_tokens.shape[0])

    if seed_from_episode_start:
        # Use the start of the full episode. Length scales with episode
        # size (capped by seed_length_max), with a sensible floor of 8
        # tokens so the model has enough context to anchor.
        seed_len: int = min(config.seed_length_max, max(8, full_length // 3))
        seed_start: int = 0
    else:
        # Legacy: seed from the tagged span position.
        seed_len = min(config.seed_length_max, original_length // 4)
        seed_len = max(seed_len, 4)  # at least 4 tokens
        seed_start = span_start

    # Check whether we actually have enough tokens for a usable seed.
    available: int = full_length - seed_start
    if available < 2:
        logger.debug(
            "Skipping replay for tag %d: only %d tokens available for seed",
            id(tag), available,
        )
        return None

    seed_len = min(seed_len, available)
    if seed_len < 2:
        logger.debug(
            "Skipping replay for tag %d: computed seed_len %d < 2",
            id(tag), seed_len,
        )
        return None

    seed_tokens: torch.Tensor = original_tokens[seed_start : seed_start + seed_len]

    # ---- Step 2: Prepend "Recall: " prefix ----
    prefix_ids: torch.Tensor = tokenizer.encode(
        REPLAY_PREFIX, add_special_tokens=False, return_tensors="pt",
    ).squeeze(0)  # (prefix_len,)

    # Move to the correct device and build the full input.
    prefix_ids = prefix_ids.to(device)
    seed_tokens = seed_tokens.to(device)
    input_ids: torch.Tensor = torch.cat([prefix_ids, seed_tokens], dim=0)
    input_ids = input_ids.unsqueeze(0)  # (1, seq_len) — batch dim for generate()

    # ---- Step 3: Compute target length ----
    target_length: int = max(
        config.min_replay_length,
        original_length // config.compression_target,
    )

    # ---- Step 4: Generate ----
    with torch.no_grad():
        output_ids: torch.Tensor = model.generate(
            input_ids=input_ids,
            do_sample=True,
            temperature=config.replay_temperature,
            top_p=config.replay_top_p,
            max_new_tokens=target_length,
        )  # (1, prefix_len + seed_len + generated_len)

    # Flatten back to 1-D and strip the prefix so text_ids = seed + generated.
    full_output: torch.Tensor = output_ids.squeeze(0)  # (total_len,)
    prefix_len: int = prefix_ids.shape[0]
    text_ids: torch.Tensor = full_output[prefix_len:]  # seed + generated

    logger.debug(
        "Generated replay for tag %d (source=%s): seed=%d, generated=%d, "
        "target=%d, original=%d",
        id(tag), source_id, seed_len, text_ids.shape[0] - seed_len,
        target_length, original_length,
    )

    # ---- Step 5: Package as ReplaySample ----
    return ReplaySample(
        text_ids=text_ids.cpu(),
        tag_id=id(tag),
        prp_score=tag.S_score,
        original_length=original_length,
        seed_length=seed_len,
    )
