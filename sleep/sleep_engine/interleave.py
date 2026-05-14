"""
Interleaving strategy for SLEEP sleep-cycle training batches.

Implements Q4.3 (interleaving strategy) from SLEEP_Formalization.md.

Each sleep training batch mixes three sources:
  1. **New replay** -- generated from W_fast using PRP-allocated tags.
  2. **Old knowledge (nearby)** -- generated from W_target on topics related
     to the new replay (proximity-weighted protection).
  3. **Old knowledge (random)** -- generated from W_target on diverse generic
     prompts (broad catastrophic-forgetting protection).

A three-phase curriculum controls the old-to-new ratio eta across the sleep
cycle: warmup (eta=9), consolidate (eta=4), stabilize (eta=9).
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import torch

from sleep.config import SleepConfig
from sleep.sleep_engine.replay import ReplaySample
from sleep.utils.logging import get_logger, metrics

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

logger = get_logger("sleep.sleep_engine.interleave")

# ---------------------------------------------------------------------------
# Generic seed prompts for random old-knowledge generation (Q4.3)
# ---------------------------------------------------------------------------

GENERIC_PROMPTS: list[str] = [
    "The history of",
    "In mathematics,",
    "According to research,",
    "The process of",
    "One important aspect of",
    "Scientists have discovered",
    "The relationship between",
    "In recent years,",
    "A common approach to",
    "The fundamental principle of",
]


# ---------------------------------------------------------------------------
# Curriculum helper
# ---------------------------------------------------------------------------

def get_curriculum_eta(step: int, total_steps: int, config: SleepConfig) -> int:
    """Return the old-to-new ratio eta for the current step in the sleep cycle.

    The curriculum has three phases:

    * **Phase 1 -- warmup** (first ``config.curriculum_warmup`` fraction of
      steps): ``eta = 9`` (90 % old, 10 % new).
    * **Phase 2 -- consolidate** (middle ``config.curriculum_consolidate``
      fraction): ``eta = config.eta_default`` (typically 4, i.e. 80/20).
    * **Phase 3 -- stabilize** (remaining fraction): ``eta = 9`` (90/10).

    Args:
        step: Current training step within this sleep cycle (0-indexed).
        total_steps: Total number of training steps in this sleep cycle.
        config: :class:`SleepConfig` containing the curriculum fractions and
                default eta.

    Returns:
        The integer old-to-new ratio for this step.
    """
    if total_steps <= 0:
        return config.eta_default

    progress: float = step / total_steps

    warmup_end: float = config.curriculum_warmup
    consolidate_end: float = warmup_end + config.curriculum_consolidate

    if progress < warmup_end:
        # Phase 1: warmup -- very conservative
        return 9
    elif progress < consolidate_end:
        # Phase 2: consolidation -- main training
        return config.eta_default
    else:
        # Phase 3: stabilize -- conservative again
        return 9


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def sample_new_replay(
    replay_dataset: list[ReplaySample],
    n_samples: int,
) -> list[torch.Tensor]:
    """Sample from the replay dataset, weighted by PRP score.

    Uses weighted random sampling **with replacement** so that high-PRP
    memories are replayed more frequently, matching the formalization's
    intent that priority determines rehearsal frequency.

    Args:
        replay_dataset: Available replay samples produced by
                        :func:`~sleep.sleep_engine.replay.generate_replay`.
        n_samples: Number of samples to draw.

    Returns:
        A list of token-ID tensors (each 1-D, variable length).
    """
    if not replay_dataset or n_samples <= 0:
        return []

    weights: list[float] = [s.prp_score for s in replay_dataset]

    # random.choices does weighted sampling WITH replacement.
    chosen: list[ReplaySample] = random.choices(
        replay_dataset, weights=weights, k=n_samples,
    )

    return [s.text_ids for s in chosen]


def generate_old_knowledge(
    model: object,
    tokenizer: PreTrainedTokenizer,
    n_samples: int,
    seed_texts: list[str] | None = None,
    max_length: int = 256,
    temperature: float = 1.0,
    top_p: float = 0.95,
    device: str = "cpu",
) -> list[torch.Tensor]:
    """Generate old-knowledge samples from W_target (W_slow + W_cons).

    Two modes of operation:

    * **Proximity-weighted** (``seed_texts`` provided): generate a
      continuation of each seed text, producing old knowledge that is
      semantically near the new replay.
    * **Random** (``seed_texts is None``): pick diverse seeds from
      :data:`GENERIC_PROMPTS` and generate continuations.

    Args:
        model: The model configured in *target-inference* mode
               (W_slow + W_cons adapters active, **no** W_fast).
        tokenizer: Tokenizer for *model*.
        n_samples: Number of old-knowledge samples to generate.
        seed_texts: Optional list of seed strings. When provided the
                    function cycles through them to generate continuations.
                    When ``None``, generic prompts are used instead.
        max_length: Maximum number of new tokens per sample.
        temperature: Sampling temperature (high = diverse).
        top_p: Nucleus-sampling threshold.
        device: Device for generation (e.g. ``"cpu"``, ``"cuda:0"``).

    Returns:
        A list of token-ID tensors (each 1-D, variable length).
    """
    if n_samples <= 0:
        return []

    # Decide which seeds to use.
    if seed_texts is not None and len(seed_texts) > 0:
        seeds: list[str] = seed_texts
    else:
        seeds = GENERIC_PROMPTS

    results: list[torch.Tensor] = []

    for i in range(n_samples):
        seed: str = seeds[i % len(seeds)]

        input_ids: torch.Tensor = tokenizer.encode(
            seed,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(device)  # (1, seq_len)

        with torch.no_grad():
            output_ids: torch.Tensor = model.generate(
                input_ids=input_ids,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_length,
            )  # (1, seq_len + generated)

        # Flatten to 1-D and move to CPU for uniformity with replay tensors.
        results.append(output_ids.squeeze(0).cpu())

    return results


# ---------------------------------------------------------------------------
# Main batch builder
# ---------------------------------------------------------------------------

def build_sleep_batch(
    replay_dataset: list[ReplaySample],
    model: object,
    tokenizer: PreTrainedTokenizer,
    step: int,
    total_steps: int,
    config: SleepConfig,
    device: str = "cpu",
) -> list[torch.Tensor]:
    """Build one training batch for the sleep cycle.

    The procedure (per Q4.3):

    1. Determine eta from the warmup/consolidate/stabilize curriculum.
    2. Compute counts: ``n_new``, ``n_old_nearby``, ``n_old_random``.
    3. Sample new replay from *replay_dataset* weighted by PRP score.
    4. Generate nearby old knowledge (seeded from replay topics).
    5. Generate random old knowledge (seeded from generic prompts).
    6. Shuffle and return.

    Args:
        replay_dataset: Replay samples produced during this sleep cycle.
        model: The model in *target-inference* mode (W_slow + W_cons).
        tokenizer: Tokenizer matching *model*.
        step: Current training step within this sleep cycle (0-indexed).
        total_steps: Total training steps in this sleep cycle.
        config: :class:`SleepConfig` with interleaving and training params.
        device: Device for generation (e.g. ``"cpu"``, ``"cuda:0"``).

    Returns:
        A list of token-ID tensors (variable length; caller is responsible
        for padding/collation).
    """
    batch_size: int = config.batch_size

    # ---- Step 1: curriculum eta ----
    eta: int = get_curriculum_eta(step, total_steps, config)

    # ---- Step 2: compute counts ----
    n_new: int = max(1, batch_size // (1 + eta))
    n_old: int = batch_size - n_new
    n_old_nearby: int = round(n_old * config.proximity_fraction)
    n_old_random: int = n_old - n_old_nearby

    # Handle empty replay dataset -- produce only old knowledge.
    if not replay_dataset:
        logger.warning(
            "Empty replay dataset at step %d/%d; batch will contain only "
            "old knowledge (%d samples).",
            step, total_steps, batch_size,
        )
        n_new = 0
        n_old_nearby = 0
        n_old_random = batch_size

    logger.debug(
        "step=%d/%d  eta=%d  n_new=%d  n_old_nearby=%d  n_old_random=%d",
        step, total_steps, eta, n_new, n_old_nearby, n_old_random,
    )

    # ---- Step 3: new replay (PRP-weighted) ----
    new_samples: list[torch.Tensor] = sample_new_replay(
        replay_dataset, n_new,
    )

    # ---- Step 4: nearby old knowledge ----
    # Extract seed texts from the sampled new replay for proximity seeding.
    seed_texts: list[str] | None = None
    if new_samples and n_old_nearby > 0:
        seed_texts = []
        for tensor in new_samples:
            # Decode the first tokens of each new replay as a topic seed.
            n_seed_tokens: int = min(16, tensor.shape[0])
            decoded: str = tokenizer.decode(
                tensor[:n_seed_tokens], skip_special_tokens=True,
            )
            seed_texts.append(decoded)

    old_nearby: list[torch.Tensor] = generate_old_knowledge(
        model=model,
        tokenizer=tokenizer,
        n_samples=n_old_nearby,
        seed_texts=seed_texts,
        max_length=256,
        temperature=1.0,
        top_p=0.95,
        device=device,
    )

    # ---- Step 5: random old knowledge ----
    old_random: list[torch.Tensor] = generate_old_knowledge(
        model=model,
        tokenizer=tokenizer,
        n_samples=n_old_random,
        seed_texts=None,  # uses GENERIC_PROMPTS
        max_length=256,
        temperature=1.0,
        top_p=0.95,
        device=device,
    )

    # ---- Step 6: combine and shuffle ----
    batch: list[torch.Tensor] = new_samples + old_nearby + old_random
    random.shuffle(batch)

    # ---- Metrics ----
    metrics.log(
        {
            "interleave/eta": eta,
            "interleave/n_new": len(new_samples),
            "interleave/n_old_nearby": len(old_nearby),
            "interleave/n_old_random": len(old_random),
            "interleave/batch_size": len(batch),
        },
        step=step,
    )

    return batch
