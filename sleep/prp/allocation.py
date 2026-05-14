"""
PRP Competitive Allocation with Hysteresis.

Implements Q2.3 (budget), Q2.4 (competitive allocation with stealing), and Q2.5 (adaptive
threshold) from SLEEP_Formalization.md.

Allocates PRPs to the highest-scoring tags within a fixed budget. An already-allocated tag
does NOT need to re-exceed the threshold to keep its PRP — it only loses allocation if it
falls outside the top-B by score. This hysteresis prevents oscillation.
"""

from __future__ import annotations

import math
import statistics
from typing import List

from sleep.config import PRPConfig
from sleep.tagging.tags import Tag


# ---------------------------------------------------------------------------
# Adaptive threshold  (Q2.5)
# ---------------------------------------------------------------------------

def compute_threshold(scores: list[float], config: PRPConfig) -> float:
    """Compute adaptive PRP threshold from score distribution.

    θ_PRP(t) = max(θ_floor, μ_S + κ_PRP · σ_S)

    Args:
        scores: PRP composite scores for all tags in the buffer.
        config: PRPConfig with kappa_prp and theta_floor.

    Returns:
        The adaptive threshold. Returns theta_floor when fewer than 2 scores
        are available (stdev undefined).
    """
    if len(scores) < 2:
        return config.theta_floor

    mu = statistics.mean(scores)
    sigma = statistics.stdev(scores)

    return max(config.theta_floor, mu + config.kappa_prp * sigma)


# ---------------------------------------------------------------------------
# Competitive allocation  (Q2.4)
# ---------------------------------------------------------------------------

def allocate_prps(tags: list[Tag], budget: int, config: PRPConfig) -> dict:
    """Run competitive PRP allocation with hysteresis.

    Algorithm:
        1. Compute adaptive threshold from score distribution.
        2. Sort tags by S_score descending.
        3. Walk sorted list — first B tags that are EITHER already allocated
           OR score >= threshold get p=1. Everything else gets p=0.
        4. Post-pass: check δ_steal condition for unallocated high-scorers
           that can displace the lowest allocated tag.

    Updates each tag's ``p`` field in-place (0 or 1).

    Args:
        tags:   All tags in the buffer. Each must have S_score already computed.
        budget: Maximum number of tags that may be allocated (B).
        config: PRPConfig with delta_steal, kappa_prp, theta_floor.

    Returns:
        Dict with allocation statistics::

            {
                "allocated": int,       # number of tags with p=1
                "threshold": float,     # the computed threshold
                "newly_allocated": int, # tags that went 0 -> 1
                "deallocated": int,     # tags that went 1 -> 0
            }
    """
    # --- Edge cases --------------------------------------------------------
    if not tags:
        return {
            "allocated": 0,
            "threshold": config.theta_floor,
            "newly_allocated": 0,
            "deallocated": 0,
        }

    if budget <= 0:
        deallocated = 0
        for tag in tags:
            if tag.p == 1:
                deallocated += 1
                tag.p = 0
        return {
            "allocated": 0,
            "threshold": config.theta_floor,
            "newly_allocated": 0,
            "deallocated": deallocated,
        }

    # --- Snapshot previous allocation state --------------------------------
    prev_alloc: dict[int, int] = {id(tag): tag.p for tag in tags}

    # --- Step 1: Compute threshold -----------------------------------------
    scores: list[float] = [tag.S_score for tag in tags]
    threshold: float = compute_threshold(scores, config)

    # --- Step 2: Sort by S_score descending --------------------------------
    ranked: list[Tag] = sorted(tags, key=lambda t: t.S_score, reverse=True)

    # --- Step 3: Allocate with hysteresis ----------------------------------
    allocated_count: int = 0
    for tag in ranked:
        if allocated_count < budget:
            if tag.p == 1:
                # Already allocated — keep it (hysteresis: no threshold check)
                allocated_count += 1
            elif tag.S_score >= threshold:
                # New allocation — must meet threshold
                tag.p = 1
                allocated_count += 1
            else:
                # Below threshold, not allocated
                tag.p = 0
        else:
            # Budget exhausted
            tag.p = 0

    # --- Step 4: Stealing with minimum differential (δ_steal) --------------
    # After the main pass, check if any unallocated tag can displace the
    # lowest-scoring allocated tag by at least δ_steal.
    _apply_stealing(ranked, budget, threshold, config)

    # --- Compute stats -----------------------------------------------------
    newly_allocated: int = 0
    deallocated: int = 0
    total_allocated: int = 0

    for tag in tags:
        was: int = prev_alloc[id(tag)]
        if tag.p == 1:
            total_allocated += 1
            if was == 0:
                newly_allocated += 1
        else:
            if was == 1:
                deallocated += 1

    return {
        "allocated": total_allocated,
        "threshold": threshold,
        "newly_allocated": newly_allocated,
        "deallocated": deallocated,
    }


# ---------------------------------------------------------------------------
# Stealing helper
# ---------------------------------------------------------------------------

def _apply_stealing(
    ranked: list[Tag],
    budget: int,
    threshold: float,
    config: PRPConfig,
) -> None:
    """Post-allocation stealing pass.

    For each unallocated tag (in descending score order), check if it can
    steal the PRP slot from the lowest-scoring currently-allocated tag.

    A steal happens when:
        candidate.S_score > victim.S_score + δ_steal

    The candidate must also meet the threshold (since it was not previously
    allocated — no hysteresis benefit for a fresh steal).

    Args:
        ranked:    Tags sorted by S_score descending (modified in-place).
        budget:    Maximum allocation count.
        threshold: Current adaptive threshold.
        config:    PRPConfig with delta_steal.
    """
    delta: float = config.delta_steal

    # Keep iterating until no more steals are possible in a single pass.
    changed: bool = True
    while changed:
        changed = False

        # Gather currently allocated and unallocated candidates
        allocated: list[Tag] = [t for t in ranked if t.p == 1]
        if not allocated:
            break

        candidates: list[Tag] = [
            t for t in ranked
            if t.p == 0 and t.S_score >= threshold
        ]
        if not candidates:
            break

        # Victim = allocated tag with the lowest score
        victim: Tag = min(allocated, key=lambda t: t.S_score)

        # Best candidate = unallocated tag with highest score (first in
        # ranked order that is unallocated and above threshold)
        best: Tag = max(candidates, key=lambda t: t.S_score)

        if best.S_score > victim.S_score + delta:
            victim.p = 0
            best.p = 1
            changed = True
        # If no steal happened, we're done.
