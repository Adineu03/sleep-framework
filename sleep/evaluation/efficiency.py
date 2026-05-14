"""
Consolidation efficiency and PRP allocation quality for the SLEEP evaluation suite.

Implements Metrics 4 and 5 from Q5.4 (SLEEP_Formalization.md):
    CE  = |successfully_consolidated| / |PRP_allocated|       (target > 0.8)
    PAQ = |{t : t.p=1 AND t was later accessed}| / B         (precision@B)
    PAQ_relative = PAQ / PAQ_oracle                           (target > 0.6)
"""

from __future__ import annotations

from typing import Any

from sleep.tagging.tags import Tag
from sleep.utils.logging import get_logger, metrics

logger = get_logger("sleep.evaluation.efficiency")


# ---------------------------------------------------------------------------
# Consolidation Efficiency (CE)
# ---------------------------------------------------------------------------

def compute_consolidation_efficiency(
    n_consolidated: int,
    n_prp_allocated: int,
) -> float:
    """Compute consolidation efficiency.

    CE = n_consolidated / n_prp_allocated

    This is the fraction of PRP-allocated memories that passed validation
    during sleep. Target: CE > 0.8 (80% success rate).

    Args:
        n_consolidated:  Number of memories that successfully consolidated.
        n_prp_allocated: Number of memories that had PRP allocation.

    Returns:
        CE ratio (float). Returns 0.0 if n_prp_allocated is 0.
    """
    if n_prp_allocated <= 0:
        logger.warning("No PRP-allocated memories; CE is undefined (returning 0.0)")
        return 0.0

    ce = n_consolidated / n_prp_allocated

    logger.info(
        "Consolidation efficiency: CE=%.4f (%d/%d)",
        ce, n_consolidated, n_prp_allocated,
    )
    metrics.log({
        "evaluation/consolidation_efficiency": ce,
        "evaluation/n_consolidated": n_consolidated,
        "evaluation/n_prp_allocated": n_prp_allocated,
    })

    return ce


# ---------------------------------------------------------------------------
# PRP Allocation Quality (PAQ)
# ---------------------------------------------------------------------------

def compute_prp_allocation_quality(
    allocated_tags: list[Tag],
    accessed_after: dict[int, int],
    total_budget: int,
) -> dict:
    """Compute PRP allocation quality.

    PAQ measures whether the PRP allocator is selecting the right memories
    to consolidate — i.e., ones that will actually be accessed later.

    PAQ         = |{t : t.p=1 AND accessed_after[t_id] > 0}| / total_budget
    PAQ_oracle  = min(|{t_id : accessed_after[t_id] > 0}|, total_budget) / total_budget
    PAQ_relative = PAQ / PAQ_oracle

    The oracle is the hindsight-optimal allocator: it would allocate PRPs to
    whichever memories actually get accessed later, up to the budget.

    Args:
        allocated_tags: List of tags that were PRP-allocated (p=1).
        accessed_after: Mapping from tag index (positional in the full tag
                        buffer) to post-allocation access count. Tags not
                        present are assumed to have 0 accesses.
        total_budget:   Total PRP budget B.

    Returns:
        Dictionary with keys:
            - ``paq`` (float): Precision of PRP allocation.
            - ``paq_oracle`` (float): Oracle (hindsight-optimal) precision.
            - ``paq_relative`` (float): paq / paq_oracle.
    """
    if total_budget <= 0:
        logger.warning("Total budget is 0; PAQ is undefined")
        return {"paq": 0.0, "paq_oracle": 0.0, "paq_relative": 0.0}

    # Count allocated tags that were later accessed
    n_allocated_and_accessed = 0
    for tag in allocated_tags:
        tag_id = id(tag)
        if accessed_after.get(tag_id, 0) > 0:
            n_allocated_and_accessed += 1

    paq = n_allocated_and_accessed / total_budget

    # Oracle: how many tags in the entire population were accessed?
    n_total_accessed = sum(1 for count in accessed_after.values() if count > 0)
    # Oracle allocates PRPs to up to budget of those that were actually accessed
    paq_oracle = min(n_total_accessed, total_budget) / total_budget

    # Relative quality
    if paq_oracle > 0:
        paq_relative = paq / paq_oracle
    else:
        # No tags were accessed at all — allocation is vacuously perfect
        paq_relative = 1.0

    result = {
        "paq": paq,
        "paq_oracle": paq_oracle,
        "paq_relative": paq_relative,
    }

    logger.info(
        "PRP allocation quality: PAQ=%.4f, oracle=%.4f, relative=%.4f",
        paq, paq_oracle, paq_relative,
    )
    metrics.log({
        "evaluation/paq": paq,
        "evaluation/paq_oracle": paq_oracle,
        "evaluation/paq_relative": paq_relative,
    })

    return result


# ---------------------------------------------------------------------------
# Tag population statistics
# ---------------------------------------------------------------------------

def compute_tag_stats(tags: list[Tag]) -> dict:
    """Compute summary statistics for a tag population.

    Useful for monitoring the health of the tag buffer across experiments.

    Args:
        tags: List of :class:`Tag` instances.

    Returns:
        Dictionary with keys:
            - ``n_tags`` (int): Total number of tags.
            - ``n_allocated`` (int): Number with PRP allocation (p=1).
            - ``n_novelty`` (int): Number of novelty-type tags.
            - ``n_revision`` (int): Number of revision-type tags.
            - ``mean_strength`` (float): Mean current strength.
            - ``mean_initial_strength`` (float): Mean initial strength (s0).
            - ``mean_score`` (float): Mean PRP composite score.
            - ``mean_access_count`` (float): Mean access count.
            - ``mean_error`` (float): Mean initial prediction error (e0).
            - ``total_accesses`` (int): Sum of all access counts.
    """
    if not tags:
        return {
            "n_tags": 0,
            "n_allocated": 0,
            "n_novelty": 0,
            "n_revision": 0,
            "mean_strength": 0.0,
            "mean_initial_strength": 0.0,
            "mean_score": 0.0,
            "mean_access_count": 0.0,
            "mean_error": 0.0,
            "total_accesses": 0,
        }

    n = len(tags)
    n_allocated = sum(1 for t in tags if t.p == 1)
    n_novelty = sum(1 for t in tags if t.tag_type == "novelty")
    n_revision = sum(1 for t in tags if t.tag_type == "revision")

    mean_strength = sum(t.s for t in tags) / n
    mean_initial_strength = sum(t.s0 for t in tags) / n
    mean_score = sum(t.S_score for t in tags) / n
    mean_access = sum(t.a for t in tags) / n
    mean_error = sum(t.e0 for t in tags) / n
    total_accesses = sum(t.a for t in tags)

    stats = {
        "n_tags": n,
        "n_allocated": n_allocated,
        "n_novelty": n_novelty,
        "n_revision": n_revision,
        "mean_strength": mean_strength,
        "mean_initial_strength": mean_initial_strength,
        "mean_score": mean_score,
        "mean_access_count": mean_access,
        "mean_error": mean_error,
        "total_accesses": total_accesses,
    }

    logger.info(
        "Tag stats: n=%d, allocated=%d, mean_s=%.4f, mean_score=%.4f",
        n, n_allocated, mean_strength, mean_score,
    )

    return stats
