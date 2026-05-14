"""Tests for sleep.prp.allocation — competitive PRP allocation with hysteresis."""

from __future__ import annotations

import torch
import pytest

from sleep.config import PRPConfig
from sleep.tagging.tags import Tag
from sleep.prp.allocation import allocate_prps, compute_threshold


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tag(
    *,
    S_score: float = 0.0,
    p: int = 0,
    d_tag: int = 32,
) -> Tag:
    """Create a minimal Tag with a pre-set S_score and allocation flag."""
    return Tag(
        k=torch.randn(d_tag),
        s=1.0,
        s0=1.0,
        s_reinforced=0.0,
        t0=0,
        e0=0.0,
        a=0,
        rho=0.0,
        ctx=(0, 10, "test"),
        p=p,
        S_score=S_score,
        R=0.0,
        xref_count=0,
        tag_type="novelty",
    )


# ---------------------------------------------------------------------------
# Tests — Part 2 worked example (20 tags, budget of 5)
# ---------------------------------------------------------------------------

class TestWorkedExample:
    """Reproduce the Part 2 worked example: 20 tags, budget 5, top 5 win."""

    def test_top5_get_allocated(self):
        scores = [i * 0.05 for i in range(20)]  # 0.0, 0.05, ..., 0.95
        tags = [make_tag(S_score=sc) for sc in scores]

        # Use a low theta_floor and kappa so that the top 5 clearly pass.
        config = PRPConfig(theta_floor=0.0, kappa_prp=0.0, delta_steal=0.05)
        result = allocate_prps(tags, budget=5, config=config)

        assert result["allocated"] == 5
        # The 5 highest-scored tags (indices 15-19) should be allocated.
        allocated_scores = sorted(
            [t.S_score for t in tags if t.p == 1], reverse=True
        )
        expected_top5 = sorted(scores, reverse=True)[:5]
        assert allocated_scores == pytest.approx(expected_top5)


# ---------------------------------------------------------------------------
# Tests — Hysteresis
# ---------------------------------------------------------------------------

class TestHysteresis:
    """Tags that are already allocated keep their PRP even below threshold."""

    def test_preallocated_keeps_slot(self):
        # Pre-allocate a tag with a score *below* threshold but inside the
        # budget.  Because it is already allocated, hysteresis keeps it.
        config = PRPConfig(theta_floor=0.5, kappa_prp=0.0, delta_steal=1.0)

        # 3 tags: one pre-allocated at 0.45, two unallocated at 0.6 and 0.7.
        preallocated = make_tag(S_score=0.45, p=1)
        high1 = make_tag(S_score=0.7, p=0)
        high2 = make_tag(S_score=0.6, p=0)
        tags = [preallocated, high1, high2]

        allocate_prps(tags, budget=3, config=config)

        # All three fit within budget=3, so the pre-allocated tag stays.
        assert preallocated.p == 1


# ---------------------------------------------------------------------------
# Tests — Stealing (delta_steal)
# ---------------------------------------------------------------------------

class TestStealing:
    """An unallocated tag with sufficiently higher score can steal a PRP slot."""

    def test_steal_succeeds(self):
        config = PRPConfig(theta_floor=0.0, kappa_prp=0.0, delta_steal=0.05)

        # Budget of 2.  Allocate two mediocre tags, then add one strong one
        # that exceeds the weakest by > delta_steal.
        weak = make_tag(S_score=0.3, p=1)
        medium = make_tag(S_score=0.5, p=1)
        strong = make_tag(S_score=0.8, p=0)  # 0.8 > 0.3 + 0.05

        tags = [weak, medium, strong]
        allocate_prps(tags, budget=2, config=config)

        # Strong should steal weak's slot.
        assert strong.p == 1
        assert medium.p == 1
        assert weak.p == 0

    def test_steal_blocked_by_guard(self):
        # Use a high theta_floor so the challenger doesn't qualify in the
        # main allocation pass — only the stealing path could promote it.
        # With delta_steal=1.0, 0.65 < 0.5 + 1.0 so steal is blocked.
        config = PRPConfig(theta_floor=0.8, kappa_prp=0.0, delta_steal=1.0)

        low = make_tag(S_score=0.5, p=1)
        mid = make_tag(S_score=0.6, p=1)
        challenger = make_tag(S_score=0.65, p=0)  # below threshold (0.8), so no main-pass allocation

        tags = [low, mid, challenger]
        allocate_prps(tags, budget=2, config=config)

        # Challenger is below threshold and delta_steal blocks stealing -> stays out.
        assert challenger.p == 0
        assert low.p == 1
        assert mid.p == 1


# ---------------------------------------------------------------------------
# Tests — Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases for allocation."""

    def test_budget_zero(self):
        config = PRPConfig()
        tags = [make_tag(S_score=0.9) for _ in range(5)]
        result = allocate_prps(tags, budget=0, config=config)
        assert result["allocated"] == 0
        for t in tags:
            assert t.p == 0

    def test_empty_tags(self):
        config = PRPConfig()
        result = allocate_prps([], budget=5, config=config)
        assert result["allocated"] == 0


# ---------------------------------------------------------------------------
# Tests — compute_threshold
# ---------------------------------------------------------------------------

class TestComputeThreshold:
    """Verify the adaptive threshold formula."""

    def test_threshold_formula(self):
        import statistics

        scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        config = PRPConfig(kappa_prp=1.0, theta_floor=0.0)

        mu = statistics.mean(scores)
        sigma = statistics.stdev(scores)
        expected = max(0.0, mu + 1.0 * sigma)

        assert compute_threshold(scores, config) == pytest.approx(expected)

    def test_threshold_respects_floor(self):
        scores = [0.01, 0.02, 0.01, 0.02]  # very low mean + std
        config = PRPConfig(kappa_prp=0.0, theta_floor=0.5)
        threshold = compute_threshold(scores, config)
        assert threshold >= 0.5

    def test_threshold_single_score(self):
        config = PRPConfig(theta_floor=0.3)
        assert compute_threshold([0.9], config) == config.theta_floor

    def test_threshold_empty(self):
        config = PRPConfig(theta_floor=0.3)
        assert compute_threshold([], config) == config.theta_floor
