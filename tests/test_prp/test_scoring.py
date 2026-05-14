"""Tests for sleep.prp.scoring — PRP composite scoring."""

from __future__ import annotations

import torch
import pytest

from sleep.config import PRPConfig
from sleep.tagging.tags import Tag


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tag(
    *,
    e0: float = 0.0,
    s: float = 1.0,
    s0: float = 1.0,
    rho: float = 0.0,
    R: float = 0.0,
    xref_count: int = 0,
    tag_type: str = "novelty",
    d_tag: int = 32,
) -> Tag:
    """Create a minimal Tag with controllable scoring-relevant fields."""
    return Tag(
        k=torch.randn(d_tag),
        s=s,
        s0=s0,
        s_reinforced=0.0,
        t0=0,
        e0=e0,
        a=0,
        rho=rho,
        ctx=(0, 10, "test"),
        p=0,
        S_score=0.0,
        R=R,
        xref_count=xref_count,
        tag_type=tag_type,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestComputePrpScores:
    """Tests for compute_prp_scores."""

    def test_scores_in_unit_interval(self):
        """All composite scores (before revision bonus) should be in [0, 1]."""
        from sleep.prp.scoring import compute_prp_scores

        tags = [
            make_tag(e0=0.5, rho=0.3, R=0.2, xref_count=1),
            make_tag(e0=1.0, rho=1.0, R=1.0, xref_count=3),
            make_tag(e0=0.1, rho=0.0, R=0.0, xref_count=0),
            make_tag(e0=0.8, rho=0.5, R=0.5, xref_count=2),
            make_tag(e0=0.3, rho=0.2, R=0.1, xref_count=0),
        ]
        config = PRPConfig()
        scores = compute_prp_scores(tags, config, revision_bonus=0.0)

        assert len(scores) == 5
        for sc in scores:
            assert 0.0 <= sc <= 1.0, f"Score {sc} outside [0, 1]"

    def test_revision_bonus_applied(self):
        """Revision tags should receive an additive bonus."""
        from sleep.prp.scoring import compute_prp_scores

        novelty = make_tag(e0=0.5, rho=0.3, R=0.3, xref_count=1, tag_type="novelty")
        revision = make_tag(e0=0.5, rho=0.3, R=0.3, xref_count=1, tag_type="revision")
        tags = [novelty, revision]

        config = PRPConfig()
        bonus = 0.3
        scores = compute_prp_scores(tags, config, revision_bonus=bonus)

        # The revision tag's score should exceed the novelty tag's by exactly the bonus.
        assert scores[1] == pytest.approx(scores[0] + bonus, abs=1e-7)

    def test_high_values_beat_zeros(self):
        """A tag with high e0, rho, R, xref_count should outscore an all-zero tag."""
        from sleep.prp.scoring import compute_prp_scores

        high = make_tag(e0=2.0, rho=5.0, R=3.0, xref_count=5)
        low = make_tag(e0=0.0, rho=0.0, R=0.0, xref_count=0)
        tags = [high, low]

        config = PRPConfig()
        scores = compute_prp_scores(tags, config, revision_bonus=0.0)

        assert scores[0] > scores[1]

    def test_empty_tag_list(self):
        """Empty input should return empty output without error."""
        from sleep.prp.scoring import compute_prp_scores

        config = PRPConfig()
        scores = compute_prp_scores([], config, revision_bonus=0.3)
        assert scores == []

    def test_all_zeros_returns_all_zeros(self):
        """Tags with every component at zero should all score zero."""
        from sleep.prp.scoring import compute_prp_scores

        tags = [
            make_tag(e0=0.0, s=0.0, s0=1.0, rho=0.0, R=0.0, xref_count=0)
            for _ in range(4)
        ]
        config = PRPConfig()
        scores = compute_prp_scores(tags, config, revision_bonus=0.0)

        for sc in scores:
            assert sc == pytest.approx(0.0, abs=1e-10)
