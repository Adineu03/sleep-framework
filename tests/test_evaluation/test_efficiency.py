"""Tests for consolidation efficiency and tag stats (Module 6 — efficiency.py)."""

import pytest
import torch

from sleep.evaluation.efficiency import (
    compute_consolidation_efficiency,
    compute_tag_stats,
)
from sleep.tagging.tags import Tag


# --------------------------------------------------------------------- #
# Consolidation Efficiency (CE)
# --------------------------------------------------------------------- #

def test_consolidation_efficiency_normal():
    """8 consolidated out of 10 allocated -> CE = 0.8."""
    ce = compute_consolidation_efficiency(n_consolidated=8, n_prp_allocated=10)
    assert ce == pytest.approx(0.8)


def test_consolidation_efficiency_perfect():
    """All allocated memories consolidated -> CE = 1.0."""
    ce = compute_consolidation_efficiency(n_consolidated=10, n_prp_allocated=10)
    assert ce == pytest.approx(1.0)


def test_consolidation_efficiency_zero_allocated():
    """No allocated memories -> CE = 0.0."""
    ce = compute_consolidation_efficiency(n_consolidated=0, n_prp_allocated=0)
    assert ce == 0.0


# --------------------------------------------------------------------- #
# Tag stats
# --------------------------------------------------------------------- #

def _make_tag(
    tag_type: str = "novelty",
    p: int = 0,
    s: float = 0.5,
    s0: float = 0.6,
    S_score: float = 0.3,
    a: int = 2,
    e0: float = 1.0,
) -> Tag:
    """Helper to create a minimal Tag for testing."""
    return Tag(
        k=torch.randn(128),
        s=s,
        s0=s0,
        s_reinforced=0.0,
        t0=0,
        e0=e0,
        a=a,
        rho=0.0,
        ctx=(0, 10, "test_source"),
        p=p,
        S_score=S_score,
        tag_type=tag_type,
    )


def test_compute_tag_stats_expected_keys():
    """compute_tag_stats should return all expected keys."""
    tags = [_make_tag()]
    stats = compute_tag_stats(tags)
    expected_keys = {
        "n_tags", "n_allocated", "n_novelty", "n_revision",
        "mean_strength", "mean_initial_strength", "mean_score",
        "mean_access_count", "mean_error", "total_accesses",
    }
    assert set(stats.keys()) == expected_keys


def test_compute_tag_stats_counts():
    """Verify correct counting of tag types and allocation."""
    tags = [
        _make_tag(tag_type="novelty", p=1),
        _make_tag(tag_type="novelty", p=0),
        _make_tag(tag_type="revision", p=1),
    ]
    stats = compute_tag_stats(tags)
    assert stats["n_tags"] == 3
    assert stats["n_allocated"] == 2
    assert stats["n_novelty"] == 2
    assert stats["n_revision"] == 1


def test_compute_tag_stats_means():
    """Verify mean calculations."""
    tags = [
        _make_tag(s=0.4, e0=1.0, a=2, S_score=0.2, s0=0.5),
        _make_tag(s=0.6, e0=2.0, a=4, S_score=0.4, s0=0.7),
    ]
    stats = compute_tag_stats(tags)
    assert stats["mean_strength"] == pytest.approx(0.5)
    assert stats["mean_error"] == pytest.approx(1.5)
    assert stats["mean_access_count"] == pytest.approx(3.0)
    assert stats["mean_score"] == pytest.approx(0.3)
    assert stats["mean_initial_strength"] == pytest.approx(0.6)
    assert stats["total_accesses"] == 6


def test_compute_tag_stats_empty():
    """Empty tag list should return zeroed stats."""
    stats = compute_tag_stats([])
    assert stats["n_tags"] == 0
    assert stats["mean_strength"] == 0.0
