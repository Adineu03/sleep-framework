"""Tests for sleep.prp.crossref -- cross-reference density computation."""

from __future__ import annotations

import torch
import pytest

from sleep.tagging.tags import Tag
from sleep.prp.crossref import compute_cross_references


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tag(
    *,
    k: torch.Tensor | None = None,
    d_tag: int = 128,
    e0: float = 1.0,
    s: float = 0.8,
    s0: float = 0.9,
    rho: float = 0.0,
    a: int = 0,
    R: float = 0.0,
    xref_count: int = 0,
    tag_type: str = "novelty",
    p: int = 0,
    S_score: float = 0.0,
) -> Tag:
    """Create a Tag with sensible defaults for testing."""
    if k is None:
        k = torch.randn(d_tag)
    return Tag(
        k=k, s=s, s0=s0, s_reinforced=0.0, t0=0, e0=e0,
        a=a, rho=rho, ctx=(0, 10, "test"), p=p, S_score=S_score,
        R=R, xref_count=xref_count, tag_type=tag_type,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestComputeCrossReferences:
    """Tests for compute_cross_references."""

    def test_identical_keys_xref_count(self):
        """Two tags with identical keys should each have xref_count = 1."""
        shared_key = torch.randn(128)
        tag_a = make_tag(k=shared_key.clone())
        tag_b = make_tag(k=shared_key.clone())
        tags = [tag_a, tag_b]

        compute_cross_references(tags, theta_xref=0.5)

        # Identical keys -> cosine similarity = 1.0 > 0.5
        assert tag_a.xref_count == 1
        assert tag_b.xref_count == 1

    def test_orthogonal_keys_xref_count(self):
        """Two tags with orthogonal keys should each have xref_count = 0."""
        # Construct exactly orthogonal vectors
        k1 = torch.zeros(128)
        k1[0] = 1.0
        k2 = torch.zeros(128)
        k2[1] = 1.0

        tag_a = make_tag(k=k1)
        tag_b = make_tag(k=k2)
        tags = [tag_a, tag_b]

        compute_cross_references(tags, theta_xref=0.5)

        assert tag_a.xref_count == 0
        assert tag_b.xref_count == 0

    def test_cluster_structure(self):
        """4 tags in 2 clusters: within-cluster similarity high, between-cluster low."""
        # Cluster A: keys dominated by first half of dimensions
        base_a = torch.zeros(128)
        base_a[:64] = 1.0
        k_a1 = base_a + torch.randn(128) * 0.05
        k_a2 = base_a + torch.randn(128) * 0.05

        # Cluster B: keys dominated by second half of dimensions
        base_b = torch.zeros(128)
        base_b[64:] = 1.0
        k_b1 = base_b + torch.randn(128) * 0.05
        k_b2 = base_b + torch.randn(128) * 0.05

        tags = [
            make_tag(k=k_a1),
            make_tag(k=k_a2),
            make_tag(k=k_b1),
            make_tag(k=k_b2),
        ]

        compute_cross_references(tags, theta_xref=0.5)

        # Each tag should cross-reference its cluster-mate (count=1)
        # but NOT the tags in the other cluster.
        assert tags[0].xref_count == 1, f"Cluster A tag 0: expected 1, got {tags[0].xref_count}"
        assert tags[1].xref_count == 1, f"Cluster A tag 1: expected 1, got {tags[1].xref_count}"
        assert tags[2].xref_count == 1, f"Cluster B tag 0: expected 1, got {tags[2].xref_count}"
        assert tags[3].xref_count == 1, f"Cluster B tag 1: expected 1, got {tags[3].xref_count}"

    def test_single_tag_xref_zero(self):
        """A single tag should have xref_count = 0 (no pairs to compare)."""
        tag = make_tag()
        compute_cross_references([tag], theta_xref=0.5)
        assert tag.xref_count == 0

    def test_empty_list_no_crash(self):
        """Empty input should return without error."""
        compute_cross_references([], theta_xref=0.5)  # should not raise
