"""Tests for sleep.tagging.buffer — TagBuffer capacity, decay, reinforcement, GC."""

import math

import pytest
import torch

from sleep.config import TaggingConfig
from sleep.tagging.tags import Tag
from sleep.tagging.buffer import TagBuffer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tag(
    d_tag: int = 64,
    s: float = 0.8,
    s0: float = 0.8,
    e0: float = 1.0,
    t0: int = 0,
    rho: float = 0.0,
    a: int = 0,
    s_reinforced: float = 0.0,
    key: torch.Tensor | None = None,
) -> Tag:
    """Create a Tag with sensible defaults for testing."""
    k = key if key is not None else torch.randn(d_tag)
    return Tag(
        k=k, s=s, s0=s0, s_reinforced=s_reinforced,
        t0=t0, e0=e0, a=a, rho=rho,
        ctx=(0, 10, "test"),
        R_last_update=t0,
    )


# ---------------------------------------------------------------------------
# Add and capacity eviction
# ---------------------------------------------------------------------------

class TestAddAndCapacity:

    def test_add_within_capacity(self):
        """Adding fewer tags than n_max should keep all of them."""
        config = TaggingConfig()
        buf = TagBuffer(config=config, n_max=10)
        tags = [_make_tag() for _ in range(5)]
        buf.add(tags)
        assert buf.n_active == 5

    def test_eviction_removes_lowest_priority(self):
        """When exceeding n_max, lowest s*(1+rho) tags are evicted."""
        config = TaggingConfig()
        buf = TagBuffer(config=config, n_max=3)

        # Add 5 tags with distinct priorities s*(1+rho)
        tags = [
            _make_tag(s=0.1, rho=0.0),  # priority = 0.1
            _make_tag(s=0.5, rho=0.0),  # priority = 0.5
            _make_tag(s=0.9, rho=0.0),  # priority = 0.9
            _make_tag(s=0.3, rho=1.0),  # priority = 0.6
            _make_tag(s=0.8, rho=1.0),  # priority = 1.6
        ]
        buf.add(tags)

        assert buf.n_active == 3
        # Surviving tags should be the top-3 by priority: 1.6, 0.9, 0.6
        surviving_priorities = sorted(
            [t.s * (1.0 + t.rho) for t in buf.tags], reverse=True
        )
        assert surviving_priorities[0] == pytest.approx(1.6, rel=1e-5)
        assert surviving_priorities[1] == pytest.approx(0.9, rel=1e-5)
        assert surviving_priorities[2] == pytest.approx(0.6, rel=1e-5)

    def test_occupancy(self):
        """occupancy should reflect current fill fraction."""
        config = TaggingConfig()
        buf = TagBuffer(config=config, n_max=10)
        buf.add([_make_tag() for _ in range(7)])
        assert buf.occupancy == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Decay and garbage collection
# ---------------------------------------------------------------------------

class TestDecayAndGC:

    def test_decay_follows_exponential_formula(self):
        """Strength should follow s_base = (s0 - eps)*exp(-dt/tau_decay) + eps."""
        config = TaggingConfig(
            tau_base=1000, gamma_decay=0.5, epsilon=0.01, epsilon_gc=0.001,
        )
        buf = TagBuffer(config=config, n_max=100)

        s0 = 0.9
        e0 = 2.0
        t0 = 0
        tag = _make_tag(s=s0, s0=s0, e0=e0, t0=t0)
        buf.add([tag])

        current_step = 500
        buf.decay_and_gc(current_step)

        tau_decay = config.tau_base * (1.0 + config.gamma_decay * e0)
        dt = current_step - t0
        expected_s = (s0 - config.epsilon) * math.exp(-dt / tau_decay) + config.epsilon

        surviving = buf.tags
        assert len(surviving) == 1
        assert surviving[0].s == pytest.approx(expected_s, rel=1e-6)

    def test_decay_with_reinforcement(self):
        """s = min(s_base + s_reinforced, 1.0) when tag has been reinforced."""
        config = TaggingConfig(
            tau_base=1000, gamma_decay=0.5, epsilon=0.01, epsilon_gc=0.001,
        )
        buf = TagBuffer(config=config, n_max=100)

        s0 = 0.5
        e0 = 1.0
        s_reinforced = 0.3
        tag = _make_tag(s=s0, s0=s0, e0=e0, t0=0, s_reinforced=s_reinforced)
        buf.add([tag])

        current_step = 200
        buf.decay_and_gc(current_step)

        tau_decay = config.tau_base * (1.0 + config.gamma_decay * e0)
        s_base = (s0 - config.epsilon) * math.exp(-current_step / tau_decay) + config.epsilon
        expected = min(s_base + s_reinforced, 1.0)

        assert buf.tags[0].s == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Reinforcement via process_query
# ---------------------------------------------------------------------------

class TestReinforcement:

    def test_process_query_increases_access_and_rho(self):
        """Accessing a tag should increment a, rho, and s_reinforced."""
        config = TaggingConfig(theta_access=0.0, delta_s=0.3)  # theta=0 → always access
        buf = TagBuffer(config=config, n_max=100)

        key = torch.randn(64)
        key = key / key.norm()  # unit vector
        tag = _make_tag(s=0.5, s0=0.5, key=key.clone())
        buf.add([tag])

        # Query with the same key → cosine sim ≈ 1.0 (well above theta=0)
        accessed = buf.process_query(key.clone(), current_step=10)

        assert len(accessed) == 1
        assert accessed[0].a == 1
        assert accessed[0].rho > 0.0
        assert accessed[0].s_reinforced > 0.0
        assert accessed[0].s > 0.5  # strength increased

    def test_no_access_below_threshold(self):
        """Tags whose cosine similarity is below theta_access should not be accessed."""
        config = TaggingConfig(theta_access=0.99)  # very high threshold
        buf = TagBuffer(config=config, n_max=100)

        tag = _make_tag(key=torch.randn(64))
        buf.add([tag])

        # Random query key — unlikely to have cosine > 0.99
        query = torch.randn(64)
        accessed = buf.process_query(query, current_step=10)
        assert len(accessed) == 0
        assert buf.tags[0].a == 0


# ---------------------------------------------------------------------------
# Diminishing returns (sqrt(n) scaling)
# ---------------------------------------------------------------------------

class TestDiminishingReturns:

    def test_rho_grows_sublinearly(self):
        """After many accesses, rho should grow as ~sqrt(n), not linearly."""
        config = TaggingConfig(theta_access=0.0, delta_s=0.3)
        buf = TagBuffer(config=config, n_max=100)

        key = torch.randn(64)
        key = key / key.norm()
        tag = _make_tag(s=0.5, s0=0.5, key=key.clone())
        buf.add([tag])

        n_accesses = 100
        for i in range(n_accesses):
            buf.process_query(key.clone(), current_step=i + 1)

        final_rho = buf.tags[0].rho
        final_a = buf.tags[0].a
        assert final_a == n_accesses

        # rho = sum_{i=1}^{n} sim / sqrt(i)
        # For sim ~ 1.0, rho ~ sum 1/sqrt(i) ~ 2*sqrt(n)
        # Linear would be ~ n. Check rho << n.
        assert final_rho < n_accesses * 0.5, (
            f"rho={final_rho} should be much less than {n_accesses} (sublinear growth)"
        )


# ---------------------------------------------------------------------------
# GC removes dead tags
# ---------------------------------------------------------------------------

class TestGCRemovesDeadTags:

    def test_gc_removes_tag_after_sufficient_decay(self):
        """A tag with low e0 should decay below epsilon_gc and be removed."""
        config = TaggingConfig(
            tau_base=100,
            gamma_decay=0.0,   # tau_decay = tau_base (no error bonus)
            epsilon=0.01,
            epsilon_gc=0.02,
        )
        buf = TagBuffer(config=config, n_max=100)

        # s0 just above epsilon — will decay to epsilon quickly
        tag = _make_tag(s=0.05, s0=0.05, e0=0.1, t0=0)
        buf.add([tag])
        assert buf.n_active == 1

        # Advance far enough that s_base ≈ epsilon < epsilon_gc
        # s_base = (0.05 - 0.01)*exp(-dt/100) + 0.01
        # We need s_base < 0.02 → (0.04)*exp(-dt/100) < 0.01
        # → exp(-dt/100) < 0.25 → dt > 100*ln(4) ≈ 139
        removed = buf.decay_and_gc(current_step=200)

        assert removed == 1
        assert buf.n_active == 0

    def test_gc_keeps_strong_tags(self):
        """Tags with high initial strength should survive GC."""
        config = TaggingConfig(
            tau_base=1000, gamma_decay=0.5,
            epsilon=0.01, epsilon_gc=0.02,
        )
        buf = TagBuffer(config=config, n_max=100)

        tag = _make_tag(s=0.95, s0=0.95, e0=2.0, t0=0)
        buf.add([tag])

        # Modest time advance — strong tag should survive
        removed = buf.decay_and_gc(current_step=100)
        assert removed == 0
        assert buf.n_active == 1
