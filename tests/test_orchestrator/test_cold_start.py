"""Tests for cold-start calibration (Module 5 — cold_start.py)."""

import pytest

from sleep.config import ColdStartConfig
from sleep.orchestrator.cold_start import ColdStartManager


@pytest.fixture
def config() -> ColdStartConfig:
    """Default ColdStartConfig: kappa_cold=3.0, n_burnin=50, n_ramp=50, n_mature=500."""
    return ColdStartConfig()


@pytest.fixture
def manager(config) -> ColdStartManager:
    """ColdStartManager with normal_kappa=1.5."""
    return ColdStartManager(config=config, normal_kappa=1.5)


# --------------------------------------------------------------------- #
# Kappa calibration
# --------------------------------------------------------------------- #

def test_burnin_phase_kappa(manager):
    """During burn-in (interaction 0), kappa should equal kappa_cold (3.0)."""
    kappa = manager.get_effective_kappa(0)
    assert kappa == 3.0


def test_ramp_phase_midpoint_kappa(manager, config):
    """At the midpoint of the ramp, kappa should be between cold and normal."""
    midpoint = config.n_burnin + config.n_ramp // 2  # 50 + 25 = 75
    kappa = manager.get_effective_kappa(midpoint)
    # Should be between 1.5 (normal) and 3.0 (cold)
    assert 1.5 < kappa < 3.0
    # At midpoint of ramp: progress=0.5, kappa = 3.0 + 0.5*(1.5-3.0) = 2.25
    assert kappa == pytest.approx(2.25, abs=0.01)


def test_mature_phase_kappa(manager, config):
    """After ramp completes (interaction >= n_burnin+n_ramp), kappa = normal."""
    interaction = config.n_burnin + config.n_ramp  # 100
    kappa = manager.get_effective_kappa(interaction)
    assert kappa == 1.5

    # Well beyond ramp
    kappa = manager.get_effective_kappa(200)
    assert kappa == 1.5


# --------------------------------------------------------------------- #
# Budget scaling
# --------------------------------------------------------------------- #

def test_budget_scale_at_zero(manager):
    """At interaction 0, budget_scale should be 0.0."""
    scale = manager.get_budget_scale(0)
    assert scale == 0.0


def test_budget_scale_at_mature(manager, config):
    """At interaction n_mature, budget_scale should be 1.0."""
    scale = manager.get_budget_scale(config.n_mature)
    assert scale == 1.0


def test_budget_scale_midpoint(manager, config):
    """At half of n_mature, budget_scale should be 0.5."""
    scale = manager.get_budget_scale(config.n_mature // 2)
    assert scale == pytest.approx(0.5, abs=0.01)


# --------------------------------------------------------------------- #
# is_mature property
# --------------------------------------------------------------------- #

def test_is_mature_false_initially(manager):
    """System is not mature at start."""
    assert manager.is_mature is False


def test_is_mature_true_after_n_mature(manager, config):
    """System is mature after n_mature interactions."""
    manager.interaction_count = config.n_mature
    assert manager.is_mature is True
