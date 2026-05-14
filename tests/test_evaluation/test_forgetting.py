"""Tests for forgetting curve analysis (Module 6 — forgetting.py)."""

import pytest

from sleep.evaluation.forgetting import classify_forgetting, compute_forgetting_curve


# --------------------------------------------------------------------- #
# classify_forgetting
# --------------------------------------------------------------------- #

def test_classify_stable():
    """relative_forgetting < 0.1 -> 'stable'."""
    assert classify_forgetting(0.05, n_cycles=10) == "stable"


def test_classify_gradual():
    """0.1 <= relative_forgetting < 0.5 -> 'gradual'."""
    assert classify_forgetting(0.3, n_cycles=50) == "gradual"


def test_classify_catastrophic():
    """relative_forgetting >= 0.5 -> 'catastrophic'."""
    assert classify_forgetting(0.6, n_cycles=5) == "catastrophic"


# --------------------------------------------------------------------- #
# compute_forgetting_curve: stable
# --------------------------------------------------------------------- #

def test_stable_curve():
    """Minor recall decline -> curve_type = 'stable'."""
    history = [(0, 0.9), (10, 0.88), (50, 0.85)]
    result = compute_forgetting_curve(history)
    assert result["curve_type"] == "stable"
    assert result["initial_recall"] == 0.9
    assert result["final_recall"] == 0.85
    assert result["absolute_forgetting"] == pytest.approx(0.05, abs=1e-6)
    # relative = 0.05 / 0.9 ≈ 0.0556 < 0.1 -> stable
    assert result["relative_forgetting"] < 0.1


# --------------------------------------------------------------------- #
# compute_forgetting_curve: catastrophic
# --------------------------------------------------------------------- #

def test_catastrophic_curve():
    """Large recall drop -> curve_type = 'catastrophic'."""
    history = [(0, 0.9), (5, 0.3)]
    result = compute_forgetting_curve(history)
    assert result["curve_type"] == "catastrophic"
    # relative = (0.9 - 0.3) / 0.9 ≈ 0.667
    assert result["relative_forgetting"] >= 0.5


# --------------------------------------------------------------------- #
# compute_forgetting_curve: gradual
# --------------------------------------------------------------------- #

def test_gradual_curve():
    """Moderate recall decline -> curve_type = 'gradual'."""
    history = [(0, 0.9), (50, 0.55)]
    result = compute_forgetting_curve(history)
    assert result["curve_type"] == "gradual"
    # relative = (0.9 - 0.55) / 0.9 ≈ 0.389
    assert 0.1 <= result["relative_forgetting"] < 0.5


# --------------------------------------------------------------------- #
# Half-life computation
# --------------------------------------------------------------------- #

def test_half_life_when_reached():
    """DRA drops below 50% of initial -> half_life is that cycle."""
    history = [(0, 1.0), (3, 0.6), (7, 0.4)]
    result = compute_forgetting_curve(history)
    # 50% of 1.0 = 0.5; first cycle where DRA <= 0.5 is cycle 7
    assert result["half_life"] == 7


def test_half_life_never_reached():
    """DRA never drops below 50% of initial -> half_life is None."""
    history = [(0, 1.0), (10, 0.8), (50, 0.7)]
    result = compute_forgetting_curve(history)
    assert result["half_life"] is None


# --------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------- #

def test_empty_history_raises():
    """Empty recall_history -> ValueError."""
    with pytest.raises(ValueError):
        compute_forgetting_curve([])


def test_single_entry():
    """Single entry: no forgetting, stable."""
    result = compute_forgetting_curve([(0, 0.9)])
    assert result["absolute_forgetting"] == 0.0
    assert result["curve_type"] == "stable"
