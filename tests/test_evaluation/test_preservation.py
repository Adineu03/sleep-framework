"""Tests for Base Capability Preservation (Module 6 — preservation.py)."""

import pytest

from sleep.evaluation.preservation import compute_bcp, check_degradation


# --------------------------------------------------------------------- #
# compute_bcp
# --------------------------------------------------------------------- #

def test_bcp_no_degradation():
    """ppl_current == ppl_original -> BCP = 1.0."""
    bcp = compute_bcp(ppl_current=10.0, ppl_original=10.0)
    assert bcp == pytest.approx(1.0)


def test_bcp_slight_degradation():
    """ppl_current=11, ppl_original=10 -> BCP = 1.1."""
    bcp = compute_bcp(ppl_current=11.0, ppl_original=10.0)
    assert bcp == pytest.approx(1.1)


def test_bcp_improvement():
    """ppl_current < ppl_original -> BCP < 1.0."""
    bcp = compute_bcp(ppl_current=9.0, ppl_original=10.0)
    assert bcp == pytest.approx(0.9)


def test_bcp_zero_original_raises():
    """ppl_original=0 -> ValueError."""
    with pytest.raises(ValueError):
        compute_bcp(ppl_current=10.0, ppl_original=0.0)


def test_bcp_negative_original_raises():
    """ppl_original < 0 -> ValueError."""
    with pytest.raises(ValueError):
        compute_bcp(ppl_current=10.0, ppl_original=-1.0)


# --------------------------------------------------------------------- #
# check_degradation
# --------------------------------------------------------------------- #

def test_check_degradation_not_degraded():
    """BCP=1.03 is within default threshold 1.05 -> not degraded."""
    degraded, message = check_degradation(bcp=1.03)
    assert degraded is False
    assert "within threshold" in message.lower() or "bcp=" in message.lower()


def test_check_degradation_degraded():
    """BCP=1.06 exceeds default threshold 1.05 -> degraded."""
    degraded, message = check_degradation(bcp=1.06)
    assert degraded is True
    assert "degradation" in message.lower()


def test_check_degradation_exactly_at_threshold():
    """BCP exactly equal to threshold (1.05) -> not degraded (uses >)."""
    degraded, _ = check_degradation(bcp=1.05)
    assert degraded is False


def test_check_degradation_custom_threshold():
    """Custom threshold of 1.10 -> BCP=1.08 is not degraded."""
    degraded, _ = check_degradation(bcp=1.08, threshold=1.10)
    assert degraded is False
