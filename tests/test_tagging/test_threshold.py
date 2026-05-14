"""Tests for sleep.tagging.threshold — AdaptiveThreshold z-score flagging."""

import math

import pytest

from sleep.tagging.threshold import AdaptiveThreshold


class TestConstantSurprises:

    def test_mu_converges_to_constant(self):
        """With constant input, EMA mean should converge to that constant."""
        thresh = AdaptiveThreshold(beta=0.95, kappa=1.5)
        constant = 3.0
        surprises = [constant] * 200

        thresh.update_and_flag(surprises)

        assert thresh.mu == pytest.approx(constant, rel=1e-2)

    def test_all_flags_false_for_constant_input(self):
        """Nothing is surprising when every token has the same surprise."""
        thresh = AdaptiveThreshold(beta=0.95, kappa=1.5)
        constant = 3.0
        # Feed enough tokens for the EMA to stabilise
        warmup = [constant] * 200
        thresh.update_and_flag(warmup)

        # Now test a fresh batch of the same constant
        flags = thresh.update_and_flag([constant] * 50)
        assert not any(flags), "Constant surprises should produce no flags"


class TestOutlierDetection:

    def test_spike_is_flagged(self):
        """A large surprise spike after a constant baseline should be flagged."""
        thresh = AdaptiveThreshold(beta=0.99, kappa=1.5)
        baseline = [2.0] * 300
        thresh.update_and_flag(baseline)

        spike_value = 20.0  # far above baseline
        flags = thresh.update_and_flag([spike_value])
        assert flags[0] is True, "A large spike should be flagged"

    def test_baseline_after_spike_is_not_flagged(self):
        """Tokens returning to baseline after a spike should not be flagged."""
        thresh = AdaptiveThreshold(beta=0.99, kappa=1.5)
        baseline = [2.0] * 300
        thresh.update_and_flag(baseline)

        # Spike then return to baseline
        thresh.update_and_flag([20.0])
        flags = thresh.update_and_flag([2.0] * 50)
        # After EMA re-adjusts, baseline tokens should not be flagged
        assert not any(flags[-20:]), "Baseline tokens should not be flagged"


class TestKappaOverride:

    def test_lower_kappa_produces_more_flags(self):
        """Lowering kappa should flag more tokens for the same data."""
        # Use two independent thresholds with the same state
        data = [2.0] * 200 + [4.0, 5.0, 6.0, 3.5, 4.5]

        thresh_high = AdaptiveThreshold(beta=0.99, kappa=2.0)
        thresh_low = AdaptiveThreshold(beta=0.99, kappa=2.0)

        # Warm up identically
        thresh_high.update_and_flag(data[:200])
        thresh_low.update_and_flag(data[:200])

        test_batch = data[200:]
        flags_high_kappa = thresh_high.update_and_flag(test_batch, kappa_override=2.0)
        flags_low_kappa = thresh_low.update_and_flag(test_batch, kappa_override=0.5)

        n_high = sum(flags_high_kappa)
        n_low = sum(flags_low_kappa)
        assert n_low >= n_high, (
            f"Lower kappa should produce >= flags: got {n_low} vs {n_high}"
        )


class TestSigmaFloor:

    def test_sigma_does_not_reach_zero(self):
        """Even with zero-variance input, sigma should stay above epsilon floor."""
        thresh = AdaptiveThreshold(beta=0.99, kappa=1.5)
        # Feed many identical values — variance should approach 0 but the
        # z-score denominator is max(sigma, 1e-8), so no division by zero.
        thresh.update_and_flag([5.0] * 500)

        # sigma itself can get very small, but the z-score computation
        # clamps via max(sigma, 1e-8). Verify sigma is a finite number.
        assert math.isfinite(thresh.sigma)
        # The z-score path uses max(sigma, 1e-8) so verify no flag is raised
        # (which would happen if sigma were truly zero and z blew up).
        flags = thresh.update_and_flag([5.0])
        assert flags[0] is False

    def test_sigma_remains_positive_after_constant_input(self):
        """sigma should not literally become negative."""
        thresh = AdaptiveThreshold(beta=0.99, kappa=1.5)
        thresh.update_and_flag([1.0] * 1000)
        assert thresh.sigma >= 0.0
