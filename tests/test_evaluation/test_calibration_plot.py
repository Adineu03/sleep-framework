"""Tests for sleep.evaluation.calibration_plot."""

from __future__ import annotations

import os

import pytest

from sleep.evaluation.calibration_plot import (
    plot_tagged_vs_untagged_calibration,
    reliability_bins,
)


def _fact(fact_id, correct, predicted, probs):
    return {
        "fact_id": fact_id,
        "correct_letter": correct,
        "predicted_letter": predicted,
        "is_correct": correct == predicted,
        "option_probs": probs,
    }


def _sample(n, correct_prob, seedbase):
    # Deterministic per-fact dicts; correct option carries `correct_prob`.
    facts = []
    for i in range(n):
        rest = (1.0 - correct_prob) / 3
        probs = {"A": correct_prob, "B": rest, "C": rest, "D": rest}
        predicted = "A" if (i + seedbase) % 2 == 0 else "B"
        facts.append(_fact(f"f{seedbase}_{i}", "A", predicted, probs))
    return facts


class TestReliabilityBins:

    def test_counts_sum_to_n(self):
        conf = [0.1, 0.35, 0.35, 0.9]
        correct = [0, 1, 0, 1]
        _, _, count = reliability_bins(conf, correct, n_bins=10)
        assert sum(count) == 4

    def test_conf_one_lands_in_last_bin(self):
        _, _, count = reliability_bins([1.0], [1], n_bins=10)
        assert count[-1] == 1


class TestPlot:

    def test_writes_pdf_and_png(self, tmp_path):
        tagged = _sample(20, 0.40, 0)
        untagged = _sample(20, 0.24, 1)
        out = str(tmp_path / "cal.pdf")
        path = plot_tagged_vs_untagged_calibration(tagged, untagged, out)
        assert path == out
        assert os.path.exists(out)
        assert os.path.exists(out[:-4] + ".png")
        assert os.path.getsize(out) > 0

    def test_writes_png_only(self, tmp_path):
        tagged = _sample(10, 0.5, 0)
        out = str(tmp_path / "cal.png")
        plot_tagged_vs_untagged_calibration(tagged, [], out)
        assert os.path.exists(out)

    def test_both_empty_raises(self, tmp_path):
        with pytest.raises(ValueError):
            plot_tagged_vs_untagged_calibration([], [], str(tmp_path / "x.pdf"))
