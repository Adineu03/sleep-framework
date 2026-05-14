"""Tests for sleep.evaluation.calibration.

Pure-Python tests using synthetic per-fact dicts shaped like the output of
``multiple_choice_recall``. No model is needed.
"""

import pytest

from sleep.evaluation.calibration import (
    compute_calibration_metrics,
    compute_stratified_calibration,
    format_calibration_table,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_per_fact(
    fact_id: str,
    correct_letter: str,
    predicted_letter: str,
    option_probs: dict,
) -> dict:
    """Build a per-fact dict matching the multiple_choice_recall schema."""
    return {
        "fact_id": fact_id,
        "correct_letter": correct_letter,
        "predicted_letter": predicted_letter,
        "is_correct": (correct_letter == predicted_letter),
        "option_probs": option_probs,
        "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
    }


def _uniform() -> dict:
    return {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}


def _peaked(letter: str, p: float) -> dict:
    """Probabilities heavily concentrated on `letter`, rest uniform."""
    others = (1.0 - p) / 3.0
    return {L: (p if L == letter else others) for L in ("A", "B", "C", "D")}


# ---------------------------------------------------------------------------
# compute_calibration_metrics
# ---------------------------------------------------------------------------

class TestComputeCalibrationMetrics:

    def test_empty_input_returns_empty_metrics(self):
        m = compute_calibration_metrics([])
        assert m["n_facts"] == 0
        # All other documented keys present
        for key in (
            "accuracy",
            "mean_correct_prob",
            "mean_predicted_prob",
            "mean_correct_when_wrong",
            "mean_correct_when_right",
            "ece",
            "brier_score",
        ):
            assert key in m
            # Each is None for empty input
            assert m[key] is None

    def test_perfect_predictions(self):
        """All correct, all p(correct)=1.0 -> ECE=0, accuracy=1.0, brier=0."""
        per_fact = [
            _make_per_fact("f1", "A", "A", _peaked("A", 1.0)),
            _make_per_fact("f2", "B", "B", _peaked("B", 1.0)),
            _make_per_fact("f3", "C", "C", _peaked("C", 1.0)),
        ]
        m = compute_calibration_metrics(per_fact)
        assert m["accuracy"] == pytest.approx(1.0)
        assert m["mean_correct_prob"] == pytest.approx(1.0)
        assert m["ece"] == pytest.approx(0.0, abs=1e-9)
        assert m["brier_score"] == pytest.approx(0.0, abs=1e-9)
        # No wrong facts -> mean_correct_when_wrong is None
        assert m["mean_correct_when_wrong"] is None
        assert m["mean_correct_when_right"] == pytest.approx(1.0)

    def test_random_predictions(self):
        """Uniform option_probs -> mean_correct_prob ≈ 0.25."""
        per_fact = [
            _make_per_fact("f1", "A", "A", _uniform()),
            _make_per_fact("f2", "B", "A", _uniform()),
            _make_per_fact("f3", "C", "A", _uniform()),
            _make_per_fact("f4", "D", "A", _uniform()),
        ]
        m = compute_calibration_metrics(per_fact)
        assert m["mean_correct_prob"] == pytest.approx(0.25)
        # Predicted (argmax) is always A in this construction with prob 0.25.
        assert m["mean_predicted_prob"] == pytest.approx(0.25)

    def test_mean_correct_when_wrong_and_right(self):
        """2 right (p_correct=0.7), 2 wrong (p_correct=0.3)."""
        per_fact = [
            # Right: predict A, correct A, p(A)=0.7
            _make_per_fact("r1", "A", "A", _peaked("A", 0.7)),
            _make_per_fact("r2", "A", "A", _peaked("A", 0.7)),
            # Wrong: predict B, correct A, p(A)=0.3.
            # Build dict so B has highest mass and A has 0.3.
            _make_per_fact("w1", "A", "B",
                           {"A": 0.3, "B": 0.5, "C": 0.1, "D": 0.1}),
            _make_per_fact("w2", "A", "B",
                           {"A": 0.3, "B": 0.5, "C": 0.1, "D": 0.1}),
        ]
        m = compute_calibration_metrics(per_fact)
        assert m["mean_correct_when_right"] == pytest.approx(0.7)
        assert m["mean_correct_when_wrong"] == pytest.approx(0.3)
        assert m["accuracy"] == pytest.approx(0.5)

    def test_brier_score_known(self):
        """2 facts: one right p=0.8, one wrong p=0.2.
        Brier = ((0.8-1)^2 + (0.2-0)^2)/2 = 0.04.
        """
        per_fact = [
            # Right: correct A, predict A, p(A)=0.8
            _make_per_fact("r", "A", "A", _peaked("A", 0.8)),
            # Wrong: correct A (p=0.2), predict B (p=0.5).
            _make_per_fact("w", "A", "B",
                           {"A": 0.2, "B": 0.5, "C": 0.15, "D": 0.15}),
        ]
        m = compute_calibration_metrics(per_fact)
        assert m["brier_score"] == pytest.approx(0.04, abs=1e-9)

    def test_ece_perfect_calibration(self):
        """Confidence ≈ accuracy in each bin -> ECE near 0.

        Construct: 10 facts where confidence is 1.0 and all are correct.
        That puts all mass in the highest bin with conf=1.0, acc=1.0 -> ECE=0.
        """
        per_fact = [
            _make_per_fact(f"f{i}", "A", "A", _peaked("A", 1.0))
            for i in range(10)
        ]
        m = compute_calibration_metrics(per_fact)
        assert m["ece"] == pytest.approx(0.0, abs=1e-9)

    def test_ece_overconfident(self):
        """All wrong but model assigns 0.9 to its choice -> high ECE.

        For each fact: model picks B with p(B)=0.9, but correct is A.
        Confidence = p(predicted) = 0.9, accuracy = 0.0 in that bin.
        ECE = |0.9 - 0.0| = 0.9.
        """
        per_fact = [
            _make_per_fact(
                f"f{i}", "A", "B",
                {"A": 0.05, "B": 0.9, "C": 0.025, "D": 0.025},
            )
            for i in range(10)
        ]
        m = compute_calibration_metrics(per_fact)
        # Confidence ~0.9, accuracy 0 -> ECE close to 0.9.
        assert m["ece"] == pytest.approx(0.9, abs=1e-6)


# ---------------------------------------------------------------------------
# compute_stratified_calibration
# ---------------------------------------------------------------------------

class TestStratifiedCalibration:

    def test_groups_correctly(self):
        per_fact = [
            _make_per_fact("f1", "A", "A", _peaked("A", 0.8)),
            _make_per_fact("f2", "A", "A", _peaked("A", 0.7)),
            _make_per_fact("f3", "A", "B",
                           {"A": 0.3, "B": 0.5, "C": 0.1, "D": 0.1}),
            _make_per_fact("f4", "A", "B",
                           {"A": 0.2, "B": 0.6, "C": 0.1, "D": 0.1}),
        ]
        groups = {"f1": "G1", "f2": "G1", "f3": "G2", "f4": "G2"}
        out = compute_stratified_calibration(per_fact, groups)

        assert set(out.keys()) == {"G1", "G2"}
        assert out["G1"]["n_facts"] == 2
        assert out["G2"]["n_facts"] == 2
        # G1 is the all-right group.
        assert out["G1"]["accuracy"] == pytest.approx(1.0)
        # G2 is the all-wrong group.
        assert out["G2"]["accuracy"] == pytest.approx(0.0)

    def test_unknown_fact_id_skipped(self):
        """fact_id missing from group_assignments doesn't crash; just skipped."""
        per_fact = [
            _make_per_fact("known", "A", "A", _peaked("A", 0.9)),
            _make_per_fact("unknown", "A", "B",
                           {"A": 0.1, "B": 0.7, "C": 0.1, "D": 0.1}),
        ]
        groups = {"known": "G1"}
        out = compute_stratified_calibration(per_fact, groups)
        assert "G1" in out
        assert out["G1"]["n_facts"] == 1
        # 'unknown' fact was excluded; no second group should appear.
        assert set(out.keys()) == {"G1"}

    def test_empty_group_returns_nones(self):
        """A group named in assignments with no matching facts returns Nones."""
        per_fact = [
            _make_per_fact("f1", "A", "A", _peaked("A", 0.8)),
        ]
        # 'phantom' group is in assignments but has no fact pointing at it.
        groups = {"f1": "G1", "ghost_id_not_in_per_fact": "phantom"}
        out = compute_stratified_calibration(per_fact, groups)
        assert "phantom" in out
        assert out["phantom"]["n_facts"] == 0
        # All metric values None.
        for key in (
            "accuracy", "mean_correct_prob", "mean_predicted_prob",
            "mean_correct_when_wrong", "mean_correct_when_right",
            "ece", "brier_score",
        ):
            assert out["phantom"][key] is None

    def test_all_facts_one_group(self):
        """All facts in one group -> stratified result == unstratified."""
        per_fact = [
            _make_per_fact("f1", "A", "A", _peaked("A", 0.9)),
            _make_per_fact("f2", "B", "B", _peaked("B", 0.6)),
            _make_per_fact("f3", "C", "A",
                           {"A": 0.4, "B": 0.2, "C": 0.3, "D": 0.1}),
        ]
        groups = {"f1": "ALL", "f2": "ALL", "f3": "ALL"}
        strat = compute_stratified_calibration(per_fact, groups)
        flat = compute_calibration_metrics(per_fact)
        for key in (
            "n_facts",
            "accuracy",
            "mean_correct_prob",
            "mean_predicted_prob",
            "ece",
            "brier_score",
        ):
            if flat[key] is None:
                assert strat["ALL"][key] is None
            else:
                assert strat["ALL"][key] == pytest.approx(flat[key])


# ---------------------------------------------------------------------------
# format_calibration_table
# ---------------------------------------------------------------------------

class TestFormatTable:

    def _example_metrics(self):
        return {
            "GroupA": compute_calibration_metrics([
                _make_per_fact("a1", "A", "A", _peaked("A", 0.9)),
                _make_per_fact("a2", "B", "B", _peaked("B", 0.6)),
            ]),
            "GroupB": compute_calibration_metrics([
                _make_per_fact("b1", "A", "B",
                               {"A": 0.3, "B": 0.5, "C": 0.1, "D": 0.1}),
            ]),
        }

    def test_includes_all_groups(self):
        table = format_calibration_table(self._example_metrics())
        assert "GroupA" in table
        assert "GroupB" in table

    def test_includes_metric_columns(self):
        table = format_calibration_table(self._example_metrics())
        assert "Acc" in table
        assert "ECE" in table
        assert "Brier" in table

    def test_handles_none_values(self):
        """A group with all-None metrics renders without crashing.

        Each None cell should appear as a dash. We only assert the function
        returns a non-empty string and contains the group name.
        """
        metrics = {
            "Empty": {
                "accuracy": None,
                "mean_correct_prob": None,
                "mean_predicted_prob": None,
                "mean_correct_when_wrong": None,
                "mean_correct_when_right": None,
                "ece": None,
                "brier_score": None,
                "n_facts": 0,
            },
        }
        table = format_calibration_table(metrics)
        assert "Empty" in table
        assert "-" in table  # at least one None rendered as dash
        # Should be multiple lines (header + separator + 1 row).
        assert len(table.splitlines()) >= 3

    def test_n_facts_displayed(self):
        """Counts are present in the table."""
        metrics = self._example_metrics()
        table = format_calibration_table(metrics)
        # GroupA has 2 facts, GroupB has 1.
        # Just check both numerals appear somewhere in the rendered table.
        assert " 2 " in table or "2\n" in table or "2 " in table
        assert " 1 " in table or "1\n" in table or "1 " in table
