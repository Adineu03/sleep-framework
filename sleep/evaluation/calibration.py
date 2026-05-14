"""
Calibration metrics for SLEEP multiple-choice recall.

Multiple-choice accuracy answers a single binary question per fact: did the
model pick the right option? But two models with the same accuracy can carry
very different amounts of latent encoding signal. A model that guesses at 25%
and is uniformly diffuse (~0.25 on every option) has effectively no memory.
A model that guesses at 25% but assigns 0.40 to the correct option even when
it picks something else is *encoding the trace* — it just isn't acting on it.
This module surfaces that distinction.

Three families of metric are computed from the ``option_probs`` field produced
by :func:`sleep.evaluation.recall_formats.multiple_choice_recall`:

  1. **Mean correct-option probability**, overall and split by whether the
     model's hard prediction was right or wrong. ``mean_correct_when_wrong``
     is the most diagnostic for SLEEP — it isolates encoding signal that
     hasn't yet won the argmax.
  2. **Expected Calibration Error (ECE)** with 10 equal-width bins, measuring
     the gap between the model's stated confidence and its actual accuracy.
  3. **Brier score** of the correct-option probability against binary
     correctness, a proper scoring rule that rewards both calibration and
     resolution.

A stratified variant slices these by group (e.g. Consolidated / Failed /
TaggedNoPRP / Untagged) so that downstream analyses can compare the
encoding-vs-action gap across the SLEEP pipeline stages.
"""

from __future__ import annotations

import math
from typing import Optional

from sleep.utils.logging import get_logger

logger = get_logger("sleep.evaluation.calibration")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ECE_N_BINS = 10
_METRIC_KEYS = (
    "accuracy",
    "mean_correct_prob",
    "mean_predicted_prob",
    "mean_correct_when_wrong",
    "mean_correct_when_right",
    "ece",
    "brier_score",
    "n_facts",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_metrics() -> dict[str, Optional[float]]:
    """Return a metrics dict with all values set to ``None`` (n_facts=0).

    Used as the safe default for empty groups so that downstream consumers
    can always index into the same key set.
    """
    out: dict[str, Optional[float]] = {k: None for k in _METRIC_KEYS}
    out["n_facts"] = 0
    return out


def _expected_calibration_error(
    confidences: list[float],
    correctness: list[int],
    n_bins: int = _ECE_N_BINS,
) -> float:
    """Compute Expected Calibration Error with equal-width bins on [0, 1].

    Bins are ``[0, 1/n_bins), [1/n_bins, 2/n_bins), ..., [(n-1)/n_bins, 1]``
    (the final bin is closed on the right so that ``conf == 1.0`` lands in
    bin ``n_bins - 1`` rather than overflowing).

    Args:
        confidences: Model confidence on its argmax choice, one per fact.
        correctness: Binary correctness (0/1) for the same facts.
        n_bins:      Number of equal-width bins (default 10).

    Returns:
        ECE = sum_{b} (|b| / N) * |conf(b) - acc(b)|.
    """
    n = len(confidences)
    if n == 0:
        return 0.0

    bin_conf_sum = [0.0] * n_bins
    bin_correct_sum = [0.0] * n_bins
    bin_count = [0] * n_bins

    for conf, correct in zip(confidences, correctness):
        # Map confidence to a bin index in [0, n_bins-1].
        # The right-closed final bin is achieved by clamping the index.
        idx = int(conf * n_bins)
        if idx >= n_bins:
            idx = n_bins - 1
        elif idx < 0:
            idx = 0
        bin_conf_sum[idx] += conf
        bin_correct_sum[idx] += correct
        bin_count[idx] += 1

    ece = 0.0
    for b in range(n_bins):
        if bin_count[b] == 0:
            continue
        mean_conf = bin_conf_sum[b] / bin_count[b]
        mean_acc = bin_correct_sum[b] / bin_count[b]
        ece += (bin_count[b] / n) * abs(mean_conf - mean_acc)
    return ece


def _safe_mean(values: list[float]) -> Optional[float]:
    """Mean of ``values``, or ``None`` if the list is empty."""
    if not values:
        return None
    return sum(values) / len(values)


# ---------------------------------------------------------------------------
# 1. Per-set calibration
# ---------------------------------------------------------------------------

def compute_calibration_metrics(
    per_fact_results: list[dict],
) -> dict[str, Optional[float]]:
    """Compute calibration metrics over a list of per-fact MC results.

    Each input dict is expected to follow the schema produced by
    :func:`sleep.evaluation.recall_formats.multiple_choice_recall`:
    it must contain ``correct_letter``, ``predicted_letter``, ``is_correct``,
    and ``option_probs`` (a dict keyed by ``"A"``/``"B"``/``"C"``/``"D"``).

    Args:
        per_fact_results: List of per-fact dicts. May be empty.

    Returns:
        Dictionary with keys:
          - ``accuracy``: fraction of facts with ``is_correct=True``. Carried
            here (in addition to being available upstream) so that the
            stratified table can report it without a second pass over facts.
          - ``mean_correct_prob``: mean ``P(correct option)`` over all facts.
          - ``mean_predicted_prob``: mean ``P(model's chosen option)`` (i.e.
            mean confidence on the argmax).
          - ``mean_correct_when_wrong``: mean ``P(correct option)`` restricted
            to facts where the model's hard prediction was wrong. Best
            diagnostic for "knowledge present but not acted on".
          - ``mean_correct_when_right``: same, restricted to facts where the
            hard prediction was correct.
          - ``ece``: 10-bin Expected Calibration Error of model confidence
            vs. accuracy.
          - ``brier_score``: mean of ``(P(correct) - is_correct)^2``.
          - ``n_facts``: number of facts contributing.

        Any metric whose underlying sample is empty (e.g. no wrong facts)
        is returned as ``None``. ``n_facts`` is always an int.
    """
    n = len(per_fact_results)
    if n == 0:
        logger.warning("compute_calibration_metrics: received 0 facts")
        return _empty_metrics()

    correct_probs: list[float] = []
    predicted_probs: list[float] = []
    correct_probs_when_wrong: list[float] = []
    correct_probs_when_right: list[float] = []
    confidences: list[float] = []
    correctness: list[int] = []
    brier_terms: list[float] = []

    for r in per_fact_results:
        option_probs = r["option_probs"]
        correct_letter = r["correct_letter"]
        predicted_letter = r["predicted_letter"]
        is_correct_bool = bool(r["is_correct"])
        is_correct_int = 1 if is_correct_bool else 0

        p_correct = float(option_probs[correct_letter])
        p_predicted = float(option_probs[predicted_letter])

        correct_probs.append(p_correct)
        predicted_probs.append(p_predicted)
        confidences.append(p_predicted)
        correctness.append(is_correct_int)
        brier_terms.append((p_correct - is_correct_int) ** 2)

        if is_correct_bool:
            correct_probs_when_right.append(p_correct)
        else:
            correct_probs_when_wrong.append(p_correct)

    accuracy = sum(correctness) / n
    metrics: dict[str, Optional[float]] = {
        "accuracy": accuracy,
        "mean_correct_prob": _safe_mean(correct_probs),
        "mean_predicted_prob": _safe_mean(predicted_probs),
        "mean_correct_when_wrong": _safe_mean(correct_probs_when_wrong),
        "mean_correct_when_right": _safe_mean(correct_probs_when_right),
        "ece": _expected_calibration_error(confidences, correctness, _ECE_N_BINS),
        "brier_score": _safe_mean(brier_terms),
        "n_facts": n,
    }

    logger.info(
        "Calibration: n=%d, mean_correct=%.4f, mean_correct|wrong=%s, ece=%.4f, brier=%.4f",
        n,
        metrics["mean_correct_prob"] if metrics["mean_correct_prob"] is not None else float("nan"),
        f"{metrics['mean_correct_when_wrong']:.4f}"
        if metrics["mean_correct_when_wrong"] is not None
        else "n/a",
        metrics["ece"] if metrics["ece"] is not None else float("nan"),
        metrics["brier_score"] if metrics["brier_score"] is not None else float("nan"),
    )

    return metrics


# ---------------------------------------------------------------------------
# 2. Stratified calibration
# ---------------------------------------------------------------------------

def compute_stratified_calibration(
    per_fact_results: list[dict],
    group_assignments: dict[str, str],
) -> dict[str, dict[str, Optional[float]]]:
    """Compute calibration metrics for each group of facts.

    Facts are assigned to groups via ``group_assignments`` (keyed by
    ``fact_id``). Facts whose ``fact_id`` is not in the mapping are silently
    excluded — the caller is expected to populate ``group_assignments`` for
    every fact it cares about. Groups present in ``group_assignments`` but
    with zero matching facts are returned with all-``None`` metrics so that
    downstream code can iterate uniformly.

    Args:
        per_fact_results:  List of per-fact MC dicts (see
                           :func:`compute_calibration_metrics`).
        group_assignments: Mapping ``fact_id -> group_name``.

    Returns:
        Mapping ``group_name -> calibration metrics dict``.
    """
    # Bucket facts by group.
    buckets: dict[str, list[dict]] = {}
    for r in per_fact_results:
        fact_id = r.get("fact_id")
        if fact_id is None:
            continue
        group = group_assignments.get(fact_id)
        if group is None:
            continue
        buckets.setdefault(group, []).append(r)

    # Ensure every group named in the assignments shows up in the output,
    # even if no fact in per_fact_results landed in it.
    all_groups = set(group_assignments.values())
    for group in all_groups:
        buckets.setdefault(group, [])

    out: dict[str, dict[str, Optional[float]]] = {}
    for group, facts in buckets.items():
        if not facts:
            logger.info("Stratified calibration: group %r has 0 facts", group)
            out[group] = _empty_metrics()
        else:
            out[group] = compute_calibration_metrics(facts)

    return out


# ---------------------------------------------------------------------------
# 3. Pretty-printed table
# ---------------------------------------------------------------------------

def _fmt_value(value: Optional[float], width: int, precision: int = 2) -> str:
    """Format a metric value for the calibration table.

    ``None`` is rendered as an em dash, left-justified in ``width`` columns,
    matching the formatting of numeric cells.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        text = "-"
    else:
        text = f"{value:.{precision}f}"
    return text.ljust(width)


def format_calibration_table(
    stratified_metrics: dict[str, dict[str, Optional[float]]],
) -> str:
    """Render stratified calibration metrics as an ASCII table.

    Columns: ``Group | N | Acc | MeanCorrP | MeanCorrP|Wrong | ECE | Brier``.
    Rows are sorted by descending ``n_facts`` (with ties broken by group name)
    so that the most informative rows are on top. ``None`` cells are rendered
    as a single dash ``-``.

    Args:
        stratified_metrics: Output of :func:`compute_stratified_calibration`.

    Returns:
        Multi-line string suitable for printing or logging.
    """
    # Column layout. Widths are chosen to comfortably hold any plausible
    # group name and the formatted metric values.
    col_group = 18
    col_n = 5
    col_acc = 7
    col_mcp = 11
    col_mcpw = 17
    col_ece = 7
    col_brier = 7

    header = (
        "Group".ljust(col_group)
        + "N".ljust(col_n)
        + "Acc".ljust(col_acc)
        + "MeanCorrP".ljust(col_mcp)
        + "MeanCorrP|Wrong".ljust(col_mcpw)
        + "ECE".ljust(col_ece)
        + "Brier".ljust(col_brier)
    )
    total_width = col_group + col_n + col_acc + col_mcp + col_mcpw + col_ece + col_brier
    separator = "-" * total_width

    lines = [header, separator]

    # Sort groups for stable output: largest n_facts first, then alphabetical.
    def _sort_key(item: tuple[str, dict[str, Optional[float]]]) -> tuple[int, str]:
        name, m = item
        n = m.get("n_facts") or 0
        return (-int(n), name)

    for group, m in sorted(stratified_metrics.items(), key=_sort_key):
        n_facts = m.get("n_facts") or 0
        n_text = str(int(n_facts)).ljust(col_n)

        row = (
            group[:col_group - 1].ljust(col_group)
            + n_text
            + _fmt_value(m.get("accuracy"), col_acc)
            + _fmt_value(m.get("mean_correct_prob"), col_mcp)
            + _fmt_value(m.get("mean_correct_when_wrong"), col_mcpw)
            + _fmt_value(m.get("ece"), col_ece)
            + _fmt_value(m.get("brier_score"), col_brier)
        )
        lines.append(row)

    return "\n".join(lines)
