"""
Calibration plotting for the SLEEP recognition signal (mentor feedback item #10).

:mod:`sleep.evaluation.calibration` already computes the numbers — mean correct-
option probability, ECE, Brier, and the diagnostic ``mean_correct_when_wrong``.
What the reviewer asked for is the *picture*: a confidence-vs-accuracy /
P(correct) view that makes the recognition signal legible even where it does not
flip the argmax. This module turns the existing per-fact MC results into two
publication-ready panels.

It is deliberately import-light: ``matplotlib`` is imported inside the plotting
function so that importing this module (e.g. in a headless test) does not require
a display backend, and the rest of ``sleep.evaluation`` stays plot-free.

Panels produced by :func:`plot_tagged_vs_untagged_calibration`:

  1. **P(correct option) distribution**, tagged vs untagged, as overlaid
     histograms with group-mean lines. This is the "is there latent signal"
     view: if tagged facts pool probability mass on the correct option — even
     below the 0.5 needed to win a 4-way argmax — the recognition signal is
     real.
  2. **Reliability curve** (confidence vs empirical accuracy) with the y=x
     ideal, per group, annotated with ECE.
"""

from __future__ import annotations

import os
from typing import Optional

from sleep.evaluation.calibration import (
    _ECE_N_BINS,
    compute_calibration_metrics,
)
from sleep.utils.logging import get_logger

logger = get_logger("sleep.evaluation.calibration_plot")

__all__ = ["plot_tagged_vs_untagged_calibration", "reliability_bins"]


def reliability_bins(
    confidences: list[float],
    correctness: list[int],
    n_bins: int = _ECE_N_BINS,
) -> tuple[list[float], list[float], list[int]]:
    """Bin confidences and return per-bin (mean_conf, mean_acc, count).

    Mirrors the binning used by
    :func:`sleep.evaluation.calibration._expected_calibration_error` so the
    reliability curve and the reported ECE are computed on identical bins.

    Args:
        confidences: Model confidence on its argmax choice, one per fact.
        correctness: Binary correctness (0/1) for the same facts.
        n_bins:      Number of equal-width bins on ``[0, 1]``.

    Returns:
        ``(mean_conf, mean_acc, count)`` lists of length ``n_bins``; empty bins
        carry ``mean_conf = mean_acc = float("nan")`` and ``count = 0``.
    """
    conf_sum = [0.0] * n_bins
    acc_sum = [0.0] * n_bins
    count = [0] * n_bins
    for conf, correct in zip(confidences, correctness):
        idx = int(conf * n_bins)
        if idx >= n_bins:
            idx = n_bins - 1
        elif idx < 0:
            idx = 0
        conf_sum[idx] += conf
        acc_sum[idx] += correct
        count[idx] += 1

    mean_conf = [
        (conf_sum[b] / count[b]) if count[b] else float("nan") for b in range(n_bins)
    ]
    mean_acc = [
        (acc_sum[b] / count[b]) if count[b] else float("nan") for b in range(n_bins)
    ]
    return mean_conf, mean_acc, count


def _correct_probs(per_fact: list[dict]) -> list[float]:
    """Extract P(correct option) for each fact."""
    return [float(r["option_probs"][r["correct_letter"]]) for r in per_fact]


def _confidence_and_correctness(per_fact: list[dict]) -> tuple[list[float], list[int]]:
    """Extract (P(argmax option), is_correct) per fact for the reliability curve."""
    conf = [float(r["option_probs"][r["predicted_letter"]]) for r in per_fact]
    correct = [1 if r["is_correct"] else 0 for r in per_fact]
    return conf, correct


def plot_tagged_vs_untagged_calibration(
    tagged_per_fact: list[dict],
    untagged_per_fact: list[dict],
    out_path: str,
    *,
    title: Optional[str] = None,
    n_bins: int = _ECE_N_BINS,
) -> str:
    """Render the two-panel tagged-vs-untagged calibration figure.

    Args:
        tagged_per_fact:   Per-fact MC dicts for tagged facts (schema from
                           :func:`sleep.evaluation.recall_formats.multiple_choice_recall`).
        untagged_per_fact: Same, for untagged facts.
        out_path:          Where to write the figure. The extension chooses the
                           format (``.pdf`` for LaTeX, ``.png`` for docs). Both
                           are written when ``out_path`` ends in ``.pdf``.
        title:             Optional suptitle; a sensible default is used if
                           ``None``.
        n_bins:            Bins for the reliability curve.

    Returns:
        The path actually written (the ``out_path`` argument).

    Raises:
        ValueError: If both fact lists are empty (nothing to plot).
    """
    if not tagged_per_fact and not untagged_per_fact:
        raise ValueError("Both fact lists are empty — nothing to plot.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tagged_cal = compute_calibration_metrics(tagged_per_fact) if tagged_per_fact else None
    untagged_cal = (
        compute_calibration_metrics(untagged_per_fact) if untagged_per_fact else None
    )

    fig, (ax_hist, ax_rel) = plt.subplots(1, 2, figsize=(11, 4.4))

    # -- Panel 1: P(correct option) distribution --------------------------
    bins = [i / 20.0 for i in range(21)]  # 0.05-wide bins on [0, 1]
    for per_fact, label, color in (
        (tagged_per_fact, "Tagged", "#1f77b4"),
        (untagged_per_fact, "Untagged", "#d62728"),
    ):
        if not per_fact:
            continue
        probs = _correct_probs(per_fact)
        ax_hist.hist(
            probs, bins=bins, alpha=0.55, label=f"{label} (n={len(probs)})",
            color=color, density=True,
        )
        mean_p = sum(probs) / len(probs)
        ax_hist.axvline(mean_p, color=color, linestyle="--", linewidth=1.3)
    ax_hist.axvline(
        0.25, color="grey", linestyle=":", linewidth=1.0, label="Chance (0.25)",
    )
    ax_hist.set_xlabel("P(correct option)")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("Correct-option probability mass")
    ax_hist.legend(fontsize=9, framealpha=0.9)

    # -- Panel 2: reliability curve ---------------------------------------
    ax_rel.plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=1.0, label="Ideal")
    for per_fact, cal, label, color in (
        (tagged_per_fact, tagged_cal, "Tagged", "#1f77b4"),
        (untagged_per_fact, untagged_cal, "Untagged", "#d62728"),
    ):
        if not per_fact:
            continue
        conf, correct = _confidence_and_correctness(per_fact)
        mean_conf, mean_acc, count = reliability_bins(conf, correct, n_bins)
        xs = [c for c, n in zip(mean_conf, count) if n > 0]
        ys = [a for a, n in zip(mean_acc, count) if n > 0]
        ece = cal["ece"] if cal and cal["ece"] is not None else float("nan")
        ax_rel.plot(
            xs, ys, "o-", color=color, linewidth=1.6, markersize=6,
            label=f"{label} (ECE={ece:.3f})",
        )
    ax_rel.set_xlabel("Confidence (P assigned to chosen option)")
    ax_rel.set_ylabel("Empirical accuracy")
    ax_rel.set_title("Reliability")
    ax_rel.set_xlim(0, 1)
    ax_rel.set_ylim(0, 1)
    ax_rel.legend(fontsize=9, framealpha=0.9)

    if title is None:
        title = "KV memory recognition signal: tagged vs untagged calibration"
    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    # Mirror a PNG next to a PDF so the docs pipeline has a raster copy.
    if out_path.lower().endswith(".pdf"):
        fig.savefig(out_path[:-4] + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    logger.info(
        "Wrote calibration plot to %s (tagged n=%d, untagged n=%d)",
        out_path, len(tagged_per_fact), len(untagged_per_fact),
    )
    return out_path
