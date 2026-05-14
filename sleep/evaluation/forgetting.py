"""
Forgetting curve computation for the SLEEP evaluation suite.

Implements Metric 2 from Q5.4 (SLEEP_Formalization.md):
    F(t)     = DRA(0) - DRA(t)           (absolute forgetting)
    F_rel(t) = F(t) / DRA(0)             (relative forgetting)

Analyzes the shape of recall decay over successive sleep cycles, classifying
the behavior as stable, gradual, or catastrophic.
"""

from __future__ import annotations

import math

from sleep.utils.logging import get_logger, metrics

logger = get_logger("sleep.evaluation.forgetting")


# ---------------------------------------------------------------------------
# Forgetting classification
# ---------------------------------------------------------------------------

def classify_forgetting(relative_forgetting: float, n_cycles: int) -> str:
    """Classify forgetting behavior based on relative forgetting.

    Classification thresholds (from Q5.4):
        - ``"stable"``:       relative_forgetting < 0.1  (less than 10% lost)
        - ``"gradual"``:      0.1 <= relative_forgetting < 0.5
        - ``"catastrophic"``: relative_forgetting >= 0.5

    Args:
        relative_forgetting: (initial_recall - final_recall) / initial_recall.
        n_cycles:            Number of sleep cycles between first and last
                             measurement (used for logging context only).

    Returns:
        One of ``"stable"``, ``"gradual"``, or ``"catastrophic"``.
    """
    if relative_forgetting < 0.1:
        return "stable"
    elif relative_forgetting < 0.5:
        return "gradual"
    else:
        return "catastrophic"


# ---------------------------------------------------------------------------
# Forgetting curve analysis
# ---------------------------------------------------------------------------

def compute_forgetting_curve(
    recall_history: list[tuple[int, float]],
) -> dict:
    """Analyze the forgetting curve from a recall history.

    The recall history is a chronologically ordered list of
    ``(sleep_cycle, dra_score)`` pairs, where ``sleep_cycle`` is the cycle
    number at which recall was measured.

    The function computes:
        - **absolute_forgetting**: DRA(first) - DRA(last)
        - **relative_forgetting**: absolute / DRA(first)
        - **half_life**: the first cycle at which DRA drops to 50% of initial
          (None if it never does)
        - **curve_type**: classification via :func:`classify_forgetting`

    Args:
        recall_history: List of ``(cycle, dra_score)`` tuples, sorted by cycle.
                        Must contain at least one entry.

    Returns:
        Dictionary with keys:
            - ``initial_recall`` (float)
            - ``final_recall`` (float)
            - ``absolute_forgetting`` (float)
            - ``relative_forgetting`` (float)
            - ``half_life`` (int | None)
            - ``curve_type`` (str)

    Raises:
        ValueError: If recall_history is empty.
    """
    if not recall_history:
        raise ValueError("recall_history must contain at least one entry")

    # Sort by cycle to be safe
    history = sorted(recall_history, key=lambda x: x[0])

    initial_cycle, initial_recall = history[0]
    final_cycle, final_recall = history[-1]

    absolute_forgetting = initial_recall - final_recall
    if initial_recall > 0:
        relative_forgetting = absolute_forgetting / initial_recall
    else:
        relative_forgetting = 0.0

    # Compute half-life: first cycle where DRA <= 0.5 * initial_recall
    half_life: int | None = None
    if initial_recall > 0:
        half_threshold = 0.5 * initial_recall
        for cycle, dra in history:
            if dra <= half_threshold:
                half_life = cycle
                break

    n_cycles = final_cycle - initial_cycle if len(history) > 1 else 0
    curve_type = classify_forgetting(relative_forgetting, n_cycles)

    result = {
        "initial_recall": initial_recall,
        "final_recall": final_recall,
        "absolute_forgetting": absolute_forgetting,
        "relative_forgetting": relative_forgetting,
        "half_life": half_life,
        "curve_type": curve_type,
    }

    logger.info(
        "Forgetting curve: initial=%.4f, final=%.4f, relative=%.4f, type=%s, half_life=%s",
        initial_recall,
        final_recall,
        relative_forgetting,
        curve_type,
        half_life,
    )
    metrics.log({
        "evaluation/initial_recall": initial_recall,
        "evaluation/final_recall": final_recall,
        "evaluation/absolute_forgetting": absolute_forgetting,
        "evaluation/relative_forgetting": relative_forgetting,
        "evaluation/curve_type_is_stable": 1.0 if curve_type == "stable" else 0.0,
    })

    return result
