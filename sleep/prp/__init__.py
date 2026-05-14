"""
PRP (Plasticity-Related Protein) allocation system.

Orchestrates scoring, cross-reference computation, and competitive allocation
into a single PRPSystem class that the sleep orchestrator calls periodically.
"""

from __future__ import annotations

from sleep.config import PRPConfig
from sleep.utils.logging import get_logger, metrics
from sleep.prp.allocation import allocate_prps, compute_threshold
from sleep.prp.crossref import compute_cross_references
from sleep.prp.scoring import compute_prp_scores
from sleep.tagging.tags import Tag

logger = get_logger("sleep.prp")

__all__ = [
    "PRPSystem",
    "compute_prp_scores",
    "compute_cross_references",
    "allocate_prps",
]


class PRPSystem:
    """Orchestrator for the full PRP scoring and allocation pipeline.

    Ties together scoring (Q2.2), cross-reference computation (Q2.2),
    and competitive allocation (Q2.3-Q2.5) into a single update loop
    called periodically by the sleep orchestrator.
    """

    def __init__(
        self,
        config: PRPConfig,
        budget: int,
        revision_bonus: float = 0.3,
    ) -> None:
        """
        Args:
            config: PRPConfig with all PRP hyperparameters.
            budget: B = c_prp * model_params_billions (computed externally).
            revision_bonus: Bonus score for revision-type tags (from RevisionConfig).
        """
        self._config = config
        self._budget = budget
        self._revision_bonus = revision_bonus

        # Internal tracking
        self._last_crossref_step: int = -1
        self._total_updates: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def budget(self) -> int:
        """The PRP budget B."""
        return self._budget

    @property
    def total_updates(self) -> int:
        """Count of update() calls so far."""
        return self._total_updates

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def update(
        self,
        tags: list[Tag],
        current_step: int,
        force_crossref: bool = False,
    ) -> dict:
        """Run the full PRP scoring and allocation pipeline.

        Called periodically (every ``allocation_interval`` steps from the
        orchestrator).

        Steps:
            1. If due for cross-reference update (every ``crossref_interval``
               steps) or *force_crossref*, recompute cross-reference edges.
            2. Compute PRP composite scores for all tags.
            3. Run competitive allocation against the budget.
            4. Collect and return allocation statistics.

        Args:
            tags: Current list of live tags.
            current_step: Global inference step counter.
            force_crossref: If True, recompute cross-references regardless
                of interval.

        Returns:
            Dict with allocation statistics::

                {
                    "allocated": int,
                    "threshold": float,
                    "newly_allocated": int,
                    "deallocated": int,
                    "budget": int,
                    "budget_utilization": float,
                    "mean_score": float,
                    "crossref_updated": bool,
                }
        """
        config = self._config

        # Snapshot allocation state before this round
        prev_allocated: set[int] = {id(t) for t in tags if t.p == 1}

        # 1. Cross-reference update (batch, every crossref_interval steps)
        crossref_due = (
            self._last_crossref_step < 0
            or (current_step - self._last_crossref_step) >= config.crossref_interval
        )
        crossref_updated = False
        if crossref_due or force_crossref:
            compute_cross_references(tags, config.theta_xref)
            self._last_crossref_step = current_step
            crossref_updated = True

        # 2. Score all tags
        compute_prp_scores(tags, config, self._revision_bonus)

        # 3. Competitive allocation
        allocate_prps(tags, self._budget, config)

        # 4. Gather stats
        curr_allocated: set[int] = {id(t) for t in tags if t.p == 1}
        newly_allocated = len(curr_allocated - prev_allocated)
        deallocated = len(prev_allocated - curr_allocated)
        n_allocated = len(curr_allocated)

        scores = [t.S_score for t in tags]
        mean_score = sum(scores) / len(scores) if scores else 0.0

        self._total_updates += 1

        logger.info(
            "step=%d | PRP update: %d/%d allocated (%.0f%%) | +%d -%d | threshold=%.3f | mean_score=%.3f",
            current_step, n_allocated, self._budget,
            (n_allocated / self._budget * 100) if self._budget > 0 else 0,
            newly_allocated, deallocated,
            compute_threshold(scores, config), mean_score,
        )
        metrics.log({
            "prp/allocated": n_allocated,
            "prp/budget_utilization": n_allocated / self._budget if self._budget > 0 else 0.0,
            "prp/newly_allocated": newly_allocated,
            "prp/deallocated": deallocated,
            "prp/mean_score": mean_score,
            "prp/threshold": compute_threshold(scores, config),
        }, step=current_step)

        return {
            "allocated": n_allocated,
            "threshold": compute_threshold(scores, config),
            "newly_allocated": newly_allocated,
            "deallocated": deallocated,
            "budget": self._budget,
            "budget_utilization": n_allocated / self._budget if self._budget > 0 else 0.0,
            "mean_score": mean_score,
            "crossref_updated": crossref_updated,
        }

    # ------------------------------------------------------------------
    # Consolidation helpers
    # ------------------------------------------------------------------

    def get_consolidation_candidates(self, tags: list[Tag]) -> list[Tag]:
        """Return PRP-allocated tags sorted by score descending.

        These are the candidates for the next sleep cycle.
        """
        return sorted(
            [t for t in tags if t.p == 1],
            key=lambda t: t.S_score,
            reverse=True,
        )
