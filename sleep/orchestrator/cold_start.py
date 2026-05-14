"""
Cold start calibration (Q6.2).

During the burn-in period the system has not yet established reliable surprise
statistics.  The ``ColdStartManager`` provides elevated thresholds (higher kappa)
and reduced PRP budgets to prevent over-tagging and premature consolidation.
"""

from __future__ import annotations

from sleep.config import ColdStartConfig
from sleep.utils.logging import get_logger

logger = get_logger("sleep.orchestrator.cold_start")


class ColdStartManager:
    """Manages threshold calibration during the burn-in period.

    Three phases:
        1. **Burn-in** (interactions < ``n_burnin``):
           Uses ``kappa_cold`` (elevated, e.g. 3.0) to suppress noisy tags.
        2. **Ramp** (``n_burnin`` <= interactions < ``n_burnin + n_ramp``):
           Linearly interpolates kappa from ``kappa_cold`` down to ``normal_kappa``.
        3. **Mature** (interactions >= ``n_burnin + n_ramp``):
           Uses ``normal_kappa`` — system has stable surprise statistics.

    PRP budget is similarly ramped from 0 to full over ``n_mature`` interactions.

    Args:
        config:       ``ColdStartConfig`` with ``kappa_cold``, ``n_burnin``,
                      ``n_ramp``, ``n_mature``.
        normal_kappa: The normal (non-cold-start) kappa value to ramp toward.
    """

    def __init__(self, config: ColdStartConfig, normal_kappa: float = 1.5) -> None:
        self._config = config
        self._normal_kappa = normal_kappa
        self._interaction_count: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def interaction_count(self) -> int:
        """Number of interactions tracked so far."""
        return self._interaction_count

    @interaction_count.setter
    def interaction_count(self, value: int) -> None:
        self._interaction_count = value

    @property
    def is_mature(self) -> bool:
        """True if past the maturation period (both kappa ramp and budget ramp complete)."""
        return self._interaction_count >= self._config.n_mature

    @property
    def is_past_burnin(self) -> bool:
        """True if past the initial burn-in period."""
        return self._interaction_count >= self._config.n_burnin

    # ------------------------------------------------------------------
    # Kappa calibration
    # ------------------------------------------------------------------

    def get_effective_kappa(self, interaction_count: int) -> float:
        """Return the effective kappa for the current interaction.

        Args:
            interaction_count: Current number of completed interactions.

        Returns:
            Effective kappa value:
            - ``kappa_cold`` during burn-in (interactions < ``n_burnin``)
            - Linear interpolation ``kappa_cold`` -> ``normal_kappa`` during ramp
            - ``normal_kappa`` after ramp completes
        """
        cfg = self._config

        if interaction_count < cfg.n_burnin:
            return cfg.kappa_cold

        ramp_start = cfg.n_burnin
        ramp_end = cfg.n_burnin + cfg.n_ramp

        if interaction_count >= ramp_end:
            return self._normal_kappa

        # Linear interpolation within the ramp window
        progress = (interaction_count - ramp_start) / max(cfg.n_ramp, 1)
        effective = cfg.kappa_cold + progress * (self._normal_kappa - cfg.kappa_cold)
        return effective

    # ------------------------------------------------------------------
    # Budget scaling
    # ------------------------------------------------------------------

    def get_budget_scale(self, interaction_count: int) -> float:
        """Return PRP budget scaling factor (0.0 to 1.0).

        Linearly ramps from 0 to 1 over ``n_mature`` interactions.  This
        prevents the system from allocating PRP slots before it has enough
        data to make meaningful priority decisions.

        Args:
            interaction_count: Current number of completed interactions.

        Returns:
            A float in [0.0, 1.0].
        """
        n_mature = self._config.n_mature
        if n_mature <= 0:
            return 1.0
        return min(1.0, interaction_count / n_mature)

    # ------------------------------------------------------------------
    # Convenience: advance counter
    # ------------------------------------------------------------------

    def record_interaction(self) -> None:
        """Increment the interaction counter and log phase transitions."""
        self._interaction_count += 1
        n = self._interaction_count
        cfg = self._config

        # Log phase transitions
        if n == cfg.n_burnin:
            logger.info(
                "Cold start: burn-in complete at interaction %d — "
                "beginning kappa ramp (%.2f -> %.2f over %d interactions)",
                n, cfg.kappa_cold, self._normal_kappa, cfg.n_ramp,
            )
        elif n == cfg.n_burnin + cfg.n_ramp:
            logger.info(
                "Cold start: kappa ramp complete at interaction %d — "
                "using normal kappa=%.2f",
                n, self._normal_kappa,
            )
        elif n == cfg.n_mature:
            logger.info(
                "Cold start: system mature at interaction %d — "
                "full PRP budget now available",
                n,
            )
