"""
Adaptive z-score threshold for surprise flagging (Q1.2, Step 2).

Maintains exponential moving average statistics of per-token surprise and
flags tokens whose z-score exceeds the sensitivity parameter kappa.

    mu_t   = beta * mu_{t-1}   + (1 - beta) * e_t
    var_t  = beta * var_{t-1}  + (1 - beta) * (e_t - mu_t)^2
    flag_t = (e_t - mu_t) / max(sigma_t, 1e-8) > kappa
"""

from __future__ import annotations

import math


class AdaptiveThreshold:
    """Running EMA statistics with z-score flagging.

    Parameters
    ----------
    beta:
        EMA smoothing factor (default 0.99, ~100-token effective window).
    kappa:
        Default sensitivity — number of standard deviations above the
        running mean required to flag a token.
    """

    def __init__(self, beta: float = 0.99, kappa: float = 1.5) -> None:
        self._beta: float = beta
        self._kappa: float = kappa

        # Initialise with mu=0, var=1.0 to avoid division-by-zero
        # before enough data has been observed.
        self._mu: float = 0.0
        self._var: float = 1.0

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def mu(self) -> float:
        """Current running mean of surprise."""
        return self._mu

    @property
    def sigma(self) -> float:
        """Current running standard deviation of surprise."""
        return math.sqrt(self._var)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def update_and_flag(
        self,
        surprises: list[float],
        kappa_override: float | None = None,
    ) -> list[bool]:
        """Update running stats token-by-token and return flags.

        Parameters
        ----------
        surprises:
            Per-token surprise values (nats) for the current chunk.
        kappa_override:
            If provided, overrides the instance-level kappa for this call
            (useful for cold-start elevated threshold, kappa_cold = 3.0).

        Returns
        -------
        list[bool]
            One flag per token — ``True`` if the token exceeds the
            adaptive threshold.
        """
        kappa: float = kappa_override if kappa_override is not None else self._kappa
        beta: float = self._beta
        flags: list[bool] = []

        for e_t in surprises:
            # Update running mean.
            self._mu = beta * self._mu + (1.0 - beta) * e_t

            # Update running variance.
            self._var = beta * self._var + (1.0 - beta) * (e_t - self._mu) ** 2

            # Z-score test.
            sigma_t: float = math.sqrt(self._var)
            z: float = (e_t - self._mu) / max(sigma_t, 1e-8)
            flags.append(z > kappa)

        return flags