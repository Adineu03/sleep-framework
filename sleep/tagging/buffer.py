"""
Tag Buffer — manages the collection of active tags.

Handles decay (Q1.4), reinforcement on access (Q1.5), garbage collection,
and capacity eviction (Q1.6).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from sleep.tagging.tags import Tag

if TYPE_CHECKING:
    from sleep.config import TaggingConfig


class TagBuffer:
    """Manages a bounded collection of Tag objects.

    Provides decay, reinforcement, garbage collection, and capacity eviction
    following the SLEEP formalization (Q1.4-Q1.6).
    """

    def __init__(
        self,
        config: TaggingConfig,
        n_max: int,
        tau_recency: int = 500,
    ) -> None:
        self._config: TaggingConfig = config
        self._n_max: int = n_max
        self._tau_recency: int = tau_recency
        self._tags: list[Tag] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_active(self) -> int:
        """Number of tags currently in the buffer."""
        return len(self._tags)

    @property
    def n_prp_allocated(self) -> int:
        """Number of tags with PRP allocation flag set."""
        return sum(1 for t in self._tags if t.p == 1)

    @property
    def occupancy(self) -> float:
        """Fraction of buffer capacity in use."""
        if self._n_max <= 0:
            return 0.0
        return self.n_active / self._n_max

    @property
    def tags(self) -> list[Tag]:
        """Read-only access to the tag list (returns a shallow copy)."""
        return list(self._tags)

    # ------------------------------------------------------------------
    # 1. Add tags (with capacity eviction)
    # ------------------------------------------------------------------

    def add(self, tags: list[Tag]) -> None:
        """Add new tags to the buffer, evicting lowest-priority tags if needed.

        Eviction priority (low = evict first): ``s * (1 + rho)``.
        """
        self._tags.extend(tags)

        if len(self._tags) > self._n_max:
            # Sort descending by priority so the tail contains eviction candidates.
            self._tags.sort(key=lambda t: t.s * (1.0 + t.rho), reverse=True)
            self._tags = self._tags[: self._n_max]

    # ------------------------------------------------------------------
    # 2. Decay & garbage collection
    # ------------------------------------------------------------------

    def decay_and_gc(self, current_step: int) -> int:
        """Recompute decayed strengths and remove tags below the GC threshold.

        Decay formula (Q1.4)::

            tau_decay = tau_base * (1 + gamma * e0)
            s_base(t) = (s0 - epsilon) * exp(-(t - t0) / tau_decay) + epsilon
            s(t)      = min(s_base(t) + s_reinforced, 1.0)

        Returns the number of tags removed.
        """
        cfg = self._config
        tau_base: int = cfg.tau_base
        gamma: float = cfg.gamma_decay
        epsilon: float = cfg.epsilon
        epsilon_gc: float = cfg.epsilon_gc

        surviving: list[Tag] = []

        for tag in self._tags:
            dt: int = current_step - tag.t0
            tau_decay: float = tau_base * (1.0 + gamma * tag.e0)

            s_base: float = (tag.s0 - epsilon) * math.exp(-dt / tau_decay) + epsilon
            s_eff: float = min(s_base + tag.s_reinforced, 1.0)
            tag.s = s_eff

            if s_eff >= epsilon_gc:
                surviving.append(tag)

        removed: int = len(self._tags) - len(surviving)
        self._tags = surviving
        return removed

    # ------------------------------------------------------------------
    # 3. Process query — access detection & reinforcement (Q1.5)
    # ------------------------------------------------------------------

    def process_query(
        self,
        query_key: torch.Tensor,
        current_step: int,
    ) -> list[Tag]:
        """Detect accessed tags and apply reinforcement.

        For each tag whose cosine similarity with *query_key* exceeds
        ``theta_access``, apply the reinforcement update::

            a       += 1
            rho     += sim * (1 / sqrt(a))
            boost    = delta_s * (1 - s) * (1 / sqrt(a))
            s_reinforced += boost
            s        = min(s + boost, 1.0)

        Also updates the recency-weighted utility *R*::

            R = R * exp(-(current_step - R_last_update) / tau_recency) + sim
            R_last_update = current_step

        Returns the list of accessed tags.
        """
        if not self._tags:
            return []

        cfg = self._config
        theta_access: float = cfg.theta_access
        delta_s: float = cfg.delta_s
        tau_recency: int = self._tau_recency

        # Stack all tag keys into a matrix for batched cosine similarity.
        # query_key: (d_tag,)  ->  (1, d_tag)
        # keys:      (N, d_tag)
        keys = torch.stack([t.k for t in self._tags], dim=0)  # (N, d_tag)
        query_2d = query_key.unsqueeze(0)                      # (1, d_tag)

        # cosine_similarity along dim=1 returns (N,)
        sims = F.cosine_similarity(keys, query_2d, dim=1)      # (N,)

        accessed: list[Tag] = []

        for idx, tag in enumerate(self._tags):
            sim: float = sims[idx].item()

            if sim <= theta_access:
                continue

            # --- Reinforcement ---
            tag.a += 1
            inv_sqrt_a: float = 1.0 / math.sqrt(tag.a)

            tag.rho += sim * inv_sqrt_a

            boost: float = delta_s * (1.0 - tag.s) * inv_sqrt_a
            tag.s_reinforced += boost
            tag.s = min(tag.s + boost, 1.0)

            # --- Recency-weighted utility ---
            dt: int = current_step - tag.R_last_update
            if dt > 0 and tau_recency > 0:
                tag.R = tag.R * math.exp(-dt / tau_recency) + sim
            else:
                tag.R += sim
            tag.R_last_update = current_step

            accessed.append(tag)

        return accessed