"""
PRP composite scoring for the SLEEP tagging buffer.

Implements Q2.2 (composite scoring function) and Q6.1 (revision bonus)
from SLEEP_Formalization.md.

The composite score determines which tags get allocated PRPs and queued
for consolidation during the sleep phase:

    S(τ) = w₁ · Ê(τ) + w₂ · Â(τ) + w₃ · X̂(τ) + w₄ · R̂(τ)

Components:
    Ê  — Cumulative prediction error (normalized)
    Â  — Access frequency via cumulative utility ρ (normalized)
    X̂  — Cross-reference density (normalized by active tag count)
    R̂  — Recency-weighted utility (normalized)

All components are in [0, 1]. Weights come from PRPConfig and sum to 1.
"""

from __future__ import annotations

from sleep.config import PRPConfig
from sleep.tagging.tags import Tag


# Minimum denominator to avoid division by zero.
_EPS: float = 1e-10


def compute_prp_scores(
    tags: list[Tag],
    config: PRPConfig,
    revision_bonus: float = 0.3,
) -> list[float]:
    """Compute PRP composite scores for all tags in the buffer.

    Updates each tag's ``S_score`` field in-place and returns the list of
    scores in the same order as *tags*.

    Scoring formula (Q2.2)::

        E(τ)  = e₀ · s / s₀
        Ê(τ)  = E(τ) / max(E(τⱼ) for τⱼ in buffer)

        Â(τ)  = ρ / (1 + max(ρⱼ for τⱼ in buffer))

        X̂(τ)  = xref_count / N_active

        R̂(τ)  = R / max(Rⱼ for τⱼ in buffer)

        S(τ)  = w_error · Ê + w_access · Â + w_crossref · X̂ + w_recency · R̂

    Revision bonus (Q6.1): tags with ``tag_type == "revision"`` receive an
    additive bonus of *revision_bonus* on top of the composite score.

    Args:
        tags: Active tags in the buffer.
        config: PRPConfig supplying the four scoring weights.
        revision_bonus: Additive score bonus for revision-type tags.

    Returns:
        List of composite scores, one per tag.
    """
    if not tags:
        return []

    n_active: int = len(tags)

    # ------------------------------------------------------------------
    # Pre-compute raw (un-normalised) values for each component
    # ------------------------------------------------------------------

    # Component 1: cumulative prediction error  E(τ) = e₀ · s / s₀
    raw_errors: list[float] = [
        tag.e0 * (tag.s / max(tag.s0, _EPS)) for tag in tags
    ]
    max_error: float = max(raw_errors) if raw_errors else 0.0

    # Component 2: access frequency via cumulative utility ρ
    max_rho: float = max((tag.rho for tag in tags), default=0.0)

    # Component 4: recency-weighted utility R
    max_r: float = max((tag.R for tag in tags), default=0.0)

    # ------------------------------------------------------------------
    # Compute normalised components and composite score per tag
    # ------------------------------------------------------------------
    scores: list[float] = []

    for i, tag in enumerate(tags):
        # Ê — normalised prediction error
        e_hat: float = raw_errors[i] / max(max_error, _EPS)

        # Â — normalised access frequency
        a_hat: float = tag.rho / (1.0 + max_rho)

        # X̂ — cross-reference density (already count-based, normalise by N)
        x_hat: float = tag.xref_count / max(n_active, 1)

        # R̂ — normalised recency-weighted utility
        r_hat: float = tag.R / max(max_r, _EPS)

        # Combined score
        s_score: float = (
            config.w_error * e_hat
            + config.w_access * a_hat
            + config.w_crossref * x_hat
            + config.w_recency * r_hat
        )

        # Revision bonus (Q6.1)
        if tag.tag_type == "revision":
            s_score += revision_bonus

        # Persist on the tag
        tag.S_score = s_score
        scores.append(s_score)

    return scores
