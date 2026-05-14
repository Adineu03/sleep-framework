"""
Cross-reference density computation for the SLEEP PRP scoring system.

Implements Q2.2 Component 3 (Cross-Reference Density) from SLEEP_Formalization.md.

X(τᵢ) = Σⱼ≠ᵢ 𝟙[cos(kᵢ, kⱼ) > θ_xref] / N_active

This is a batch operation run periodically (every crossref_interval steps).
The raw count (numerator, before dividing by N_active) is stored in each tag's
xref_count field — normalization happens downstream in scoring.py.
"""

from __future__ import annotations

import torch
from torch import Tensor

from sleep.tagging.tags import Tag


def compute_cross_references(tags: list[Tag], theta_xref: float = 0.5) -> None:
    """Compute cross-reference counts for all tags.

    Updates each tag's xref_count field in-place.

    Uses batched cosine similarity for efficiency:
    1. Stack all key vectors into matrix K ∈ ℝ^(N × d_tag)
    2. Compute cosine similarity matrix S = K_norm @ K_norm^T
    3. Count entries > θ_xref per row (excluding self-similarity on the diagonal)

    Args:
        tags:       List of active Tag objects whose xref_count will be updated.
        theta_xref: Cosine similarity threshold for counting a cross-reference
                    edge.  Default 0.5 (from PRPConfig.theta_xref).
    """
    n: int = len(tags)

    # --- Edge cases ---
    if n == 0:
        return

    if n == 1:
        tags[0].xref_count = 0
        return

    # --- Stack keys onto a common device ---
    device: torch.device = tags[0].k.device
    keys: list[Tensor] = [tag.k.to(device) for tag in tags]
    K: Tensor = torch.stack(keys, dim=0)  # (N, d_tag)

    # --- L2-normalise rows ---
    K_norm: Tensor = torch.nn.functional.normalize(K, p=2, dim=1)  # (N, d_tag)

    # --- Pairwise cosine similarity via matmul ---
    S: Tensor = K_norm @ K_norm.T  # (N, N)

    # --- Mask out self-similarity (diagonal) ---
    mask: Tensor = ~torch.eye(n, dtype=torch.bool, device=device)  # (N, N)

    # --- Threshold and count per row ---
    above_threshold: Tensor = (S > theta_xref) & mask  # (N, N)
    counts: Tensor = above_threshold.sum(dim=1)  # (N,)

    # --- Write back raw counts ---
    counts_list: list[int] = counts.tolist()
    for i, tag in enumerate(tags):
        tag.xref_count = counts_list[i]
