"""
Tag data structure and key projection for the SLEEP tagging layer.

Implements Q1.1 (tag definition) and Q1.3 (tag creation function) from SLEEP_Formalization.md.

A tag τ = (k, s, s0, s_reinforced, t0, e0, a, ρ, ctx, ...) captures a single surprising span
detected during inference. The key vector k is a low-dimensional projection of the span's
mean-pooled hidden state, used for similarity-based retrieval and access detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from sleep.config import TaggingConfig


# ---------------------------------------------------------------------------
# Tag dataclass
# ---------------------------------------------------------------------------

@dataclass
class Tag:
    """A tagged memory from a surprising span.

    Core fields (from formalization):
        k              — Key vector, projected hidden state (d_tag,)
        s              — Current strength ∈ [0, 1], subject to decay and reinforcement
        s0             — Initial strength = sigmoid(α_init * E_span). Never modified.
        s_reinforced   — Cumulative reinforcement bonus from access events
        t0             — Creation timestamp (inference step)
        e0             — Initial prediction error (E_span)
        a              — Access count
        rho            — Cumulative utility ρ
        ctx            — (span_start, span_end, source_id)

    Implementation tracking fields:
        p              — PRP allocation flag (0=not allocated, 1=allocated)
        S_score        — Latest PRP composite score
        R              — Recency-weighted utility (exponentially decaying)
        R_last_update  — Step of last R update
        xref_count     — Cross-reference density count (cached)
        fail_count     — Consolidation failure count
        tag_type       — "novelty" or "revision" (Q6.1)
    """

    # --- Core fields (Q1.1) ---
    k: Tensor                                       # (d_tag,)
    s: float                                        # current strength ∈ [0, 1]
    s0: float                                       # initial strength (immutable after creation)
    s_reinforced: float                             # cumulative reinforcement ≥ 0
    t0: int                                         # creation step
    e0: float                                       # initial prediction error (E_span)
    a: int                                          # access count ≥ 0
    rho: float                                      # cumulative utility ≥ 0
    ctx: Tuple[int, int, str]                       # (span_start, span_end, source_id)

    # --- Implementation tracking fields ---
    p: int = 0                                      # PRP allocation flag {0, 1}
    S_score: float = 0.0                            # latest PRP composite score
    R: float = 0.0                                  # recency-weighted utility
    R_last_update: int = 0                          # step of last R update
    xref_count: int = 0                             # cross-reference density (cached)
    fail_count: int = 0                             # consolidation failure count
    tag_type: str = "novelty"                       # "novelty" or "revision"

    def to_dict(self) -> dict:
        """Serialize to a plain dict for checkpointing."""
        return {
            "k": self.k.detach().cpu().tolist(),
            "s": self.s,
            "s0": self.s0,
            "s_reinforced": self.s_reinforced,
            "t0": self.t0,
            "e0": self.e0,
            "a": self.a,
            "rho": self.rho,
            "ctx": list(self.ctx),
            "p": self.p,
            "S_score": self.S_score,
            "R": self.R,
            "R_last_update": self.R_last_update,
            "xref_count": self.xref_count,
            "fail_count": self.fail_count,
            "tag_type": self.tag_type,
        }

    @classmethod
    def from_dict(cls, d: dict, device: str = "cpu") -> Tag:
        """Deserialize from a plain dict (inverse of to_dict)."""
        return cls(
            k=torch.tensor(d["k"], dtype=torch.float32, device=device),
            s=float(d["s"]),
            s0=float(d["s0"]),
            s_reinforced=float(d["s_reinforced"]),
            t0=int(d["t0"]),
            e0=float(d["e0"]),
            a=int(d["a"]),
            rho=float(d["rho"]),
            ctx=tuple(d["ctx"]),
            p=int(d.get("p", 0)),
            S_score=float(d.get("S_score", 0.0)),
            R=float(d.get("R", 0.0)),
            R_last_update=int(d.get("R_last_update", 0)),
            xref_count=int(d.get("xref_count", 0)),
            fail_count=int(d.get("fail_count", 0)),
            tag_type=str(d.get("tag_type", "novelty")),
        )


# ---------------------------------------------------------------------------
# Tag key projection  (k = W_proj · h_bar + b_proj)
# ---------------------------------------------------------------------------

class TagKeyProjection(nn.Module):
    """Linear projection from model hidden states to tag key vectors.

    k = W_proj @ h_bar + b_proj

    W_proj ∈ R^(d_tag x d_model), initialized with Xavier uniform.
    This is a thin wrapper around nn.Linear so it participates in
    standard PyTorch serialization (state_dict / load_state_dict).
    """

    def __init__(self, d_model: int, d_tag: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, d_tag, bias=True)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h_bar: Tensor) -> Tensor:
        """Project a hidden state (or batch) to a tag key vector.

        Args:
            h_bar: Tensor of shape (..., d_model).

        Returns:
            Tensor of shape (..., d_tag).
        """
        return self.proj(h_bar)


# ---------------------------------------------------------------------------
# Tag creation factory  (Q1.3)
# ---------------------------------------------------------------------------

def create_tag(
    h_bar: Tensor,
    E_span: float,
    step: int,
    ctx: Tuple[int, int, str],
    config: TaggingConfig,
    key_projection: TagKeyProjection,
    tag_type: str = "novelty",
) -> Tag:
    """Create a new tag from a surprising span.

    Implements Q1.3:
        k  = W_proj · h_bar + b_proj
        s0 = sigmoid(alpha_init * E_span)
        τ  = (k, s0, s0, 0, step, E_span, 0, 0, ctx, ...)

    Args:
        h_bar:          Mean-pooled hidden state of the span, shape (d_model,).
        E_span:         Mean excess surprise of the span (> 0).
        step:           Current inference step (creation timestamp).
        ctx:            (span_start, span_end, source_id).
        config:         TaggingConfig with alpha_init and other hyperparameters.
        key_projection: TagKeyProjection module for h_bar -> k.
        tag_type:       "novelty" (default) or "revision" (Q6.1).

    Returns:
        A fully initialized Tag.
    """
    with torch.no_grad():
        k = key_projection(h_bar)  # (d_tag,)

    s0 = torch.sigmoid(torch.tensor(config.alpha_init * E_span)).item()

    return Tag(
        k=k.detach(),
        s=s0,
        s0=s0,
        s_reinforced=0.0,
        t0=step,
        e0=E_span,
        a=0,
        rho=0.0,
        ctx=ctx,
        p=0,
        S_score=0.0,
        R=0.0,
        R_last_update=step,
        xref_count=0,
        fail_count=0,
        tag_type=tag_type,
    )