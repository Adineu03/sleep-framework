"""
Span segmentation of surprising tokens (Q1.2, Step 3).

Takes per-token boolean flags and merges them into surprising spans,
discarding short fragments and computing per-span aggregate statistics.

Algorithm:
    1. Flag tokens where e_t > theta_t  (done upstream)
    2. Merge adjacent flagged tokens within gap tolerance g
    3. Discard spans shorter than min_span
    4. For each surviving span [t_start, t_end]:
       E_span  = mean({e_t - mu_t : t in [t_start, t_end]})   (mean excess surprise)
       h_bar   = mean({h_t : t in [t_start, t_end]})           (mean-pooled hidden state)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Span:
    """A contiguous region of surprising tokens.

    Attributes:
        start: Inclusive start index in the token sequence.
        end: Inclusive end index in the token sequence.
        E_span: Mean excess surprise over the span (nats above running mean).
        h_bar: Mean-pooled final-layer hidden state, shape ``(d_model,)``.
    """

    start: int
    end: int
    E_span: float
    h_bar: torch.Tensor


def segment_spans(
    flags: list[bool],
    surprises: list[float],
    hidden_states: list[torch.Tensor],
    running_mean: float,
    gap_tolerance: int = 3,
    min_span: int = 4,
) -> list[Span]:
    """Merge boolean flags into surprising spans with aggregate statistics.

    Parameters
    ----------
    flags:
        Per-token boolean flags (True = token exceeded adaptive threshold).
    surprises:
        Per-token surprise values in nats.
    hidden_states:
        Per-token final-layer hidden states, each of shape ``(d_model,)``.
    running_mean:
        The current running mean mu at the time of flagging.  Used to
        compute excess surprise ``e_t - mu``.
    gap_tolerance:
        Maximum number of unflagged tokens allowed between two flagged
        groups before they are merged into a single span.
    min_span:
        Minimum span length (inclusive) — spans shorter than this are
        discarded.

    Returns
    -------
    list[Span]
        Surviving spans sorted by start position.
    """
    # ------------------------------------------------------------------
    # 1. Collect raw flagged intervals: [(start, end), ...]
    # ------------------------------------------------------------------
    raw_intervals: list[tuple[int, int]] = []
    i: int = 0
    n: int = len(flags)

    while i < n:
        if flags[i]:
            start = i
            # Extend as far as the flag is True.
            while i < n and flags[i]:
                i += 1
            raw_intervals.append((start, i - 1))  # inclusive end
        else:
            i += 1

    if not raw_intervals:
        return []

    # ------------------------------------------------------------------
    # 2. Merge intervals whose gap <= gap_tolerance
    # ------------------------------------------------------------------
    merged: list[tuple[int, int]] = [raw_intervals[0]]

    for start, end in raw_intervals[1:]:
        prev_start, prev_end = merged[-1]
        gap = start - prev_end - 1
        if gap <= gap_tolerance:
            # Extend the previous interval.
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    # ------------------------------------------------------------------
    # 3. Discard spans shorter than min_span
    # ------------------------------------------------------------------
    kept: list[tuple[int, int]] = [
        (s, e) for s, e in merged if (e - s + 1) >= min_span
    ]

    # ------------------------------------------------------------------
    # 4. Compute per-span aggregates
    # ------------------------------------------------------------------
    spans: list[Span] = []

    for s, e in kept:
        span_len: int = e - s + 1

        # Mean excess surprise: E_span = mean(e_t - mu for t in span)
        excess_sum: float = sum(surprises[t] - running_mean for t in range(s, e + 1))
        E_span: float = excess_sum / span_len

        # Mean-pooled hidden state: h_bar = mean(h_t for t in span)
        h_bar: torch.Tensor = torch.stack(
            [hidden_states[t] for t in range(s, e + 1)]
        ).mean(dim=0)

        spans.append(Span(start=s, end=e, E_span=E_span, h_bar=h_bar))

    return spans