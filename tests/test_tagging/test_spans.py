"""Tests for sleep.tagging.spans — span segmentation of surprising tokens."""

import pytest
import torch

from sleep.tagging.spans import segment_spans


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_hidden_states(n: int, d_model: int = 32) -> list[torch.Tensor]:
    """Create n deterministic hidden-state vectors."""
    return [torch.full((d_model,), float(i)) for i in range(n)]


def _make_surprises(n: int, base: float = 5.0) -> list[float]:
    """Create n surprise values equal to base + index (so they're distinct)."""
    return [base + i for i in range(n)]


# ---------------------------------------------------------------------------
# Basic merging
# ---------------------------------------------------------------------------

class TestBasicMerging:

    def test_adjacent_flags_with_small_gap_merge(self):
        """[T,T,F,T,T] with gap_tolerance=1 should produce one span."""
        flags = [True, True, False, True, True]
        surprises = _make_surprises(5, base=10.0)
        hidden = _make_hidden_states(5)

        spans = segment_spans(
            flags=flags,
            surprises=surprises,
            hidden_states=hidden,
            running_mean=0.0,
            gap_tolerance=1,
            min_span=1,  # don't filter by length
        )

        assert len(spans) == 1
        assert spans[0].start == 0
        assert spans[0].end == 4


class TestGapTolerance:

    def test_large_gap_produces_separate_spans(self):
        """Flags separated by gap > tolerance should yield separate spans."""
        # T T F F F F T T  — gap of 4 between index 1 and 6
        flags = [True, True, False, False, False, False, True, True]
        surprises = _make_surprises(8, base=10.0)
        hidden = _make_hidden_states(8)

        spans = segment_spans(
            flags=flags,
            surprises=surprises,
            hidden_states=hidden,
            running_mean=0.0,
            gap_tolerance=2,
            min_span=1,
        )

        assert len(spans) == 2
        assert spans[0].end < spans[1].start


class TestMinSpanFiltering:

    def test_short_spans_are_discarded(self):
        """Spans shorter than min_span should be removed."""
        # Two flagged regions: length 2 and length 5
        flags = [True, True, False, False, False, False, False,
                 True, True, True, True, True]
        surprises = _make_surprises(12, base=10.0)
        hidden = _make_hidden_states(12)

        spans = segment_spans(
            flags=flags,
            surprises=surprises,
            hidden_states=hidden,
            running_mean=0.0,
            gap_tolerance=0,
            min_span=4,
        )

        # Only the length-5 span should survive
        assert len(spans) == 1
        assert spans[0].start == 7
        assert spans[0].end == 11

    def test_no_spans_when_all_too_short(self):
        """If every span is below min_span, return empty list."""
        flags = [True, False, True]
        spans = segment_spans(
            flags=flags,
            surprises=_make_surprises(3),
            hidden_states=_make_hidden_states(3),
            running_mean=0.0,
            gap_tolerance=0,
            min_span=10,
        )
        assert spans == []


class TestESpanComputation:

    def test_E_span_is_mean_excess_surprise(self):
        """E_span should equal mean(surprise_t - running_mean) over the span."""
        running_mean = 3.0
        surprises = [5.0, 7.0, 9.0, 11.0]
        flags = [True, True, True, True]
        hidden = _make_hidden_states(4)

        spans = segment_spans(
            flags=flags,
            surprises=surprises,
            hidden_states=hidden,
            running_mean=running_mean,
            gap_tolerance=0,
            min_span=1,
        )

        assert len(spans) == 1
        expected = sum(s - running_mean for s in surprises) / len(surprises)
        assert spans[0].E_span == pytest.approx(expected, rel=1e-6)


class TestHBarComputation:

    def test_h_bar_is_mean_of_hidden_states(self):
        """h_bar should be the element-wise mean of hidden states in the span."""
        d_model = 16
        # Use predictable values: h_t = [t, t, ..., t]
        hidden = [torch.full((d_model,), float(i)) for i in range(4)]
        flags = [True, True, True, True]
        surprises = [10.0] * 4

        spans = segment_spans(
            flags=flags,
            surprises=surprises,
            hidden_states=hidden,
            running_mean=0.0,
            gap_tolerance=0,
            min_span=1,
        )

        assert len(spans) == 1
        expected_h_bar = torch.stack(hidden).mean(dim=0)
        assert torch.allclose(spans[0].h_bar, expected_h_bar, atol=1e-6)

    def test_h_bar_shape(self):
        """h_bar should have shape (d_model,)."""
        d_model = 64
        hidden = _make_hidden_states(5, d_model=d_model)
        flags = [True] * 5

        spans = segment_spans(
            flags=flags,
            surprises=[10.0] * 5,
            hidden_states=hidden,
            running_mean=0.0,
            gap_tolerance=0,
            min_span=1,
        )

        assert spans[0].h_bar.shape == (d_model,)
