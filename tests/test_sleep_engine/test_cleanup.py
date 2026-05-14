"""Tests for sleep.sleep_engine.cleanup — validation and tag cleanup."""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sleep.config import SleepConfig
from sleep.sleep_engine.cleanup import cleanup_tags, compute_span_surprise
from sleep.tagging.tags import Tag


@pytest.fixture(scope="module")
def sleep_config():
    return SleepConfig()


@pytest.fixture(scope="module")
def gpt2_model():
    """Load GPT-2 once for the entire module."""
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()
    return model


@pytest.fixture(scope="module")
def tokenizer():
    """Load GPT-2 tokenizer once for the entire module."""
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _make_tag(fail_count: int = 0, s: float = 0.8) -> Tag:
    """Helper to create a tag with specific fail_count and strength."""
    return Tag(
        k=torch.randn(128),
        s=s,
        s0=0.8,
        s_reinforced=0.0,
        t0=0,
        e0=1.5,
        a=0,
        rho=0.0,
        ctx=(10, 50, "test_source"),
        S_score=0.7,
        fail_count=fail_count,
    )


class TestCleanupTagsAllPassed:
    """Test cleanup_tags when all tags pass validation."""

    def test_all_in_passed_list(self, sleep_config):
        tags = [_make_tag() for _ in range(5)]
        validation = {id(t): True for t in tags}

        result = cleanup_tags(tags, validation, sleep_config)

        assert len(result["passed"]) == 5
        assert len(result["failed"]) == 0
        assert len(result["permanently_removed"]) == 0


class TestCleanupTagsAllFailed:
    """Test cleanup_tags when all tags fail validation."""

    def test_all_in_failed_list(self, sleep_config):
        tags = [_make_tag(fail_count=0, s=0.8) for _ in range(5)]
        original_strengths = [t.s for t in tags]
        validation = {id(t): False for t in tags}

        result = cleanup_tags(tags, validation, sleep_config)

        assert len(result["passed"]) == 0
        assert len(result["failed"]) == 5
        assert len(result["permanently_removed"]) == 0

        # Check that s was halved and fail_count incremented
        for i, tag in enumerate(tags):
            assert tag.s == pytest.approx(original_strengths[i] * 0.5)
            assert tag.fail_count == 1


class TestCleanupTagsPermanentRemoval:
    """Test that tags with 3 failures are permanently removed."""

    def test_three_failures_permanently_removed(self, sleep_config):
        # Tag already has fail_count=2, so one more failure -> 3 -> removed
        tags = [_make_tag(fail_count=2)]
        validation = {id(tags[0]): False}

        result = cleanup_tags(tags, validation, sleep_config)

        assert len(result["passed"]) == 0
        assert len(result["failed"]) == 0
        assert len(result["permanently_removed"]) == 1
        assert tags[0].fail_count == 3


class TestComputeSpanSurprise:
    """Test compute_span_surprise returns a positive float."""

    def test_returns_positive_float(self, gpt2_model, tokenizer):
        text = "The quick brown fox jumps over the lazy dog and runs away quickly."
        ids = tokenizer.encode(text, add_special_tokens=False)
        token_ids = torch.tensor(ids, dtype=torch.long)

        surprise = compute_span_surprise(
            model=gpt2_model,
            token_ids=token_ids,
            span_start=2,
            span_end=min(10, len(ids)),
            device="cpu",
        )
        assert isinstance(surprise, float)
        assert surprise > 0.0
