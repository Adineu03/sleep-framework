"""Tests for sleep.sleep_engine.replay — replay generation for the sleep engine."""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sleep.config import SleepConfig
from sleep.sleep_engine.replay import ReplaySample, generate_replay
from sleep.tagging.tags import Tag


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


@pytest.fixture(scope="module")
def sleep_config():
    return SleepConfig()


@pytest.fixture(scope="module")
def sample_tag():
    """A valid tag with a span [10, 60) in a 100-token sequence."""
    return Tag(
        k=torch.randn(128),
        s=0.8,
        s0=0.8,
        s_reinforced=0.0,
        t0=0,
        e0=1.5,
        a=0,
        rho=0.0,
        ctx=(10, 60, "test_source"),
        S_score=0.7,
    )


@pytest.fixture(scope="module")
def original_tokens(tokenizer):
    """A 100-token tensor from encoding sample text."""
    text = "The quick brown fox jumps over the lazy dog. " * 20
    ids = tokenizer.encode(text, add_special_tokens=False)
    return torch.tensor(ids[:100], dtype=torch.long)


class TestGenerateReplayValid:
    """Test generate_replay with valid inputs returns a proper ReplaySample."""

    def test_returns_replay_sample(
        self, gpt2_model, tokenizer, sleep_config, sample_tag, original_tokens
    ):
        result = generate_replay(
            tag=sample_tag,
            model=gpt2_model,
            tokenizer=tokenizer,
            original_tokens=original_tokens,
            config=sleep_config,
            device="cpu",
        )
        assert result is not None
        assert isinstance(result, ReplaySample)

    def test_replay_has_correct_fields(
        self, gpt2_model, tokenizer, sleep_config, sample_tag, original_tokens
    ):
        result = generate_replay(
            tag=sample_tag,
            model=gpt2_model,
            tokenizer=tokenizer,
            original_tokens=original_tokens,
            config=sleep_config,
            device="cpu",
        )
        assert result is not None
        assert isinstance(result.text_ids, torch.Tensor)
        assert result.text_ids.dim() == 1
        assert result.tag_id == id(sample_tag)
        assert result.prp_score == sample_tag.S_score
        assert result.original_length == 50  # span_end - span_start = 60 - 10


class TestGenerateReplayNoneTokens:
    """Test generate_replay with None original_tokens returns None."""

    def test_none_original_tokens_returns_none(
        self, gpt2_model, tokenizer, sleep_config, sample_tag
    ):
        result = generate_replay(
            tag=sample_tag,
            model=gpt2_model,
            tokenizer=tokenizer,
            original_tokens=None,
            config=sleep_config,
            device="cpu",
        )
        assert result is None


class TestGenerateReplayOutOfBounds:
    """Test generate_replay with out-of-bounds span returns None."""

    def test_span_exceeds_token_length(
        self, gpt2_model, tokenizer, sleep_config, original_tokens
    ):
        oob_tag = Tag(
            k=torch.randn(128),
            s=0.8,
            s0=0.8,
            s_reinforced=0.0,
            t0=0,
            e0=1.5,
            a=0,
            rho=0.0,
            ctx=(50, 200, "test_source"),  # span_end=200 > len=100
            S_score=0.7,
        )
        result = generate_replay(
            tag=oob_tag,
            model=gpt2_model,
            tokenizer=tokenizer,
            original_tokens=original_tokens,
            config=sleep_config,
            device="cpu",
        )
        assert result is None


class TestReplaySampleSeedLength:
    """Test that ReplaySample has correct seed_length."""

    def test_seed_length_is_positive(
        self, gpt2_model, tokenizer, sleep_config, sample_tag, original_tokens
    ):
        result = generate_replay(
            tag=sample_tag,
            model=gpt2_model,
            tokenizer=tokenizer,
            original_tokens=original_tokens,
            config=sleep_config,
            device="cpu",
        )
        assert result is not None
        assert result.seed_length >= 2
        assert result.seed_length <= sleep_config.seed_length_max


class TestReplayCompression:
    """Test that replay is shorter than original for long spans."""

    def test_replay_shorter_than_original(
        self, gpt2_model, tokenizer, sleep_config
    ):
        # Create a long span tag (0, 500) with 500 tokens
        text = "The quick brown fox jumps over the lazy dog. " * 100
        tok = AutoTokenizer.from_pretrained("gpt2")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        ids = tok.encode(text, add_special_tokens=False)
        long_tokens = torch.tensor(ids[:500], dtype=torch.long)

        long_tag = Tag(
            k=torch.randn(128),
            s=0.8,
            s0=0.8,
            s_reinforced=0.0,
            t0=0,
            e0=1.5,
            a=0,
            rho=0.0,
            ctx=(0, 500, "test_source"),
            S_score=0.7,
        )

        result = generate_replay(
            tag=long_tag,
            model=gpt2_model,
            tokenizer=tok,
            original_tokens=long_tokens,
            config=sleep_config,
            device="cpu",
        )
        assert result is not None
        # Replay (text_ids) should be shorter than original span length (500)
        assert result.text_ids.shape[0] < result.original_length
