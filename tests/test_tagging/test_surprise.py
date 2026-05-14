"""Tests for sleep.tagging.surprise — per-token surprise computation."""

import pytest
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from sleep.tagging.surprise import compute_surprise


# ---------------------------------------------------------------------------
# Module-scoped fixture: load GPT-2 once for all tests in this file
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gpt2_tokenizer():
    return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture(scope="module")
def gpt2_model():
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestComputeSurprise:

    def test_output_lengths_match_input(self, gpt2_tokenizer, gpt2_model):
        """surprises, hidden_states, and tokens all have length == seq_len."""
        text = "The quick brown fox jumps over the lazy dog"
        token_ids = gpt2_tokenizer.encode(text, return_tensors="pt").squeeze(0)
        seq_len = token_ids.shape[0]

        result = compute_surprise(gpt2_model, token_ids)

        assert len(result.surprises) == seq_len
        assert len(result.hidden_states) == seq_len
        assert result.tokens.shape == (seq_len,)

    def test_first_token_surprise_is_zero(self, gpt2_tokenizer, gpt2_model):
        """The first token has no prior context so its surprise must be 0."""
        token_ids = gpt2_tokenizer.encode("Hello world", return_tensors="pt").squeeze(0)
        result = compute_surprise(gpt2_model, token_ids)
        assert result.surprises[0] == 0.0

    def test_surprises_are_non_negative(self, gpt2_tokenizer, gpt2_model):
        """Shannon surprise is -log p >= 0 (probabilities <= 1)."""
        text = "Machine learning is a subfield of artificial intelligence"
        token_ids = gpt2_tokenizer.encode(text, return_tensors="pt").squeeze(0)
        result = compute_surprise(gpt2_model, token_ids)

        for s in result.surprises:
            assert s >= 0.0, f"Surprise should be non-negative, got {s}"

    def test_hidden_states_have_correct_d_model(self, gpt2_tokenizer, gpt2_model):
        """Each hidden state vector should have dimension d_model (768 for gpt2)."""
        d_model = gpt2_model.config.hidden_size  # 768 for gpt2

        token_ids = gpt2_tokenizer.encode("Testing hidden states", return_tensors="pt").squeeze(0)
        result = compute_surprise(gpt2_model, token_ids)

        for h in result.hidden_states:
            assert h.shape == (d_model,), f"Expected ({d_model},), got {h.shape}"

    def test_tokens_match_input(self, gpt2_tokenizer, gpt2_model):
        """Returned token IDs should match the input (moved to cpu)."""
        token_ids = gpt2_tokenizer.encode("A B C", return_tensors="pt").squeeze(0)
        result = compute_surprise(gpt2_model, token_ids)
        assert torch.equal(result.tokens, token_ids.cpu())
