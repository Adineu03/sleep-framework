"""Tests for delayed recall accuracy evaluation (Module 6 — recall.py).

All tests mock the model's generate() to avoid loading GPT-2.
"""

import pytest
from unittest.mock import MagicMock, patch

import torch

from sleep.evaluation.recall import (
    RecallTestCase,
    _score_response,
    evaluate_recall,
)


# --------------------------------------------------------------------- #
# _score_response unit tests (no model needed)
# --------------------------------------------------------------------- #

def test_score_response_all_found():
    """All expected keywords present -> score = 1.0."""
    score = _score_response("The revenue was $4.2M in Q3", ["revenue", "Q3"])
    assert score == 1.0


def test_score_response_none_found():
    """No expected keywords present -> score = 0.0."""
    score = _score_response("The weather is nice today", ["revenue", "Q3"])
    assert score == 0.0


def test_score_response_partial():
    """One of two keywords present -> score = 0.5."""
    score = _score_response("The revenue increased", ["revenue", "Q3"])
    assert score == 0.5


def test_score_response_empty_keywords():
    """Empty expected_keywords -> score = 1.0 (vacuously true)."""
    score = _score_response("anything", [])
    assert score == 1.0


def test_score_response_case_insensitive():
    """Keyword matching should be case-insensitive."""
    score = _score_response("The REVENUE was high", ["revenue"])
    assert score == 1.0


# --------------------------------------------------------------------- #
# evaluate_recall with mocked model
# --------------------------------------------------------------------- #

def _make_mock_model_and_tokenizer(response_text: str):
    """Create mock model and tokenizer that 'generate' a fixed response.

    The mock model.generate() returns a tensor whose new tokens, when
    decoded by the mock tokenizer, produce ``response_text``.
    """
    model = MagicMock()
    model.eval = MagicMock()

    tokenizer = MagicMock()
    tokenizer.eos_token_id = 50256

    # tokenizer(prompt, return_tensors="pt") -> {"input_ids": tensor of shape (1, 5)}
    prompt_ids = torch.tensor([[1, 2, 3, 4, 5]])
    tokenizer.return_value = {"input_ids": prompt_ids}

    # model.generate returns prompt_ids + 3 "generated" tokens
    generated_tokens = torch.tensor([10, 11, 12])
    full_output = torch.cat([prompt_ids[0], generated_tokens]).unsqueeze(0)
    model.generate = MagicMock(return_value=full_output)

    # tokenizer.decode(generated_ids, ...) -> response_text
    tokenizer.decode = MagicMock(return_value=response_text)

    return model, tokenizer


def test_evaluate_recall_all_keywords_present():
    """Model generates text containing all keywords -> DRA = 1.0."""
    model, tokenizer = _make_mock_model_and_tokenizer("The revenue was $4.2M")

    test_cases = [
        RecallTestCase(
            source_id="doc1",
            prompt="What was the revenue?",
            expected_keywords=["revenue", "4.2M"],
        ),
    ]

    result = evaluate_recall(model, tokenizer, test_cases, device="cpu")
    assert result["dra"] == 1.0
    assert result["n_cases"] == 1
    assert result["per_case"][0]["score"] == 1.0


def test_evaluate_recall_no_keywords_present():
    """Model generates text missing all keywords -> DRA = 0.0."""
    model, tokenizer = _make_mock_model_and_tokenizer("I don't know")

    test_cases = [
        RecallTestCase(
            source_id="doc1",
            prompt="What was the revenue?",
            expected_keywords=["revenue", "4.2M"],
        ),
    ]

    result = evaluate_recall(model, tokenizer, test_cases, device="cpu")
    assert result["dra"] == 0.0


def test_evaluate_recall_partial():
    """Model generates text with 1 of 2 keywords -> DRA = 0.5."""
    model, tokenizer = _make_mock_model_and_tokenizer("The revenue increased significantly")

    test_cases = [
        RecallTestCase(
            source_id="doc1",
            prompt="What was the revenue?",
            expected_keywords=["revenue", "4.2M"],
        ),
    ]

    result = evaluate_recall(model, tokenizer, test_cases, device="cpu")
    assert result["dra"] == 0.5


def test_evaluate_recall_empty_cases():
    """Empty test_cases list -> DRA = 0.0, n_cases = 0."""
    model, tokenizer = _make_mock_model_and_tokenizer("")
    result = evaluate_recall(model, tokenizer, [], device="cpu")
    assert result["dra"] == 0.0
    assert result["n_cases"] == 0
