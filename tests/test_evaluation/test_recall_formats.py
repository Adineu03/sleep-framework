"""Tests for sleep.evaluation.recall_formats.

Covers ``group_facts_by_template``, ``multiple_choice_recall``,
``cloze_recall``, and ``free_form_recall``. Uses real GPT-2 (small) at
module-scope, matching the fixture pattern in tests/test_tagging/test_surprise.py.
"""

import pytest
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from sleep.evaluation.recall_formats import (
    cloze_recall,
    free_form_recall,
    group_facts_by_template,
    multiple_choice_recall,
)


# ---------------------------------------------------------------------------
# Module-scoped fixtures: load GPT-2 once for all tests in this file
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gpt2_tokenizer():
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return tok


@pytest.fixture(scope="module")
def gpt2_model():
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_alpha_facts():
    """Build a small set of facts using the 'alpha' template."""
    return [
        {
            "id": "a1",
            "template": "alpha",
            "text": "Alice visited Tokyo recently for a conference.",
            "test_prompt": "Where did Alice visit?",
            "keywords": ["Tokyo", "Alice"],
        },
        {
            "id": "a2",
            "template": "alpha",
            "text": "Bob travelled to Berlin last summer.",
            "test_prompt": "Where did Bob travel?",
            "keywords": ["Berlin", "Bob"],
        },
        {
            "id": "a3",
            "template": "alpha",
            "text": "Carol arrived in Madrid on Friday.",
            "test_prompt": "Where did Carol go?",
            "keywords": ["Madrid", "Carol"],
        },
        {
            "id": "a4",
            "template": "alpha",
            "text": "Dan flew to Paris in December.",
            "test_prompt": "Where did Dan fly?",
            "keywords": ["Paris", "Dan"],
        },
    ]


def _make_beta_facts():
    """Build a small set of facts using the 'beta' template."""
    return [
        {
            "id": "b1",
            "template": "beta",
            "text": "The price of widgets rose to $42 last quarter.",
            "test_prompt": "What was the widget price?",
            "keywords": ["$42", "widgets"],
        },
        {
            "id": "b2",
            "template": "beta",
            "text": "Gadget revenue hit $99 in Q3.",
            "test_prompt": "What was the gadget revenue?",
            "keywords": ["$99", "gadgets"],
        },
        {
            "id": "b3",
            "template": "beta",
            "text": "Sprocket sales totaled $15 last week.",
            "test_prompt": "What were sprocket sales?",
            "keywords": ["$15", "sprockets"],
        },
        {
            "id": "b4",
            "template": "beta",
            "text": "Cog earnings reached $77 this year.",
            "test_prompt": "What were cog earnings?",
            "keywords": ["$77", "cogs"],
        },
    ]


# ---------------------------------------------------------------------------
# group_facts_by_template
# ---------------------------------------------------------------------------

class TestGroupByTemplate:

    def test_groups_by_template_field(self):
        facts = _make_alpha_facts()[:2] + _make_beta_facts()[:3]
        grouped = group_facts_by_template(facts)
        assert set(grouped.keys()) == {"alpha", "beta"}
        assert len(grouped["alpha"]) == 2
        assert len(grouped["beta"]) == 3

    def test_handles_single_template(self):
        facts = _make_alpha_facts()
        grouped = group_facts_by_template(facts)
        assert list(grouped.keys()) == ["alpha"]
        assert len(grouped["alpha"]) == len(facts)

    def test_handles_empty_list(self):
        assert group_facts_by_template([]) == {}

    def test_facts_without_template_bucket(self):
        """Facts missing a 'template' key fall into '_untemplated'."""
        facts = [{"id": "x", "keywords": ["foo"]}]
        grouped = group_facts_by_template(facts)
        assert "_untemplated" in grouped
        assert len(grouped["_untemplated"]) == 1


# ---------------------------------------------------------------------------
# multiple_choice_recall
# ---------------------------------------------------------------------------

class TestMultipleChoiceRecall:

    def test_returns_correct_schema(self, gpt2_model, gpt2_tokenizer):
        facts = _make_alpha_facts() + _make_beta_facts()[:1]  # 5 facts
        grouped = group_facts_by_template(facts)
        result = multiple_choice_recall(gpt2_model, gpt2_tokenizer, facts, grouped)

        # Top-level schema
        assert set(result.keys()) >= {"per_fact", "accuracy", "mean_correct_prob", "n_facts"}
        assert isinstance(result["per_fact"], list)
        assert len(result["per_fact"]) == len(facts)

        # Per-fact schema
        for entry in result["per_fact"]:
            assert {"fact_id", "correct_letter", "predicted_letter",
                    "is_correct", "option_probs", "options"} <= set(entry.keys())

    def test_option_probs_sum_to_one(self, gpt2_model, gpt2_tokenizer):
        facts = _make_alpha_facts()
        grouped = group_facts_by_template(facts)
        result = multiple_choice_recall(gpt2_model, gpt2_tokenizer, facts, grouped)
        for entry in result["per_fact"]:
            total = sum(entry["option_probs"].values())
            assert total == pytest.approx(1.0, abs=1e-5)

    def test_correct_letter_is_one_of_ABCD(self, gpt2_model, gpt2_tokenizer):
        facts = _make_alpha_facts()
        grouped = group_facts_by_template(facts)
        result = multiple_choice_recall(gpt2_model, gpt2_tokenizer, facts, grouped)
        for entry in result["per_fact"]:
            assert entry["correct_letter"] in {"A", "B", "C", "D"}
            assert entry["predicted_letter"] in {"A", "B", "C", "D"}

    def test_n_facts_matches_input(self, gpt2_model, gpt2_tokenizer):
        facts = _make_alpha_facts()  # 4 facts, all have keywords
        grouped = group_facts_by_template(facts)
        result = multiple_choice_recall(gpt2_model, gpt2_tokenizer, facts, grouped)
        assert result["n_facts"] == len(facts)

    def test_distractors_come_from_same_template(self, gpt2_model, gpt2_tokenizer):
        """Distractors for an alpha fact should be drawn from other alpha facts."""
        alpha = _make_alpha_facts()
        beta = _make_beta_facts()
        all_facts = alpha + beta
        grouped = group_facts_by_template(all_facts)

        # Evaluate just the first alpha fact
        result = multiple_choice_recall(gpt2_model, gpt2_tokenizer, [alpha[0]], grouped)
        entry = result["per_fact"][0]

        # The 4 displayed options: 1 correct + 3 distractors. All four should be
        # first-keywords of alpha facts (none from beta).
        alpha_first_keywords = {f["keywords"][0] for f in alpha}
        beta_first_keywords = {f["keywords"][0] for f in beta}
        for letter, opt in entry["options"].items():
            assert opt in alpha_first_keywords
            assert opt not in beta_first_keywords

    def test_handles_template_with_few_facts(self, gpt2_model, gpt2_tokenizer):
        """Single-fact template falls back to generic distractors without crashing."""
        lonely = [{
            "id": "lone1",
            "template": "lonely",
            "text": "The answer is 42.",
            "test_prompt": "What is the answer?",
            "keywords": ["42"],
        }]
        grouped = group_facts_by_template(lonely)
        result = multiple_choice_recall(gpt2_model, gpt2_tokenizer, lonely, grouped)
        assert result["n_facts"] == 1
        # Schema still intact, four options present.
        assert len(result["per_fact"][0]["options"]) == 4

    def test_deterministic_across_runs(self, gpt2_model, gpt2_tokenizer):
        """Identical inputs should yield identical per-fact predictions."""
        facts = _make_alpha_facts()
        grouped = group_facts_by_template(facts)
        r1 = multiple_choice_recall(gpt2_model, gpt2_tokenizer, facts, grouped)
        r2 = multiple_choice_recall(gpt2_model, gpt2_tokenizer, facts, grouped)
        for e1, e2 in zip(r1["per_fact"], r2["per_fact"]):
            assert e1["fact_id"] == e2["fact_id"]
            assert e1["correct_letter"] == e2["correct_letter"]
            assert e1["predicted_letter"] == e2["predicted_letter"]
            assert e1["options"] == e2["options"]

    def test_empty_facts_returns_zero_n(self, gpt2_model, gpt2_tokenizer):
        result = multiple_choice_recall(gpt2_model, gpt2_tokenizer, [], {})
        assert result["n_facts"] == 0
        assert result["per_fact"] == []


# ---------------------------------------------------------------------------
# cloze_recall
# ---------------------------------------------------------------------------

class TestClozeRecall:

    def test_returns_correct_schema(self, gpt2_model, gpt2_tokenizer):
        facts = _make_alpha_facts() + _make_beta_facts()[:1]  # 5 facts
        result = cloze_recall(gpt2_model, gpt2_tokenizer, facts)

        assert set(result.keys()) >= {"per_fact", "accuracy", "n_facts"}
        # All facts have keywords appearing mid-text -> all should be evaluated.
        assert result["n_facts"] == len(facts)
        for entry in result["per_fact"]:
            assert {"fact_id", "prompt", "expected_keyword", "generation",
                    "is_correct"} <= set(entry.keys())

    def test_keyword_in_text_creates_prompt(self, gpt2_model, gpt2_tokenizer):
        """For 'Alice visited Tokyo recently...' with keyword 'Tokyo',
        the prompt is the prefix up to (but not including) 'Tokyo'."""
        fact = {
            "id": "k1",
            "template": "alpha",
            "text": "Alice visited Tokyo recently for a conference.",
            "test_prompt": "Where did Alice visit?",
            "keywords": ["Tokyo"],
        }
        result = cloze_recall(gpt2_model, gpt2_tokenizer, [fact])
        assert result["n_facts"] == 1
        entry = result["per_fact"][0]
        assert entry["prompt"] == "Alice visited "
        assert entry["expected_keyword"] == "Tokyo"

    def test_skips_when_keyword_at_start(self, gpt2_model, gpt2_tokenizer):
        """A fact whose keyword starts at index 0 should be skipped (empty prompt)."""
        fact = {
            "id": "skip1",
            "template": "alpha",
            "text": "Tokyo is the capital of Japan.",
            "test_prompt": "What city?",
            "keywords": ["Tokyo"],
        }
        result = cloze_recall(gpt2_model, gpt2_tokenizer, [fact])
        assert result["n_facts"] == 0
        assert result["per_fact"] == []

    def test_no_keyword_in_text_skipped(self, gpt2_model, gpt2_tokenizer):
        """A fact whose keyword does not appear in text should be skipped."""
        fact = {
            "id": "skip2",
            "template": "alpha",
            "text": "Some unrelated sentence about clouds.",
            "test_prompt": "What is mentioned?",
            "keywords": ["Tokyo"],
        }
        result = cloze_recall(gpt2_model, gpt2_tokenizer, [fact])
        assert result["n_facts"] == 0

    def test_max_facts(self, gpt2_model, gpt2_tokenizer):
        """Mix valid and skip-worthy facts; per_fact length matches n_facts."""
        facts = _make_alpha_facts()  # 4 valid
        # Add 2 skip-worthy facts.
        facts.append({
            "id": "skipA",
            "template": "alpha",
            "text": "Tokyo starts at index 0 here.",
            "test_prompt": "x?",
            "keywords": ["Tokyo"],
        })
        facts.append({
            "id": "skipB",
            "template": "alpha",
            "text": "No relevant token in this string.",
            "test_prompt": "x?",
            "keywords": ["Vienna"],
        })
        result = cloze_recall(gpt2_model, gpt2_tokenizer, facts)
        assert result["n_facts"] <= len(facts)
        assert len(result["per_fact"]) == result["n_facts"]
        # The 4 valid alpha facts should all have made it through.
        assert result["n_facts"] == 4

    def test_empty_facts(self, gpt2_model, gpt2_tokenizer):
        result = cloze_recall(gpt2_model, gpt2_tokenizer, [])
        assert result["n_facts"] == 0
        assert result["per_fact"] == []


# ---------------------------------------------------------------------------
# free_form_recall
# ---------------------------------------------------------------------------

class TestFreeFormRecall:

    def test_returns_correct_schema(self, gpt2_model, gpt2_tokenizer):
        facts = _make_alpha_facts()[:3]
        torch.manual_seed(0)
        result = free_form_recall(gpt2_model, gpt2_tokenizer, facts)

        assert set(result.keys()) >= {"per_fact", "mean_score", "accuracy", "n_facts"}
        assert result["n_facts"] == len(facts)
        for entry in result["per_fact"]:
            assert {"fact_id", "score", "is_correct", "found_keywords",
                    "expected_keywords", "generation", "prompt"} <= set(entry.keys())
            assert 0.0 <= entry["score"] <= 1.0

    def test_score_fraction_is_correct(self, gpt2_model, gpt2_tokenizer, monkeypatch):
        """A generation containing 2 of 4 expected keywords should score 0.5."""
        # Patch tokenizer.decode to inject a controlled "generation" containing
        # exactly two of the four expected keywords.
        fact = {
            "id": "ff1",
            "template": "alpha",
            "text": "irrelevant",
            "test_prompt": "Generate something:",
            "keywords": ["alpha", "bravo", "charlie", "delta"],
        }
        # Wrap the real decode: when decoding the generated suffix, return a
        # canned string. Call original for any other decode use.
        original_decode = gpt2_tokenizer.decode

        call_count = {"n": 0}

        def fake_decode(ids, *args, **kwargs):
            call_count["n"] += 1
            return "alpha and bravo only"

        monkeypatch.setattr(gpt2_tokenizer, "decode", fake_decode)
        try:
            result = free_form_recall(gpt2_model, gpt2_tokenizer, [fact])
        finally:
            monkeypatch.setattr(gpt2_tokenizer, "decode", original_decode)

        assert result["n_facts"] == 1
        entry = result["per_fact"][0]
        assert entry["score"] == pytest.approx(0.5)
        assert set(entry["found_keywords"]) == {"alpha", "bravo"}
        # 0.5 >= 1/3 -> is_correct True
        assert entry["is_correct"] is True

    def test_is_correct_threshold(self, gpt2_model, gpt2_tokenizer, monkeypatch):
        """is_correct iff score >= 1/3 of expected keywords."""
        # Three keywords: 1/3 found -> score = 1/3 -> is_correct True.
        fact_pass = {
            "id": "fp",
            "template": "alpha",
            "text": "irrelevant",
            "test_prompt": "P:",
            "keywords": ["foo", "bar", "baz"],
        }
        # Three keywords: 0/3 found -> score = 0 -> is_correct False.
        fact_fail = {
            "id": "ff",
            "template": "alpha",
            "text": "irrelevant",
            "test_prompt": "P:",
            "keywords": ["foo", "bar", "baz"],
        }

        # Decode returns different content depending on which call it is.
        # In free_form_recall, decode is called once per fact.
        responses = iter(["foo and nothing else", "completely unrelated"])

        def fake_decode(ids, *args, **kwargs):
            return next(responses)

        monkeypatch.setattr(gpt2_tokenizer, "decode", fake_decode)

        result = free_form_recall(
            gpt2_model, gpt2_tokenizer, [fact_pass, fact_fail]
        )
        assert result["n_facts"] == 2
        e_pass = result["per_fact"][0]
        e_fail = result["per_fact"][1]

        assert e_pass["score"] == pytest.approx(1.0 / 3.0)
        assert e_pass["is_correct"] is True
        assert e_fail["score"] == pytest.approx(0.0)
        assert e_fail["is_correct"] is False

    def test_empty_facts(self, gpt2_model, gpt2_tokenizer):
        result = free_form_recall(gpt2_model, gpt2_tokenizer, [])
        assert result["n_facts"] == 0
        assert result["per_fact"] == []
