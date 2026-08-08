"""Tests for the two-stage-validation mini-recall gate (mentor item #6).

We test :func:`sleep.sleep_engine.cleanup.mini_recall_check` — the external
free-form gate that Stage 2 adds on top of the surprise-reduction proxy — with
a controllable stub model/tokenizer so the gate logic (keyword hit, fail-closed
on missing data) is verified deterministically without a real forward pass.
"""

from __future__ import annotations

import torch

from sleep.sleep_engine.cleanup import mini_recall_check


class _StubTokenizer:
    eos_token_id = 0

    def __init__(self, generated_text: str):
        self._generated = generated_text

    def __call__(self, prompt, return_tensors="pt"):
        return {"input_ids": torch.zeros((1, 3), dtype=torch.long)}

    def decode(self, ids, skip_special_tokens=True):
        # The gate slices output[:, input_len:]; we ignore ids and return the
        # controlled generation so the keyword-matching logic is what's tested.
        return self._generated


class _StubModel:
    def generate(self, input_ids, max_new_tokens=40, **kwargs):
        # Return a sequence longer than the input so the slice is non-empty.
        pad = torch.zeros((1, max_new_tokens), dtype=torch.long)
        return torch.cat([input_ids, pad], dim=1)


class _Tag:
    ctx = (0, 3, "src")


def _fact(**kw):
    base = {"test_prompt": "What is the capital?", "keywords": ["Poseidonville"]}
    base.update(kw)
    return base


class TestMiniRecallCheck:

    def test_pass_when_keyword_present(self):
        model = _StubModel()
        tok = _StubTokenizer("The answer is Poseidonville, established 2026.")
        assert mini_recall_check(_Tag(), model, _fact(), tok) is True

    def test_fail_when_keyword_absent(self):
        model = _StubModel()
        tok = _StubTokenizer("I am not sure about that.")
        assert mini_recall_check(_Tag(), model, _fact(), tok) is False

    def test_case_insensitive(self):
        model = _StubModel()
        tok = _StubTokenizer("it is POSEIDONVILLE.")
        assert mini_recall_check(_Tag(), model, _fact(), tok) is True

    def test_none_fact_fails_closed(self):
        assert mini_recall_check(_Tag(), _StubModel(), None, _StubTokenizer("x")) is False

    def test_missing_keywords_fails_closed(self):
        model = _StubModel()
        tok = _StubTokenizer("anything")
        assert mini_recall_check(_Tag(), model, _fact(keywords=[]), tok) is False

    def test_missing_prompt_fails_closed(self):
        model = _StubModel()
        tok = _StubTokenizer("anything")
        assert mini_recall_check(_Tag(), model, _fact(test_prompt=""), tok) is False

    def test_any_of_multiple_keywords_hits(self):
        model = _StubModel()
        tok = _StubTokenizer("mentions only 2026 here")
        fact = _fact(keywords=["Poseidonville", "2026", "Atlantis"])
        assert mini_recall_check(_Tag(), model, fact, tok) is True
