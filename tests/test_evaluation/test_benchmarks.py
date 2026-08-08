"""Tests for sleep.evaluation.benchmarks standard-benchmark loaders."""

from __future__ import annotations

import json

import pytest

from sleep.evaluation.benchmarks import (
    REQUIRED_FACT_FIELDS,
    load_counterfactual_facts,
    load_lama_facts,
    load_real_facts,
    normalize_facts,
)


class TestNormalizeFacts:

    def test_fills_template(self):
        facts = normalize_facts([
            {"id": "1", "text": "t", "test_prompt": "p", "keywords": ["k"]},
        ])
        assert facts[0]["template"] == "default"

    def test_drops_incomplete(self):
        facts = normalize_facts([
            {"id": "1", "text": "t", "test_prompt": "p"},  # no keywords
            {"id": "2", "text": "t", "test_prompt": "p", "keywords": ["k"]},
        ])
        assert len(facts) == 1 and facts[0]["id"] == "2"

    def test_string_keyword_coerced_to_list(self):
        facts = normalize_facts([
            {"id": "1", "text": "t", "test_prompt": "p", "keywords": "k"},
        ])
        assert facts[0]["keywords"] == ["k"]


class TestLoadLama:

    def test_loads_masked_sentence(self, tmp_path):
        p = tmp_path / "lama.json"
        p.write_text(json.dumps([
            {"sub_label": "Paris", "obj_label": "France",
             "masked_sentence": "[X] is the capital of [MASK]."},
        ]))
        facts = load_lama_facts(str(p))
        assert len(facts) == 1
        f = facts[0]
        assert all(f.get(k) for k in REQUIRED_FACT_FIELDS)
        assert "France" in f["text"]
        assert f["keywords"] == ["France"]
        assert "____" in f["test_prompt"]

    def test_jsonl(self, tmp_path):
        p = tmp_path / "lama.jsonl"
        p.write_text(
            '{"sub_label":"A","obj_label":"B","masked_sentence":"[X] rel [MASK]."}\n'
            '{"sub_label":"C","obj_label":"D","masked_sentence":"[X] rel [MASK]."}\n'
        )
        facts = load_lama_facts(str(p))
        assert len(facts) == 2


class TestLoadCounterfactual:

    def test_requested_rewrite(self, tmp_path):
        p = tmp_path / "cf.json"
        p.write_text(json.dumps([
            {"case_id": 42, "requested_rewrite": {
                "prompt": "The capital of {} is",
                "subject": "Atlantis",
                "target_new": {"str": "Poseidonville"},
                "relation_id": "P36",
            }},
        ]))
        facts = load_counterfactual_facts(str(p))
        assert len(facts) == 1
        f = facts[0]
        assert f["keywords"] == ["Poseidonville"]
        assert "Atlantis" in f["test_prompt"]
        assert f["template"] == "P36"


class TestLoadReal:

    def test_passthrough(self, tmp_path):
        p = tmp_path / "real.json"
        p.write_text(json.dumps([
            {"id": "r1", "text": "X happened in 2026.",
             "question": "When did X happen?", "keywords": ["2026"]},
        ]))
        facts = load_real_facts(str(p))
        assert facts[0]["test_prompt"] == "When did X happen?"
        assert facts[0]["template"] == "real"

    def test_unknown_benchmark_raises(self):
        from sleep.evaluation.benchmarks import load_benchmark
        with pytest.raises(ValueError):
            load_benchmark("nonexistent", "x.json")
