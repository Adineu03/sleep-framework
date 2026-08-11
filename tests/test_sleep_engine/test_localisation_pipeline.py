"""Integration test: the localisation consolidation path through SleepEngine.

Exercises the exact seam Phase B runs on GPU — paraphrase replay + distill
training into a mid-MLP w_cons under the full safety machinery, with two-stage
validation — end-to-end on a tiny Qwen2, so the wiring is ground-truth-checked
before any GPU spend.
"""

from __future__ import annotations

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from sleep.config import SleepConfig, WeightsConfig
from sleep.sleep_engine import SleepEngine
from sleep.weights import DualWeightSystem


class _Tok:
    """Char-level stub tokenizer with the interfaces the engine touches."""

    eos_token_id = 0
    pad_token_id = 0

    def __call__(self, text, return_tensors="pt", **kw):
        ids = [(ord(c) % 96) + 2 for c in text[:48]] or [2, 3]
        return type("Enc", (), {"input_ids": torch.tensor([ids])})()

    def decode(self, ids, skip_special_tokens=True):
        return "decoded"


class _Tag:
    """Minimal tag satisfying the engine's contract."""

    def __init__(self, source_id):
        self.ctx = (0, 4, source_id)
        self.p = 1.0
        self.s = 1.0
        self.fail_count = 0


def _system(train_mode="distill"):
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=100, hidden_size=32, intermediate_size=64,
        num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=256, sliding_window=None,
        attn_implementation="eager", torch_dtype=torch.float32,
    ))
    wcfg = WeightsConfig(
        lora_rank=2, lora_alpha=4,
        adapted_matrices=["up_proj", "down_proj"],
        layer_selection="middle",
        injection_selection="top",
        alpha_slow=5e-3,  # tiny model: make movement visible in few steps
        delta_max=1.0,    # don't clip away the movement in a 10-step test
    )
    dws = DualWeightSystem(
        model, wcfg, use_kv_memory_for_fast=True, kv_max_total_tokens=128,
    )
    engine = SleepEngine(
        dual_weights=dws,
        tokenizer=_Tok(),
        sleep_config=SleepConfig(),
        weights_config=wcfg,
        mu_surprise=1.0,
        device="cpu",
        replay_strategy="paraphrase",
        train_mode=train_mode,
        train_steps_override=10,
        two_stage_validation=True,
    )
    return dws, engine


_FACTS = {
    "f1": {"id": "f1", "text": "alpha beta gamma delta epsilon",
           "test_prompt": "which greek letters?",
           "keywords": ["zzz_never_generated"],
           "paraphrases": ["alpha beta gamma", "Question: greek?\nAnswer: alpha beta",
                           "the letters are alpha beta gamma delta"]},
    "f2": {"id": "f2", "text": "one two three four five six",
           "test_prompt": "which numbers?",
           "keywords": ["zzz_never_generated"],
           "paraphrases": ["one two three", "Question: numbers?\nAnswer: one two",
                           "counting one two three four"]},
}


def _tokens(text):
    return _Tok()(text).input_ids[0]


class TestLocalisationPipeline:

    def test_paraphrase_distill_cycle_runs_and_moves_wcons(self):
        dws, engine = _system("distill")
        before = dws.save_cons_checkpoint()

        candidates = [_Tag("f1"), _Tag("f2")]
        tokens_map = {k: _tokens(v["text"]) for k, v in _FACTS.items()}
        result = engine.run_cycle(
            candidates=candidates,
            original_tokens_map=tokens_map,
            key_projection=None,
            fact_map=_FACTS,
        )

        # Paraphrase replay: 2 sources x 3 wordings = 6 accepted samples.
        assert result["n_replays_generated"] == 6
        assert result["n_replays_accepted"] == 6
        # Distill training ran the overridden number of steps.
        assert result["training_stats"]["n_steps"] == 10
        assert result["training_stats"]["mean_loss"] == result["training_stats"]["mean_loss"]
        # w_cons moved.
        after = dws.save_cons_checkpoint()
        moved = any(
            not torch.allclose(before[k], after[k]) for k in before
        )
        assert moved, "distill cycle did not move w_cons"
        # Two-stage bookkeeping present (recall gate can't pass with the
        # impossible keyword, which is the point: fail-closed counts).
        assert result["n_passed_recall_gate"] is not None

    def test_paraphrase_ce_cycle_runs(self):
        dws, engine = _system("ce")
        candidates = [_Tag("f1"), _Tag("f2")]
        tokens_map = {k: _tokens(v["text"]) for k, v in _FACTS.items()}
        result = engine.run_cycle(
            candidates=candidates,
            original_tokens_map=tokens_map,
            key_projection=None,
            fact_map=_FACTS,
        )
        assert result["n_replays_accepted"] == 6
        assert result["training_stats"]["n_steps"] == 10

    def test_paraphrase_without_fact_map_raises(self):
        _, engine = _system("distill")
        with pytest.raises(RuntimeError, match="fact_map"):
            engine.run_cycle(
                candidates=[_Tag("f1")],
                original_tokens_map={"f1": _tokens("alpha beta gamma")},
                key_projection=None,
                fact_map=None,
            )

    def test_invalid_train_mode_rejected(self):
        dws, _ = _system("ce")
        with pytest.raises(ValueError, match="train_mode"):
            SleepEngine(
                dual_weights=dws, tokenizer=_Tok(),
                sleep_config=SleepConfig(), weights_config=WeightsConfig(),
                mu_surprise=1.0, train_mode="rlhf",
            )
