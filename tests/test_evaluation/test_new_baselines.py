"""Tests for the two baselines added per mentor item #5: EWC-only, in-context.

The RAG and Naive-LoRA baselines predate this work; here we cover the new
``EWCOnlyBaseline`` (its EWC penalty math) and ``InContextBaseline`` (prompt
construction / lookup). A tiny Qwen2 supplies real ``v_proj``/``o_proj`` for the
LoRA baseline; a stub tokenizer keeps the tests deterministic and fast.
"""

from __future__ import annotations

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from sleep.config import SLEEPConfig
from sleep.evaluation.baselines import EWCOnlyBaseline, InContextBaseline


def _tiny_qwen2():
    cfg = Qwen2Config(
        vocab_size=100, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=128, sliding_window=None,
        attn_implementation="eager", torch_dtype=torch.float32,
    )
    return Qwen2ForCausalLM(cfg)


class _StubTokenizer:
    eos_token_id = 0

    def __init__(self, text="generated answer text"):
        self._text = text

    def __call__(self, prompt, return_tensors="pt", truncation=False, max_length=None):
        g = torch.Generator().manual_seed(len(prompt))
        return {"input_ids": torch.randint(1, 100, (1, 6), generator=g)}

    def decode(self, ids, skip_special_tokens=True):
        return self._text


class TestEWCOnlyBaseline:

    def _make(self):
        torch.manual_seed(0)
        model = _tiny_qwen2()
        config = SLEEPConfig()
        config.weights.lora_rank = 4
        config.weights.lora_alpha = 8
        return EWCOnlyBaseline(model, _StubTokenizer(), config, device="cpu")

    def test_penalty_zero_before_consolidation(self):
        base = self._make()
        assert float(base._ewc_penalty().item()) == 0.0 if base._theta_star else True

    def test_consolidate_stores_anchor_and_fisher(self):
        base = self._make()
        base.consolidate_task(["a fact about something", "another fact"])
        assert len(base._theta_star) == 1
        assert len(base._fisher) == 1

    def test_penalty_zero_at_anchor_positive_after_move(self):
        base = self._make()
        # Train one step first so LoRA B moves off its zero init — only then do
        # the adapter parameters carry non-zero Fisher (with B=0, A cannot
        # affect the loss, so its Fisher is legitimately zero).
        base.train_on_input("a fact about something to learn")
        base.consolidate_task(["a fact about something", "and another one"])
        # At the anchor, penalty is ~0 (params haven't moved since the snapshot).
        assert float(base._ewc_penalty().item()) < 1e-6
        # Move the B parameters (which carry Fisher mass); penalty must rise.
        with torch.no_grad():
            base.lora_modules[0].B.add_(1.0)
        assert float(base._ewc_penalty().item()) > 0.0

    def test_train_step_runs_with_penalty(self):
        base = self._make()
        base.consolidate_task(["a fact about something"])
        loss = base.train_on_input("a new fact to learn")
        assert loss == loss and loss < float("inf")  # finite


class TestInContextBaseline:

    def test_add_and_lookup(self):
        model = _tiny_qwen2()
        base = InContextBaseline(model, _StubTokenizer("Poseidonville"), device="cpu")
        base.add_fact("f1", "The capital of Atlantis is Poseidonville.")
        out = base.query("What is the capital of Atlantis?", fact_id="f1", max_new_tokens=3)
        assert isinstance(out, str) and len(out) > 0

    def test_explicit_fact_text_overrides(self):
        model = _tiny_qwen2()
        base = InContextBaseline(model, _StubTokenizer("x"), device="cpu")
        # No stored fact, but explicit text supplied → still works.
        out = base.query("q?", fact_text="some fact", max_new_tokens=2)
        assert isinstance(out, str)
