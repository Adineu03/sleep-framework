"""Smoke + behaviour tests for sleep.warmup.WarmupTrainer.

These use a tiny Qwen2 wrapped in a real DualWeightSystem with KV memory
enabled, so they exercise the true training path (bank writes, gated injected
forward, backward through the gate) end-to-end on CPU. They assert the loop
runs, produces finite losses, actually moves the gate off identity, and leaves
the bank empty afterwards.
"""

from __future__ import annotations

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from sleep.config import WeightsConfig
from sleep.warmup import WarmupTrainer
from sleep.weights import DualWeightSystem


def _make_tiny_qwen2() -> Qwen2ForCausalLM:
    config = Qwen2Config(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        sliding_window=None,
        attn_implementation="eager",
        torch_dtype=torch.float32,
    )
    return Qwen2ForCausalLM(config)


def _make_dual_weights() -> DualWeightSystem:
    torch.manual_seed(0)
    model = _make_tiny_qwen2()
    cfg = WeightsConfig(lora_rank=4, lora_alpha=8, adapted_fraction=0.5)
    return DualWeightSystem(
        model, cfg,
        use_kv_memory_for_fast=True,
        kv_max_total_tokens=256,
        kv_top_k=0,
    )


class _StubTokenizer:
    pad_token_id = 0


def _corpus(n_seqs=6, length=24, vocab=100, seed=1) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, vocab, (length,), generator=g) for _ in range(n_seqs)]


class TestWarmupTrainer:

    def test_requires_kv_memory(self):
        torch.manual_seed(0)
        model = _make_tiny_qwen2()
        dw = DualWeightSystem(model, WeightsConfig(lora_rank=4, lora_alpha=8))
        try:
            WarmupTrainer(dw, _StubTokenizer(), device="cpu")
            raised = False
        except RuntimeError:
            raised = True
        assert raised, "WarmupTrainer must reject a non-KV DualWeightSystem"

    def test_run_smoke_and_finite_loss(self):
        dw = _make_dual_weights()
        trainer = WarmupTrainer(dw, _StubTokenizer(), device="cpu")
        result = trainer.run(_corpus(), n_steps=8, lr=1e-2, seed=0)

        assert result.n_steps >= 1
        assert result.final_loss == result.final_loss  # not NaN
        assert result.final_loss < float("inf")
        assert set(result.gate_scales.keys()) == set(dw.adapted_layers)

    def test_gate_moves_off_identity(self):
        dw = _make_dual_weights()
        trainer = WarmupTrainer(dw, _StubTokenizer(), device="cpu")
        # Identity at start.
        assert torch.allclose(
            trainer.gate.log_scale, torch.zeros_like(trainer.gate.log_scale)
        )
        trainer.run(_corpus(), n_steps=12, lr=5e-2, seed=0)
        # After training the gate should have moved.
        assert not torch.allclose(
            trainer.gate.log_scale, torch.zeros_like(trainer.gate.log_scale)
        )

    def test_gate_frozen_and_bank_clear_after_run(self):
        dw = _make_dual_weights()
        trainer = WarmupTrainer(dw, _StubTokenizer(), device="cpu")
        trainer.run(_corpus(), n_steps=6, lr=1e-2, seed=0)

        # Gate frozen for downstream sleep/eval use.
        assert all(not p.requires_grad for p in trainer.gate.parameters())
        # Warm-up episode cleared; bank back to empty so eval starts clean.
        assert dw.kv_bank.n_tags == 0

    def test_gate_persists_on_injector(self):
        dw = _make_dual_weights()
        trainer = WarmupTrainer(dw, _StubTokenizer(), device="cpu")
        trainer.run(_corpus(), n_steps=4, lr=1e-2, seed=0)
        assert dw.kv_injector.memory_gate is trainer.gate

    def test_empty_corpus_raises(self):
        dw = _make_dual_weights()
        trainer = WarmupTrainer(dw, _StubTokenizer(), device="cpu")
        try:
            trainer.run([torch.randint(0, 100, (3,))], n_steps=4, min_seq_len=8)
            raised = False
        except ValueError:
            raised = True
        assert raised
