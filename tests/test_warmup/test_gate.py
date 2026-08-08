"""Tests for the retrieval-aware memory gate (sleep.warmup.gate).

The load-bearing test is :class:`TestGateInjectorEquality.test_identity_gate_is_bit_identical`
— the hook-based ground-truth check the project requires for any change to the
KV-injection critical path. An identity-initialised gate must leave the patched
attention forward bit-identical to the ungated path; if it does not, every
warm-up result would be confounded by a silent change to injection itself.
"""

from __future__ import annotations

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from sleep.warmup.gate import MemoryGate
from sleep.weights.kv_injection import KVInjector
from sleep.weights.kv_memory import KVMemoryBank


# ---------------------------------------------------------------------------
# Fixtures — mirror tests/test_weights/test_kv_injection.py
# ---------------------------------------------------------------------------

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
    model = Qwen2ForCausalLM(config)
    model.eval()
    return model


def _make_bank(layers=(2, 3)) -> KVMemoryBank:
    return KVMemoryBank(
        adapted_layer_indices=list(layers),
        num_kv_heads=2,
        head_dim=8,
        max_total_tokens=32,
        device="cpu",
        dtype=torch.float32,
    )


def _populate(bank: KVMemoryBank, layers=(2, 3), n_tokens=4, seed=99) -> None:
    torch.manual_seed(seed)
    layer_kvs = {}
    for layer_idx in layers:
        k = torch.randn(n_tokens, 2, 8) * 0.5
        v = torch.randn(n_tokens, 2, 8) * 0.5
        layer_kvs[layer_idx] = (k, v)
    bank.append("tag", layer_kvs)


def _logits(model, input_ids) -> torch.Tensor:
    with torch.no_grad():
        return model(input_ids=input_ids).logits


# ---------------------------------------------------------------------------
# Gate math
# ---------------------------------------------------------------------------

class TestGateMath:

    def test_identity_init_scales_are_one(self):
        gate = MemoryGate(adapted_layers=[2, 3], num_kv_heads=2)
        assert torch.allclose(gate.log_scale, torch.zeros_like(gate.log_scale))
        for layer in (2, 3):
            assert torch.allclose(
                gate.scale_for_layer(layer), torch.ones(2)
            )

    def test_identity_forward_returns_input_unchanged(self):
        gate = MemoryGate(adapted_layers=[2], num_kv_heads=2)
        mem_v = torch.randn(1, 2, 5, 8)
        out = gate(mem_v, layer_idx=2)
        assert torch.allclose(out, mem_v)

    def test_known_scale_multiplies_per_head(self):
        gate = MemoryGate(adapted_layers=[2], num_kv_heads=2, init_scale=1.0)
        # Set head 0 -> scale 2, head 1 -> scale 3.
        with torch.no_grad():
            import math
            gate.log_scale[0, 0] = math.log(2.0)
            gate.log_scale[0, 1] = math.log(3.0)
        mem_v = torch.ones(1, 2, 4, 8)
        out = gate(mem_v, layer_idx=2)
        assert torch.allclose(out[0, 0], torch.full((4, 8), 2.0))
        assert torch.allclose(out[0, 1], torch.full((4, 8), 3.0))

    def test_init_scale_parameter(self):
        gate = MemoryGate(adapted_layers=[1], num_kv_heads=1, init_scale=2.0)
        assert torch.allclose(gate.scale_for_layer(1), torch.full((1,), 2.0))

    def test_param_count_is_layers_times_heads(self):
        gate = MemoryGate(adapted_layers=[0, 1, 2], num_kv_heads=4)
        n = sum(p.numel() for p in gate.parameters())
        assert n == 3 * 4


# ---------------------------------------------------------------------------
# Injector equality — the critical hook-based ground-truth check
# ---------------------------------------------------------------------------

class TestGateInjectorEquality:

    def test_identity_gate_is_bit_identical(self):
        """An identity gate must not change injected-attention output at all.

        Ground truth = the injector's own populated-bank forward with no gate.
        The identity gate is required to reproduce it to within fp tolerance.
        """
        model = _make_tiny_qwen2()
        torch.manual_seed(10)
        input_ids = torch.randint(0, 100, (1, 7))

        # Ground truth: populated bank, NO gate.
        bank = _make_bank()
        injector = KVInjector(model, bank)
        injector.install()
        try:
            _populate(bank)
            ungated = _logits(model, input_ids)

            # Same bank, install an identity gate.
            gate = MemoryGate(adapted_layers=[2, 3], num_kv_heads=2)
            injector.set_memory_gate(gate)
            gated_identity = _logits(model, input_ids)
        finally:
            injector.uninstall()

        assert torch.allclose(ungated, gated_identity, atol=1e-6), (
            "identity gate changed the injected-attention output"
        )

    def test_non_identity_gate_changes_output(self):
        model = _make_tiny_qwen2()
        torch.manual_seed(11)
        input_ids = torch.randint(0, 100, (1, 7))

        bank = _make_bank()
        injector = KVInjector(model, bank)
        injector.install()
        try:
            _populate(bank)
            ungated = _logits(model, input_ids)

            gate = MemoryGate(adapted_layers=[2, 3], num_kv_heads=2, init_scale=3.0)
            injector.set_memory_gate(gate)
            gated = _logits(model, input_ids)
        finally:
            injector.uninstall()

        assert not torch.allclose(ungated, gated, atol=1e-3)

    def test_clearing_gate_restores_ungated_output(self):
        model = _make_tiny_qwen2()
        torch.manual_seed(12)
        input_ids = torch.randint(0, 100, (1, 6))

        bank = _make_bank()
        injector = KVInjector(model, bank)
        injector.install()
        try:
            _populate(bank)
            ungated = _logits(model, input_ids)

            injector.set_memory_gate(
                MemoryGate(adapted_layers=[2, 3], num_kv_heads=2, init_scale=2.5)
            )
            _ = _logits(model, input_ids)

            injector.set_memory_gate(None)
            restored = _logits(model, input_ids)
        finally:
            injector.uninstall()

        assert torch.allclose(ungated, restored, atol=1e-6)

    def test_gate_attr_cleaned_up_on_uninstall(self):
        model = _make_tiny_qwen2()
        bank = _make_bank(layers=(2,))
        injector = KVInjector(model, bank)
        injector.set_memory_gate(MemoryGate(adapted_layers=[2], num_kv_heads=2))
        injector.install()
        attn = model.model.layers[2].self_attn
        assert hasattr(attn, "_sleep_memory_gate")
        injector.uninstall()
        assert not hasattr(attn, "_sleep_memory_gate")

    def test_gate_is_differentiable_through_forward(self):
        """Loss on the gated forward must produce a gradient on the gate."""
        model = _make_tiny_qwen2()
        torch.manual_seed(13)
        input_ids = torch.randint(0, 100, (1, 6))

        bank = _make_bank()
        injector = KVInjector(model, bank)
        injector.install()
        try:
            _populate(bank)
            gate = MemoryGate(adapted_layers=[2, 3], num_kv_heads=2)
            injector.set_memory_gate(gate)
            out = model(input_ids=input_ids, labels=input_ids)
            out.loss.backward()
        finally:
            injector.uninstall()

        assert gate.log_scale.grad is not None
        assert torch.isfinite(gate.log_scale.grad).all()
