"""Integration tests for KV memory features in DualWeightSystem.

The most important test in this file is ``test_extracted_kv_matches_in_attention_kv``
— it verifies that the K/V we extract via separate-pass output_hidden_states +
input_layernorm + k_proj/v_proj exactly matches what the model computes inside
its own attention forward. Without this, an off-by-one in the layer-norm or
indexing would produce close-but-wrong K/V values that silently degrade
attention scores.
"""

from __future__ import annotations

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.models.qwen2 import modeling_qwen2 as _qwen2_mod

from sleep.config import WeightsConfig
from sleep.weights import DualWeightSystem


# ---------------------------------------------------------------------------
# Tiny Qwen2 fixture
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


def _make_weights_config() -> WeightsConfig:
    """Config that adapts the top 1/3 of layers (= layers 3 of 4)."""
    return WeightsConfig(
        lora_rank=4,
        lora_alpha=8,
        adapted_fraction=0.333,
        adapted_matrices=["v_proj", "o_proj"],
        alpha_fast=1e-4,
        momentum_fast=0.9,
        alpha_slow=1e-4,
        delta_max=0.01,
        phi_min=0.10,
        lambda_ewc=10.0,
        fisher_refresh_interval=10,
        fisher_calibration_mix=0.7,
        epsilon_degrade=0.02,
    )


@pytest.fixture
def tiny_dws_no_kv():
    """DualWeightSystem with KV memory DISABLED. (Default behaviour.)"""
    torch.manual_seed(100)
    return DualWeightSystem(_make_tiny_qwen2(), _make_weights_config())


@pytest.fixture
def tiny_dws_kv():
    """DualWeightSystem with KV memory ENABLED."""
    torch.manual_seed(100)
    dws = DualWeightSystem(
        _make_tiny_qwen2(),
        _make_weights_config(),
        use_kv_memory_for_fast=True,
        kv_max_total_tokens=100,
    )
    yield dws
    dws.cleanup()  # uninstall injector after the test


# ---------------------------------------------------------------------------
# Backward compatibility — KV mode disabled by default
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    def test_kv_mode_disabled_by_default(self, tiny_dws_no_kv):
        assert tiny_dws_no_kv.use_kv_memory_for_fast is False
        assert tiny_dws_no_kv.kv_bank is None
        assert tiny_dws_no_kv.kv_injector is None

    def test_write_raises_when_disabled(self, tiny_dws_no_kv):
        with pytest.raises(RuntimeError, match="not enabled"):
            tiny_dws_no_kv.write_to_kv_bank(
                "tag", torch.tensor([1, 2, 3, 4, 5]), 1, 4,
            )

    def test_evict_raises_when_disabled(self, tiny_dws_no_kv):
        with pytest.raises(RuntimeError, match="not enabled"):
            tiny_dws_no_kv.evict_from_kv_bank("tag")

    def test_clear_raises_when_disabled(self, tiny_dws_no_kv):
        with pytest.raises(RuntimeError, match="not enabled"):
            tiny_dws_no_kv.clear_kv_bank()


# ---------------------------------------------------------------------------
# KV mode initialization
# ---------------------------------------------------------------------------


class TestKVModeInit:
    def test_kv_bank_constructed(self, tiny_dws_kv):
        assert tiny_dws_kv.use_kv_memory_for_fast is True
        bank = tiny_dws_kv.kv_bank
        assert bank is not None
        assert bank.num_kv_heads == 2
        assert bank.head_dim == 8
        assert bank.max_total_tokens == 100
        # adapted_layers should match the top 1/3 of 4 layers = [3]
        # (math.ceil((1 - 0.333) * 4) = ceil(2.668) = 3, so adapted = [3])
        assert tuple(bank.adapted_layers) == tuple(tiny_dws_kv.adapted_layers)

    def test_injector_installed(self, tiny_dws_kv):
        injector = tiny_dws_kv.kv_injector
        assert injector is not None
        assert injector.is_installed

    def test_lora_w_fast_still_exists(self, tiny_dws_kv):
        # Strategy Y: LoRA W_fast is preserved for backward compat / ablation
        assert tiny_dws_kv.w_fast_params > 0


# ---------------------------------------------------------------------------
# write_to_kv_bank semantics
# ---------------------------------------------------------------------------


class TestWriteToBank:
    def test_basic_write(self, tiny_dws_kv):
        token_ids = torch.randint(0, 100, (10,))
        n_written = tiny_dws_kv.write_to_kv_bank("tag_a", token_ids, 2, 7)
        assert n_written == 5
        assert tiny_dws_kv.kv_bank.n_tags == 1
        assert tiny_dws_kv.kv_bank.n_total_tokens == 5

    def test_multi_tag_writes(self, tiny_dws_kv):
        token_ids = torch.randint(0, 100, (15,))
        tiny_dws_kv.write_to_kv_bank("a", token_ids, 0, 3)
        tiny_dws_kv.write_to_kv_bank("b", token_ids, 5, 10)
        assert tiny_dws_kv.kv_bank.n_tags == 2
        assert tiny_dws_kv.kv_bank.n_total_tokens == 8

    def test_invalid_span(self, tiny_dws_kv):
        token_ids = torch.randint(0, 100, (5,))
        with pytest.raises(ValueError, match="span_end"):
            tiny_dws_kv.write_to_kv_bank("a", token_ids, 3, 3)
        with pytest.raises(ValueError, match="exceeds"):
            tiny_dws_kv.write_to_kv_bank("a", token_ids, 0, 10)

    def test_evict(self, tiny_dws_kv):
        token_ids = torch.randint(0, 100, (10,))
        tiny_dws_kv.write_to_kv_bank("a", token_ids, 0, 5)
        assert tiny_dws_kv.evict_from_kv_bank("a") is True
        assert tiny_dws_kv.evict_from_kv_bank("a") is False
        assert tiny_dws_kv.kv_bank.n_tags == 0

    def test_clear(self, tiny_dws_kv):
        token_ids = torch.randint(0, 100, (10,))
        tiny_dws_kv.write_to_kv_bank("a", token_ids, 0, 3)
        tiny_dws_kv.write_to_kv_bank("b", token_ids, 4, 7)
        tiny_dws_kv.clear_kv_bank()
        assert tiny_dws_kv.kv_bank.n_tags == 0


# ---------------------------------------------------------------------------
# THE CRITICAL TEST — extraction-point verification
# ---------------------------------------------------------------------------


class TestExtractionPointVerification:
    """The most important test in Phase A.

    Verifies that K/V extracted via the separate-pass approach (output_hidden_states
    -> input_layernorm -> k_proj/v_proj) exactly equals what the model
    computes inside its own attention forward pass.

    If this fails, our extraction is wrong by some transform — most likely
    a missing layer-norm or an off-by-one in indexing — and downstream
    "KV injection doesn't recall well" diagnoses would be misleading.
    """

    def test_extracted_kv_matches_in_attention_kv(self, tiny_dws_kv):
        """For every adapted layer, K/V extracted by write_to_kv_bank must
        equal K/V observed inside that layer's attention forward, modulo:
            - K is stored pre-RoPE (we capture pre-RoPE inside attention)
            - V is stored verbatim (V is never RoPE'd)
        """
        torch.manual_seed(42)
        token_ids = torch.randint(0, 100, (12,))
        span_start, span_end = 3, 9
        span_len = span_end - span_start

        # (1) Capture in-attention pre-RoPE K and V via a hook on apply_rotary_pos_emb.
        # Strategy: patch _qwen2_mod.apply_rotary_pos_emb with a wrapper that
        # records its inputs, then runs the original. Restore after the test.

        captured: dict[int, dict[str, torch.Tensor]] = {}

        original_apply = _qwen2_mod.apply_rotary_pos_emb

        # We need to know which layer index each call is for.
        # Solution: use forward hooks on each adapted self_attn module.
        # The hook fires AFTER attention forward returns — too late for pre-RoPE K.
        # Instead, hook k_proj and v_proj outputs on each adapted layer.
        # k_proj output is (B, T, n_kv_heads * head_dim) — this is pre-RoPE K
        # (still needs view+transpose to match the bank's shape).

        adapted_layers = tiny_dws_kv.adapted_layers
        layers = tiny_dws_kv._locate_decoder_layers()

        kv_proj_capture: dict[int, dict[str, torch.Tensor]] = {}

        def make_k_hook(layer_idx):
            def hook(_module, _inputs, output):
                # Output of k_proj: (B=1, T=full_seq_len, n_kv_heads * head_dim)
                kv_proj_capture.setdefault(layer_idx, {})["k_flat"] = output.detach().clone()
            return hook

        def make_v_hook(layer_idx):
            def hook(_module, _inputs, output):
                kv_proj_capture.setdefault(layer_idx, {})["v_flat"] = output.detach().clone()
            return hook

        handles = []
        for layer_idx in adapted_layers:
            attn = layers[layer_idx].self_attn
            handles.append(attn.k_proj.register_forward_hook(make_k_hook(layer_idx)))
            handles.append(attn.v_proj.register_forward_hook(make_v_hook(layer_idx)))

        try:
            # Run a normal forward pass on the same input (must NOT use the
            # KV injector since it modifies forward — uninstall first).
            tiny_dws_kv.kv_injector.uninstall()
            try:
                with torch.no_grad():
                    tiny_dws_kv.model(
                        input_ids=token_ids[:span_end].unsqueeze(0).to(
                            next(tiny_dws_kv.model.parameters()).device,
                        ),
                        use_cache=False,
                    )
            finally:
                tiny_dws_kv.kv_injector.install()
        finally:
            for h in handles:
                h.remove()

        # Slice the captured K/V to the span [span_start:span_end] and reshape.
        in_attn_kv: dict[int, dict[str, torch.Tensor]] = {}
        n_kv_heads = tiny_dws_kv.kv_bank.num_kv_heads
        head_dim = tiny_dws_kv.kv_bank.head_dim
        for layer_idx in adapted_layers:
            k_flat = kv_proj_capture[layer_idx]["k_flat"][0]  # (full_seq, hidden)
            v_flat = kv_proj_capture[layer_idx]["v_flat"][0]
            k_span = k_flat[span_start:span_end].view(span_len, n_kv_heads, head_dim)
            v_span = v_flat[span_start:span_end].view(span_len, n_kv_heads, head_dim)
            in_attn_kv[layer_idx] = {"k": k_span, "v": v_span}

        # (2) Run the separate-pass extraction via write_to_kv_bank
        tiny_dws_kv.write_to_kv_bank("verify", token_ids, span_start, span_end)

        # (3) Compare every adapted layer's K/V
        for layer_idx in adapted_layers:
            stored_k, stored_v = tiny_dws_kv.kv_bank._entries["verify"].layer_kvs[layer_idx]
            expected_k = in_attn_kv[layer_idx]["k"].to(
                dtype=stored_k.dtype, device=stored_k.device,
            )
            expected_v = in_attn_kv[layer_idx]["v"].to(
                dtype=stored_v.dtype, device=stored_v.device,
            )
            assert torch.allclose(stored_k, expected_k, atol=1e-6), (
                f"Layer {layer_idx}: extracted K does not match in-attention K. "
                f"Max diff: {(stored_k - expected_k).abs().max().item()}"
            )
            assert torch.allclose(stored_v, expected_v, atol=1e-6), (
                f"Layer {layer_idx}: extracted V does not match in-attention V. "
                f"Max diff: {(stored_v - expected_v).abs().max().item()}"
            )


# ---------------------------------------------------------------------------
# End-to-end: write then probe attention output
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_write_changes_attention_output(self, tiny_dws_kv):
        """Verify writing a tag actually affects subsequent forward passes."""
        torch.manual_seed(7)
        probe = torch.randint(0, 100, (1, 5))

        with torch.no_grad():
            before = tiny_dws_kv.model(input_ids=probe).logits

        torch.manual_seed(8)
        ctx = torch.randint(0, 100, (10,))
        tiny_dws_kv.write_to_kv_bank("ctx", ctx, 2, 8)

        with torch.no_grad():
            after = tiny_dws_kv.model(input_ids=probe).logits

        assert before.shape == after.shape
        assert not torch.allclose(before, after, atol=1e-3)

    def test_clear_restores_baseline(self, tiny_dws_kv):
        """After write + clear, attention should match the empty-bank baseline."""
        torch.manual_seed(9)
        probe = torch.randint(0, 100, (1, 5))

        with torch.no_grad():
            baseline = tiny_dws_kv.model(input_ids=probe).logits

        torch.manual_seed(10)
        ctx = torch.randint(0, 100, (10,))
        tiny_dws_kv.write_to_kv_bank("a", ctx, 1, 6)
        tiny_dws_kv.clear_kv_bank()

        with torch.no_grad():
            after_clear = tiny_dws_kv.model(input_ids=probe).logits

        assert torch.allclose(baseline, after_clear, atol=1e-5)
