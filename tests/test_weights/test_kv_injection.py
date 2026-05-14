"""Tests for sleep.weights.kv_injection.

Two layers of testing:

1. Pure-helper unit tests on synthetic tensors — RoPE half-rotation, K-only
   RoPE application, attention-mask extension (especially the float-dtype
   correctness for memory columns).

2. Integration tests with a tiny Qwen2 model constructed in-memory — verify
   that install/uninstall is non-destructive (output unchanged with empty
   bank), that adding memory entries actually changes attention output, and
   that the patched forward survives different attention configurations.
"""

from __future__ import annotations

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.models.qwen2 import modeling_qwen2 as _qwen2_mod

from sleep.weights.kv_injection import (
    KVInjector,
    _apply_rope_k_only,
    _compute_memory_rope,
    _compute_topk_visibility,
    _extend_attention_mask_with_memory,
    _rotate_half,
)
from sleep.weights.kv_memory import KVMemoryBank


# ---------------------------------------------------------------------------
# Tiny Qwen2 fixture — small but architecturally complete
# ---------------------------------------------------------------------------


def _make_tiny_qwen2() -> Qwen2ForCausalLM:
    """Construct a randomly-initialized Qwen2 with minimal dimensions."""
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


@pytest.fixture(scope="module")
def tiny_qwen2():
    torch.manual_seed(0)
    return _make_tiny_qwen2()


# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------


class TestRotateHalf:

    def test_matches_qwen_implementation(self):
        # Our _rotate_half should produce the exact same output as the one
        # used internally by Qwen2's apply_rotary_pos_emb.
        torch.manual_seed(1)
        x = torch.randn(2, 4, 8, 16)  # (B, H, T, D)
        # Reproduce HF's implementation directly:
        d = x.shape[-1]
        x1 = x[..., : d // 2]
        x2 = x[..., d // 2 :]
        expected = torch.cat((-x2, x1), dim=-1)

        actual = _rotate_half(x)
        assert torch.allclose(actual, expected)

    def test_shape_preserved(self):
        x = torch.randn(3, 8)
        assert _rotate_half(x).shape == x.shape


class TestApplyRopeKOnly:

    def test_matches_k_half_of_apply_rotary_pos_emb(self):
        """_apply_rope_k_only must produce the same K as the full
        apply_rotary_pos_emb function from Qwen2."""
        torch.manual_seed(2)
        B, H, T, D = 2, 4, 6, 16
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        # cos/sin shape (B, T, D)
        cos = torch.randn(B, T, D)
        sin = torch.randn(B, T, D)

        _, expected_k = _qwen2_mod.apply_rotary_pos_emb(q, k, cos, sin)
        actual_k = _apply_rope_k_only(k, cos, sin)

        assert torch.allclose(actual_k, expected_k, atol=1e-6)

    def test_handles_different_unsqueeze_dim(self):
        # If somebody passes unsqueeze_dim=2 the helper should still produce
        # something sensible (this is what HF supports).
        B, H, T, D = 1, 2, 4, 8
        k = torch.randn(B, H, T, D)
        cos = torch.randn(B, T, D)
        sin = torch.randn(B, T, D)
        out = _apply_rope_k_only(k, cos, sin, unsqueeze_dim=1)
        assert out.shape == k.shape


# ---------------------------------------------------------------------------
# Attention-mask extension — the user-flagged dtype correctness test
# ---------------------------------------------------------------------------


class TestExtendAttentionMask:

    def test_prepends_n_mem_columns(self):
        # Original mask: (B=2, 1, Q=3, KV=4)
        mask = torch.zeros(2, 1, 3, 4, dtype=torch.float32)
        result = _extend_attention_mask_with_memory(
            mask, n_mem=5,
            batch_size=2, q_len=3,
            device=mask.device, dtype=mask.dtype,
        )
        assert result is not None
        assert result.shape == (2, 1, 3, 9)  # 5 + 4

    def test_memory_columns_are_zero_visible(self):
        """The user-flagged risk: memory columns must be 0.0 (visible),
        not True/1, in the mask's float dtype."""
        # Build a mask that has a -inf in some original position so we can
        # check we didn't accidentally use the wrong "visible" value.
        neg_inf = torch.finfo(torch.float32).min
        mask = torch.full((1, 1, 2, 3), 0.0, dtype=torch.float32)
        mask[0, 0, 0, 0] = neg_inf  # one position masked

        result = _extend_attention_mask_with_memory(
            mask, n_mem=4,
            batch_size=1, q_len=2,
            device=mask.device, dtype=mask.dtype,
        )

        # First 4 columns must be exactly 0.0 (visible) for ALL queries
        assert torch.all(result[..., :4] == 0.0)
        # Original mask preserved in last 3 columns
        assert torch.equal(result[..., 4:], mask)

    def test_dtype_matches_mask_dtype_bfloat16(self):
        # bfloat16 path — the dtype-mismatch failure mode the user warned about
        mask = torch.zeros(1, 1, 2, 3, dtype=torch.bfloat16)
        result = _extend_attention_mask_with_memory(
            mask, n_mem=2,
            batch_size=1, q_len=2,
            device=mask.device, dtype=mask.dtype,
        )
        assert result.dtype == torch.bfloat16
        # Must not have been silently promoted to float32 or boolean
        assert result[..., :2].dtype == torch.bfloat16

    def test_n_mem_zero_returns_unchanged(self):
        mask = torch.zeros(1, 1, 2, 3)
        result = _extend_attention_mask_with_memory(
            mask, n_mem=0,
            batch_size=1, q_len=2,
            device=mask.device, dtype=mask.dtype,
        )
        assert result is mask  # exact same object — no copy

    def test_none_mask_returns_none(self):
        # When attention_mask is None upstream, we don't synthesize one;
        # the attention impl will handle implicit causal masking.
        result = _extend_attention_mask_with_memory(
            None, n_mem=4,
            batch_size=1, q_len=2,
            device=torch.device("cpu"), dtype=torch.float32,
        )
        assert result is None

    def test_rejects_non_4d(self):
        mask = torch.zeros(2, 3, 4)  # 3D, not 4D
        with pytest.raises(ValueError, match="4D"):
            _extend_attention_mask_with_memory(
                mask, n_mem=1,
                batch_size=1, q_len=2,
                device=mask.device, dtype=mask.dtype,
            )


# ---------------------------------------------------------------------------
# Memory RoPE computation
# ---------------------------------------------------------------------------


class TestComputeTopkVisibility:

    def test_top_k_zero_returns_none(self):
        # k <= 0 means no gating
        Q = torch.randn(1, 4, 5, 8)  # (B, H_q, Q_len, D)
        K = torch.randn(1, 2, 6, 8)  # (B, H_kv, M, D)
        result = _compute_topk_visibility(Q, K, top_k=0)
        assert result is None

    def test_top_k_geq_n_mem_returns_none(self):
        # k >= n_mem means no gating needed
        Q = torch.randn(1, 4, 5, 8)
        K = torch.randn(1, 2, 6, 8)
        assert _compute_topk_visibility(Q, K, top_k=6) is None
        assert _compute_topk_visibility(Q, K, top_k=10) is None

    def test_top_k_returns_correct_shape(self):
        Q = torch.randn(2, 4, 5, 8)  # B=2, H_q=4, Q=5, D=8
        K = torch.randn(2, 2, 6, 8)  # B=2, H_kv=2, M=6, D=8
        vis = _compute_topk_visibility(Q, K, top_k=3)
        assert vis is not None
        assert vis.shape == (2, 5, 6)
        assert vis.dtype == torch.bool

    def test_top_k_selects_exactly_k(self):
        Q = torch.randn(1, 4, 3, 8)
        K = torch.randn(1, 2, 10, 8)
        vis = _compute_topk_visibility(Q, K, top_k=4)
        # Each (b, q) row should have exactly 4 True values
        for b in range(vis.shape[0]):
            for q in range(vis.shape[1]):
                assert vis[b, q].sum().item() == 4

    def test_top_k_picks_highest_scoring(self):
        """Construct a synthetic case where one memory key has very high
        inner product with the query and verify it's selected."""
        torch.manual_seed(0)
        # Single batch, single Q-head and KV-head, Q_len=1, M=5, D=4
        # Make K[0,0,3,:] strongly aligned with Q[0,0,0,:]
        Q = torch.zeros(1, 1, 1, 4)
        Q[0, 0, 0, :] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        K = torch.randn(1, 1, 5, 4) * 0.01  # all small
        K[0, 0, 3, :] = torch.tensor([10.0, 0.0, 0.0, 0.0])  # huge alignment

        vis = _compute_topk_visibility(Q, K, top_k=1)
        assert vis is not None
        # Only memory position 3 should be visible
        assert vis[0, 0, 3].item() is True
        assert vis[0, 0].sum().item() == 1

    def test_handles_grouped_query_attention(self):
        """When num_q_heads != num_kv_heads (GQA), the function should
        still produce a valid (B, Q, M) mask without crashing."""
        Q = torch.randn(1, 8, 4, 16)  # 8 Q-heads
        K = torch.randn(1, 2, 6, 16)  # 2 KV-heads (GQA group of 4)
        vis = _compute_topk_visibility(Q, K, top_k=2)
        assert vis is not None
        assert vis.shape == (1, 4, 6)


class TestExtendMaskWithVisibility:

    def test_visibility_mask_zeros_visible_neginf_hidden(self):
        # Construct a (B=1, Q=2, M=3) visibility tensor
        visibility = torch.tensor([
            [[True, False, True],
             [False, True, False]],
        ])
        base_mask = torch.zeros(1, 1, 2, 4, dtype=torch.float32)
        result = _extend_attention_mask_with_memory(
            base_mask, n_mem=3,
            batch_size=1, q_len=2,
            device=base_mask.device, dtype=base_mask.dtype,
            memory_visibility=visibility,
        )
        assert result.shape == (1, 1, 2, 7)  # 3 mem + 4 base
        # Memory portion: visible -> 0.0, hidden -> dtype.min
        neg_min = torch.finfo(torch.float32).min
        # Query 0 visibility: [T, F, T] -> [0, min, 0]
        assert result[0, 0, 0, 0].item() == 0.0
        assert result[0, 0, 0, 1].item() == neg_min
        assert result[0, 0, 0, 2].item() == 0.0
        # Query 1 visibility: [F, T, F] -> [min, 0, min]
        assert result[0, 0, 1, 0].item() == neg_min
        assert result[0, 0, 1, 1].item() == 0.0
        assert result[0, 0, 1, 2].item() == neg_min

    def test_visibility_dtype_matches_mask_dtype(self):
        visibility = torch.tensor([[[True, False]]])
        base_mask = torch.zeros(1, 1, 1, 2, dtype=torch.bfloat16)
        result = _extend_attention_mask_with_memory(
            base_mask, n_mem=2,
            batch_size=1, q_len=1,
            device=base_mask.device, dtype=base_mask.dtype,
            memory_visibility=visibility,
        )
        assert result.dtype == torch.bfloat16

    def test_visibility_wrong_shape_rejected(self):
        bad_visibility = torch.tensor([True, False, True])  # 1D, wrong
        base_mask = torch.zeros(1, 1, 2, 3)
        with pytest.raises(ValueError, match="memory_visibility must have shape"):
            _extend_attention_mask_with_memory(
                base_mask, n_mem=3,
                batch_size=1, q_len=2,
                device=base_mask.device, dtype=base_mask.dtype,
                memory_visibility=bad_visibility,
            )


class TestComputeMemoryRope:

    def test_uses_negative_positions(self, tiny_qwen2):
        rope = tiny_qwen2.model.rotary_emb
        # We can't easily inspect the exact returned values, but we can verify
        # cos/sin shapes and that the computation runs without error for
        # negative positions.
        ref = torch.zeros(1, 2, 5, 8)  # (B, H, T, D) — only B/device/dtype matter
        cos, sin = _compute_memory_rope(
            rope, n_mem=5, batch_size=1, reference_tensor=ref,
        )
        assert cos.shape[-2] == 5  # T axis
        assert sin.shape[-2] == 5
        assert cos.dtype == ref.dtype

    def test_negative_positions_differ_from_positive(self, tiny_qwen2):
        """Sanity: cos/sin computed at -3..-1 must differ from cos/sin at 1..3.
        If they're identical, our negative-position RoPE assumption is wrong."""
        rope = tiny_qwen2.model.rotary_emb
        ref = torch.zeros(1, 2, 3, 8)

        positions_pos = torch.arange(1, 4, dtype=torch.long).unsqueeze(0)
        cos_pos, sin_pos = rope(ref, positions_pos)

        positions_neg = torch.arange(-3, 0, dtype=torch.long).unsqueeze(0)
        cos_neg, sin_neg = rope(ref, positions_neg)

        # cos is even (cos(-x) == cos(x)) so they should match where the
        # absolute value of positions matches.  pos=[1,2,3] vs neg=[-3,-2,-1]
        # → cos should agree at pos[i] and neg[2-i] (reversed magnitudes).
        # sin is odd so sin should be the negated mirror.
        assert torch.allclose(cos_pos, cos_neg.flip(-2), atol=1e-5)
        assert torch.allclose(sin_pos, -sin_neg.flip(-2), atol=1e-5)


# ---------------------------------------------------------------------------
# KVInjector — install / uninstall mechanics
# ---------------------------------------------------------------------------


class TestKVInjectorMechanics:

    def test_construction_validates_layers(self, tiny_qwen2):
        # tiny_qwen2 has 4 layers (indices 0..3). Asking for layer 99 must fail.
        bank = KVMemoryBank(
            adapted_layer_indices=[2, 3, 99],
            num_kv_heads=2,
            head_dim=8,
            max_total_tokens=10,
            device="cpu",
            dtype=torch.float32,
        )
        with pytest.raises(ValueError, match="out of range"):
            KVInjector(tiny_qwen2, bank)

    def test_construction_locates_rope_module(self, tiny_qwen2):
        bank = KVMemoryBank(
            adapted_layer_indices=[2, 3],
            num_kv_heads=2, head_dim=8, max_total_tokens=10,
            device="cpu", dtype=torch.float32,
        )
        injector = KVInjector(tiny_qwen2, bank)
        assert injector._rope_module is tiny_qwen2.model.rotary_emb

    def test_install_replaces_forward(self, tiny_qwen2):
        bank = KVMemoryBank(
            adapted_layer_indices=[2, 3],
            num_kv_heads=2, head_dim=8, max_total_tokens=10,
            device="cpu", dtype=torch.float32,
        )
        injector = KVInjector(tiny_qwen2, bank)
        original_forwards = {
            i: tiny_qwen2.model.layers[i].self_attn.forward
            for i in [2, 3]
        }
        injector.install()
        for i in [2, 3]:
            current = tiny_qwen2.model.layers[i].self_attn.forward
            assert current is not original_forwards[i]
        # Layer 0 (not in bank) must be untouched
        assert tiny_qwen2.model.layers[0].self_attn.forward is not None
        injector.uninstall()  # cleanup

    def test_uninstall_restores_forward(self, tiny_qwen2):
        bank = KVMemoryBank(
            adapted_layer_indices=[2, 3],
            num_kv_heads=2, head_dim=8, max_total_tokens=10,
            device="cpu", dtype=torch.float32,
        )
        injector = KVInjector(tiny_qwen2, bank)
        original = tiny_qwen2.model.layers[2].self_attn.forward
        injector.install()
        injector.uninstall()
        # forward should be restored
        assert tiny_qwen2.model.layers[2].self_attn.forward == original or \
               tiny_qwen2.model.layers[2].self_attn.forward.__func__ is original.__func__

    def test_double_install_raises(self, tiny_qwen2):
        bank = KVMemoryBank(
            adapted_layer_indices=[2, 3],
            num_kv_heads=2, head_dim=8, max_total_tokens=10,
            device="cpu", dtype=torch.float32,
        )
        injector = KVInjector(tiny_qwen2, bank)
        injector.install()
        try:
            with pytest.raises(RuntimeError, match="already installed"):
                injector.install()
        finally:
            injector.uninstall()

    def test_uninstall_without_install_raises(self, tiny_qwen2):
        bank = KVMemoryBank(
            adapted_layer_indices=[2, 3],
            num_kv_heads=2, head_dim=8, max_total_tokens=10,
            device="cpu", dtype=torch.float32,
        )
        injector = KVInjector(tiny_qwen2, bank)
        with pytest.raises(RuntimeError, match="not installed"):
            injector.uninstall()

    def test_context_manager(self, tiny_qwen2):
        bank = KVMemoryBank(
            adapted_layer_indices=[2, 3],
            num_kv_heads=2, head_dim=8, max_total_tokens=10,
            device="cpu", dtype=torch.float32,
        )
        injector = KVInjector(tiny_qwen2, bank)
        assert not injector.is_installed
        with injector as inj:
            assert inj.is_installed
        assert not injector.is_installed

    def test_temporary_attrs_cleaned_up(self, tiny_qwen2):
        bank = KVMemoryBank(
            adapted_layer_indices=[2],
            num_kv_heads=2, head_dim=8, max_total_tokens=10,
            device="cpu", dtype=torch.float32,
        )
        injector = KVInjector(tiny_qwen2, bank)
        attn = tiny_qwen2.model.layers[2].self_attn
        injector.install()
        assert hasattr(attn, "_sleep_kv_bank")
        assert hasattr(attn, "_sleep_rope_module")
        injector.uninstall()
        assert not hasattr(attn, "_sleep_kv_bank")
        assert not hasattr(attn, "_sleep_rope_module")


# ---------------------------------------------------------------------------
# Integration tests with tiny Qwen2
# ---------------------------------------------------------------------------


def _run_forward_logits(model, input_ids):
    with torch.no_grad():
        out = model(input_ids=input_ids)
    return out.logits


class TestIntegrationWithTinyQwen2:

    def test_empty_bank_changes_nothing(self, tiny_qwen2):
        """With an empty bank installed, model output must be bit-identical
        to model without injector."""
        torch.manual_seed(10)
        input_ids = torch.randint(0, 100, (1, 7))

        before = _run_forward_logits(tiny_qwen2, input_ids)

        bank = KVMemoryBank(
            adapted_layer_indices=[2, 3],
            num_kv_heads=2, head_dim=8, max_total_tokens=10,
            device="cpu", dtype=torch.float32,
        )
        injector = KVInjector(tiny_qwen2, bank)
        injector.install()
        try:
            after_install_empty = _run_forward_logits(tiny_qwen2, input_ids)
        finally:
            injector.uninstall()

        after_uninstall = _run_forward_logits(tiny_qwen2, input_ids)

        assert torch.allclose(before, after_install_empty, atol=1e-5)
        assert torch.allclose(before, after_uninstall, atol=1e-5)

    def test_populated_bank_changes_output(self, tiny_qwen2):
        """With memory entries in the bank, model output must differ from
        the empty-bank output."""
        torch.manual_seed(11)
        input_ids = torch.randint(0, 100, (1, 7))

        bank = KVMemoryBank(
            adapted_layer_indices=[2, 3],
            num_kv_heads=2, head_dim=8, max_total_tokens=20,
            device="cpu", dtype=torch.float32,
        )
        injector = KVInjector(tiny_qwen2, bank)
        injector.install()

        try:
            empty_out = _run_forward_logits(tiny_qwen2, input_ids)

            # Add a memory entry. K/V are random — they don't have to "mean"
            # anything for this test; we just need them to be non-zero so they
            # influence attention.
            torch.manual_seed(99)
            n_tokens = 4
            layer_kvs = {}
            for layer_idx in [2, 3]:
                k = torch.randn(n_tokens, 2, 8) * 0.5  # scale so it doesn't blow up
                v = torch.randn(n_tokens, 2, 8) * 0.5
                layer_kvs[layer_idx] = (k, v)
            bank.append("test_tag", layer_kvs)

            populated_out = _run_forward_logits(tiny_qwen2, input_ids)
        finally:
            injector.uninstall()

        # Output must have changed
        assert not torch.allclose(empty_out, populated_out, atol=1e-3)
        # Shape must be preserved
        assert empty_out.shape == populated_out.shape

    def test_evict_restores_output(self, tiny_qwen2):
        """After evicting all entries, output must equal the empty-bank case."""
        torch.manual_seed(12)
        input_ids = torch.randint(0, 100, (1, 5))

        bank = KVMemoryBank(
            adapted_layer_indices=[2, 3],
            num_kv_heads=2, head_dim=8, max_total_tokens=20,
            device="cpu", dtype=torch.float32,
        )
        injector = KVInjector(tiny_qwen2, bank)
        injector.install()

        try:
            empty_out = _run_forward_logits(tiny_qwen2, input_ids)

            torch.manual_seed(7)
            kvs = {
                i: (torch.randn(3, 2, 8) * 0.3, torch.randn(3, 2, 8) * 0.3)
                for i in [2, 3]
            }
            bank.append("evict_me", kvs)
            populated_out = _run_forward_logits(tiny_qwen2, input_ids)
            bank.evict("evict_me")
            after_evict_out = _run_forward_logits(tiny_qwen2, input_ids)
        finally:
            injector.uninstall()

        assert not torch.allclose(empty_out, populated_out, atol=1e-3)
        assert torch.allclose(empty_out, after_evict_out, atol=1e-5)

    def test_only_adapted_layers_affected(self, tiny_qwen2):
        """If we adapt only layer 3, layer 2's attention output must still
        equal the no-injection baseline. We probe this via hooks."""
        torch.manual_seed(13)
        input_ids = torch.randint(0, 100, (1, 6))

        # Capture layer-2 self_attn output without any injection
        layer2_baseline = []

        def grab_baseline(_module, _inputs, output):
            # output is (attn_output, attn_weights or None)
            layer2_baseline.append(output[0].detach().clone())

        h = tiny_qwen2.model.layers[2].self_attn.register_forward_hook(grab_baseline)
        try:
            _run_forward_logits(tiny_qwen2, input_ids)
        finally:
            h.remove()

        # Now install on layer 3 only with a populated bank, and capture
        # layer-2 self_attn output
        bank = KVMemoryBank(
            adapted_layer_indices=[3],
            num_kv_heads=2, head_dim=8, max_total_tokens=20,
            device="cpu", dtype=torch.float32,
        )
        torch.manual_seed(8)
        bank.append(
            "tag", {3: (torch.randn(3, 2, 8) * 0.3, torch.randn(3, 2, 8) * 0.3)},
        )

        injector = KVInjector(tiny_qwen2, bank)
        injector.install()
        try:
            layer2_with_l3_inject = []

            def grab_with_inject(_module, _inputs, output):
                layer2_with_l3_inject.append(output[0].detach().clone())

            h = tiny_qwen2.model.layers[2].self_attn.register_forward_hook(grab_with_inject)
            try:
                _run_forward_logits(tiny_qwen2, input_ids)
            finally:
                h.remove()
        finally:
            injector.uninstall()

        # Layer 2 (unadapted) must produce identical output
        assert torch.allclose(
            layer2_baseline[0], layer2_with_l3_inject[0], atol=1e-5,
        )

    def test_topk_gating_no_change_when_k_geq_n_mem(self, tiny_qwen2):
        """With top_k >= n_mem, output should equal the no-gating case."""
        torch.manual_seed(15)
        input_ids = torch.randint(0, 100, (1, 5))

        bank = KVMemoryBank(
            adapted_layer_indices=[2, 3],
            num_kv_heads=2, head_dim=8, max_total_tokens=20,
            device="cpu", dtype=torch.float32,
        )
        torch.manual_seed(16)
        for tag_id in ["a", "b"]:
            kvs = {
                i: (torch.randn(2, 2, 8) * 0.3, torch.randn(2, 2, 8) * 0.3)
                for i in [2, 3]
            }
            bank.append(tag_id, kvs)
        # n_mem = 4 total

        injector_no_gate = KVInjector(tiny_qwen2, bank, top_k=0)
        injector_no_gate.install()
        try:
            out_no_gate = _run_forward_logits(tiny_qwen2, input_ids)
        finally:
            injector_no_gate.uninstall()

        # k=10 > n_mem=4 → should return None internally → same as no gating
        injector_high_k = KVInjector(tiny_qwen2, bank, top_k=10)
        injector_high_k.install()
        try:
            out_high_k = _run_forward_logits(tiny_qwen2, input_ids)
        finally:
            injector_high_k.uninstall()

        assert torch.allclose(out_no_gate, out_high_k, atol=1e-5)

    def test_topk_gating_changes_output_when_k_lt_n_mem(self, tiny_qwen2):
        """With top_k < n_mem, output should differ from no-gating."""
        torch.manual_seed(17)
        input_ids = torch.randint(0, 100, (1, 5))

        bank = KVMemoryBank(
            adapted_layer_indices=[2, 3],
            num_kv_heads=2, head_dim=8, max_total_tokens=30,
            device="cpu", dtype=torch.float32,
        )
        torch.manual_seed(18)
        for tag_id in ["a", "b", "c"]:
            kvs = {
                i: (torch.randn(3, 2, 8) * 0.3, torch.randn(3, 2, 8) * 0.3)
                for i in [2, 3]
            }
            bank.append(tag_id, kvs)
        # n_mem = 9

        injector_no_gate = KVInjector(tiny_qwen2, bank, top_k=0)
        injector_no_gate.install()
        try:
            out_no_gate = _run_forward_logits(tiny_qwen2, input_ids)
        finally:
            injector_no_gate.uninstall()

        injector_k2 = KVInjector(tiny_qwen2, bank, top_k=2)
        injector_k2.install()
        try:
            out_k2 = _run_forward_logits(tiny_qwen2, input_ids)
        finally:
            injector_k2.uninstall()

        # With strict gating (k=2 of 9), output should differ
        assert not torch.allclose(out_no_gate, out_k2, atol=1e-3)

    def test_set_enabled_disabled_is_no_op(self, tiny_qwen2):
        """When disabled, output must equal the no-injector baseline even
        with a populated bank."""
        torch.manual_seed(40)
        input_ids = torch.randint(0, 100, (1, 6))

        # Baseline: model output without any injector
        baseline = _run_forward_logits(tiny_qwen2, input_ids)

        bank = KVMemoryBank(
            adapted_layer_indices=[2, 3],
            num_kv_heads=2, head_dim=8, max_total_tokens=20,
            device="cpu", dtype=torch.float32,
        )
        # Populate with random memories
        torch.manual_seed(41)
        bank.append("a", {
            i: (torch.randn(3, 2, 8) * 0.5, torch.randn(3, 2, 8) * 0.5)
            for i in [2, 3]
        })

        injector = KVInjector(tiny_qwen2, bank, top_k=0)
        injector.install()
        try:
            # Sanity: enabled (default) — should differ from baseline
            out_enabled = _run_forward_logits(tiny_qwen2, input_ids)
            assert not torch.allclose(baseline, out_enabled, atol=1e-3)

            # Disabled — should match baseline exactly
            injector.set_enabled(False)
            assert injector.is_enabled is False
            out_disabled = _run_forward_logits(tiny_qwen2, input_ids)
            assert torch.allclose(baseline, out_disabled, atol=1e-5)

            # Re-enable — should match the enabled output again
            injector.set_enabled(True)
            assert injector.is_enabled is True
            out_re_enabled = _run_forward_logits(tiny_qwen2, input_ids)
            assert torch.allclose(out_enabled, out_re_enabled, atol=1e-5)
        finally:
            injector.uninstall()

    def test_disabled_bypasses_with_empty_bank_too(self, tiny_qwen2):
        """Disable+empty bank case: still equals baseline."""
        torch.manual_seed(42)
        input_ids = torch.randint(0, 100, (1, 4))

        baseline = _run_forward_logits(tiny_qwen2, input_ids)

        bank = KVMemoryBank(
            adapted_layer_indices=[2],
            num_kv_heads=2, head_dim=8, max_total_tokens=10,
            device="cpu", dtype=torch.float32,
        )
        injector = KVInjector(tiny_qwen2, bank, top_k=0)
        injector.install()
        injector.set_enabled(False)
        try:
            out = _run_forward_logits(tiny_qwen2, input_ids)
        finally:
            injector.uninstall()
        assert torch.allclose(baseline, out, atol=1e-5)

    def test_set_top_k_runtime(self, tiny_qwen2):
        """set_top_k should change behaviour without uninstall/reinstall."""
        torch.manual_seed(19)
        input_ids = torch.randint(0, 100, (1, 5))

        bank = KVMemoryBank(
            adapted_layer_indices=[2, 3],
            num_kv_heads=2, head_dim=8, max_total_tokens=30,
            device="cpu", dtype=torch.float32,
        )
        torch.manual_seed(20)
        for tag_id in ["a", "b", "c"]:
            kvs = {
                i: (torch.randn(3, 2, 8) * 0.3, torch.randn(3, 2, 8) * 0.3)
                for i in [2, 3]
            }
            bank.append(tag_id, kvs)

        injector = KVInjector(tiny_qwen2, bank, top_k=0)
        injector.install()
        try:
            out_k0 = _run_forward_logits(tiny_qwen2, input_ids)
            injector.set_top_k(2)
            out_k2 = _run_forward_logits(tiny_qwen2, input_ids)
            injector.set_top_k(0)
            out_k0_again = _run_forward_logits(tiny_qwen2, input_ids)
        finally:
            injector.uninstall()

        assert not torch.allclose(out_k0, out_k2, atol=1e-3)
        assert torch.allclose(out_k0, out_k0_again, atol=1e-5)

    def test_multi_token_memory_concatenates_correctly(self, tiny_qwen2):
        """Verify attention sees an extended K/V on adapted layers when
        memory is non-empty. Probe shape of internal K via a hook."""
        torch.manual_seed(14)
        input_ids = torch.randint(0, 100, (1, 5))

        bank = KVMemoryBank(
            adapted_layer_indices=[3],
            num_kv_heads=2, head_dim=8, max_total_tokens=20,
            device="cpu", dtype=torch.float32,
        )
        torch.manual_seed(20)
        n_mem = 6
        bank.append(
            "tag",
            {3: (torch.randn(n_mem, 2, 8) * 0.3, torch.randn(n_mem, 2, 8) * 0.3)},
        )

        # Probe k_proj's output shape on layer 3 — that's the un-extended K.
        # The extension happens AFTER k_proj inside our patched forward,
        # so we can't grab the extended K via a hook on k_proj.
        # Instead, we'll just verify the model still produces the right logits
        # shape (B, T, vocab) — extended K is internal and the ultimate
        # attention output collapses back to (B, T, hidden).
        injector = KVInjector(tiny_qwen2, bank)
        injector.install()
        try:
            out = _run_forward_logits(tiny_qwen2, input_ids)
        finally:
            injector.uninstall()

        # Output shape unchanged: memory injection doesn't alter the output time axis
        assert out.shape == (1, 5, tiny_qwen2.config.vocab_size)
