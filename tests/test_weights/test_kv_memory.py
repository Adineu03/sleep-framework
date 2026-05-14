"""Tests for sleep.weights.kv_memory.KVMemoryBank.

Pure unit tests with no model dependency. Verify:
- Append/evict/clear semantics
- Shape, dtype, device validation
- Capacity bookkeeping
- Concatenation order matches insertion order
- Per-tag offset bookkeeping
- Cache invalidation on mutation
"""

from __future__ import annotations

import pytest
import torch

from sleep.weights.kv_memory import KVEntry, KVMemoryBank


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


# Use small but realistic dimensions: 3 layers (sparse adapted set), 4 KV heads,
# head_dim 8. Keeps test tensors tiny while exercising the full API.
ADAPTED_LAYERS = [19, 20, 21]
NUM_KV_HEADS = 4
HEAD_DIM = 8
MAX_TOKENS = 100
DEVICE = torch.device("cpu")
DTYPE = torch.float32


def _make_bank(
    max_tokens: int = MAX_TOKENS,
    layers: list[int] = None,
) -> KVMemoryBank:
    return KVMemoryBank(
        adapted_layer_indices=layers if layers is not None else ADAPTED_LAYERS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        max_total_tokens=max_tokens,
        device=DEVICE,
        dtype=DTYPE,
    )


def _make_layer_kvs(
    n_tokens: int,
    layers: list[int] = None,
    *,
    fill: float | None = None,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Build a layer_kvs dict for ``append``.

    If fill is given, both K and V are filled with that value (useful for
    checking concatenation order). Otherwise random.
    """
    if layers is None:
        layers = ADAPTED_LAYERS
    kvs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer in layers:
        if fill is None:
            k = torch.randn(n_tokens, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE)
            v = torch.randn(n_tokens, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE)
        else:
            k = torch.full(
                (n_tokens, NUM_KV_HEADS, HEAD_DIM),
                fill_value=fill,
                dtype=DTYPE,
            )
            v = torch.full(
                (n_tokens, NUM_KV_HEADS, HEAD_DIM),
                fill_value=fill + 100.0,  # distinguish K from V
                dtype=DTYPE,
            )
        kvs[layer] = (k, v)
    return kvs


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:

    def test_basic_construction(self):
        bank = _make_bank()
        assert bank.n_tags == 0
        assert bank.n_total_tokens == 0
        assert bank.is_empty
        assert not bank.at_capacity
        assert bank.adapted_layers == tuple(ADAPTED_LAYERS)
        assert bank.num_kv_heads == NUM_KV_HEADS
        assert bank.head_dim == HEAD_DIM
        assert bank.max_total_tokens == MAX_TOKENS

    def test_empty_layers_rejected(self):
        with pytest.raises(ValueError, match="adapted_layer_indices"):
            KVMemoryBank(
                adapted_layer_indices=[],
                num_kv_heads=4,
                head_dim=8,
                max_total_tokens=100,
                device="cpu",
                dtype=torch.float32,
            )

    def test_invalid_dimensions_rejected(self):
        with pytest.raises(ValueError, match="num_kv_heads"):
            KVMemoryBank(ADAPTED_LAYERS, 0, 8, 100, "cpu", torch.float32)
        with pytest.raises(ValueError, match="head_dim"):
            KVMemoryBank(ADAPTED_LAYERS, 4, 0, 100, "cpu", torch.float32)
        with pytest.raises(ValueError, match="max_total_tokens"):
            KVMemoryBank(ADAPTED_LAYERS, 4, 8, 0, "cpu", torch.float32)


# ---------------------------------------------------------------------------
# Append
# ---------------------------------------------------------------------------


class TestAppend:

    def test_single_append(self):
        bank = _make_bank()
        bank.append("tag_001", _make_layer_kvs(n_tokens=5))
        assert bank.n_tags == 1
        assert bank.n_total_tokens == 5
        assert "tag_001" in bank
        assert bank.tag_ids() == ["tag_001"]

    def test_multi_append(self):
        bank = _make_bank()
        bank.append("tag_001", _make_layer_kvs(3))
        bank.append("tag_002", _make_layer_kvs(7))
        bank.append("tag_003", _make_layer_kvs(2))
        assert bank.n_tags == 3
        assert bank.n_total_tokens == 12
        assert bank.tag_ids() == ["tag_001", "tag_002", "tag_003"]

    def test_duplicate_tag_id_rejected(self):
        bank = _make_bank()
        bank.append("tag_001", _make_layer_kvs(5))
        with pytest.raises(ValueError, match="already in bank"):
            bank.append("tag_001", _make_layer_kvs(3))

    def test_missing_layer_rejected(self):
        bank = _make_bank()
        kvs = _make_layer_kvs(5)
        del kvs[ADAPTED_LAYERS[0]]
        with pytest.raises(ValueError, match="missing"):
            bank.append("tag_001", kvs)

    def test_extra_layer_rejected(self):
        bank = _make_bank()
        kvs = _make_layer_kvs(5)
        kvs[99] = (
            torch.randn(5, NUM_KV_HEADS, HEAD_DIM),
            torch.randn(5, NUM_KV_HEADS, HEAD_DIM),
        )
        with pytest.raises(ValueError, match="extra"):
            bank.append("tag_001", kvs)

    def test_shape_mismatch_rejected(self):
        bank = _make_bank()
        kvs = _make_layer_kvs(5)
        # Mangle one layer's K shape
        kvs[ADAPTED_LAYERS[0]] = (
            torch.randn(5, NUM_KV_HEADS + 1, HEAD_DIM),
            torch.randn(5, NUM_KV_HEADS, HEAD_DIM),
        )
        with pytest.raises(ValueError, match="K shape"):
            bank.append("tag_001", kvs)

    def test_kv_shape_disagree_rejected(self):
        bank = _make_bank()
        kvs = _make_layer_kvs(5)
        kvs[ADAPTED_LAYERS[0]] = (
            torch.randn(5, NUM_KV_HEADS, HEAD_DIM),
            torch.randn(7, NUM_KV_HEADS, HEAD_DIM),  # different n_tokens for V
        )
        with pytest.raises(ValueError, match=r"K shape .* != V shape"):
            bank.append("tag_001", kvs)

    def test_per_layer_n_tokens_must_match(self):
        bank = _make_bank()
        kvs = _make_layer_kvs(5)
        # Replace one layer's tensors with different n_tokens
        kvs[ADAPTED_LAYERS[1]] = (
            torch.randn(8, NUM_KV_HEADS, HEAD_DIM),
            torch.randn(8, NUM_KV_HEADS, HEAD_DIM),
        )
        with pytest.raises(ValueError, match="disagrees"):
            bank.append("tag_001", kvs)

    def test_zero_tokens_rejected(self):
        bank = _make_bank()
        with pytest.raises(ValueError, match="must be > 0"):
            bank.append("tag_001", _make_layer_kvs(0))

    def test_dtype_mismatch_rejected(self):
        bank = _make_bank()
        kvs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for layer in ADAPTED_LAYERS:
            kvs[layer] = (
                torch.randn(5, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float16),
                torch.randn(5, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float16),
            )
        with pytest.raises(ValueError, match="dtype"):
            bank.append("tag_001", kvs)

    def test_non_tensor_rejected(self):
        bank = _make_bank()
        kvs = _make_layer_kvs(5)
        kvs[ADAPTED_LAYERS[0]] = (
            "not a tensor",  # type: ignore
            torch.randn(5, NUM_KV_HEADS, HEAD_DIM),
        )
        with pytest.raises(ValueError, match="must be torch.Tensor"):
            bank.append("tag_001", kvs)


# ---------------------------------------------------------------------------
# Capacity
# ---------------------------------------------------------------------------


class TestCapacity:

    def test_under_capacity(self):
        bank = _make_bank(max_tokens=10)
        bank.append("tag_001", _make_layer_kvs(5))
        assert not bank.at_capacity
        assert bank.n_total_tokens == 5

    def test_at_capacity(self):
        bank = _make_bank(max_tokens=10)
        bank.append("tag_001", _make_layer_kvs(10))
        assert bank.at_capacity

    def test_overflow_raises(self):
        bank = _make_bank(max_tokens=10)
        bank.append("tag_001", _make_layer_kvs(7))
        with pytest.raises(RuntimeError, match="exceeding max_total_tokens"):
            bank.append("tag_002", _make_layer_kvs(5))
        # Failed append must not corrupt state
        assert bank.n_tags == 1
        assert bank.n_total_tokens == 7
        assert "tag_002" not in bank

    def test_overflow_then_evict_then_append(self):
        bank = _make_bank(max_tokens=10)
        bank.append("tag_001", _make_layer_kvs(7))
        with pytest.raises(RuntimeError):
            bank.append("tag_002", _make_layer_kvs(5))
        bank.evict("tag_001")
        bank.append("tag_002", _make_layer_kvs(5))  # fits now
        assert bank.n_tags == 1
        assert bank.n_total_tokens == 5


# ---------------------------------------------------------------------------
# Eviction
# ---------------------------------------------------------------------------


class TestEvict:

    def test_evict_existing(self):
        bank = _make_bank()
        bank.append("tag_001", _make_layer_kvs(5))
        result = bank.evict("tag_001")
        assert result is True
        assert bank.is_empty

    def test_evict_nonexistent(self):
        bank = _make_bank()
        result = bank.evict("does_not_exist")
        assert result is False

    def test_evict_middle_preserves_order(self):
        bank = _make_bank()
        bank.append("a", _make_layer_kvs(2))
        bank.append("b", _make_layer_kvs(2))
        bank.append("c", _make_layer_kvs(2))
        bank.evict("b")
        assert bank.tag_ids() == ["a", "c"]

    def test_clear_removes_all(self):
        bank = _make_bank()
        bank.append("a", _make_layer_kvs(2))
        bank.append("b", _make_layer_kvs(3))
        bank.clear()
        assert bank.is_empty
        assert bank.n_total_tokens == 0


# ---------------------------------------------------------------------------
# Read API
# ---------------------------------------------------------------------------


class TestGetForLayer:

    def test_empty_bank_returns_none(self):
        bank = _make_bank()
        assert bank.get_for_layer(ADAPTED_LAYERS[0]) is None

    def test_invalid_layer_raises(self):
        bank = _make_bank()
        with pytest.raises(ValueError, match="not in adapted_layers"):
            bank.get_for_layer(999)

    def test_single_entry_shape(self):
        bank = _make_bank()
        bank.append("tag_001", _make_layer_kvs(5))
        result = bank.get_for_layer(ADAPTED_LAYERS[0])
        assert result is not None
        k, v = result
        assert k.shape == (5, NUM_KV_HEADS, HEAD_DIM)
        assert v.shape == (5, NUM_KV_HEADS, HEAD_DIM)

    def test_concat_in_insertion_order(self):
        # Use distinguishable fill values to verify ordering
        bank = _make_bank()
        bank.append("a", _make_layer_kvs(2, fill=1.0))
        bank.append("b", _make_layer_kvs(3, fill=2.0))
        bank.append("c", _make_layer_kvs(1, fill=3.0))
        layer = ADAPTED_LAYERS[0]
        k, v = bank.get_for_layer(layer)

        assert k.shape[0] == 6
        # First 2 tokens come from "a" (fill 1.0)
        assert torch.all(k[0:2] == 1.0)
        # Next 3 from "b" (fill 2.0)
        assert torch.all(k[2:5] == 2.0)
        # Last 1 from "c" (fill 3.0)
        assert torch.all(k[5:6] == 3.0)

        # V uses fill + 100
        assert torch.all(v[0:2] == 101.0)
        assert torch.all(v[2:5] == 102.0)
        assert torch.all(v[5:6] == 103.0)

    def test_get_after_evict_reflects_change(self):
        bank = _make_bank()
        bank.append("a", _make_layer_kvs(2, fill=1.0))
        bank.append("b", _make_layer_kvs(3, fill=2.0))
        bank.evict("a")
        k, _ = bank.get_for_layer(ADAPTED_LAYERS[0])
        assert k.shape[0] == 3
        assert torch.all(k == 2.0)

    def test_cache_invalidation_on_append(self):
        # First read populates cache; subsequent append must invalidate it.
        bank = _make_bank()
        bank.append("a", _make_layer_kvs(2, fill=1.0))
        first = bank.get_for_layer(ADAPTED_LAYERS[0])[0]
        assert first.shape[0] == 2

        bank.append("b", _make_layer_kvs(3, fill=2.0))
        second = bank.get_for_layer(ADAPTED_LAYERS[0])[0]

        # Cache must have rebuilt: total length grew, and new chunk has fill=2
        assert second.shape[0] == 5
        assert torch.all(second[2:5] == 2.0)
        # The original "a" portion is still 1.0 in the rebuilt cache
        assert torch.all(second[0:2] == 1.0)

    def test_cache_invalidation_on_evict(self):
        bank = _make_bank()
        bank.append("a", _make_layer_kvs(2, fill=1.0))
        bank.append("b", _make_layer_kvs(3, fill=2.0))
        first = bank.get_for_layer(ADAPTED_LAYERS[0])[0]
        assert first.shape[0] == 5

        bank.evict("a")
        second = bank.get_for_layer(ADAPTED_LAYERS[0])[0]
        assert second.shape[0] == 3
        assert torch.all(second == 2.0)

    def test_get_for_each_adapted_layer(self):
        bank = _make_bank()
        bank.append("a", _make_layer_kvs(4))
        for layer in ADAPTED_LAYERS:
            result = bank.get_for_layer(layer)
            assert result is not None
            k, v = result
            assert k.shape == (4, NUM_KV_HEADS, HEAD_DIM)
            assert v.shape == (4, NUM_KV_HEADS, HEAD_DIM)


# ---------------------------------------------------------------------------
# Per-tag offsets
# ---------------------------------------------------------------------------


class TestEntryOffsets:

    def test_empty_bank_returns_empty_dict(self):
        bank = _make_bank()
        assert bank.get_entry_offsets() == {}

    def test_offsets_match_insertion_order(self):
        bank = _make_bank()
        bank.append("a", _make_layer_kvs(2))
        bank.append("b", _make_layer_kvs(3))
        bank.append("c", _make_layer_kvs(4))
        offsets = bank.get_entry_offsets()
        assert offsets == {
            "a": (0, 2),
            "b": (2, 5),
            "c": (5, 9),
        }

    def test_offsets_consistent_with_concatenated_tensor(self):
        bank = _make_bank()
        bank.append("a", _make_layer_kvs(2, fill=1.0))
        bank.append("b", _make_layer_kvs(3, fill=2.0))
        layer = ADAPTED_LAYERS[0]
        k, _ = bank.get_for_layer(layer)
        offsets = bank.get_entry_offsets()
        a_start, a_end = offsets["a"]
        b_start, b_end = offsets["b"]
        assert torch.all(k[a_start:a_end] == 1.0)
        assert torch.all(k[b_start:b_end] == 2.0)


# ---------------------------------------------------------------------------
# Diagnostics / repr
# ---------------------------------------------------------------------------


class TestDiagnostics:

    def test_stats_empty(self):
        bank = _make_bank()
        stats = bank.stats()
        assert stats["n_tags"] == 0
        assert stats["n_total_tokens"] == 0
        assert stats["occupancy"] == 0.0
        assert stats["at_capacity"] is False

    def test_stats_populated(self):
        bank = _make_bank(max_tokens=20)
        bank.append("a", _make_layer_kvs(8))
        stats = bank.stats()
        assert stats["n_tags"] == 1
        assert stats["n_total_tokens"] == 8
        assert stats["occupancy"] == 0.4
        assert stats["at_capacity"] is False
        assert stats["n_adapted_layers"] == len(ADAPTED_LAYERS)

    def test_repr_contains_useful_info(self):
        bank = _make_bank()
        bank.append("a", _make_layer_kvs(3))
        s = repr(bank)
        assert "n_tags=1" in s
        assert "n_total_tokens=3" in s

    def test_len_and_contains(self):
        bank = _make_bank()
        bank.append("a", _make_layer_kvs(2))
        bank.append("b", _make_layer_kvs(3))
        assert len(bank) == 2
        assert "a" in bank
        assert "z" not in bank


# ---------------------------------------------------------------------------
# Detach: appended tensors should not retain autograd graph
# ---------------------------------------------------------------------------


class TestDetach:

    def test_appended_tensors_are_detached(self):
        bank = _make_bank()
        # Build K/V that require grad — the bank must detach them.
        kvs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for layer in ADAPTED_LAYERS:
            k = torch.randn(
                3, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, requires_grad=True,
            )
            v = torch.randn(
                3, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, requires_grad=True,
            )
            kvs[layer] = (k, v)
        bank.append("a", kvs)

        retrieved_k, retrieved_v = bank.get_for_layer(ADAPTED_LAYERS[0])
        # After concat, the result should not require grad because inputs were detached.
        assert not retrieved_k.requires_grad
        assert not retrieved_v.requires_grad
