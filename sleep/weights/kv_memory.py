"""KV memory bank — the substrate of W_fast under the KV-injection architecture.

A bounded, per-tag-indexed store of (K, V) tensors at each adapted attention
layer. Supports append (one-shot write at tag creation time), per-layer
retrieval (concatenation of stored entries for attention injection), and
per-tag eviction (called by PRP cleanup or sleep consolidation).

Stored K is **pre-RoPE** — the position-independent representation. RoPE
gets applied at read time during attention injection so memories occupy
negative positions [-n_mem, -1] in the read context. V is stored verbatim
(V does not get RoPE).

See ``docs/KV_INJECTION_DESIGN.md`` for the full architectural rationale.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch
from torch import Tensor

from sleep.utils.logging import get_logger

logger = get_logger("sleep.weights.kv_memory")


# ---------------------------------------------------------------------------
# Per-tag entry
# ---------------------------------------------------------------------------


@dataclass
class KVEntry:
    """One tag's stored K/V across all adapted layers.

    Attributes:
        tag_id: Unique identifier (typically the source_id from the tag).
        n_tokens: Number of tokens in the stored span (same across layers).
        layer_kvs: Maps adapted layer index -> (K, V) tensor pair.
            Each tensor has shape (n_tokens, num_kv_heads, head_dim).
            K is **pre-RoPE**. V is raw.
    """

    tag_id: str
    n_tokens: int
    layer_kvs: dict[int, tuple[Tensor, Tensor]]


# ---------------------------------------------------------------------------
# The bank
# ---------------------------------------------------------------------------


class KVMemoryBank:
    """Per-tag, per-layer storage of pre-RoPE K and raw V tensors.

    Capacity is bounded by ``max_total_tokens`` (sum across all stored entries).
    The bank does NOT auto-evict on overflow — callers (typically the PRP
    system) are responsible for eviction policy. The bank exposes
    :attr:`at_capacity` and :attr:`n_total_tokens` so callers can check
    before appending.

    Concatenated per-layer K/V tensors are cached lazily and invalidated
    on append/evict/clear.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        adapted_layer_indices: list[int],
        num_kv_heads: int,
        head_dim: int,
        max_total_tokens: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        if not adapted_layer_indices:
            raise ValueError("adapted_layer_indices must be non-empty")
        if num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be positive")
        if head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if max_total_tokens <= 0:
            raise ValueError("max_total_tokens must be positive")

        self._adapted_layers: tuple[int, ...] = tuple(adapted_layer_indices)
        self._adapted_layer_set: frozenset[int] = frozenset(adapted_layer_indices)
        self._num_kv_heads: int = num_kv_heads
        self._head_dim: int = head_dim
        self._max_total_tokens: int = max_total_tokens
        self._device: torch.device = torch.device(device)
        self._dtype: torch.dtype = dtype

        # OrderedDict preserves insertion order for FIFO-style debug iteration
        # but we don't auto-evict — callers manage eviction.
        self._entries: OrderedDict[str, KVEntry] = OrderedDict()

        # Lazy-cached concatenated K/V per layer.
        # None means "no entries OR cache stale."
        # We rebuild on next get_for_layer call when stale.
        self._cached: dict[int, tuple[Tensor, Tensor]] = {}
        self._dirty_layers: set[int] = set()

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def adapted_layers(self) -> tuple[int, ...]:
        """Tuple of layer indices this bank stores K/V for."""
        return self._adapted_layers

    @property
    def num_kv_heads(self) -> int:
        return self._num_kv_heads

    @property
    def head_dim(self) -> int:
        return self._head_dim

    @property
    def max_total_tokens(self) -> int:
        return self._max_total_tokens

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def n_tags(self) -> int:
        """Number of distinct tag entries in the bank."""
        return len(self._entries)

    @property
    def n_total_tokens(self) -> int:
        """Sum of n_tokens across all stored entries."""
        return sum(entry.n_tokens for entry in self._entries.values())

    @property
    def at_capacity(self) -> bool:
        """True if total stored tokens >= max_total_tokens.

        Callers should check this before appending. The bank does not
        auto-evict; callers (PRP system, sleep cleanup) decide what to drop.
        """
        return self.n_total_tokens >= self._max_total_tokens

    @property
    def is_empty(self) -> bool:
        return len(self._entries) == 0

    def tag_ids(self) -> list[str]:
        """Insertion-ordered list of tag_ids currently in the bank."""
        return list(self._entries.keys())

    def __len__(self) -> int:
        return self.n_tags

    def __contains__(self, tag_id: str) -> bool:
        return tag_id in self._entries

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def append(
        self,
        tag_id: str,
        layer_kvs: dict[int, tuple[Tensor, Tensor]],
    ) -> None:
        """Store K/V for a tagged span across all adapted layers.

        Args:
            tag_id: Unique identifier for this tagged span. Must not already
                exist in the bank.
            layer_kvs: Mapping from adapted layer index to (K, V) tensor pair.
                Both tensors must have shape ``(n_tokens, num_kv_heads,
                head_dim)``, dtype ``self.dtype``, on ``self.device``.
                Must contain an entry for every layer in
                ``self.adapted_layers``.

        Raises:
            ValueError: On duplicate tag_id, missing/extra layer keys, shape
                mismatch, dtype mismatch, device mismatch, or zero-length span.
            RuntimeError: If appending would exceed ``max_total_tokens``.
                (Caller must evict first.)
        """
        if tag_id in self._entries:
            raise ValueError(f"tag_id {tag_id!r} already in bank")

        provided_layers = set(layer_kvs.keys())
        if provided_layers != self._adapted_layer_set:
            missing = self._adapted_layer_set - provided_layers
            extra = provided_layers - self._adapted_layer_set
            raise ValueError(
                f"layer_kvs keys must exactly match adapted_layers. "
                f"missing={sorted(missing)} extra={sorted(extra)}"
            )

        # Validate shapes/dtypes/devices, infer n_tokens from first layer
        n_tokens: int | None = None
        expected_shape_suffix = (self._num_kv_heads, self._head_dim)

        for layer_idx, (k, v) in layer_kvs.items():
            if not isinstance(k, Tensor) or not isinstance(v, Tensor):
                raise ValueError(
                    f"layer {layer_idx}: K and V must be torch.Tensor"
                )

            if k.shape != v.shape:
                raise ValueError(
                    f"layer {layer_idx}: K shape {tuple(k.shape)} != "
                    f"V shape {tuple(v.shape)}"
                )

            if k.dim() != 3 or k.shape[1:] != expected_shape_suffix:
                raise ValueError(
                    f"layer {layer_idx}: expected K shape "
                    f"(n_tokens, {self._num_kv_heads}, {self._head_dim}), "
                    f"got {tuple(k.shape)}"
                )

            if n_tokens is None:
                n_tokens = k.shape[0]
                if n_tokens == 0:
                    raise ValueError(
                        f"layer {layer_idx}: n_tokens must be > 0"
                    )
            elif k.shape[0] != n_tokens:
                raise ValueError(
                    f"layer {layer_idx}: n_tokens={k.shape[0]} disagrees "
                    f"with first layer's n_tokens={n_tokens}"
                )

            if k.dtype != self._dtype or v.dtype != self._dtype:
                raise ValueError(
                    f"layer {layer_idx}: expected dtype {self._dtype}, "
                    f"got K={k.dtype} V={v.dtype}"
                )

            if k.device != self._device or v.device != self._device:
                raise ValueError(
                    f"layer {layer_idx}: expected device {self._device}, "
                    f"got K={k.device} V={v.device}"
                )

        assert n_tokens is not None  # type narrowing

        # Capacity check
        new_total = self.n_total_tokens + n_tokens
        if new_total > self._max_total_tokens:
            raise RuntimeError(
                f"appending tag_id={tag_id!r} (n_tokens={n_tokens}) would "
                f"bring total to {new_total}, exceeding max_total_tokens="
                f"{self._max_total_tokens}. Evict first."
            )

        # Store. Detach to ensure no autograd graph is retained.
        stored_kvs: dict[int, tuple[Tensor, Tensor]] = {
            layer_idx: (k.detach(), v.detach())
            for layer_idx, (k, v) in layer_kvs.items()
        }
        entry = KVEntry(
            tag_id=tag_id,
            n_tokens=n_tokens,
            layer_kvs=stored_kvs,
        )
        self._entries[tag_id] = entry

        # Invalidate all layer caches
        self._dirty_layers.update(self._adapted_layers)

        logger.debug(
            "append tag_id=%s n_tokens=%d total_tokens=%d/%d n_tags=%d",
            tag_id,
            n_tokens,
            self.n_total_tokens,
            self._max_total_tokens,
            self.n_tags,
        )

    def evict(self, tag_id: str) -> bool:
        """Remove all K/V for a tag.

        Args:
            tag_id: Identifier of the tag to remove.

        Returns:
            True if the tag was found and removed; False if it wasn't in the bank.
        """
        if tag_id not in self._entries:
            return False

        entry = self._entries.pop(tag_id)
        self._dirty_layers.update(self._adapted_layers)

        logger.debug(
            "evict tag_id=%s freed %d tokens; remaining %d tags / %d tokens",
            tag_id,
            entry.n_tokens,
            self.n_tags,
            self.n_total_tokens,
        )
        return True

    def clear(self) -> None:
        """Remove all entries.

        Called at the end of a successful sleep cycle, when consolidated
        knowledge has been transferred from the bank to W_cons.
        """
        n_freed_tags = len(self._entries)
        n_freed_tokens = self.n_total_tokens
        self._entries.clear()
        self._cached.clear()
        self._dirty_layers.clear()

        if n_freed_tags > 0:
            logger.info(
                "bank cleared: freed %d tags / %d tokens",
                n_freed_tags,
                n_freed_tokens,
            )

    # ------------------------------------------------------------------
    # Read API for attention injection
    # ------------------------------------------------------------------

    def get_for_layer(
        self,
        layer_idx: int,
    ) -> tuple[Tensor, Tensor] | None:
        """Concatenated K, V across all stored entries for one layer.

        Args:
            layer_idx: An index in ``self.adapted_layers``.

        Returns:
            Tuple ``(K, V)`` of stacked tensors, each of shape
            ``(n_total_tokens, num_kv_heads, head_dim)``, where
            ``n_total_tokens`` is the sum across stored entries. Order matches
            insertion order. Returns ``None`` if the bank is empty.

        Raises:
            ValueError: If ``layer_idx`` is not in adapted_layers.
        """
        if layer_idx not in self._adapted_layer_set:
            raise ValueError(
                f"layer_idx={layer_idx} is not in adapted_layers="
                f"{sorted(self._adapted_layers)}"
            )

        if not self._entries:
            return None

        # Rebuild cache if dirty for this layer (or never built)
        if layer_idx in self._dirty_layers or layer_idx not in self._cached:
            ks: list[Tensor] = []
            vs: list[Tensor] = []
            for entry in self._entries.values():
                k, v = entry.layer_kvs[layer_idx]
                ks.append(k)
                vs.append(v)
            self._cached[layer_idx] = (
                torch.cat(ks, dim=0),
                torch.cat(vs, dim=0),
            )
            self._dirty_layers.discard(layer_idx)

        return self._cached[layer_idx]

    def get_entry_offsets(self) -> dict[str, tuple[int, int]]:
        """Map of tag_id -> (start_offset, end_offset) in concatenated tensors.

        Useful for callers who need to know which slice of a layer's
        concatenated K/V tensor corresponds to which original tag (e.g. for
        per-tag attention attribution debugging).

        Returns:
            Dict mapping tag_id to ``(start, end)`` where the entry's tokens
            occupy ``[start:end]`` in the concatenated layer tensors.
        """
        offsets: dict[str, tuple[int, int]] = {}
        cursor = 0
        for tag_id, entry in self._entries.items():
            offsets[tag_id] = (cursor, cursor + entry.n_tokens)
            cursor += entry.n_tokens
        return offsets

    # ------------------------------------------------------------------
    # Diagnostics / repr
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, int | float | bool]:
        """Snapshot of bank statistics for logging."""
        return {
            "n_tags": self.n_tags,
            "n_total_tokens": self.n_total_tokens,
            "max_total_tokens": self._max_total_tokens,
            "occupancy": self.n_total_tokens / self._max_total_tokens,
            "at_capacity": self.at_capacity,
            "n_adapted_layers": len(self._adapted_layers),
        }

    def __repr__(self) -> str:
        return (
            f"KVMemoryBank("
            f"n_tags={self.n_tags}, "
            f"n_total_tokens={self.n_total_tokens}/{self._max_total_tokens}, "
            f"adapted_layers={list(self._adapted_layers)}, "
            f"device={self._device}, dtype={self._dtype}"
            f")"
        )
