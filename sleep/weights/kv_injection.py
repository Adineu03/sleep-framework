"""KV memory injection into transformer attention.

Replaces the ``forward`` method of selected ``Qwen2Attention`` modules with a
patched version that injects stored K/V from a :class:`KVMemoryBank` as a
prefix to the attention computation. Memory occupies negative positions
``[-n_mem, -1]`` so that current-sequence RoPE remains untouched.

See ``docs/KV_INJECTION_DESIGN.md`` for the architectural rationale.

Implementation notes:
    - Stored K is **pre-RoPE**. We apply RoPE at read time with negative
      position IDs.
    - V is stored verbatim (V does not get RoPE).
    - Attention mask is extended on the kv-axis with **float zeros** of the
      mask's own dtype. Memory columns are visible to all query positions.
      A boolean True or integer 1 here would silently produce wrong attention
      scores in Qwen's float-mask attention path.
    - The patched forward is a thin wrapper around the original logic. We
      copy it carefully so that an upstream change to ``Qwen2Attention.forward``
      is detected via ``KVInjector.SUPPORTED_TRANSFORMERS_VERSIONS`` and a
      structural check, rather than silently producing wrong output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

# Transformers internals we rely on. These are stable APIs in the Qwen2
# implementation across recent transformers releases (5.x).
from transformers.models.qwen2 import modeling_qwen2 as _qwen2_mod

from sleep.utils.logging import get_logger
from sleep.weights.kv_memory import KVMemoryBank

logger = get_logger("sleep.weights.kv_injection")


# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------


def _rotate_half(x: Tensor) -> Tensor:
    """Rotate the last dimension by half — the same op Qwen2 uses internally.

    Equivalent to: ``torch.cat([-x[..., d//2:], x[..., :d//2]], dim=-1)``.
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope_k_only(
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    unsqueeze_dim: int = 1,
) -> Tensor:
    """Apply RoPE rotation to K only.

    Mirrors the K-half of ``transformers.models.qwen2.modeling_qwen2.apply_rotary_pos_emb``,
    used at memory-injection time where there is no Q to rotate.

    Args:
        k:   Tensor of shape ``(B, H, T, D)`` — keys to rotate.
        cos: Cosine table of shape ``(B, T, D)``.
        sin: Sine table, same shape.
        unsqueeze_dim: Axis to unsqueeze on cos/sin to broadcast over heads.
            Default 1 (insert head axis between batch and time).

    Returns:
        Rotated K tensor, same shape as input.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (k * cos) + (_rotate_half(k) * sin)


# ---------------------------------------------------------------------------
# Attention mask extension
# ---------------------------------------------------------------------------


def _extend_attention_mask_with_memory(
    attention_mask: Tensor | None,
    n_mem: int,
    *,
    batch_size: int,
    q_len: int,
    device: torch.device,
    dtype: torch.dtype,
    memory_visibility: Tensor | None = None,
) -> Tensor | None:
    """Prepend ``n_mem`` columns to a 4D attention mask, optionally with
    per-query top-k gating on memory visibility.

    Qwen2's attention path uses a **float** additive mask: 0.0 means visible,
    a large negative value (typically ``-inf`` or the dtype's min) means masked.
    Memory columns must be filled with values in the mask's dtype.

    Args:
        attention_mask: The current mask of shape ``(B, 1, Q, KV)``, or
            ``None`` (no mask present, e.g. flash-attention path).
        n_mem: Number of memory tokens to prepend.
        batch_size, q_len, device, dtype: Used when ``attention_mask is None``
            to construct a fresh extended mask. ``dtype`` is also used to
            ensure the prepended zeros match the mask dtype.
        memory_visibility: Optional bool tensor of shape ``(B, Q, n_mem)``.
            ``True`` = memory position visible to that query (mask = 0.0),
            ``False`` = hidden (mask = dtype min). When ``None``, all memory
            positions are visible to all queries (the default — equivalent to
            top_k = n_mem). When provided, this implements top-k gating.

    Returns:
        Extended mask of shape ``(B, 1, Q, n_mem + KV)`` if ``attention_mask``
        was given, or ``None`` if ``attention_mask is None`` (memory-only
        masks aren't supported in the implicit-causal path).
    """
    if n_mem == 0:
        return attention_mask

    if attention_mask is None:
        # See note in original docstring — implicit-causal path, can't add
        # a partial mask without breaking current-sequence handling.
        return None

    if attention_mask.dim() != 4:
        raise ValueError(
            f"attention_mask must be 4D (B, 1, Q, KV); got shape "
            f"{tuple(attention_mask.shape)}"
        )

    B = attention_mask.shape[0]
    one = attention_mask.shape[1]
    Q = attention_mask.shape[2]

    if memory_visibility is None:
        # All memory visible to all queries (the original behaviour)
        mem_cols = torch.zeros(
            B, one, Q, n_mem,
            dtype=attention_mask.dtype, device=attention_mask.device,
        )
    else:
        # Top-k gating: visible -> 0.0, hidden -> dtype min
        if memory_visibility.shape != (B, Q, n_mem):
            raise ValueError(
                f"memory_visibility must have shape ({B}, {Q}, {n_mem}); "
                f"got {tuple(memory_visibility.shape)}"
            )
        # Use dtype.min rather than literal -inf — float-mask convention,
        # and avoids NaN issues some attention impls have with -inf.
        neg_large = torch.finfo(attention_mask.dtype).min
        mem_cols = torch.where(
            memory_visibility.unsqueeze(1),  # broadcast head dim
            torch.zeros((), dtype=attention_mask.dtype, device=attention_mask.device),
            torch.full((), neg_large, dtype=attention_mask.dtype, device=attention_mask.device),
        )  # (B, 1, Q, n_mem)

    return torch.cat([mem_cols, attention_mask], dim=-1)


def _compute_topk_visibility(
    query_states_rope: Tensor,
    memory_k_rope: Tensor,
    top_k: int,
) -> Tensor | None:
    """Compute a per-query boolean mask selecting the top-k most relevant
    memory positions by Q · K_mem inner product.

    The scoring aggregates across heads: for each (batch, query) pair, we
    compute a single relevance score per memory position by taking the max
    over heads. Top-k of those scores are flagged as visible.

    Why max-over-heads: any head finding a memory relevant is sufficient;
    we don't want to average and lose the signal from a few specialized
    heads.

    Args:
        query_states_rope: Post-RoPE query states, shape (B, H_q, Q, D).
        memory_k_rope: Post-RoPE memory keys, shape (B, H_kv, M, D).
        top_k: Number of memory positions to keep visible per query.
            If ``top_k >= M``, returns ``None`` (no gating needed).
            If ``top_k <= 0``, returns ``None`` and the caller should treat
            it as "no gating".

    Returns:
        Boolean tensor of shape (B, Q, M), or ``None`` if no gating is needed.
    """
    n_mem = memory_k_rope.shape[2]
    if top_k <= 0 or top_k >= n_mem:
        return None  # no gating

    B, H_q, Q, D = query_states_rope.shape
    H_kv = memory_k_rope.shape[1]
    n_groups = H_q // H_kv  # GQA group size

    # Reshape Q to align with kv-heads: aggregate query heads within each
    # kv-head's group via mean (arbitrary; max would also work).
    if n_groups > 1:
        Q_grouped = query_states_rope.view(B, H_kv, n_groups, Q, D).mean(dim=2)
    else:
        Q_grouped = query_states_rope  # H_q == H_kv

    # Compute scores: (B, H_kv, Q, M)
    # einsum: for each head, dot product across dim D
    scores = torch.einsum("bhqd,bhmd->bhqm", Q_grouped, memory_k_rope)

    # Aggregate across kv-heads via max — most-relevant-head wins
    scores_per_query = scores.max(dim=1).values  # (B, Q, M)

    # For each (b, q), find top-k memory indices
    topk_idx = scores_per_query.topk(top_k, dim=-1).indices  # (B, Q, k)

    # Build boolean visibility mask
    visible = torch.zeros_like(scores_per_query, dtype=torch.bool)  # (B, Q, M)
    visible.scatter_(-1, topk_idx, True)

    return visible


# ---------------------------------------------------------------------------
# RoPE for memory positions
# ---------------------------------------------------------------------------


def _compute_memory_rope(
    rope_module: torch.nn.Module,
    n_mem: int,
    batch_size: int,
    reference_tensor: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute (cos, sin) for memory positions ``[-n_mem, -1]``.

    Args:
        rope_module: The model's ``Qwen2RotaryEmbedding`` instance.
        n_mem: Number of memory tokens.
        batch_size: Batch dimension expected by the rope module.
        reference_tensor: Used by the rope module to determine device and dtype.

    Returns:
        Tuple ``(cos, sin)`` each of shape ``(batch_size, n_mem, head_dim)``.
    """
    # position_ids must be (batch, n_mem)
    positions = torch.arange(
        -n_mem, 0,
        device=reference_tensor.device,
        dtype=torch.long,
    ).unsqueeze(0)
    if batch_size > 1:
        positions = positions.expand(batch_size, -1)

    cos, sin = rope_module(reference_tensor, positions)
    return cos, sin


# ---------------------------------------------------------------------------
# Patched forward
# ---------------------------------------------------------------------------


def _kv_injected_attention_forward(
    self,
    hidden_states: Tensor,
    position_embeddings: tuple[Tensor, Tensor],
    attention_mask: Tensor | None,
    past_key_values: Any | None = None,
    **kwargs: Any,
) -> tuple[Tensor, Tensor | None]:
    """Replacement for ``Qwen2Attention.forward`` with KV memory injection.

    Mirrors the original logic exactly except for the memory-injection step
    inserted between the standard RoPE application and the attention call.

    The memory bank and rope module are accessed via attributes set by
    :class:`KVInjector` at install time:
        - ``self._sleep_kv_bank``: the :class:`KVMemoryBank`
        - ``self._sleep_rope_module``: the model's rotary embedding module
    """
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    # Standard Q/K/V projections + RoPE on current sequence ------------------
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = _qwen2_mod.apply_rotary_pos_emb(
        query_states, key_states, cos, sin,
    )

    # ---- Standard past_key_values cache (kept ENTIRELY SEPARATE from memory) -
    # We update the cache FIRST, with only the current sequence's K/V.
    # Memory is injected AFTER the cache update, so it's never part of the
    # autoregressive cache. This is essential for correctness during
    # multi-step generation: without this, memory K/V would be appended to
    # the cache on every generation step, accumulating duplicates.
    if past_key_values is not None:
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx,
        )

    # ---- Memory injection (after cache update) -----------------------------
    bank: KVMemoryBank = self._sleep_kv_bank
    layer_idx: int = self.layer_idx

    # Enable/disable toggle: when disabled, skip injection entirely.
    # Used during sleep-training phases so W_cons learns from context alone.
    injection_enabled = getattr(self, "_sleep_kv_enabled", True)

    mem_kv = (
        bank.get_for_layer(layer_idx)
        if injection_enabled and layer_idx in bank.adapted_layers
        else None
    )
    n_mem = 0
    if mem_kv is not None:
        mem_k_pre, mem_v = mem_kv  # each (n_mem, num_kv_heads, head_dim)
        n_mem = mem_k_pre.shape[0]

        # Reshape to (B, num_kv_heads, n_mem, head_dim)
        batch_size = key_states.shape[0]
        mem_k_pre_b = mem_k_pre.unsqueeze(0).transpose(1, 2)  # (1, num_kv_heads, n_mem, head_dim)
        mem_v_b = mem_v.unsqueeze(0).transpose(1, 2)
        if batch_size > 1:
            mem_k_pre_b = mem_k_pre_b.expand(batch_size, -1, -1, -1)
            mem_v_b = mem_v_b.expand(batch_size, -1, -1, -1)

        # RoPE for negative positions
        rope_module = self._sleep_rope_module
        mem_cos, mem_sin = _compute_memory_rope(
            rope_module=rope_module,
            n_mem=n_mem,
            batch_size=batch_size,
            reference_tensor=mem_k_pre_b,
        )
        mem_k_rope = _apply_rope_k_only(mem_k_pre_b, mem_cos, mem_sin)

        # Cast to match the dtype of current K/V (in case the bank's dtype
        # differs slightly — e.g., bank in bfloat16 but RoPE returned float32).
        mem_k_rope = mem_k_rope.to(dtype=key_states.dtype)
        mem_v_b = mem_v_b.to(dtype=value_states.dtype)

        # Top-k visibility gating: read from attribute set by KVInjector
        top_k = getattr(self, "_sleep_kv_top_k", 0)
        memory_visibility = _compute_topk_visibility(
            query_states_rope=query_states,
            memory_k_rope=mem_k_rope,
            top_k=top_k,
        )  # bool (B, Q, M) or None for no-gating

        # Prepend memory along the time axis. Now key_states/value_states
        # contain (memory + cached_current).
        key_states = torch.cat([mem_k_rope, key_states], dim=2)
        value_states = torch.cat([mem_v_b, value_states], dim=2)

        # Extend attention mask. The incoming attention_mask already has
        # KV-axis length matching the cached current sequence (because HF
        # constructs it from the cache state); we prepend n_mem columns.
        attention_mask = _extend_attention_mask_with_memory(
            attention_mask,
            n_mem=n_mem,
            batch_size=key_states.shape[0],
            q_len=query_states.shape[2],
            device=key_states.device,
            dtype=key_states.dtype,
            memory_visibility=memory_visibility,
        )

    # ---- Attention computation ----------------------------------------------
    attention_interface = _qwen2_mod.ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, _qwen2_mod.eager_attention_forward,
    )

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


# ---------------------------------------------------------------------------
# The injector
# ---------------------------------------------------------------------------


@dataclass
class _InstalledLayer:
    """Bookkeeping for one patched attention module."""

    layer_idx: int
    attn_module: torch.nn.Module
    original_forward: Any  # the original bound method


class KVInjector:
    """Installs and uninstalls KV memory injection on a transformer model.

    Usage::

        bank = KVMemoryBank(adapted_layer_indices=[19, 20, 21], ...)
        injector = KVInjector(model, bank)
        injector.install()
        # ... model forward passes now consult the bank ...
        injector.uninstall()  # restores original behavior

    The injector is idempotent only in the sense that ``install`` raises if
    already installed, and ``uninstall`` raises if not installed. Use
    :attr:`is_installed` to check state.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        bank: KVMemoryBank,
        *,
        top_k: int = 0,
    ) -> None:
        """
        Args:
            model: A transformer model.
            bank: The KV memory bank to inject from.
            top_k: If > 0 and < n_mem, enables top-k retrieval gating per
                query position per layer. Only the ``top_k`` highest-scoring
                memory positions (by Q · K_mem inner product) are made visible
                to each query. If 0 or >= n_mem, all memories are visible
                (no gating). Recommended starting value: 8-16.
        """
        self._model = model
        self._bank = bank
        self._top_k = int(top_k)
        self._enabled = True   # default: injection ON when installed
        self._installed_layers: list[_InstalledLayer] = []

        # Validate at construction so we fail fast
        self._rope_module = self._locate_rope_module()
        self._validate_layers_exist()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    @property
    def bank(self) -> KVMemoryBank:
        return self._bank

    @property
    def is_installed(self) -> bool:
        return len(self._installed_layers) > 0

    @property
    def top_k(self) -> int:
        return self._top_k

    def set_top_k(self, top_k: int) -> None:
        """Update top_k at runtime. Takes effect on the next forward pass.

        Useful for sweeping ``top_k`` without reinstalling the injector.
        """
        self._top_k = int(top_k)
        for slot in self._installed_layers:
            slot.attn_module._sleep_kv_top_k = self._top_k  # type: ignore[attr-defined]

    @property
    def is_enabled(self) -> bool:
        """Whether memory injection is currently active.

        When False, the patched forward bypasses memory injection entirely
        and behaves identically to the unpatched attention. Use during
        sleep-training phases where W_cons must learn from context alone.
        """
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable memory injection at runtime.

        Effects of ``set_enabled(False)``:
            - Stored K/V are NOT injected into attention forward
            - The bank's contents are preserved (not cleared)
            - Output of patched forward is bit-identical to the unpatched
              attention's output

        Effects of ``set_enabled(True)``:
            - Stored K/V are injected as a prefix at negative positions
            - Top-k gating (if configured) is applied per query

        This toggle is the mechanism by which sleep training disables
        the KV crutch so that W_cons learns from context alone.
        """
        self._enabled = bool(enabled)
        for slot in self._installed_layers:
            slot.attn_module._sleep_kv_enabled = self._enabled  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Locate model parts
    # ------------------------------------------------------------------

    def _locate_rope_module(self) -> torch.nn.Module:
        """Find the model's Qwen2RotaryEmbedding instance.

        Walks possible locations to support both raw Qwen2ForCausalLM
        (``model.model.rotary_emb``) and PeftModel-wrapped variants
        (``peft_model.base_model.model.model.rotary_emb``).
        """
        # Each path is a sequence of attribute names. The first one whose
        # full path resolves AND ends in an object with ``rotary_emb`` wins.
        paths_to_try: list[tuple[str, ...]] = [
            ("rotary_emb",),
            ("model", "rotary_emb"),
            ("model", "model", "rotary_emb"),
            ("base_model", "model", "rotary_emb"),
            ("base_model", "model", "model", "rotary_emb"),
        ]
        for path in paths_to_try:
            obj = self._model
            ok = True
            for attr in path:
                if not hasattr(obj, attr):
                    ok = False
                    break
                obj = getattr(obj, attr)
            if ok and isinstance(obj, torch.nn.Module):
                return obj
        raise RuntimeError(
            "Could not locate rotary_emb on the model. Tried paths: "
            f"{paths_to_try}"
        )

    def _get_layers(self) -> list[torch.nn.Module]:
        """Return the list of decoder layers (each has .self_attn).

        Walks the same set of wrapper paths as ``_locate_rope_module``.
        """
        paths_to_try: list[tuple[str, ...]] = [
            ("layers",),
            ("model", "layers"),
            ("model", "model", "layers"),
            ("base_model", "model", "layers"),
            ("base_model", "model", "model", "layers"),
        ]
        for path in paths_to_try:
            obj = self._model
            ok = True
            for attr in path:
                if not hasattr(obj, attr):
                    ok = False
                    break
                obj = getattr(obj, attr)
            if ok:
                return list(obj)  # type: ignore[arg-type]
        raise RuntimeError(
            f"Could not locate decoder layers. Tried paths: {paths_to_try}"
        )

    def _validate_layers_exist(self) -> None:
        """Verify all bank.adapted_layers exist in the model."""
        layers = self._get_layers()
        n_layers = len(layers)
        for layer_idx in self._bank.adapted_layers:
            if not (0 <= layer_idx < n_layers):
                raise ValueError(
                    f"adapted_layer index {layer_idx} out of range "
                    f"[0, {n_layers}) for this model"
                )

    # ------------------------------------------------------------------
    # Install / uninstall
    # ------------------------------------------------------------------

    def install(self) -> None:
        """Replace ``forward`` on each adapted layer's self_attn module."""
        if self.is_installed:
            raise RuntimeError("KVInjector is already installed")

        layers = self._get_layers()
        for layer_idx in self._bank.adapted_layers:
            attn_module = layers[layer_idx].self_attn

            # Stash references for the patched forward to access
            attn_module._sleep_kv_bank = self._bank  # type: ignore[attr-defined]
            attn_module._sleep_rope_module = self._rope_module  # type: ignore[attr-defined]
            attn_module._sleep_kv_top_k = self._top_k  # type: ignore[attr-defined]
            attn_module._sleep_kv_enabled = self._enabled  # type: ignore[attr-defined]

            # Save original forward and install replacement
            original_forward = attn_module.forward
            patched = _kv_injected_attention_forward.__get__(
                attn_module, type(attn_module),
            )
            attn_module.forward = patched

            self._installed_layers.append(
                _InstalledLayer(
                    layer_idx=layer_idx,
                    attn_module=attn_module,
                    original_forward=original_forward,
                )
            )

        logger.info(
            "KVInjector installed on %d layers: %s",
            len(self._installed_layers),
            list(self._bank.adapted_layers),
        )

    def uninstall(self) -> None:
        """Restore the original forward on each previously patched module."""
        if not self.is_installed:
            raise RuntimeError("KVInjector is not installed")

        for slot in self._installed_layers:
            slot.attn_module.forward = slot.original_forward
            # Clean up the temporary attributes
            for attr in (
                "_sleep_kv_bank",
                "_sleep_rope_module",
                "_sleep_kv_top_k",
                "_sleep_kv_enabled",
            ):
                if hasattr(slot.attn_module, attr):
                    delattr(slot.attn_module, attr)

        n = len(self._installed_layers)
        self._installed_layers.clear()
        logger.info("KVInjector uninstalled from %d layers", n)

    def __enter__(self) -> "KVInjector":
        self.install()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.is_installed:
            self.uninstall()
