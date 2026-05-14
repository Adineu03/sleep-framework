"""
SLEEP Dual Weight System — main entry point.

Provides ``DualWeightSystem``, the single class that external code interacts with
to manage the full dual-weight architecture:

    W_effective = W_slow_base + W_cons (consolidated, updated in sleep)
                               + W_fast (hippocampal, updated on surprise)

Usage::

    from sleep.weights import DualWeightSystem

    dws = DualWeightSystem(model, config.weights)
    dws.set_mode("wake_inference")
    ...

All heavy lifting is delegated to sub-modules (lora, composition, fast_update,
plasticity).  This file is intentionally thin.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sleep.utils.logging import get_logger, metrics

logger = get_logger("sleep.weights")

from sleep.weights.lora import (
    setup_dual_adapters,
    count_adapter_params,
    get_target_modules,
)
from sleep.weights.composition import (
    set_wake_inference_mode,
    set_wfast_training_mode,
    set_sleep_generation_mode,
    set_sleep_training_mode,
    set_target_inference_mode,
    freeze_base_model,
    get_trainable_params,
)
from sleep.weights.fast_update import FastWeightUpdater
from sleep.weights.plasticity import (
    compute_plasticity_profile,
    get_adapted_layer_indices,
    apply_plasticity_scaling,
    enforce_weight_bounds,
    compute_base_model_norms,
    save_adapter_checkpoint,
    restore_adapter_checkpoint,
)
from sleep.weights.kv_memory import KVMemoryBank
from sleep.weights.kv_injection import KVInjector

if TYPE_CHECKING:
    import torch
    from peft import PeftModel
    from transformers import PreTrainedModel

from sleep.config import WeightsConfig

# ---- Mode dispatch table ----

_MODE_DISPATCH: dict[str, callable] = {
    "wake_inference": set_wake_inference_mode,
    "wfast_training": set_wfast_training_mode,
    "sleep_generation": set_sleep_generation_mode,
    "sleep_training": set_sleep_training_mode,
    "target_inference": set_target_inference_mode,
}


class DualWeightSystem:
    """Manages the complete dual-weight architecture (W_slow_base + W_cons + W_fast).

    This is the **only** class external code needs to interact with.  It wraps
    a HuggingFace ``PreTrainedModel`` with two LoRA adapters (``w_fast`` and
    ``w_cons``) and exposes clean methods for mode switching, fast weight
    updates, consolidation helpers, and plasticity-aware training.

    Args:
        model: A HuggingFace ``PreTrainedModel`` (will be wrapped with peft).
        config: ``WeightsConfig`` controlling LoRA rank, plasticity, etc.
    """

    # -- Construction --------------------------------------------------------

    def __init__(
        self,
        model: PreTrainedModel,
        config: WeightsConfig,
        *,
        use_kv_memory_for_fast: bool = False,
        kv_max_total_tokens: int = 10000,
        kv_top_k: int = 0,
    ) -> None:
        """Initialize the dual-weight system.

        Args:
            model: A HuggingFace ``PreTrainedModel`` (will be wrapped with peft).
            config: ``WeightsConfig`` controlling LoRA rank, plasticity, etc.
            use_kv_memory_for_fast: If True, also initialize a KV memory bank
                + injector for an alternative W_fast substrate. The LoRA W_fast
                adapter is still set up (for backward compat / side-by-side
                comparison); callers choose which path they exercise by which
                method they call (``update_fast_weights`` vs ``write_to_kv_bank``).
            kv_max_total_tokens: Capacity of the KV memory bank, in stored
                tokens summed across all entries. Only used when
                ``use_kv_memory_for_fast=True``.
            kv_top_k: When > 0 and < n_mem, the patched attention forward
                computes Q · K_mem scores and only makes the top-k highest-
                scoring memory positions visible per query. This is the
                Memorizing-Transformers-style retrieval gate that prevents
                irrelevant memories from drowning attention. Recommended
                starting value: 8-16. Default 0 (no gating, all memories
                visible).
        """
        self._config = config
        self._use_kv_memory_for_fast = use_kv_memory_for_fast
        self._kv_top_k = int(kv_top_k)

        # 1. Apply dual LoRA adapters (w_fast, w_cons)
        self._model: PeftModel = setup_dual_adapters(model, config)

        # 2. Freeze the base model explicitly (belt-and-suspenders)
        freeze_base_model(self._model)

        # 3. Initialize the fast weight updater
        self._fast_updater = FastWeightUpdater(self._model, config)

        # 4. Compute plasticity profile (layer_index -> phi value)
        adapted_layers = get_adapted_layer_indices(self._model, config)
        self._adapted_layers: list[int] = list(adapted_layers)
        num_layers = self._model.config.num_hidden_layers
        self._plasticity_profile: dict[int, float] = compute_plasticity_profile(
            num_layers, adapted_layers, config.phi_min,
        )

        # 5. Compute base model norms for hard clipping reference
        self._base_model_norms: dict[str, float] = compute_base_model_norms(
            self._model,
        )

        # 6. Cache parameter counts
        self._w_fast_params: int = count_adapter_params(self._model, "w_fast")
        self._w_cons_params: int = count_adapter_params(self._model, "w_cons")

        # 7. Optional KV memory bank for W_fast substrate
        self._kv_bank: KVMemoryBank | None = None
        self._kv_injector: KVInjector | None = None
        if self._use_kv_memory_for_fast:
            self._init_kv_memory(kv_max_total_tokens)

        # Log summary
        logger.info(
            "DualWeightSystem initialized | W_fast: %s params | W_cons: %s params | "
            "Adapted layers: %s | Plasticity: %.3f–%.3f%s",
            f"{self._w_fast_params:,}", f"{self._w_cons_params:,}",
            sorted(self._plasticity_profile.keys()),
            min(self._plasticity_profile.values()),
            max(self._plasticity_profile.values()),
            " | KV bank installed" if self._use_kv_memory_for_fast else "",
        )

        # 8. Set initial mode to wake inference
        self.set_mode("wake_inference")

    # -- KV memory initialization --------------------------------------------

    def _init_kv_memory(self, max_total_tokens: int) -> None:
        """Construct ``KVMemoryBank`` + ``KVInjector`` and install the injector.

        Called from ``__init__`` only when ``use_kv_memory_for_fast=True``.
        """
        # Read attention dims from the model config
        model_cfg = self._model.config
        num_kv_heads: int = model_cfg.num_key_value_heads
        head_dim: int = getattr(
            model_cfg, "head_dim",
            model_cfg.hidden_size // model_cfg.num_attention_heads,
        )

        # Pick device + dtype from a model parameter
        ref_param = next(self._model.parameters())
        device = ref_param.device
        dtype = ref_param.dtype

        self._kv_bank = KVMemoryBank(
            adapted_layer_indices=self._adapted_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_total_tokens=max_total_tokens,
            device=device,
            dtype=dtype,
        )
        self._kv_injector = KVInjector(
            self._model, self._kv_bank, top_k=self._kv_top_k,
        )
        self._kv_injector.install()

        logger.info(
            "KV memory mode enabled | adapted_layers=%s | num_kv_heads=%d | "
            "head_dim=%d | max_total_tokens=%d | top_k=%d | device=%s | dtype=%s",
            self._adapted_layers, num_kv_heads, head_dim,
            max_total_tokens, self._kv_top_k, device, dtype,
        )

    # -- Properties ----------------------------------------------------------

    @property
    def model(self) -> PeftModel:
        """The ``PeftModel`` with both adapters attached."""
        return self._model

    @property
    def fast_updater(self) -> FastWeightUpdater:
        """The ``FastWeightUpdater`` instance for online W_fast updates."""
        return self._fast_updater

    @property
    def plasticity_profile(self) -> dict[int, float]:
        """Mapping of adapted layer index to plasticity coefficient (phi)."""
        return self._plasticity_profile

    @property
    def base_model_norms(self) -> dict[str, float]:
        """Per-parameter base model norms used for hard clipping."""
        return self._base_model_norms

    @property
    def w_fast_params(self) -> int:
        """Total scalar parameters in the W_fast adapter."""
        return self._w_fast_params

    @property
    def w_cons_params(self) -> int:
        """Total scalar parameters in the W_cons adapter."""
        return self._w_cons_params

    @property
    def kv_bank(self) -> "KVMemoryBank | None":
        """The KV memory bank if ``use_kv_memory_for_fast=True``, else ``None``."""
        return self._kv_bank

    @property
    def kv_injector(self) -> "KVInjector | None":
        """The injector managing patched attention forwards, if any."""
        return self._kv_injector

    @property
    def use_kv_memory_for_fast(self) -> bool:
        """Whether KV memory is the active substrate for W_fast."""
        return self._use_kv_memory_for_fast

    @property
    def adapted_layers(self) -> list[int]:
        """Sorted list of layer indices in the adapted set."""
        return list(self._adapted_layers)

    # -- Mode switching ------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        """Switch between operational modes.

        Valid modes:
            ``"wake_inference"``  — full model eval (W_slow + W_cons + W_fast).
            ``"wfast_training"``  — W_fast trainable, rest frozen.
            ``"sleep_generation"`` — full model eval for replay generation.
            ``"sleep_training"``  — W_cons trainable, W_fast disabled.
            ``"target_inference"`` — W_slow + W_cons only, eval mode.

        Args:
            mode: One of the five mode strings listed above.

        Raises:
            ValueError: If *mode* is not recognised.
        """
        if mode not in _MODE_DISPATCH:
            raise ValueError(
                f"Unknown mode {mode!r}. Valid modes: {sorted(_MODE_DISPATCH.keys())}"
            )
        _MODE_DISPATCH[mode](self._model)

    # -- Fast weight updates -------------------------------------------------

    def update_fast_weights(
        self,
        token_ids: torch.Tensor,
        span_start: int,
        span_end: int,
        E_span: float,
        device: str = "cpu",
    ) -> float:
        """Perform a single online update to W_fast on a surprising span.

        Delegates entirely to :class:`FastWeightUpdater`.

        Args:
            token_ids: Full token sequence (1-D ``LongTensor``).
            span_start: Start index of the surprising span.
            span_end: End index (exclusive) of the surprising span.
            E_span: Aggregate prediction error for the span.
            device: Device to run the update on.

        Returns:
            The loss value computed during the update step.
        """
        loss = self._fast_updater.update(
            model=self._model,
            token_ids=token_ids,
            span_start=span_start,
            span_end=span_end,
            E_span=E_span,
            device=device,
        )
        logger.debug("W_fast update | span=[%d:%d] | E_span=%.3f | loss=%.4f", span_start, span_end, E_span, loss)
        metrics.log({"weights/wfast_loss": loss, "weights/wfast_E_span": E_span})
        return loss

    # -- KV memory writes (alternative W_fast substrate) ---------------------

    def write_to_kv_bank(
        self,
        tag_id: str,
        token_ids: "torch.Tensor",
        span_start: int,
        span_end: int,
        device: str | None = None,
    ) -> int:
        """Extract pre-RoPE K and V for a tagged span at each adapted layer
        and append them to the KV memory bank.

        This is the KV-memory analogue of :meth:`update_fast_weights`. Unlike
        the gradient-based update, this is a one-shot direct write: no
        learning rate, no optimization, no backward pass.

        Implementation:
            1. Run a forward pass on ``token_ids[:span_end]`` with
               ``output_hidden_states=True`` to get per-layer hidden states.
            2. For each adapted layer ``l``:
                a. Get ``hidden_states[l]`` — the input to layer l's attention
                   (after the previous layer, before this layer's pre-attn
                   layernorm).
                b. Apply ``model.layers[l].input_layernorm`` — **critical**:
                   K/V inside attention are computed on the layer-normed input,
                   not the raw hidden state.
                c. Apply ``self_attn.k_proj`` and ``self_attn.v_proj`` to get
                   pre-RoPE K and raw V.
                d. Slice the span tokens, reshape to
                   ``(span_len, num_kv_heads, head_dim)``.
            3. Append all per-layer K/V to the bank under ``tag_id``.

        Args:
            tag_id: Unique identifier (typically the tag's source_id).
            token_ids: Full token sequence — 1-D ``LongTensor``.
            span_start: Start index of the tagged span (inclusive).
            span_end: End index of the tagged span (exclusive).
            device: Device for the forward pass. Defaults to the model's device.

        Returns:
            Number of tokens stored.

        Raises:
            RuntimeError: If KV memory mode wasn't enabled at construction,
                or if the bank is at capacity.
            ValueError: If span indices are invalid.
        """
        if not self._use_kv_memory_for_fast:
            raise RuntimeError(
                "KV memory mode is not enabled. "
                "Construct DualWeightSystem with use_kv_memory_for_fast=True."
            )

        import torch  # noqa: PLC0415  — lazy import to keep top-level light

        if span_end <= span_start:
            raise ValueError(
                f"span_end ({span_end}) must be > span_start ({span_start})"
            )
        if span_end > token_ids.shape[0]:
            raise ValueError(
                f"span_end ({span_end}) exceeds token_ids length "
                f"({token_ids.shape[0]})"
            )

        # Pick device
        if device is None:
            device = next(self._model.parameters()).device
        else:
            device = torch.device(device)

        span_len = span_end - span_start
        n_kv_heads = self._kv_bank.num_kv_heads
        head_dim = self._kv_bank.head_dim

        # Forward pass with output_hidden_states for span context
        input_ids = token_ids[:span_end].unsqueeze(0).to(device)  # (1, span_end)
        was_training = self._model.training
        self._model.eval()
        try:
            with torch.no_grad():
                outputs = self._model(
                    input_ids=input_ids,
                    output_hidden_states=True,
                    use_cache=False,
                )
        finally:
            if was_training:
                self._model.train()

        # hidden_states is a tuple: hidden_states[i] = input to layer i
        # (hidden_states[0] = embedding output; hidden_states[i] for i>=1 = output of layer i-1)
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError(
                "Model did not return hidden_states. Check that the model "
                "supports output_hidden_states=True."
            )

        # Locate the actual decoder layers (handles PEFT wrapper)
        layers = self._locate_decoder_layers()

        layer_kvs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for layer_idx in self._adapted_layers:
            h_layer = hidden_states[layer_idx]  # (1, span_end, hidden_size)

            # CRITICAL: apply input_layernorm before k_proj/v_proj.
            # The attention's K/V inside Qwen2DecoderLayer.forward are
            # computed from input_layernorm(hidden_states), not from
            # hidden_states directly. Skipping this layer-norm produces
            # K/V values that are close-but-wrong and silently degrade
            # attention scores.
            h_normed = layers[layer_idx].input_layernorm(h_layer)

            # Slice the span tokens (only [span_start:span_end] are stored)
            h_span = h_normed[0, span_start:span_end, :]  # (span_len, hidden)

            attn_module = layers[layer_idx].self_attn
            k_flat = attn_module.k_proj(h_span)  # (span_len, n_kv_heads * head_dim)
            v_flat = attn_module.v_proj(h_span)  # (span_len, n_kv_heads * head_dim)

            # Reshape to (span_len, n_kv_heads, head_dim) and cast to bank dtype
            k = k_flat.view(span_len, n_kv_heads, head_dim).to(
                dtype=self._kv_bank.dtype, device=self._kv_bank.device,
            )
            v = v_flat.view(span_len, n_kv_heads, head_dim).to(
                dtype=self._kv_bank.dtype, device=self._kv_bank.device,
            )
            layer_kvs[layer_idx] = (k.detach(), v.detach())

        self._kv_bank.append(tag_id, layer_kvs)
        logger.debug(
            "write_to_kv_bank | tag_id=%s | n_tokens=%d | bank=%d/%d tokens",
            tag_id, span_len,
            self._kv_bank.n_total_tokens, self._kv_bank.max_total_tokens,
        )
        metrics.log({
            "weights/kv_bank_n_tokens": self._kv_bank.n_total_tokens,
            "weights/kv_bank_n_tags": self._kv_bank.n_tags,
        })
        return span_len

    def evict_from_kv_bank(self, tag_id: str) -> bool:
        """Remove a tag's K/V from the bank.

        Args:
            tag_id: The identifier passed to :meth:`write_to_kv_bank`.

        Returns:
            True if the tag was found and removed; False otherwise.

        Raises:
            RuntimeError: If KV memory mode wasn't enabled.
        """
        if not self._use_kv_memory_for_fast:
            raise RuntimeError("KV memory mode is not enabled.")
        return self._kv_bank.evict(tag_id)

    def set_kv_enabled(self, enabled: bool) -> None:
        """Enable or disable KV memory injection at runtime.

        When ``False``, all forward passes through adapted attention layers
        bypass memory injection (the bank is preserved but unused). When
        ``True``, stored memories are injected with the configured top-k
        gating.

        This is the mechanism the sleep engine uses to switch off the KV
        crutch during W_cons training: replays are generated with KV
        active so the model can attend to source material, but training
        runs with KV disabled so W_cons must encode the knowledge into
        weights rather than relying on memory.

        Raises:
            RuntimeError: If KV memory mode wasn't enabled at construction.
        """
        if not self._use_kv_memory_for_fast:
            raise RuntimeError("KV memory mode is not enabled.")
        self._kv_injector.set_enabled(enabled)

    def set_kv_top_k(self, top_k: int) -> None:
        """Update the KV-memory top-k retrieval gate at runtime.

        See :class:`KVInjector` for the semantics. Useful when sweeping
        ``top_k`` without re-creating the bank or re-writing memories.

        Raises:
            RuntimeError: If KV memory mode wasn't enabled.
        """
        if not self._use_kv_memory_for_fast:
            raise RuntimeError("KV memory mode is not enabled.")
        self._kv_top_k = int(top_k)
        self._kv_injector.set_top_k(top_k)

    def clear_kv_bank(self) -> None:
        """Remove all entries from the KV memory bank.

        Called after a successful sleep cycle when consolidated knowledge
        has been transferred from the bank into W_cons.

        Raises:
            RuntimeError: If KV memory mode wasn't enabled.
        """
        if not self._use_kv_memory_for_fast:
            raise RuntimeError("KV memory mode is not enabled.")
        self._kv_bank.clear()

    def cleanup(self) -> None:
        """Uninstall any patched forwards. Safe to call repeatedly.

        After ``cleanup()``, the underlying model returns to its un-patched
        attention behaviour. Useful in tests and at the end of an experiment.
        """
        if self._kv_injector is not None and self._kv_injector.is_installed:
            self._kv_injector.uninstall()

    def _locate_decoder_layers(self) -> list:
        """Return the list of decoder layers (each has .self_attn,
        .input_layernorm). Handles PeftModel wrapping."""
        candidates = [
            getattr(self._model, "base_model", None),
            self._model,
        ]
        for cand in candidates:
            if cand is None:
                continue
            inner = getattr(cand, "model", cand)
            if hasattr(inner, "model") and hasattr(inner.model, "layers"):
                return list(inner.model.layers)
            if hasattr(inner, "layers"):
                return list(inner.layers)
        raise RuntimeError("Could not locate decoder layers on PeftModel.")

    # -- Consolidation helpers (sleep phase) ---------------------------------

    def save_cons_checkpoint(self) -> dict:
        """Snapshot W_cons adapter state for potential rollback.

        Returns:
            A dictionary that can be passed to :meth:`restore_cons_checkpoint`.
        """
        return save_adapter_checkpoint(self._model, adapter_name="w_cons")

    def restore_cons_checkpoint(self, checkpoint: dict) -> None:
        """Restore W_cons from a previously saved checkpoint (rollback).

        Args:
            checkpoint: Dict produced by :meth:`save_cons_checkpoint`.
        """
        restore_adapter_checkpoint(self._model, checkpoint, adapter_name="w_cons")

    def apply_plasticity_scaling(self) -> None:
        """Scale W_cons gradients by the plasticity profile.

        Must be called **after** ``loss.backward()`` and **before**
        ``optimizer.step()``.
        """
        apply_plasticity_scaling(
            self._model,
            self._plasticity_profile,
            adapter_name="w_cons",
        )

    def enforce_weight_bounds(self, checkpoint: dict) -> None:
        """Hard-clip W_cons parameters within delta_max of *checkpoint*.

        Must be called **after** ``optimizer.step()``.

        Args:
            checkpoint: Pre-step checkpoint from :meth:`save_cons_checkpoint`,
                used as the reference point for maximum allowed change.
        """
        enforce_weight_bounds(
            self._model,
            checkpoint,
            self._base_model_norms,
            self._plasticity_profile,
            delta_max=self._config.delta_max,
            adapter_name="w_cons",
        )

    def get_cons_trainable_params(self) -> list[torch.nn.Parameter]:
        """Return W_cons trainable parameters for optimizer construction.

        The model must be in ``"sleep_training"`` mode (i.e., W_cons has
        ``requires_grad=True``) for this to return a non-empty list.
        """
        return get_trainable_params(self._model)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Primary interface
    "DualWeightSystem",
    # LoRA setup (for advanced / testing use)
    "setup_dual_adapters",
    "count_adapter_params",
    "get_target_modules",
    # Fast weight updater
    "FastWeightUpdater",
    # Composition mode setters
    "set_wake_inference_mode",
    "set_wfast_training_mode",
    "set_sleep_generation_mode",
    "set_sleep_training_mode",
    "set_target_inference_mode",
    "freeze_base_model",
    "get_trainable_params",
    # Plasticity utilities
    "compute_plasticity_profile",
    "get_adapted_layer_indices",
    "apply_plasticity_scaling",
    "enforce_weight_bounds",
    "compute_base_model_norms",
    "save_adapter_checkpoint",
    "restore_adapter_checkpoint",
]
