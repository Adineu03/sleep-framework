"""
Weight composition modes for the SLEEP dual-adapter system.

Manages switching between adapter configurations for different phases:
  - Wake inference:   W_slow_base + W_cons + W_fast  (eval, no grad)
  - W_fast training:  W_slow_base + W_cons + W_fast  (W_fast trainable, rest frozen)
  - Sleep generation: W_slow_base + W_cons + W_fast  (eval, no grad)
  - Sleep training:   W_slow_base + W_cons only      (W_cons trainable, rest frozen)
  - Target inference: W_slow_base + W_cons only      (eval, no grad)

Peft supports only ONE active adapter at a time. To combine both W_fast and W_cons,
we merge one into the base weights temporarily and activate the other.

Strategy:
  - "Both active" modes: merge W_cons into base, activate W_fast
  - "W_cons only" modes: unmerge W_cons, activate W_cons, W_fast inactive
  - This means W_cons merge/unmerge must be tracked carefully.

Reference: SLEEP_Formalization.md Q3.2
"""

from __future__ import annotations

import torch
from peft import PeftModel

# Canonical adapter names
ADAPTER_W_FAST: str = "w_fast"
ADAPTER_W_CONS: str = "w_cons"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _set_adapter_trainable(model: PeftModel, adapter_name: str, trainable: bool) -> None:
    """Set requires_grad for every parameter belonging to adapter_name."""
    for name, param in model.named_parameters():
        if adapter_name in name:
            param.requires_grad = trainable


def _freeze_all_adapters(model: PeftModel) -> None:
    """Freeze both W_fast and W_cons adapter parameters."""
    _set_adapter_trainable(model, ADAPTER_W_FAST, trainable=False)
    _set_adapter_trainable(model, ADAPTER_W_CONS, trainable=False)


def _freeze_all(model: PeftModel) -> None:
    """Freeze everything — base + all adapters."""
    for param in model.parameters():
        param.requires_grad = False


# ---------------------------------------------------------------------------
# State tracking: whether W_cons is currently merged into base
# We use a module-level flag. In a multi-instance scenario, this should
# be per-model, but for a single-model research setup this is fine.
# ---------------------------------------------------------------------------

_cons_merged: bool = False


def _ensure_cons_merged(model: PeftModel) -> None:
    """Merge W_cons into base weights so W_fast operates on top of W_slow+W_cons.

    After merging, W_fast is set as the active adapter. The forward pass then
    computes: (W_slow_base + W_cons_merged) + W_fast_active = W_eff.
    """
    global _cons_merged
    if not _cons_merged:
        # First, ensure any previous adapter state is clean
        # peft may auto-unmerge when switching adapters, so we handle that
        try:
            model.set_adapter(ADAPTER_W_CONS)
        except Exception:
            pass
        # Merge W_cons into base weights
        model.merge_adapter()
        _cons_merged = True
    # Activate W_fast as the live adapter on top of merged base+cons
    # Use model.base_model.set_adapter to avoid peft's auto-unmerge
    model.base_model.set_adapter(ADAPTER_W_FAST)


def _ensure_cons_unmerged(model: PeftModel) -> None:
    """Unmerge W_cons from base weights, restoring W_slow_base."""
    global _cons_merged
    if _cons_merged:
        # Switch to W_cons adapter context for unmerging
        model.base_model.set_adapter(ADAPTER_W_CONS)
        model.unmerge_adapter()
        _cons_merged = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def freeze_base_model(model: PeftModel) -> None:
    """Ensure all base model parameters have requires_grad=False."""
    for name, param in model.named_parameters():
        if ADAPTER_W_FAST not in name and ADAPTER_W_CONS not in name:
            param.requires_grad = False


def set_wake_inference_mode(model: PeftModel) -> None:
    """Configure model for wake-phase inference.

    Effective weights: W_slow_base + W_cons + W_fast.
    W_cons is merged into base; W_fast is the active adapter.
    Eval mode, no gradients.
    """
    _ensure_cons_merged(model)
    _freeze_all(model)
    model.eval()


def set_wfast_training_mode(model: PeftModel) -> None:
    """Configure model for W_fast gradient updates on surprising input.

    * W_fast adapter — trainable (active adapter in forward pass).
    * W_cons — merged into base weights (so W_fast learns residual on top).
    * W_slow_base — frozen (always).
    """
    _ensure_cons_merged(model)
    _freeze_all(model)
    _set_adapter_trainable(model, ADAPTER_W_FAST, trainable=True)
    model.train()


def set_sleep_generation_mode(model: PeftModel) -> None:
    """Configure model for replay generation during sleep.

    Effective weights: W_slow_base + W_cons + W_fast.
    Same as wake inference — both adapters contribute.
    Eval mode, no gradients.
    """
    _ensure_cons_merged(model)
    _freeze_all(model)
    model.eval()


def set_sleep_training_mode(model: PeftModel) -> None:
    """Configure model for W_cons training during sleep.

    * W_cons adapter — trainable (active adapter).
    * W_fast — NOT active (unmerged, not participating in forward pass).
    * W_slow_base — frozen (always).

    W_cons must be unmerged so it can receive gradients.
    """
    _ensure_cons_unmerged(model)
    model.set_adapter(ADAPTER_W_CONS)
    _freeze_all(model)
    _set_adapter_trainable(model, ADAPTER_W_CONS, trainable=True)
    model.train()


def set_target_inference_mode(model: PeftModel) -> None:
    """Configure model as W_target = W_slow_base + W_cons (no W_fast).

    Used for old-knowledge generation and quality checking during sleep.
    Eval mode, no gradients.
    """
    _ensure_cons_unmerged(model)
    model.set_adapter(ADAPTER_W_CONS)
    _freeze_all(model)
    model.eval()


def get_trainable_params(model: PeftModel) -> list[torch.nn.Parameter]:
    """Return currently trainable parameters (for optimizer construction)."""
    return [p for p in model.parameters() if p.requires_grad]


def is_cons_merged() -> bool:
    """Check if W_cons is currently merged into base weights."""
    return _cons_merged
