"""
Layer-specific plasticity profile and weight constraints for W_cons during sleep.

Implements the formalization from Q3.4:
  - Plasticity profile phi(l/L) = phi_min + (1 - phi_min) * (l/L)^2
  - Hard clipping of W_cons after each training step
  - Checkpoint save/restore for rollback and clipping reference

Lower layers get low plasticity (~0.1), upper layers get high plasticity (~1.0).
This reflects the biological observation that lower cortical layers encode stable
structural knowledge while upper layers are more plastic for factual updates.

Reference: SLEEP_Formalization.md Q3.4
"""

from __future__ import annotations

import copy
import logging
import re
from typing import Optional

import torch
from peft import PeftModel

from sleep.config import WeightsConfig

logger = logging.getLogger(__name__)

# Architecture-specific regex patterns for extracting layer indices from
# fully-qualified parameter names.  Matches the patterns in lora.py.
_LAYER_PATTERNS: list[re.Pattern] = [
    re.compile(r"transformer\.h\.(\d+)\."),     # GPT-2
    re.compile(r"model\.layers\.(\d+)\."),       # LLaMA / Mistral / Phi
]


def _extract_layer_index(param_name: str) -> Optional[int]:
    """Extract the transformer layer index from a fully-qualified parameter name.

    Handles both GPT-2 (``transformer.h.<idx>``) and LLaMA-style
    (``model.layers.<idx>``) naming conventions.

    Returns:
        The layer index, or ``None`` if the name doesn't match any known pattern.
    """
    for pattern in _LAYER_PATTERNS:
        m = pattern.search(param_name)
        if m:
            return int(m.group(1))
    return None


def _count_total_layers(model: PeftModel) -> int:
    """Determine the total number of transformer layers in the model.

    Scans all module names and finds the maximum layer index, then adds 1
    (since layers are 0-indexed).
    """
    max_idx: int = -1
    for name, _ in model.named_modules():
        idx = _extract_layer_index(name)
        if idx is not None and idx > max_idx:
            max_idx = idx
    if max_idx < 0:
        raise RuntimeError(
            "Could not determine layer count: no transformer layer patterns found "
            "in model module names."
        )
    return max_idx + 1


# ---------------------------------------------------------------------------
# Plasticity profile
# ---------------------------------------------------------------------------

def compute_plasticity_profile(
    num_layers: int,
    adapted_layers: list[int],
    phi_min: float = 0.1,
) -> dict[int, float]:
    """Compute the plasticity value phi(l/L) for each adapted layer.

    Formula::

        phi(l/L) = phi_min + (1 - phi_min) * (l / L)^2

    where *l* is the (0-indexed) layer index and *L* is ``num_layers``.

    Args:
        num_layers: Total number of transformer layers (L).
        adapted_layers: List of layer indices that have LoRA adapters.
        phi_min: Plasticity floor for the lowest layer (default 0.1).

    Returns:
        Dictionary mapping each adapted layer index to its plasticity value.
    """
    profile: dict[int, float] = {}
    for layer_idx in adapted_layers:
        x: float = layer_idx / num_layers
        phi: float = phi_min + (1.0 - phi_min) * (x ** 2)
        profile[layer_idx] = phi
    return profile


def get_adapted_layer_indices(
    model: PeftModel,
    config: WeightsConfig,
) -> list[int]:
    """Determine which layer indices are in the top 1/3 of the model.

    This mirrors the logic in ``lora._get_top_layer_indices`` but works from
    a ``PeftModel`` instance rather than a raw ``PreTrainedModel``.

    Args:
        model: A PeftModel wrapping the base transformer.
        config: ``WeightsConfig`` with ``adapted_fraction``.

    Returns:
        Sorted list of layer indices in the top ``adapted_fraction`` of layers.
    """
    import math

    num_layers: int = _count_total_layers(model)
    first_adapted: int = math.ceil((1.0 - config.adapted_fraction) * num_layers)
    adapted: list[int] = list(range(first_adapted, num_layers))

    logger.info(
        "Adapted layers: %s (top %.0f%% of %d layers)",
        adapted,
        config.adapted_fraction * 100,
        num_layers,
    )
    return adapted


# ---------------------------------------------------------------------------
# Gradient scaling
# ---------------------------------------------------------------------------

def apply_plasticity_scaling(
    model: PeftModel,
    plasticity_profile: dict[int, float],
    adapter_name: str = "w_cons",
) -> None:
    """Scale gradients of each adapted layer by its plasticity value.

    Must be called **after** ``loss.backward()`` and **before** ``optimizer.step()``.

    For each LoRA parameter in layer *l* belonging to ``adapter_name``::

        param.grad *= plasticity_profile[l]

    Parameters in layers not present in the profile are left untouched (this
    shouldn't happen if the profile covers all adapted layers).

    Args:
        model: PeftModel currently in sleep training mode.
        plasticity_profile: Mapping from layer index to plasticity value phi(l/L).
        adapter_name: Which adapter's parameters to scale (default ``"w_cons"``).
    """
    scaled_count: int = 0
    for name, param in model.named_parameters():
        if adapter_name not in name:
            continue
        if param.grad is None:
            continue

        layer_idx: Optional[int] = _extract_layer_index(name)
        if layer_idx is None:
            continue

        phi: Optional[float] = plasticity_profile.get(layer_idx)
        if phi is None:
            logger.warning(
                "Parameter %s in layer %d has no plasticity value; skipping scaling.",
                name,
                layer_idx,
            )
            continue

        param.grad.mul_(phi)
        scaled_count += 1

    logger.debug("Scaled gradients for %d %s parameters.", scaled_count, adapter_name)


# ---------------------------------------------------------------------------
# Hard clipping
# ---------------------------------------------------------------------------

def enforce_weight_bounds(
    model: PeftModel,
    checkpoint_state: dict[str, torch.Tensor],
    base_model_norms: dict[str, float],
    plasticity_profile: dict[int, float],
    delta_max: float = 0.001,
    adapter_name: str = "w_cons",
) -> None:
    """Hard-clip W_cons parameters to stay within the delta_max bound.

    After each sleep training step, for each adapted layer *l*::

        delta_W = W_cons[l] - W_cons_checkpoint[l]
        bound   = delta_max * phi(l/L) * ||W_slow_base[l]||_F
        if ||delta_W||_F > bound:
            W_cons[l] = W_cons_checkpoint[l] + delta_W * (bound / ||delta_W||_F)

    This is the "safety net" beyond EWC soft constraints (Q3.4).

    Args:
        model: PeftModel in sleep training mode.
        checkpoint_state: State dict of W_cons saved at the start of the sleep
            cycle (from :func:`save_adapter_checkpoint`).
        base_model_norms: Frobenius norms of base model weight matrices
            (from :func:`compute_base_model_norms`).  Keys are the base-model
            parameter names (without adapter suffixes).
        plasticity_profile: Mapping from layer index to plasticity phi(l/L).
        delta_max: Maximum relative weight change (default 0.001 from config).
        adapter_name: Which adapter to clip (default ``"w_cons"``).
    """
    clipped_count: int = 0
    named_params: dict[str, torch.nn.Parameter] = dict(model.named_parameters())

    for param_name, current_param in named_params.items():
        if adapter_name not in param_name:
            continue

        # Retrieve the checkpoint value for this parameter.
        if param_name not in checkpoint_state:
            continue

        checkpoint_value: torch.Tensor = checkpoint_state[param_name].to(
            current_param.device
        )

        # Determine layer index for plasticity lookup.
        layer_idx: Optional[int] = _extract_layer_index(param_name)
        if layer_idx is None:
            continue

        phi: Optional[float] = plasticity_profile.get(layer_idx)
        if phi is None:
            continue

        # Find the corresponding base-model norm.
        # The base model norm key is derived by mapping the adapter param name
        # back to the base weight it modifies.  We search for a base norm key
        # that shares the same layer and module path.
        base_norm: Optional[float] = _find_base_norm(param_name, base_model_norms)
        if base_norm is None or base_norm == 0.0:
            logger.debug(
                "Skipping clipping for %s: no matching base norm found.",
                param_name,
            )
            continue

        # Compute delta and check bound.
        delta_w: torch.Tensor = current_param.data - checkpoint_value
        delta_norm: float = torch.norm(delta_w, p="fro").item()
        bound: float = delta_max * phi * base_norm

        if delta_norm > bound:
            scale: float = bound / delta_norm
            current_param.data.copy_(checkpoint_value + delta_w * scale)
            clipped_count += 1

    if clipped_count > 0:
        logger.info(
            "Hard-clipped %d %s parameters to delta_max=%.4f bound.",
            clipped_count,
            adapter_name,
            delta_max,
        )


def _find_base_norm(
    adapter_param_name: str,
    base_model_norms: dict[str, float],
) -> Optional[float]:
    """Find the base-model Frobenius norm corresponding to an adapter parameter.

    Strategy: extract the layer index and module path (e.g., ``attn.c_proj`` or
    ``self_attn.v_proj``) from the adapter parameter name, then search
    ``base_model_norms`` for a key containing the same layer and module.

    Returns:
        The Frobenius norm, or ``None`` if no match is found.
    """
    layer_idx: Optional[int] = _extract_layer_index(adapter_param_name)
    if layer_idx is None:
        return None

    # Build search fragments from the adapter param name.
    # Adapter param names look like:
    #   base_model.model.transformer.h.10.attn.c_proj.lora_A.w_cons.weight
    #   base_model.model.model.layers.20.self_attn.v_proj.lora_B.w_cons.weight
    # We want the module part between the layer index and "lora_".
    # For GPT-2:  "attn.c_proj"
    # For LLaMA:  "self_attn.v_proj"

    # Find patterns like ".h.<idx>." or ".layers.<idx>."
    for pattern_str in [
        rf"\.h\.{layer_idx}\.(.*?)\.lora_",
        rf"\.layers\.{layer_idx}\.(.*?)\.lora_",
    ]:
        m = re.search(pattern_str, adapter_param_name)
        if m:
            module_path: str = m.group(1)
            # Search base norms for matching layer + module
            layer_str_variants = [f".h.{layer_idx}.", f".layers.{layer_idx}."]
            for base_key, norm_val in base_model_norms.items():
                for layer_str in layer_str_variants:
                    if layer_str in base_key and module_path in base_key:
                        return norm_val
            break

    return None


# ---------------------------------------------------------------------------
# Base model norms
# ---------------------------------------------------------------------------

def compute_base_model_norms(model: PeftModel) -> dict[str, float]:
    """Compute Frobenius norms of base model weight matrices.

    These serve as reference magnitudes for the hard clipping bound in
    :func:`enforce_weight_bounds`.  Only weight matrices (not biases) in
    transformer layers are included.

    Args:
        model: A PeftModel wrapping the base transformer.

    Returns:
        Dictionary mapping base-model parameter names to their Frobenius norms.
    """
    norms: dict[str, float] = {}

    for name, param in model.named_parameters():
        # Skip adapter parameters (they contain "lora_" in the name).
        if "lora_" in name:
            continue

        # Only include weight matrices in transformer layers.
        layer_idx: Optional[int] = _extract_layer_index(name)
        if layer_idx is None:
            continue

        # Only 2-D weight matrices (skip biases, layer norms, etc.)
        if param.ndim < 2:
            continue

        # Skip LayerNorm / RMSNorm weights (they're 1-D conceptually even
        # though some implementations store them as parameters).
        lower_name: str = name.lower()
        if "layernorm" in lower_name or "rmsnorm" in lower_name or "ln_" in lower_name:
            continue

        with torch.no_grad():
            norms[name] = torch.norm(param.float(), p="fro").item()

    logger.info("Computed Frobenius norms for %d base model weight matrices.", len(norms))
    return norms


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def save_adapter_checkpoint(
    model: PeftModel,
    adapter_name: str = "w_cons",
) -> dict[str, torch.Tensor]:
    """Save current adapter state for rollback or clipping reference.

    Args:
        model: PeftModel with the named adapter.
        adapter_name: Which adapter to checkpoint (default ``"w_cons"``).

    Returns:
        Dictionary mapping parameter names to detached CPU copies of their values.
    """
    checkpoint: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if adapter_name in name:
            checkpoint[name] = param.data.detach().cpu().clone()

    logger.info(
        "Saved checkpoint for adapter %r: %d parameters.",
        adapter_name,
        len(checkpoint),
    )
    return checkpoint


def restore_adapter_checkpoint(
    model: PeftModel,
    checkpoint: dict[str, torch.Tensor],
    adapter_name: str = "w_cons",
) -> None:
    """Restore adapter parameters from a saved checkpoint (for rollback).

    Args:
        model: PeftModel with the named adapter.
        checkpoint: State dict from :func:`save_adapter_checkpoint`.
        adapter_name: Which adapter to restore (default ``"w_cons"``).
    """
    restored_count: int = 0
    for name, param in model.named_parameters():
        if adapter_name not in name:
            continue
        if name in checkpoint:
            param.data.copy_(checkpoint[name].to(param.device))
            restored_count += 1

    logger.info(
        "Restored %d parameters for adapter %r from checkpoint.",
        restored_count,
        adapter_name,
    )
