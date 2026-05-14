"""
LoRA adapter setup for the SLEEP dual weight system.

Creates W_fast (hippocampal adapter, updated during wake) and W_cons (consolidated
adapter, updated during sleep) as low-rank adaptation matrices on selected layers
of a HuggingFace transformer model.

Architecture (from Formalization Q3.1):
    - Adapted layers: top L/3 layers only (e.g., layers 8-11 for 12-layer GPT-2)
    - Adapted matrices: value projection (V) and output projection (O) only
    - Rank r and alpha α from WeightsConfig
    - B matrices initialized to zero, A matrices from N(0, σ²)
    - Two independent adapter sets: w_fast and w_cons
"""

from __future__ import annotations

import logging
import math
import re
from typing import TYPE_CHECKING

from peft import LoraConfig, PeftModel, get_peft_model

from sleep.config import WeightsConfig

if TYPE_CHECKING:
    from transformers import PreTrainedModel

logger = logging.getLogger(__name__)

# Maps logical projection names to architecture-specific module suffixes.
# Each entry maps ("v_proj", "o_proj") to the actual module names for that arch.
_ARCH_MODULE_MAP: dict[str, dict[str, str]] = {
    "gpt2": {
        # GPT-2 uses Conv1D. c_attn is fused QKV, c_proj is output projection.
        # peft handles the fused QKV internally when targeting c_attn.
        "v_proj": "attn.c_attn",
        "o_proj": "attn.c_proj",
    },
    "llama": {
        "v_proj": "self_attn.v_proj",
        "o_proj": "self_attn.o_proj",
    },
    "mistral": {
        "v_proj": "self_attn.v_proj",
        "o_proj": "self_attn.o_proj",
    },
    "phi": {
        "v_proj": "self_attn.v_proj",
        "o_proj": "self_attn.o_proj",
    },
}

# Maps architecture type to the regex pattern for extracting layer indices
# from fully-qualified module names.
_ARCH_LAYER_PATTERN: dict[str, str] = {
    "gpt2": r"transformer\.h\.(\d+)\.",
    "llama": r"model\.layers\.(\d+)\.",
    "mistral": r"model\.layers\.(\d+)\.",
    "phi": r"model\.layers\.(\d+)\.",
}


def _detect_architecture(model: PreTrainedModel) -> str:
    """Detect the model architecture family from the model's config.

    Returns one of the keys in ``_ARCH_MODULE_MAP``.

    Raises:
        ValueError: If the architecture cannot be identified.
    """
    model_type: str = getattr(model.config, "model_type", "").lower()

    # Direct match
    if model_type in _ARCH_MODULE_MAP:
        return model_type

    # Fallback heuristics based on module names
    named_modules = {name for name, _ in model.named_modules()}
    if any("transformer.h." in n and "attn.c_attn" in n for n in named_modules):
        return "gpt2"
    if any("model.layers." in n and "self_attn.v_proj" in n for n in named_modules):
        # Could be llama, mistral, phi — they share the same mapping
        return "llama"

    raise ValueError(
        f"Unsupported model architecture: model_type={model_type!r}. "
        f"Supported: {sorted(_ARCH_MODULE_MAP.keys())}"
    )


def _count_layers(model: PreTrainedModel, arch: str) -> int:
    """Return the total number of transformer layers in the model."""
    pattern = re.compile(_ARCH_LAYER_PATTERN[arch])
    layer_indices: set[int] = set()
    for name, _ in model.named_modules():
        m = pattern.search(name)
        if m:
            layer_indices.add(int(m.group(1)))
    if not layer_indices:
        raise ValueError(f"Could not find any layers matching pattern for arch={arch!r}")
    return max(layer_indices) + 1  # layers are 0-indexed


def _get_top_layer_indices(num_layers: int, adapted_fraction: float) -> list[int]:
    """Compute the indices of the top ``adapted_fraction`` layers.

    For a 12-layer model with fraction=1/3: ceil(2*12/3) = 8, so layers [8, 9, 10, 11].
    """
    first_adapted = math.ceil((1.0 - adapted_fraction) * num_layers)
    return list(range(first_adapted, num_layers))


def get_target_modules(model: PreTrainedModel, config: WeightsConfig) -> list[str]:
    """Determine which module name suffixes to apply LoRA to.

    Only targets the top L/3 layers, on V and O projections. Rather than returning
    fully-qualified names, this returns the short target-module names that
    ``peft.LoraConfig`` expects (e.g. ``["c_attn", "c_proj"]`` for GPT-2), while
    the layer restriction is handled via ``layers_to_transform`` in the LoRA config.

    Args:
        model: A HuggingFace ``PreTrainedModel``.
        config: ``WeightsConfig`` specifying ``adapted_matrices`` and ``adapted_fraction``.

    Returns:
        De-duplicated list of short module-name suffixes for ``LoraConfig.target_modules``.
    """
    arch = _detect_architecture(model)
    arch_map = _ARCH_MODULE_MAP[arch]

    # Map logical names (v_proj, o_proj) to architecture-specific suffixes
    target_suffixes: list[str] = []
    for logical_name in config.adapted_matrices:
        if logical_name not in arch_map:
            raise ValueError(
                f"Unknown adapted matrix {logical_name!r} for arch={arch!r}. "
                f"Known: {sorted(arch_map.keys())}"
            )
        target_suffixes.append(arch_map[logical_name])

    # De-duplicate (e.g., for GPT-2 c_attn appears once even if mapped from v_proj)
    seen: set[str] = set()
    unique: list[str] = []
    for suffix in target_suffixes:
        # Extract just the final component for peft target_modules
        short = suffix.rsplit(".", maxsplit=1)[-1]
        if short not in seen:
            seen.add(short)
            unique.append(short)

    logger.info(
        "LoRA target modules for %s: %s",
        arch,
        unique,
    )
    return unique


def _build_lora_config(
    model: PreTrainedModel,
    config: WeightsConfig,
) -> LoraConfig:
    """Build a ``peft.LoraConfig`` from the SLEEP weights configuration.

    Layer selection is done via ``layers_to_transform`` so that only the top L/3
    layers receive LoRA adapters.
    """
    arch = _detect_architecture(model)
    num_layers = _count_layers(model, arch)
    top_layers = _get_top_layer_indices(num_layers, config.adapted_fraction)
    target_modules = get_target_modules(model, config)

    logger.info(
        "LoRA config: rank=%d, alpha=%d, layers=%s (of %d total), targets=%s",
        config.lora_rank,
        config.lora_alpha,
        top_layers,
        num_layers,
        target_modules,
    )

    return LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=target_modules,
        layers_to_transform=top_layers,
        bias="none",
        task_type="CAUSAL_LM",
    )


def create_lora_adapter(
    model: PreTrainedModel,
    config: WeightsConfig,
    adapter_name: str = "default",
) -> PeftModel:
    """Apply a single LoRA adapter to the model using the ``peft`` library.

    Args:
        model: A HuggingFace ``PreTrainedModel`` (not yet wrapped by peft).
        config: ``WeightsConfig`` with ``lora_rank``, ``lora_alpha``, etc.
        adapter_name: Name for the adapter (e.g. ``"w_fast"`` or ``"w_cons"``).

    Returns:
        ``PeftModel`` with the named adapter applied. The base model weights
        are frozen; only the LoRA parameters are trainable.
    """
    lora_config = _build_lora_config(model, config)
    peft_model: PeftModel = get_peft_model(model, lora_config, adapter_name=adapter_name)

    n_params = count_adapter_params(peft_model, adapter_name)
    logger.info(
        "Created LoRA adapter %r: %d trainable parameters (%.4f%% of base model)",
        adapter_name,
        n_params,
        100.0 * n_params / sum(p.numel() for p in model.parameters()),
    )
    return peft_model


def setup_dual_adapters(
    model: PreTrainedModel,
    config: WeightsConfig,
) -> PeftModel:
    """Set up both W_fast and W_cons adapters on the model.

    Uses peft's multi-adapter support. After setup, the active adapter can be
    switched with ``model.set_adapter("w_fast")`` or ``model.set_adapter("w_cons")``.

    The ``w_fast`` adapter is set as active by default (wake phase).

    Args:
        model: A HuggingFace ``PreTrainedModel`` (not yet wrapped by peft).
        config: ``WeightsConfig`` controlling LoRA hyperparameters and layer selection.

    Returns:
        ``PeftModel`` with two adapters: ``"w_fast"`` (active) and ``"w_cons"``.
    """
    # Create the first adapter (w_fast)
    peft_model = create_lora_adapter(model, config, adapter_name="w_fast")

    # Add the second adapter (w_cons) with the same architecture but independent params
    lora_config = _build_lora_config(model, config)
    peft_model.add_adapter("w_cons", lora_config)

    n_fast = count_adapter_params(peft_model, "w_fast")
    n_cons = count_adapter_params(peft_model, "w_cons")
    logger.info(
        "Dual adapters ready: w_fast=%d params, w_cons=%d params",
        n_fast,
        n_cons,
    )

    # Ensure w_fast is active (wake phase default)
    peft_model.set_adapter("w_fast")

    return peft_model


def count_adapter_params(model: PeftModel, adapter_name: str) -> int:
    """Count trainable parameters belonging to a specific adapter.

    Args:
        model: A ``PeftModel`` with one or more adapters.
        adapter_name: The adapter whose parameters to count (e.g. ``"w_fast"``).

    Returns:
        Total number of scalar parameters in the adapter.
    """
    total = 0
    for name, param in model.named_parameters():
        if adapter_name in name:
            total += param.numel()
    return total
