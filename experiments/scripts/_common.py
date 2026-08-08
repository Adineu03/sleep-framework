"""
Shared helpers for SLEEP experiment scripts.

Historically each numbered script carried its own copy of ``load_config`` and a
``dtype_map`` dict, and only script 08 set any seed (and only for its naive-LoRA
arm). The mentor review (P1) requires every key experiment to run across several
seeds and at least one additional model family / precision. This module removes
the duplication so seeding and model/precision selection are configured in one
place and behave identically everywhere.

Nothing here changes numerical behaviour for existing configs: the dtype map,
the eager-attention default for KV pipelines, and the YAML→dataclass mapping
match what the scripts already did.
"""

from __future__ import annotations

import datetime as _dt
import json
import os

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from sleep.config import PRPConfig, SleepConfig, TaggingConfig, WeightsConfig
from sleep.utils.logging import get_logger
from sleep.utils.seed import seed_everything

logger = get_logger("experiment.common")

_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def dtype_from_str(name: str) -> torch.dtype:
    """Map a config dtype string to a ``torch.dtype`` (default float32)."""
    return _DTYPE_MAP.get(name, torch.float32)


def load_experiment_config(config_path: str | None) -> dict:
    """Load a YAML experiment config into the flat dict scripts consume.

    Mirrors the historical per-script ``load_config``: with ``None`` it returns
    a GPT-2/CPU proof-of-concept config; otherwise it reads the YAML and splats
    each block into the matching dataclass. The ``**data[block]`` splat means
    every YAML key must be a valid dataclass field — this is intentional, so a
    typo fails loudly rather than being silently ignored.

    Returns a dict with keys: ``model_name``, ``device``, ``dtype``,
    ``tagging``, ``prp``, ``weights``, ``sleep``, ``use_real_mu_surprise``.
    """
    if config_path is None:
        return {
            "model_name": "gpt2",
            "device": "cpu",
            "dtype": "float32",
            "tagging": TaggingConfig(),
            "prp": PRPConfig(),
            "weights": WeightsConfig(),
            "sleep": SleepConfig(),
            "use_real_mu_surprise": False,
        }

    with open(config_path) as f:
        data = yaml.safe_load(f)

    return {
        "model_name": data["model"]["name"],
        "device": data["model"].get("device", "cpu"),
        "dtype": data["model"].get("dtype", "float32"),
        "tagging": TaggingConfig(**data.get("tagging", {})),
        "prp": PRPConfig(**data.get("prp", {})),
        "weights": WeightsConfig(**data.get("weights", {})),
        "sleep": SleepConfig(**data.get("sleep", {})),
        "use_real_mu_surprise": data.get("experiment", {}).get(
            "use_real_mu_surprise", False
        ),
    }


def build_model_and_tokenizer(
    cfg: dict,
    *,
    eager_attention: bool = True,
):
    """Load model + tokenizer per config, consistently across scripts.

    Args:
        cfg:             Output of :func:`load_experiment_config`.
        eager_attention: Force ``attn_implementation="eager"``. Required for the
            KV-injection pipelines (the SDPA path handles 4D additive masks
            differently and silently degrades top-k gating); harmless for
            non-KV runs. Pass ``False`` only to benchmark the default backend.

    Returns:
        ``(model, tokenizer)`` with the model on ``cfg["device"]`` and the
        tokenizer's ``pad_token`` guaranteed set.
    """
    dtype = dtype_from_str(cfg["dtype"])
    load_kwargs = {"dtype": dtype}
    if eager_attention:
        load_kwargs["attn_implementation"] = "eager"

    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = model.to(cfg["device"])

    logger.info(
        "Loaded %s | dtype=%s | attn=%s | device=%s | layers=%s",
        cfg["model_name"], cfg["dtype"],
        getattr(model.config, "_attn_implementation", "?"),
        cfg["device"], getattr(model.config, "num_hidden_layers", "?"),
    )
    return model, tokenizer


def results_path(basename: str, output: str | None = None) -> str:
    """Return a timestamped results path under ``experiments/results/``.

    Args:
        basename: Stem for the file, e.g. ``"warmup_extension"``.
        output:   Explicit path override; returned as-is if given.

    Returns:
        Absolute path ending in ``.json`` (no timestamp is available inside
        Workflow scripts, but these run as normal processes so ``datetime`` is
        fine here).
    """
    if output is not None:
        return output
    results_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "results")
    )
    os.makedirs(results_dir, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(results_dir, f"{basename}_{stamp}.json")


def save_results(payload: dict, path: str) -> str:
    """Write ``payload`` as pretty JSON (``default=str`` for safety)."""
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info("Wrote results to %s", path)
    return path


__all__ = [
    "dtype_from_str",
    "load_experiment_config",
    "build_model_and_tokenizer",
    "results_path",
    "save_results",
    "seed_everything",
]
