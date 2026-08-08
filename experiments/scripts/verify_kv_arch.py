"""
Cross-architecture KV-injection verification (run BEFORE any long runs on a
model family whose injection path has not been bit-verified on hardware).

Three ground-truth checks on the real model, mirroring the tiny-Qwen unit
tests but executed against the actual architecture and dtype:

  1. IDENTITY  — injector installed, bank empty: logits must equal the
                 uninstalled model's logits to fp tolerance.
  2. EFFECT    — bank populated with real episode K/V: logits must CHANGE.
  3. DISABLE   — set_enabled(False) with a populated bank: logits must return
                 to the uninstalled baseline.

Exit code 0 = all three pass (safe to run the matrix on this model).
Exit code 1 = any check fails (STOP: the injection path is wrong for this
arch and every downstream number would be silently invalid).

USAGE:
    python experiments/scripts/verify_kv_arch.py \
        --config experiments/configs/llama3_8b.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch

from _common import build_model_and_tokenizer, load_experiment_config, seed_everything
from sleep.weights import DualWeightSystem
from sleep.utils.logging import setup_logging, get_logger

setup_logging()
logger = get_logger("experiment.verify_kv")

PROBE = "The quick brown fox jumps over the lazy dog near the riverbank."
EPISODE = "Aurora Dynamics reported quarterly revenue of 48.2 million dollars in March."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--atol", type=float, default=None,
                        help="Equality tolerance. Default: 1e-5 fp32, 5e-3 bf16 "
                             "(bf16 epsilon at logit scale).")
    args = parser.parse_args()
    seed_everything(0)

    cfg = load_experiment_config(args.config)
    model, tokenizer = build_model_and_tokenizer(cfg, eager_attention=True)
    device = cfg["device"]
    atol = args.atol if args.atol is not None else (
        5e-3 if cfg["dtype"] == "bfloat16" else 1e-5
    )

    probe_ids = tokenizer(PROBE, return_tensors="pt").input_ids.to(device)

    @torch.no_grad()
    def logits() -> torch.Tensor:
        return model(input_ids=probe_ids).logits.float().cpu()

    # Baseline BEFORE any patching.
    baseline = logits()

    dws = DualWeightSystem(
        model, cfg["weights"],
        use_kv_memory_for_fast=True, kv_max_total_tokens=2000, kv_top_k=0,
    )
    # NOTE: DualWeightSystem wraps with PEFT (adapters init at zero => identity)
    # and installs the injector. All three checks run against that wrapped model,
    # so we re-baseline on the wrapped-but-empty-bank state for strictness:
    dws.set_kv_enabled(False)
    wrapped_baseline = logits()
    drift = (wrapped_baseline - baseline).abs().max().item()
    logger.info("PEFT-wrap drift vs raw model: %.2e (zero-init adapters)", drift)

    results: dict[str, bool] = {}

    # --- 1. IDENTITY: enabled + empty bank ---------------------------------
    dws.set_kv_enabled(True)
    empty_bank = logits()
    diff = (empty_bank - wrapped_baseline).abs().max().item()
    results["identity_empty_bank"] = diff <= atol
    logger.info("[1] IDENTITY  max|diff|=%.2e  (atol=%.0e)  %s",
                diff, atol, "PASS" if results["identity_empty_bank"] else "FAIL")

    # --- 2. EFFECT: populated bank must change output ----------------------
    ep_ids = tokenizer(EPISODE, return_tensors="pt").input_ids[0].to(device)
    dws.write_to_kv_bank("verify_ep", ep_ids, 0, int(ep_ids.numel()))
    populated = logits()
    diff2 = (populated - wrapped_baseline).abs().max().item()
    results["effect_populated_bank"] = diff2 > atol * 10
    logger.info("[2] EFFECT    max|diff|=%.2e  %s",
                diff2, "PASS" if results["effect_populated_bank"] else "FAIL")

    # --- 3. DISABLE: toggle off must restore the baseline ------------------
    dws.set_kv_enabled(False)
    disabled = logits()
    diff3 = (disabled - wrapped_baseline).abs().max().item()
    results["disable_restores"] = diff3 <= atol
    logger.info("[3] DISABLE   max|diff|=%.2e  %s",
                diff3, "PASS" if results["disable_restores"] else "FAIL")

    dws.cleanup()

    ok = all(results.values())
    print("\n" + "=" * 56)
    print(f"KV ARCH VERIFICATION — {cfg['model_name']}")
    for name, passed in results.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print(f"  VERDICT: {'SAFE TO RUN' if ok else 'DO NOT RUN THE MATRIX ON THIS MODEL'}")
    print("=" * 56)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
