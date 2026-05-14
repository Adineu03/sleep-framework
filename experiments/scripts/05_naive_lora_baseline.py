"""
Experiment 05: Naive LoRA Baseline (Control)

PURPOSE:
    Control baseline for the SLEEP machinery. Train a fresh LoRA adapter
    directly on the same 200 facts SLEEP saw — no tagging, no PRP, no
    generative replay, no W_fast / W_cons split — then evaluate recall under
    the same three elicitation formats (multiple choice, cloze, free-form).

    The comparison this enables:

        SLEEP (script 04)  vs  Naive LoRA (this script)
        ---------------------------------------------
        Same model, same facts, same LoRA hyperparameters, **same total
        gradient steps**. The only differences are:
          - SLEEP gates *which* facts get trained on via PRP allocation,
            and trains on *generative replays* of those facts.
          - Naive LoRA trains on the verbatim facts uniformly at random.

    Whatever margin SLEEP shows (or doesn't) over this baseline is
    attributable to the SLEEP machinery, not to "having a LoRA adapter".

MATCHED COMPUTE:
    SLEEP v5 used 100 sleep training steps total. Naive LoRA must use
    EXACTLY the same number of gradient steps. Each step samples
    ``sleep.batch_size`` facts uniformly with replacement from the 200.
    No epochs. No LR scheduler. No EWC, Fisher, or grad clipping.

USAGE (PoC, GPT-2 on CPU):
    python experiments/scripts/05_naive_lora_baseline.py \
        --facts-file experiments/data/facts_200.json --max-facts 20

USAGE (pod, Qwen2.5-7B):
    python experiments/scripts/05_naive_lora_baseline.py \
        --config experiments/configs/qwen7b.yaml \
        --facts-file experiments/data/facts_200.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import yaml
from peft import get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from sleep.config import (
    PRPConfig,
    SleepConfig,
    TaggingConfig,
    WeightsConfig,
)
from sleep.evaluation.calibration import (
    compute_calibration_metrics,
    format_calibration_table,
)
from sleep.evaluation.preservation import compute_bcp, evaluate_perplexity
from sleep.evaluation.recall_formats import (
    cloze_recall,
    free_form_recall,
    group_facts_by_template,
    multiple_choice_recall,
)
from sleep.utils.logging import get_logger, setup_logging
# We reuse the SLEEP project's layer-selection logic so the adapter geometry
# is bit-for-bit identical to SLEEP's. _build_lora_config is module-private
# but lives in the same package, and the alternative — duplicating arch
# detection + layer indexing here — would be drift-prone.
from sleep.weights.lora import _build_lora_config

setup_logging()
logger = get_logger("experiment.05")


# ============================================================================
# Control text for base capability check (same as scripts 03 / 04)
# ============================================================================

CONTROL_TEXTS = [
    "The capital of France is Paris. It is known for the Eiffel Tower and the Louvre museum.",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    "The Python programming language was created by Guido van Rossum and released in 1991.",
]


# Matched-compute target: SLEEP v5 used 100 sleep training steps total.
DEFAULT_N_STEPS = 100


# ============================================================================
# Config loading (mirrors script 04, which mirrors script 03)
# ============================================================================

def load_config(config_path):
    """Load config YAML and return all subsystem configs.

    If no config given, return PoC defaults tuned for GPT-2 on CPU.
    If config given (e.g. qwen7b.yaml), use formalization defaults from yaml.
    """
    if config_path is None:
        return {
            "model_name": "gpt2",
            "device": "cpu",
            "dtype": "float32",
            "tagging": TaggingConfig(c_tag=5000, tau_base=200, kappa=1.0),
            "prp": PRPConfig(c_prp=500, crossref_interval=10, allocation_interval=5),
            "weights": WeightsConfig(
                lora_rank=8, lora_alpha=16, alpha_slow=1e-4, delta_max=0.01,
            ),
            "sleep": SleepConfig(
                batch_size=4, steps_per_memory=10, min_replay_length=32,
                compression_target=2, replay_temperature=0.8, max_generation_attempts=2,
            ),
            "use_real_mu_surprise": False,
        }
    with open(config_path) as f:
        data = yaml.safe_load(f)
    return {
        "model_name": data["model"]["name"],
        "device": data["model"].get("device", "cpu"),
        "dtype": data["model"].get("dtype", "float32"),
        "tagging": TaggingConfig(**data["tagging"]),
        "prp": PRPConfig(**data["prp"]),
        "weights": WeightsConfig(**data["weights"]),
        "sleep": SleepConfig(**data["sleep"]),
        "use_real_mu_surprise": data.get("experiment", {}).get("use_real_mu_surprise", False),
    }


# ============================================================================
# Helpers
# ============================================================================

def _to_jsonable(obj):
    """Recursively convert tensors/numpy/etc. to JSON-serialisable forms."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (set, frozenset)):
        return [_to_jsonable(v) for v in obj]
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _to_jsonable(getattr(obj, k)) for k in obj.__dataclass_fields__}
    return obj


def _build_batch(
    facts: list,
    indices: list,
    tokenizer,
    device: str,
):
    """Tokenize and pad a batch of fact texts.

    Returns ``(input_ids, attention_mask, labels)`` where ``labels`` masks
    pad positions to ``-100`` so they don't contribute to the LM loss.
    """
    texts = [facts[i]["text"] for i in indices]
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    return input_ids, attention_mask, labels


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Naive LoRA control baseline for SLEEP."
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML (default: GPT-2 PoC on CPU)")
    parser.add_argument("--facts-file", type=str,
                        default="experiments/data/facts_200.json",
                        help="Path to facts JSON")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to results JSON (default: auto-named)")
    parser.add_argument("--max-facts", type=int, default=None,
                        help="Limit number of facts loaded (debugging)")
    parser.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS,
                        help=f"Number of gradient steps (default: {DEFAULT_N_STEPS}, "
                             "matching SLEEP v5)")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for batch sampling reproducibility")
    parser.add_argument("--skip-mc", action="store_true",
                        help="Skip multiple-choice recall format")
    parser.add_argument("--skip-cloze", action="store_true",
                        help="Skip cloze recall format")
    parser.add_argument("--skip-freeform", action="store_true",
                        help="Skip free-form recall format")
    args = parser.parse_args()

    # ---- Load facts ---------------------------------------------------------
    if not os.path.isabs(args.facts_file):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        facts_path = os.path.join(repo_root, args.facts_file)
    else:
        facts_path = args.facts_file
    with open(facts_path) as f:
        facts = json.load(f)
    if args.max_facts:
        facts = facts[: args.max_facts]
    logger.info("Loaded %d facts from %s", len(facts), facts_path)

    timestamp = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.output is None:
        results_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../results")
        )
        os.makedirs(results_dir, exist_ok=True)
        args.output = os.path.join(
            results_dir, f"naive_lora_baseline_{timestamp}.json"
        )
    logger.info("Will write results to %s", args.output)

    print("=" * 78)
    print("EXPERIMENT 05: Naive LoRA Baseline (Control)")
    print("=" * 78)

    cfg = load_config(args.config)
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(cfg["dtype"], torch.float32)

    weights_cfg = cfg["weights"]
    sleep_cfg = cfg["sleep"]

    # ====================================================================
    # SETUP
    # ====================================================================
    print("\n" + "=" * 78)
    print("SETUP")
    print("=" * 78)
    logger.info(
        "Loading model: %s (device=%s, dtype=%s)",
        cfg["model_name"], cfg["device"], cfg["dtype"],
    )
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], torch_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = model.to(cfg["device"])

    n_params_b = sum(p.numel() for p in model.parameters()) / 1e9
    logger.info("Model loaded (%.3fB parameters)", n_params_b)

    # ---- Init fresh LoRA adapter ------------------------------------------
    # Use the SLEEP project's existing helper so target modules, rank, alpha,
    # and layers_to_transform are byte-for-byte identical to what SLEEP uses.
    lora_config = _build_lora_config(model, weights_cfg)
    peft_model = get_peft_model(model, lora_config, adapter_name="naive")
    peft_model = peft_model.to(cfg["device"])

    n_trainable = sum(
        p.numel() for p in peft_model.parameters() if p.requires_grad
    )
    n_total = sum(p.numel() for p in peft_model.parameters())
    logger.info(
        "LoRA adapter ready: rank=%d, alpha=%d, adapted_fraction=%.3f, "
        "target_modules=%s, trainable=%d/%d (%.4f%%)",
        weights_cfg.lora_rank,
        weights_cfg.lora_alpha,
        weights_cfg.adapted_fraction,
        weights_cfg.adapted_matrices,
        n_trainable,
        n_total,
        100.0 * n_trainable / max(n_total, 1),
    )
    print(f"  LoRA rank:        {weights_cfg.lora_rank}")
    print(f"  LoRA alpha:       {weights_cfg.lora_alpha}")
    print(f"  Adapted fraction: {weights_cfg.adapted_fraction:.3f}")
    print(f"  Target matrices:  {weights_cfg.adapted_matrices}")
    print(f"  Trainable params: {n_trainable:,}")

    # ---- Baseline PPL on controls -----------------------------------------
    control_ids = [
        tokenizer.encode(t, return_tensors="pt").squeeze(0).to(cfg["device"])
        for t in CONTROL_TEXTS
    ]
    peft_model.eval()
    baseline_ppl = evaluate_perplexity(peft_model, control_ids, device=cfg["device"])
    logger.info("Baseline PPL on controls: %.2f", baseline_ppl)
    print(f"  Baseline PPL:     {baseline_ppl:.2f}")

    # ====================================================================
    # NAIVE LORA TRAINING (matched compute: exactly --n-steps gradient steps)
    # ====================================================================
    print("\n" + "=" * 78)
    print("TRAINING: Naive LoRA (matched compute)")
    print("=" * 78)
    print(f"  Total gradient steps: {args.n_steps}")
    print(f"  Batch size:           {sleep_cfg.batch_size}")
    print(f"  Learning rate:        {weights_cfg.alpha_slow}")
    print(f"  Weight decay:         {sleep_cfg.sleep_weight_decay}")
    print(f"  Optimizer:            AdamW (no scheduler)")
    print(f"  Sampling:             uniform with replacement over {len(facts)} facts")
    print()

    # Collect *only* the LoRA-adapter parameters as trainables. The base
    # model is frozen by get_peft_model already, so a simple filter on
    # requires_grad is sufficient.
    trainable_params = [p for p in peft_model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError(
            "No trainable parameters found in PEFT model. Adapter setup failed."
        )

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=weights_cfg.alpha_slow,
        weight_decay=sleep_cfg.sleep_weight_decay,
    )

    rng = random.Random(args.seed)
    n_facts = len(facts)
    batch_size = sleep_cfg.batch_size

    peft_model.train()
    loss_trace: list = []
    nan_seen = False

    for step in range(1, args.n_steps + 1):
        # Sample batch_size fact indices uniformly with replacement.
        indices = [rng.randrange(n_facts) for _ in range(batch_size)]
        input_ids, attention_mask, labels = _build_batch(
            facts, indices, tokenizer, cfg["device"],
        )

        outputs = peft_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss

        loss_value = float(loss.item())
        loss_trace.append({"step": step, "loss": loss_value})

        if math.isnan(loss_value) or math.isinf(loss_value):
            nan_seen = True
            logger.error("Non-finite loss at step %d: %s", step, loss_value)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % 10 == 0 or step == args.n_steps:
            print(f"    step {step:>4d}/{args.n_steps}  loss={loss_value:.4f}")

    # Quick sanity figures from the trace
    first_loss = loss_trace[0]["loss"] if loss_trace else float("nan")
    last_loss = loss_trace[-1]["loss"] if loss_trace else float("nan")
    mean_loss = (
        sum(r["loss"] for r in loss_trace) / len(loss_trace)
        if loss_trace else float("nan")
    )
    # Mean of the last 10% of steps for a more stable "final" reading.
    tail_n = max(1, len(loss_trace) // 10)
    tail_mean = (
        sum(r["loss"] for r in loss_trace[-tail_n:]) / tail_n
        if loss_trace else float("nan")
    )

    print()
    print(f"  First-step loss:       {first_loss:.4f}")
    print(f"  Final-step loss:       {last_loss:.4f}")
    print(f"  Mean loss:             {mean_loss:.4f}")
    print(f"  Mean loss (last 10%):  {tail_mean:.4f}")

    # ====================================================================
    # RECALL EVALUATIONS
    # ====================================================================
    print("\n" + "=" * 78)
    print("EVALUATION: Running recall formats")
    print("=" * 78)

    peft_model.eval()
    all_facts_by_template = group_facts_by_template(facts)

    formats_run: list = []
    mc_result = None
    cloze_result = None
    free_result = None

    if not args.skip_mc:
        print("\n[1] Multiple-choice recall ...")
        mc_result = multiple_choice_recall(
            peft_model, tokenizer, facts,
            all_facts_by_template=all_facts_by_template,
            device=cfg["device"],
        )
        print(
            f"    accuracy={mc_result['accuracy']:.4f}  "
            f"mean_correct_prob={mc_result['mean_correct_prob']:.4f}  "
            f"n={mc_result['n_facts']}"
        )
        formats_run.append("Multiple choice")

    if not args.skip_cloze:
        print("\n[2] Cloze recall ...")
        cloze_result = cloze_recall(
            peft_model, tokenizer, facts, device=cfg["device"],
        )
        print(
            f"    accuracy={cloze_result['accuracy']:.4f}  "
            f"n={cloze_result['n_facts']}"
        )
        formats_run.append("Cloze")

    if not args.skip_freeform:
        print("\n[3] Free-form recall ...")
        free_result = free_form_recall(
            peft_model, tokenizer, facts, device=cfg["device"],
        )
        print(
            f"    mean_score={free_result['mean_score']:.4f}  "
            f"accuracy={free_result['accuracy']:.4f}  "
            f"n={free_result['n_facts']}"
        )
        formats_run.append("Free-form")

    # ====================================================================
    # AGGREGATE REPORT (one row per format)
    # ====================================================================
    print("\n" + "=" * 78)
    print("AGGREGATE RECALL -- Naive LoRA, %d formats" % len(formats_run))
    print("=" * 78)

    col_fmt = 18
    col_acc = 12
    col_extra = 22
    col_n = 8
    print(
        "Format".ljust(col_fmt)
        + "Accuracy".ljust(col_acc)
        + "Other".ljust(col_extra)
        + "N".ljust(col_n)
    )
    print("-" * (col_fmt + col_acc + col_extra + col_n))

    if mc_result is not None:
        print(
            "Multiple choice".ljust(col_fmt)
            + f"{mc_result['accuracy']:.4f}".ljust(col_acc)
            + f"meanCorrP={mc_result['mean_correct_prob']:.4f}".ljust(col_extra)
            + str(mc_result["n_facts"]).ljust(col_n)
        )
    if cloze_result is not None:
        print(
            "Cloze".ljust(col_fmt)
            + f"{cloze_result['accuracy']:.4f}".ljust(col_acc)
            + "".ljust(col_extra)
            + str(cloze_result["n_facts"]).ljust(col_n)
        )
    if free_result is not None:
        print(
            "Free-form".ljust(col_fmt)
            + f"{free_result['accuracy']:.4f}".ljust(col_acc)
            + f"mean_score={free_result['mean_score']:.4f}".ljust(col_extra)
            + str(free_result["n_facts"]).ljust(col_n)
        )
    print()

    # ---- Calibration (MC only) --------------------------------------------
    calibration_metrics: dict = {}
    if mc_result is not None and mc_result["per_fact"]:
        calibration_metrics = compute_calibration_metrics(mc_result["per_fact"])
        print("=" * 78)
        print("CALIBRATION (multiple-choice only)")
        print("=" * 78)
        # compute_calibration_metrics returns a single-set dict; format_calibration_table
        # expects a per-group mapping. Wrap to match.
        print(format_calibration_table({"NaiveLoRA": calibration_metrics}))
        print()

    # ====================================================================
    # BCP MEASUREMENT (post-training PPL on controls)
    # ====================================================================
    print("=" * 78)
    print("EVALUATION: Base capability preservation")
    print("=" * 78)

    peft_model.eval()
    post_ppl = evaluate_perplexity(peft_model, control_ids, device=cfg["device"])
    bcp = compute_bcp(post_ppl, baseline_ppl)
    print(f"  Baseline PPL:        {baseline_ppl:.2f}")
    print(f"  Post-training PPL:   {post_ppl:.2f}")
    print(f"  BCP ratio:           {bcp:.4f}  ({'OK' if bcp < 1.05 else 'DEGRADED'})")

    # ====================================================================
    # SAVE RESULTS JSON
    # ====================================================================
    config_record = {
        "model_name": cfg["model_name"],
        "device": cfg["device"],
        "dtype": cfg["dtype"],
        "weights": _to_jsonable(weights_cfg),
        "sleep": _to_jsonable(sleep_cfg),
        # Tagging / PRP configs are loaded but unused by this baseline; we
        # still record them so the YAML file the run was launched with is
        # fully reproducible from the results JSON.
        "tagging": _to_jsonable(cfg["tagging"]),
        "prp": _to_jsonable(cfg["prp"]),
    }

    payload = {
        "experiment": "05_naive_lora_baseline",
        "timestamp": timestamp,
        "args": {
            "config": args.config,
            "facts_file": facts_path,
            "max_facts": args.max_facts,
            "n_steps": args.n_steps,
            "seed": args.seed,
            "skip_mc": args.skip_mc,
            "skip_cloze": args.skip_cloze,
            "skip_freeform": args.skip_freeform,
        },
        "config": config_record,
        "n_facts": len(facts),
        "training": {
            "n_steps": args.n_steps,
            "batch_size": sleep_cfg.batch_size,
            "lr": weights_cfg.alpha_slow,
            "weight_decay": sleep_cfg.sleep_weight_decay,
            "optimizer": "adamw",
            "sampling": "uniform_with_replacement",
            "first_loss": first_loss,
            "last_loss": last_loss,
            "mean_loss": mean_loss,
            "tail_mean_loss": tail_mean,
            "loss_trace": loss_trace,
            "n_trainable_params": n_trainable,
        },
        "recall_results": {
            "multiple_choice": _to_jsonable(mc_result) if mc_result else None,
            "cloze": _to_jsonable(cloze_result) if cloze_result else None,
            "free_form": _to_jsonable(free_result) if free_result else None,
        },
        "calibration": _to_jsonable(calibration_metrics),
        "bcp": {
            "baseline_ppl": baseline_ppl,
            "post_ppl": post_ppl,
            "bcp": bcp,
        },
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info("Wrote results to %s", args.output)
    print(f"\n  Results JSON: {args.output}")

    # ====================================================================
    # SANITY CHECKS
    # ====================================================================
    print("\n" + "=" * 78)
    print("SANITY CHECKS")
    print("=" * 78)

    loss_decreased = (
        len(loss_trace) >= 2
        and math.isfinite(first_loss)
        and math.isfinite(tail_mean)
        and tail_mean < first_loss
    )

    checks = [
        (
            "Trainable parameters present",
            n_trainable > 0,
            f"{n_trainable:,} params",
        ),
        (
            f"Exactly {args.n_steps} gradient steps taken (matched compute)",
            len(loss_trace) == args.n_steps,
            f"{len(loss_trace)}/{args.n_steps}",
        ),
        (
            "No NaN/Inf in training loss",
            not nan_seen,
            "all finite" if not nan_seen else "non-finite seen",
        ),
        (
            "Training loss decreased (tail mean < first)",
            loss_decreased,
            f"first={first_loss:.4f} tail={tail_mean:.4f}",
        ),
        (
            "BCP < 1.05 (no degradation)",
            bcp < 1.05,
            f"BCP={bcp:.4f}",
        ),
    ]

    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name} {detail}")

    n_passed = sum(1 for _, p, _ in checks if p)
    print(f"\n  {n_passed}/{len(checks)} sanity checks passed")

    print("\n" + "=" * 78)
    print("EXPERIMENT 05 COMPLETE")
    print("=" * 78)
    logger.info("Experiment 05 complete.")


if __name__ == "__main__":
    main()
