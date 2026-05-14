"""
Experiment 06b: KV-Memory W_fast Diagnostic (Sleep Pipeline Disabled)

PURPOSE:
    The KV-memory analogue of experiment 06. Same protocol — tag 200 facts,
    skip the sleep cycle entirely, measure recall — but the W_fast substrate
    is a direct-write KV memory bank instead of LoRA adapters trained by
    gradient descent.

    Hypothesis under test (from KV_INJECTION_DESIGN.md):
        "Replacing LoRA W_fast with direct-write KV memory injection
        produces meaningfully higher single-exposure recall while
        preserving BCP."

    Architecture difference vs experiment 06:
      - 06:  surprise -> tag -> alpha_fast LoRA gradient update (lossy compression
             through low-rank bottleneck, scaled by learning rate, susceptible
             to bfloat16 precision floor)
      - 06b: surprise -> tag -> extract pre-RoPE K and raw V from each adapted
             attention layer, append to bank (one-shot, exact, no gradient)
      - At inference, attention sees stored K/V as a prefix at negative
        positions. See KV_INJECTION_DESIGN.md for the architectural rationale.

PRE-REGISTERED DECISION RULE (Phase A.7):
    KV MC accuracy on Tagged >= 0.50, free-form score >= 0.20, BCP < 1.05
        -> KV memory works as substrate. Proceed to Phase B (sleep cycle
           integration with KV-conditioned replay).
    KV MC ~0.25 with free-form near 0%
        -> KV mechanics are sound (per A.5 verification) but the architecture
           still doesn't translate to recall.  Fall back to Path 1 with
           strongest possible narrative ("we tried both gradient-based and
           direct-write encoding mechanisms; here's why both fail and what's
           needed").
    BCP > 1.05
        -> Surprising — direct-write should not destroy preservation.
           Investigate: extraction wrong layer, mask dtype, RoPE broken.

USAGE (pod, Qwen2.5-7B):
    python experiments/scripts/06b_kv_diagnostic.py \
        --config experiments/configs/qwen7b.yaml \
        --facts-file experiments/data/facts_200.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from sleep.config import (
    PRPConfig,
    SleepConfig,
    TaggingConfig,
    WeightsConfig,
)
from sleep.evaluation.calibration import (
    compute_stratified_calibration,
    format_calibration_table,
)
from sleep.evaluation.preservation import compute_bcp, evaluate_perplexity
from sleep.evaluation.recall_formats import (
    cloze_recall,
    free_form_recall,
    group_facts_by_template,
    multiple_choice_recall,
)
from sleep.sleep_engine.quality import compute_baseline_surprise
from sleep.tagging import TaggingLayer
from sleep.utils.logging import get_logger, setup_logging
from sleep.weights import DualWeightSystem

# NOTE: SleepEngine, PRPSystem, run_cycle and prp.update / prp.allocate are
# intentionally NOT imported. This experiment runs only the wake half of the
# pipeline (tagging + W_fast updates) with no PRP allocation and no sleep cycle.

setup_logging()
logger = get_logger("experiment.06b")


# ============================================================================
# Group labels (binary: did W_fast / tagging touch this fact?)
# ============================================================================

GROUP_TAGGED = "Tagged"
GROUP_UNTAGGED = "Untagged"

GROUP_ORDER = [GROUP_TAGGED, GROUP_UNTAGGED]


# ============================================================================
# Control text for base capability check (same as scripts 03 / 04)
# ============================================================================

CONTROL_TEXTS = [
    "The capital of France is Paris. It is known for the Eiffel Tower and the Louvre museum.",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    "The Python programming language was created by Guido van Rossum and released in 1991.",
]


# ============================================================================
# Config loading (mirrors script 04)
# ============================================================================

def load_config(config_path):
    """Load config YAML and return all subsystem configs.

    If no config given, return PoC defaults tuned for GPT-2 on CPU.
    If config given (e.g. qwen7b.yaml), use formalization defaults from yaml.

    NOTE: PRPConfig and SleepConfig are still constructed even though this
    experiment does not exercise them — they are recorded in the output JSON
    for provenance, so a downstream sweep harness can faithfully reconstruct
    "what config the diagnostic was run under".
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


def _classify_group(fact_id: str, tags_by_source: dict) -> str:
    """Classify a fact as Tagged (>=1 tag) or Untagged (no tags).

    Binary classification — no PRP / sleep state is consulted, since this
    experiment does not run those phases.
    """
    return GROUP_TAGGED if tags_by_source.get(fact_id) else GROUP_UNTAGGED


def _print_dra_table(
    dra_by_group: dict,
    counts_by_group: dict,
    formats: list,
) -> None:
    """Print the master 2 groups x N formats table."""
    col0 = 17
    colw = 16

    print("=" * 78)
    print("STRATIFIED RECALL -- 2 groups (Tagged / Untagged) x %d formats" % len(formats))
    print("=" * 78)

    header = "Format".ljust(col0)
    for g in GROUP_ORDER:
        header += g.ljust(colw)
    print(header)
    print("-" * (col0 + colw * len(GROUP_ORDER)))

    for fmt in formats:
        row = fmt.ljust(col0)
        for g in GROUP_ORDER:
            n = counts_by_group.get(g, 0)
            v = dra_by_group.get(fmt, {}).get(g)
            if v is None:
                row += f"-       (n={n})".ljust(colw)
            else:
                row += f"{v:.2f} (n={n})".ljust(colw)
        print(row)
    print()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "W_fast-only diagnostic: run wake phase (tagging + W_fast) only, "
            "skip sleep cycle entirely, then probe recall."
        )
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config YAML (required)")
    parser.add_argument("--facts-file", type=str,
                        default="experiments/data/facts_200.json",
                        help="Path to facts JSON")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to results JSON (default: auto-named)")
    parser.add_argument("--max-facts", type=int, default=None,
                        help="Limit number of facts loaded (debugging)")
    parser.add_argument("--skip-mc", action="store_true",
                        help="Skip multiple-choice recall format")
    parser.add_argument("--skip-cloze", action="store_true",
                        help="Skip cloze recall format")
    parser.add_argument("--skip-freeform", action="store_true",
                        help="Skip free-form recall format")
    parser.add_argument("--override-alpha-fast", type=float, default=None,
                        help="Override weights.alpha_fast (W_fast learning rate). "
                             "Default 1e-4 may be below bfloat16 precision floor — "
                             "see Formalization Appendix A.2. Try 1e-3.")
    parser.add_argument("--kv-top-k", type=int, default=16,
                        help="Top-k retrieval gate for KV memory injection. "
                             "Per query position per layer, only the top-k "
                             "memory positions by Q·K_mem score are made "
                             "visible. 0 = no gating (all memories visible — "
                             "destroys BCP at scale). Recommended: 8-16. "
                             "Default 16.")
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
        suffix = ""
        if args.kv_top_k is not None:
            suffix += f"_k{args.kv_top_k}"
        if args.override_alpha_fast is not None:
            af_str = f"{args.override_alpha_fast:g}".replace(".", "p")
            suffix += f"_af{af_str}"
        args.output = os.path.join(
            results_dir, f"kv_diagnostic_{timestamp}{suffix}.json"
        )
    logger.info("Will write results to %s", args.output)

    print("=" * 78)
    print("EXPERIMENT 06b: KV-Memory W_fast Diagnostic (Sleep Disabled)")
    print("=" * 78)

    cfg = load_config(args.config)

    # ---- Apply alpha_fast override -------------------------------------------
    original_alpha_fast = cfg["weights"].alpha_fast
    if args.override_alpha_fast is not None:
        cfg["weights"].alpha_fast = args.override_alpha_fast
        logger.info("=" * 78)
        logger.info("HYPERPARAMETER OVERRIDE APPLIED")
        logger.info("=" * 78)
        logger.info("  weights.alpha_fast:  %s -> %s",
                    original_alpha_fast, args.override_alpha_fast)
        logger.info("=" * 78)

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(cfg["dtype"], torch.float32)

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
    # Force eager attention so our 4D additive mask with memory-visibility
    # gating is honored exactly. SDPA / flash-attn paths may handle 4D masks
    # differently and silently ignore the memory-position -inf values.
    # Eager is slower per token but at our scale (200 facts × ~30 tokens each)
    # the difference is negligible, and correctness > speed for diagnostics.
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], torch_dtype=dtype, attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    logger.info("Attention implementation: %s", model.config._attn_implementation)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = model.to(cfg["device"])

    n_params_b = sum(p.numel() for p in model.parameters()) / 1e9
    logger.info("Model loaded (%.3fB parameters)", n_params_b)

    tagging_cfg = cfg["tagging"]
    prp_cfg = cfg["prp"]
    weights_cfg = cfg["weights"]
    sleep_cfg = cfg["sleep"]

    # KEY DIFFERENCE FROM 06: enable KV memory mode for W_fast.
    # The LoRA W_fast adapter is still constructed (Strategy Y) but unused;
    # we exercise the write_to_kv_bank path instead of update_fast_weights.
    dws = DualWeightSystem(
        model, weights_cfg,
        use_kv_memory_for_fast=True,
        kv_max_total_tokens=10_000,
        kv_top_k=args.kv_top_k,
    )
    logger.info("KV top-k retrieval gate: %s", args.kv_top_k or "OFF (all visible)")
    peft_model = dws.model

    tagging = TaggingLayer(peft_model, tagging_cfg, model_params_billions=n_params_b)

    # ---- Baseline surprise (computed under target_inference, same as 04) ----
    # We still need mu_surprise for compute_baseline_surprise even though we
    # don't run the sleep cycle, because tagging.process_input may consume it
    # indirectly. For the W_fast-only path it is essentially provenance.
    logger.info("Computing baseline surprise on control texts...")
    dws.set_mode("target_inference")
    control_ids = [
        tokenizer.encode(t, return_tensors="pt").squeeze(0).to(cfg["device"])
        for t in CONTROL_TEXTS
    ]
    mu_surprise_real = compute_baseline_surprise(
        peft_model, control_ids, device=cfg["device"],
    )
    if cfg["use_real_mu_surprise"]:
        mu_surprise = mu_surprise_real
        logger.info(
            "Baseline mu_surprise: %.3f nats (using computed value)",
            mu_surprise_real,
        )
    else:
        mu_surprise = 0.0
        logger.info(
            "Baseline mu_surprise: %.3f nats (DISABLED for PoC -- set to 0)",
            mu_surprise_real,
        )

    # ---- BCP "before" PPL (must be measured BEFORE any W_fast update) -------
    # We measure under wake_inference mode because that is the mode under
    # which evaluation will happen — i.e. with W_fast active. With a
    # freshly-initialised W_fast (zero delta), this should match the
    # base-model PPL exactly; we capture it here as a sanity baseline.
    dws.set_mode("wake_inference")
    peft_model.eval()
    baseline_ppl = evaluate_perplexity(peft_model, control_ids, device=cfg["device"])
    logger.info("Baseline PPL on controls (wake_inference, pre-update): %.2f", baseline_ppl)

    # ====================================================================
    # WAKE PHASE
    # ====================================================================
    print("\n" + "=" * 78)
    print("WAKE PHASE: Processing %d facts (NO PRP, NO sleep)" % len(facts))
    print("=" * 78)

    dws.set_mode("wake_inference")
    original_tokens_map: dict = {}
    tags_by_source: dict = defaultdict(list)
    n_wfast_updates = 0

    for fact in facts:
        token_ids = tokenizer.encode(
            fact["text"], return_tensors="pt",
        ).squeeze(0).to(cfg["device"])
        original_tokens_map[fact["id"]] = token_ids

        new_tags = tagging.process_input(token_ids, source_id=fact["id"])
        tags_by_source[fact["id"]].extend(new_tags)

        for tag_idx, tag in enumerate(new_tags):
            span_start, span_end, source_id = tag.ctx
            if tag.e0 > 1.5:
                # KV-memory write: extract pre-RoPE K/V from each adapted
                # attention layer and append to the bank under a unique tag id.
                # No gradient, no learning rate — direct write.
                tag_id = f"{fact['id']}__t{tag_idx}"
                try:
                    dws.write_to_kv_bank(
                        tag_id=tag_id,
                        token_ids=token_ids,
                        span_start=span_start,
                        span_end=span_end,
                        device=cfg["device"],
                    )
                    n_wfast_updates += 1
                except RuntimeError as exc:
                    # Bank at capacity — log and skip. PRP would normally
                    # evict here; this diagnostic doesn't run PRP.
                    logger.warning(
                        "KV bank full at fact=%s tag_idx=%d: %s",
                        fact["id"], tag_idx, exc,
                    )

    all_tags = list(tagging.active_tags)
    n_tagged_facts = sum(1 for f in facts if tags_by_source.get(f["id"]))

    print(f"  Total active tags:        {len(all_tags)}")
    print(f"  Facts with >=1 tag:       {n_tagged_facts}/{len(facts)}")
    print(f"  KV bank writes performed: {n_wfast_updates}")
    print(f"  KV bank: {dws.kv_bank.n_tags} tags / {dws.kv_bank.n_total_tokens} tokens")
    print("  PRP allocation:           SKIPPED (diagnostic)")
    print("  Sleep cycle:              SKIPPED (diagnostic)")

    # ====================================================================
    # GROUP ASSIGNMENTS (binary: Tagged / Untagged)
    # ====================================================================
    print("\n" + "=" * 78)
    print("CLASSIFY: Sort facts into 2 groups (Tagged / Untagged)")
    print("=" * 78)

    group_assignments: dict = {}
    for fact in facts:
        group_assignments[fact["id"]] = _classify_group(fact["id"], tags_by_source)

    counts_by_group = Counter(group_assignments.values())
    for g in GROUP_ORDER:
        print(f"  {g.ljust(15)} {counts_by_group.get(g, 0)}")
    total = sum(counts_by_group.values())
    print(f"  {'TOTAL'.ljust(15)} {total} (expected {len(facts)})")
    if total != len(facts):
        logger.warning("Group total %d != fact count %d", total, len(facts))

    # ====================================================================
    # RECALL EVALUATIONS
    # ====================================================================
    print("\n" + "=" * 78)
    print("EVALUATION: Running recall formats with W_fast active")
    print("=" * 78)

    # IMPORTANT (deviation from the literal spec — see header docstring of
    # script 06's commit message):
    #
    # The pre-spec asked for `target_inference` mode "so W_fast is active in
    # the forward pass". In sleep/weights/composition.py, `target_inference`
    # is W_slow + W_cons ONLY (W_fast IS NOT active). The mode that *does*
    # include W_fast is `wake_inference` (W_slow + W_cons + W_fast).
    #
    # For a W_fast-only diagnostic, evaluation must be performed with W_fast
    # in the forward pass — otherwise we are measuring W_slow alone, which
    # makes the experiment vacuous. We therefore use `wake_inference` here,
    # which matches the spec's *intent* (W_fast active) rather than its
    # literal mode name.
    dws.set_mode("wake_inference")
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
    # STRATIFIED ANALYSIS
    # ====================================================================
    print("\n" + "=" * 78)
    print("STRATIFIED ANALYSIS")
    print("=" * 78)

    dra_by_group: dict = {}

    def _stratify(per_fact: list, scorer) -> dict:
        bucket: dict = defaultdict(list)
        for r in per_fact:
            fid = r.get("fact_id")
            g = group_assignments.get(fid)
            if g is None:
                continue
            bucket[g].append(scorer(r))
        return {
            g: (sum(vals) / len(vals)) if vals else None
            for g, vals in bucket.items()
        }

    if mc_result is not None:
        dra_by_group["Multiple choice"] = _stratify(
            mc_result["per_fact"], lambda r: 1.0 if r["is_correct"] else 0.0,
        )
    if cloze_result is not None:
        dra_by_group["Cloze"] = _stratify(
            cloze_result["per_fact"], lambda r: 1.0 if r["is_correct"] else 0.0,
        )
    if free_result is not None:
        dra_by_group["Free-form"] = _stratify(
            free_result["per_fact"], lambda r: float(r["score"]),
        )

    print()
    _print_dra_table(dra_by_group, counts_by_group, formats_run)

    # ---- Calibration table (MC only) ---------------------------------------
    stratified_calibration: dict = {}
    if mc_result is not None:
        stratified_calibration = compute_stratified_calibration(
            mc_result["per_fact"], group_assignments,
        )
        print("=" * 78)
        print("CALIBRATION (multiple-choice only, by Tagged/Untagged)")
        print("=" * 78)
        print(format_calibration_table(stratified_calibration))
        print()

    # ====================================================================
    # BCP MEASUREMENT (post-W_fast PPL on controls, W_fast active)
    # ====================================================================
    print("=" * 78)
    print("EVALUATION: Base capability preservation (W_fast-only)")
    print("=" * 78)

    # Measured under wake_inference so W_fast is active — this isolates the
    # BCP impact of W_fast updates alone (no W_cons updates have happened
    # because we never ran the sleep cycle).
    dws.set_mode("wake_inference")
    peft_model.eval()
    post_ppl = evaluate_perplexity(peft_model, control_ids, device=cfg["device"])
    bcp = compute_bcp(post_ppl, baseline_ppl)
    print(f"  Baseline PPL (pre-W_fast):  {baseline_ppl:.2f}")
    print(f"  Post-W_fast PPL:            {post_ppl:.2f}")
    print(f"  BCP ratio:                  {bcp:.4f}  ({'OK' if bcp < 1.05 else 'DEGRADED'})")

    # ====================================================================
    # DECISION RULE READOUT
    # ====================================================================
    print("\n" + "=" * 78)
    print("DECISION RULE READOUT")
    print("=" * 78)

    overall_mc_acc = mc_result["accuracy"] if mc_result is not None else None
    overall_free_score = free_result["mean_score"] if free_result is not None else None

    def _fmt(v):
        return f"{v:.4f}" if v is not None else "n/a"

    # Use Tagged-group MC for the decision rule (matches what the design doc
    # promised). Fall back to overall if Tagged isn't available.
    tagged_mc = dra_by_group.get("Multiple choice", {}).get(GROUP_TAGGED) or overall_mc_acc
    tagged_free = dra_by_group.get("Free-form", {}).get(GROUP_TAGGED) or overall_free_score

    print(f"  KV MC accuracy (Tagged):      {_fmt(tagged_mc)}")
    print(f"  KV MC accuracy (overall):     {_fmt(overall_mc_acc)}")
    print(f"  KV free-form score (overall): {_fmt(overall_free_score)}")
    print(f"  BCP:                          {bcp:.4f}")
    print()

    if bcp >= 1.05:
        verdict = (
            "BCP >= 1.05  -> Surprising! Direct-write should not destroy BCP. "
            "Investigate: extraction layer-norm correctness, mask dtype, "
            "RoPE handling, injector install state."
        )
    elif (
        tagged_mc is not None
        and overall_free_score is not None
        and tagged_mc >= 0.50
        and overall_free_score >= 0.20
    ):
        verdict = (
            "Tagged MC >= 0.50 AND free-form >= 0.20 with BCP < 1.05  "
            "-> KV memory works as substrate. Proceed to Phase B "
            "(sleep cycle integration with KV-conditioned replay)."
        )
    elif (
        tagged_mc is not None
        and 0.22 <= tagged_mc <= 0.30
        and overall_free_score is not None
        and overall_free_score < 0.05
    ):
        verdict = (
            "Tagged MC ~0.25 AND free-form near 0%  -> Both substrates "
            "(LoRA W_fast and KV memory) fail to produce useful single-exposure "
            "encoding. Fall back to Path 1 with two-substrate narrative."
        )
    else:
        verdict = (
            "Result is in the gray zone — does not match any pre-registered "
            "rule arm. Manual interpretation required."
        )
    print(f"  Verdict: {verdict}")

    # ====================================================================
    # SAVE RESULTS JSON
    # ====================================================================
    config_record = {
        "model_name": cfg["model_name"],
        "device": cfg["device"],
        "dtype": cfg["dtype"],
        "tagging": _to_jsonable(tagging_cfg),
        "prp": _to_jsonable(prp_cfg),
        "weights": _to_jsonable(weights_cfg),
        "sleep": _to_jsonable(sleep_cfg),
        "use_real_mu_surprise": cfg["use_real_mu_surprise"],
        "mu_surprise_used": mu_surprise,
        "mu_surprise_computed": mu_surprise_real,
    }

    payload = {
        "experiment": "06b_kv_diagnostic",
        "timestamp": timestamp,
        "args": {
            "config": args.config,
            "facts_file": facts_path,
            "max_facts": args.max_facts,
            "skip_mc": args.skip_mc,
            "skip_cloze": args.skip_cloze,
            "skip_freeform": args.skip_freeform,
            "override_alpha_fast": args.override_alpha_fast,
            "kv_top_k": args.kv_top_k,
        },
        "overrides": {
            "alpha_fast": args.override_alpha_fast,
            "alpha_fast_original": original_alpha_fast,
            "kv_top_k": args.kv_top_k,
        },
        "config": config_record,
        "n_facts": len(facts),
        "wake_summary": {
            "n_active_tags": len(all_tags),
            "n_tagged_facts": n_tagged_facts,
            "n_wfast_updates": n_wfast_updates,
            "prp_allocation": "SKIPPED",
            "sleep_cycle": "SKIPPED",
        },
        "group_assignments": group_assignments,
        "group_counts": dict(counts_by_group),
        "stratified_dra": dra_by_group,
        "stratified_calibration": _to_jsonable(stratified_calibration),
        "recall_results": {
            "multiple_choice": _to_jsonable(mc_result) if mc_result else None,
            "cloze": _to_jsonable(cloze_result) if cloze_result else None,
            "free_form": _to_jsonable(free_result) if free_result else None,
        },
        "bcp": {
            "baseline_ppl": baseline_ppl,
            "post_ppl": post_ppl,
            "bcp": bcp,
        },
        "decision_readout": {
            "overall_mc_accuracy": overall_mc_acc,
            "overall_free_form_mean_score": overall_free_score,
            "bcp": bcp,
            "verdict": verdict,
        },
        "evaluation_mode_note": (
            "All recall and post-update PPL measured under DualWeightSystem "
            "mode 'wake_inference' (W_slow + W_cons + W_fast). The literal "
            "spec said 'target_inference' but that mode excludes W_fast — "
            "see sleep/weights/composition.py. Using wake_inference matches "
            "the experiment's intent (isolate W_fast-only encoding)."
        ),
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

    tagged_dra_mc = dra_by_group.get("Multiple choice", {}).get(GROUP_TAGGED)
    untagged_dra_mc = dra_by_group.get("Multiple choice", {}).get(GROUP_UNTAGGED)

    checks = [
        ("Tags were created", len(all_tags) > 0, f"{len(all_tags)} tags"),
        (
            "At least one W_fast update occurred",
            n_wfast_updates > 0,
            f"{n_wfast_updates} updates",
        ),
        (
            "Group totals add to fact count",
            total == len(facts),
            f"{total}/{len(facts)}",
        ),
        (
            "Tagged group is non-empty",
            counts_by_group.get(GROUP_TAGGED, 0) > 0,
            f"n={counts_by_group.get(GROUP_TAGGED, 0)}",
        ),
        ("BCP < 1.05 (no degradation)", bcp < 1.05, f"BCP={bcp:.4f}"),
        (
            "MC: Tagged >= Untagged (positive W_fast signal)",
            (
                tagged_dra_mc is not None
                and untagged_dra_mc is not None
                and tagged_dra_mc >= untagged_dra_mc
            ),
            (
                f"tagged={tagged_dra_mc:.3f}, untagged={untagged_dra_mc:.3f}"
                if (tagged_dra_mc is not None and untagged_dra_mc is not None)
                else "n/a"
            ),
        ),
    ]

    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name} {detail}")

    n_passed = sum(1 for _, p, _ in checks if p)
    print(f"\n  {n_passed}/{len(checks)} sanity checks passed")

    print("\n" + "=" * 78)
    print("EXPERIMENT 06b COMPLETE")
    print("=" * 78)
    logger.info("Experiment 06 complete.")


if __name__ == "__main__":
    main()
