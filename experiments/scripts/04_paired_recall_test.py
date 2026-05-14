"""
Experiment 04: Paired Recall Test (Stratified by SLEEP Outcome)

PURPOSE:
    The gateway experiment for the next phase of SLEEP research.

    Run the full SLEEP pipeline end-to-end on a fact dataset, then classify
    each fact by what happened to it during the cycle (Consolidated, Failed,
    TaggedNoPRP, Untagged) and probe recall under three elicitation formats
    (multiple choice, cloze, free-form). The stratified comparison answers
    three concrete questions:

        - Consolidated vs TaggedNoPRP: does PRP+sleep help?
        - Consolidated vs Failed:      does validation passing predict recall?
        - Failed vs Untagged:          does PRP+sleep+failed-validation hurt?

    See docs/PROGRESS_REPORT_2026-04-30.md, "Sprint 0", for motivation.

    NOTE: Naive-LoRA / RAG / random-replay baselines are NOT included here —
    those belong to script 05.

USAGE (PoC, GPT-2 on CPU):
    python experiments/scripts/04_paired_recall_test.py \
        --facts-file experiments/data/facts_200.json --max-facts 20

USAGE (pod, Qwen2.5-7B):
    python experiments/scripts/04_paired_recall_test.py \
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
from sleep.prp import PRPSystem
from sleep.sleep_engine import SleepEngine
from sleep.sleep_engine.quality import compute_baseline_surprise
from sleep.tagging import TaggingLayer
from sleep.utils.logging import get_logger, setup_logging
from sleep.weights import DualWeightSystem

setup_logging()
logger = get_logger("experiment.04")


# ============================================================================
# Group labels (priority: Consolidated > Failed > TaggedNoPRP > Untagged)
# ============================================================================

GROUP_CONSOLIDATED = "Consolidated"
GROUP_FAILED = "Failed"
GROUP_TAGGED_NO_PRP = "TaggedNoPRP"
GROUP_UNTAGGED = "Untagged"

GROUP_ORDER = [GROUP_CONSOLIDATED, GROUP_FAILED, GROUP_TAGGED_NO_PRP, GROUP_UNTAGGED]


# ============================================================================
# Control text for base capability check (same as script 03)
# ============================================================================

CONTROL_TEXTS = [
    "The capital of France is Paris. It is known for the Eiffel Tower and the Louvre museum.",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    "The Python programming language was created by Guido van Rossum and released in 1991.",
]


# ============================================================================
# Config loading (mirrors script 03)
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


def _classify_group(
    fact_id: str,
    tags_by_source: dict,
    pre_state_by_tag_id: dict,
) -> str:
    """Classify a fact into one of the 4 SLEEP outcome groups.

    Priority: Consolidated > Failed > TaggedNoPRP > Untagged.

    Args:
        fact_id:              The fact's ``id`` (== Tag.ctx[2]).
        tags_by_source:       ``source_id -> list[Tag]``.
        pre_state_by_tag_id:  ``id(tag) -> {"p": int, "fail_count": int}``
                              snapshot taken just before SleepEngine.run_cycle.

    Returns:
        Group label string.
    """
    tags = tags_by_source.get(fact_id, [])
    if not tags:
        return GROUP_UNTAGGED

    has_prp_candidate = False
    has_consolidated = False
    has_failed = False

    for tag in tags:
        pre = pre_state_by_tag_id.get(id(tag))
        if pre is None:
            # Tag created post-cycle (shouldn't happen here, but be safe).
            continue
        if pre["p"] != 1:
            # This tag never received a PRP allocation.
            continue
        has_prp_candidate = True

        # A tag was a candidate iff p=1 going into the cycle.
        # After cleanup_tags:
        #   - passed   -> tag.p still 1, fail_count unchanged
        #   - failed   -> tag.p set to 0, fail_count += 1
        # So compare fail_count delta to detect failure.
        if tag.fail_count > pre["fail_count"]:
            has_failed = True
        else:
            has_consolidated = True

    if has_consolidated:
        return GROUP_CONSOLIDATED
    if has_failed:
        return GROUP_FAILED
    if has_prp_candidate:
        # Edge case: had p=1 but neither passed nor failed (no validation
        # happened, e.g. cycle aborted before phase 5). Bucket conservatively.
        return GROUP_FAILED
    return GROUP_TAGGED_NO_PRP


def _print_dra_table(
    dra_by_group: dict,
    counts_by_group: dict,
    formats: list,
) -> None:
    """Print the master 4 groups x N formats table."""
    col0 = 17
    colw = 14

    print("=" * 78)
    print("STRATIFIED RECALL -- All 4 SLEEP groups x %d formats" % len(formats))
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


def _print_paired_comparisons(
    dra_by_group: dict, counts_by_group: dict, formats: list,
) -> None:
    """Print the three key paired comparisons."""
    print("=" * 78)
    print("PAIRED COMPARISON -- the three key contrasts")
    print("=" * 78)

    pairs = [
        (GROUP_CONSOLIDATED, GROUP_TAGGED_NO_PRP, "does PRP+sleep help?"),
        (GROUP_CONSOLIDATED, GROUP_FAILED, "does validation passing predict recall?"),
        (GROUP_FAILED, GROUP_UNTAGGED, "does failed-validation hurt?"),
    ]

    print("Pair                                Format            A      B      A-B")
    print("-" * 78)
    for a, b, question in pairs:
        label = f"{a} vs {b}"
        for fmt in formats:
            va = dra_by_group.get(fmt, {}).get(a)
            vb = dra_by_group.get(fmt, {}).get(b)
            na = counts_by_group.get(a, 0)
            nb = counts_by_group.get(b, 0)
            if va is None or vb is None:
                row = f"  {label[:34].ljust(34)}{fmt.ljust(18)} -      -      -"
            else:
                diff = va - vb
                row = (
                    f"  {label[:34].ljust(34)}"
                    f"{fmt.ljust(18)}"
                    f"{va:.2f}   {vb:.2f}   {diff:+.2f}"
                )
            print(row)
        print(f"    -> question: {question}  (n_A={na}, n_B={nb})")
    print()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stratified paired recall test for SLEEP."
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
    parser.add_argument("--skip-mc", action="store_true",
                        help="Skip multiple-choice recall format")
    parser.add_argument("--skip-cloze", action="store_true",
                        help="Skip cloze recall format")
    parser.add_argument("--skip-freeform", action="store_true",
                        help="Skip free-form recall format")
    parser.add_argument("--override-delta-max", type=float, default=None,
                        help="Override weights.delta_max from config (e.g. 0.05)")
    parser.add_argument("--override-lambda-ewc", type=float, default=None,
                        help="Override weights.lambda_ewc from config (e.g. 10.0)")
    parser.add_argument("--override-epsilon-learn", type=float, default=None,
                        help="Override sleep.epsilon_learn from config (e.g. 0.05)")
    args = parser.parse_args()

    # ---- Load facts ---------------------------------------------------------
    if not os.path.isabs(args.facts_file):
        # Resolve relative to repo root for convenience.
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

    print("=" * 78)
    print("EXPERIMENT 04: Paired Recall Test (Stratified by SLEEP Outcome)")
    print("=" * 78)

    cfg = load_config(args.config)

    # ---- Apply hyperparameter overrides ------------------------------------
    # WeightsConfig and SleepConfig are mutable dataclasses (no frozen=True),
    # so direct assignment is fine.
    original_delta_max = cfg["weights"].delta_max
    original_lambda_ewc = cfg["weights"].lambda_ewc
    original_epsilon_learn = cfg["sleep"].epsilon_learn

    any_override_applied = (
        args.override_delta_max is not None
        or args.override_lambda_ewc is not None
        or args.override_epsilon_learn is not None
    )
    if args.override_delta_max is not None:
        cfg["weights"].delta_max = args.override_delta_max
    if args.override_lambda_ewc is not None:
        cfg["weights"].lambda_ewc = args.override_lambda_ewc
    if args.override_epsilon_learn is not None:
        cfg["sleep"].epsilon_learn = args.override_epsilon_learn

    if any_override_applied:
        logger.info("=" * 70)
        logger.info("HYPERPARAMETER OVERRIDES APPLIED")
        logger.info("=" * 70)
        if args.override_delta_max is not None:
            logger.info("  weights.delta_max:    %s -> %s", original_delta_max, args.override_delta_max)
        if args.override_lambda_ewc is not None:
            logger.info("  weights.lambda_ewc:   %s -> %s", original_lambda_ewc, args.override_lambda_ewc)
        if args.override_epsilon_learn is not None:
            logger.info("  sleep.epsilon_learn:  %s -> %s", original_epsilon_learn, args.override_epsilon_learn)
        logger.info("=" * 70)

    # ---- Auto-name output file (include override values when provided) -----
    if args.output is None:
        results_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../results")
        )
        os.makedirs(results_dir, exist_ok=True)

        def _fmt_override(v):
            # Compact, filesystem-friendly representation: 0.05 -> "0p05"
            return ("%g" % v).replace(".", "p").replace("-", "m")

        suffix = ""
        if args.override_delta_max is not None:
            suffix += f"_dm{_fmt_override(args.override_delta_max)}"
        if args.override_lambda_ewc is not None:
            suffix += f"_le{_fmt_override(args.override_lambda_ewc)}"
        if args.override_epsilon_learn is not None:
            suffix += f"_el{_fmt_override(args.override_epsilon_learn)}"

        args.output = os.path.join(
            results_dir, f"paired_recall_test_{timestamp}{suffix}.json"
        )
    logger.info("Will write results to %s", args.output)

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
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], torch_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = model.to(cfg["device"])

    n_params_b = sum(p.numel() for p in model.parameters()) / 1e9
    logger.info("Model loaded (%.3fB parameters)", n_params_b)

    tagging_cfg = cfg["tagging"]
    prp_cfg = cfg["prp"]
    weights_cfg = cfg["weights"]
    sleep_cfg = cfg["sleep"]

    dws = DualWeightSystem(model, weights_cfg)
    peft_model = dws.model

    tagging = TaggingLayer(peft_model, tagging_cfg, model_params_billions=n_params_b)
    prp = PRPSystem(
        prp_cfg, budget=int(prp_cfg.c_prp * n_params_b), revision_bonus=0.3,
    )

    # Baseline surprise
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
            "Baseline mu_surprise: %.3f nats (DISABLED for PoC — set to 0)",
            mu_surprise_real,
        )

    baseline_ppl = evaluate_perplexity(peft_model, control_ids, device=cfg["device"])
    logger.info("Baseline PPL on controls: %.2f", baseline_ppl)

    # ====================================================================
    # WAKE PHASE
    # ====================================================================
    print("\n" + "=" * 78)
    print("WAKE PHASE: Processing %d facts" % len(facts))
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

        for tag in new_tags:
            span_start, span_end, _ = tag.ctx
            if tag.e0 > 1.5:
                _ = dws.update_fast_weights(
                    token_ids, span_start, span_end, tag.e0,
                    device=cfg["device"],
                )
                n_wfast_updates += 1

    all_tags = list(tagging.active_tags)
    n_tagged_facts = sum(1 for f in facts if tags_by_source.get(f["id"]))

    print(f"  Total active tags:        {len(all_tags)}")
    print(f"  Facts with >=1 tag:       {n_tagged_facts}/{len(facts)}")
    print(f"  W_fast updates performed: {n_wfast_updates}")

    # PRP allocation
    prp_result = prp.update(all_tags, current_step=tagging.step, force_crossref=True)
    print(f"  PRP allocated:            {prp_result['allocated']}/{prp.budget}")
    print(f"  Threshold:                {prp_result['threshold']:.3f}")
    print(f"  Mean score:               {prp_result['mean_score']:.3f}")

    candidates = prp.get_consolidation_candidates(all_tags)
    print(f"  Consolidation candidates: {len(candidates)}")

    # ---- Snapshot pre-cycle state of EVERY tag (for group classification) ---
    # We need {p, fail_count} per tag id BEFORE run_cycle mutates any.
    pre_state_by_tag_id: dict = {
        id(tag): {"p": int(tag.p), "fail_count": int(tag.fail_count)}
        for tag in all_tags
    }

    if len(candidates) == 0:
        print(
            "\n  WARNING: No PRP candidates! All facts will land in either "
            "TaggedNoPRP or Untagged. Sleep cycle is skipped."
        )

    # ====================================================================
    # SLEEP PHASE
    # ====================================================================
    print("\n" + "=" * 78)
    print("SLEEP PHASE: Running consolidation")
    print("=" * 78)

    sleep_engine = SleepEngine(
        dual_weights=dws,
        tokenizer=tokenizer,
        sleep_config=sleep_cfg,
        weights_config=weights_cfg,
        mu_surprise=mu_surprise,
        device=cfg["device"],
    )

    if len(candidates) > 0:
        sleep_result = sleep_engine.run_cycle(
            candidates=candidates,
            original_tokens_map=original_tokens_map,
            key_projection=tagging.key_projection,
        )
    else:
        sleep_result = {
            "n_candidates": 0,
            "n_replays_generated": 0,
            "n_replays_accepted": 0,
            "n_consolidated": 0,
            "n_failed": 0,
            "n_permanently_removed": 0,
            "training_stats": {},
            "ppl_before": None,
            "ppl_after": None,
            "rolled_back": False,
        }

    print("\n  Sleep cycle results:")
    print(f"    Candidates:        {sleep_result.get('n_candidates', '?')}")
    print(f"    Replays generated: {sleep_result.get('n_replays_generated', '?')}")
    print(f"    Replays accepted:  {sleep_result.get('n_replays_accepted', '?')}")
    print(f"    Consolidated:      {sleep_result.get('n_consolidated', '?')}")
    print(f"    Failed:            {sleep_result.get('n_failed', '?')}")
    print(f"    Rolled back:       {sleep_result.get('rolled_back', '?')}")

    # ====================================================================
    # GROUP ASSIGNMENTS
    # ====================================================================
    print("\n" + "=" * 78)
    print("CLASSIFY: Sort facts into 4 SLEEP groups")
    print("=" * 78)

    group_assignments: dict = {}
    for fact in facts:
        group = _classify_group(
            fact["id"], tags_by_source, pre_state_by_tag_id,
        )
        group_assignments[fact["id"]] = group

    counts_by_group = Counter(group_assignments.values())
    for g in GROUP_ORDER:
        print(f"  {g.ljust(15)} {counts_by_group.get(g, 0)}")
    total = sum(counts_by_group.values())
    print(f"  {'TOTAL'.ljust(15)} {total} (expected {len(facts)})")
    if total != len(facts):
        logger.warning(
            "Group total %d != fact count %d", total, len(facts),
        )

    # ====================================================================
    # RECALL EVALUATIONS
    # ====================================================================
    print("\n" + "=" * 78)
    print("EVALUATION: Running recall formats")
    print("=" * 78)

    # All recall is performed with W_slow + W_cons (target_inference).
    dws.set_mode("target_inference")
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

    # Build per-format per-group DRA dicts.
    # For MC and cloze we use `is_correct`; for free-form we use `score`
    # (DRA is naturally a continuous keyword-coverage fraction, matching
    # the existing free-form metric in sleep.evaluation.recall).
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
    _print_paired_comparisons(dra_by_group, counts_by_group, formats_run)

    # ---- Calibration table (MC only) ---------------------------------------
    stratified_calibration: dict = {}
    if mc_result is not None:
        stratified_calibration = compute_stratified_calibration(
            mc_result["per_fact"], group_assignments,
        )
        print("=" * 78)
        print("CALIBRATION (multiple-choice only)")
        print("=" * 78)
        print(format_calibration_table(stratified_calibration))
        print()

    # ====================================================================
    # BCP MEASUREMENT (post-sleep PPL on controls)
    # ====================================================================
    print("=" * 78)
    print("EVALUATION: Base capability preservation")
    print("=" * 78)

    dws.set_mode("target_inference")
    post_ppl = evaluate_perplexity(peft_model, control_ids, device=cfg["device"])
    bcp = compute_bcp(post_ppl, baseline_ppl)
    print(f"  Baseline PPL:   {baseline_ppl:.2f}")
    print(f"  Post-sleep PPL: {post_ppl:.2f}")
    print(f"  BCP ratio:      {bcp:.4f}  ({'OK' if bcp < 1.05 else 'DEGRADED'})")

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
        "experiment": "04_paired_recall_test",
        "timestamp": timestamp,
        "args": {
            "config": args.config,
            "facts_file": facts_path,
            "max_facts": args.max_facts,
            "skip_mc": args.skip_mc,
            "skip_cloze": args.skip_cloze,
            "skip_freeform": args.skip_freeform,
        },
        "overrides": {
            "delta_max": args.override_delta_max,
            "lambda_ewc": args.override_lambda_ewc,
            "epsilon_learn": args.override_epsilon_learn,
        },
        "config": config_record,
        "n_facts": len(facts),
        "wake_summary": {
            "n_active_tags": len(all_tags),
            "n_tagged_facts": n_tagged_facts,
            "n_wfast_updates": n_wfast_updates,
            "prp": _to_jsonable(prp_result),
            "n_consolidation_candidates": len(candidates),
        },
        "sleep_result": _to_jsonable(sleep_result),
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
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info("Wrote results to %s", args.output)
    print(f"\n  Results JSON: {args.output}")

    # ====================================================================
    # SANITY CHECKS (style of script 03)
    # ====================================================================
    print("\n" + "=" * 78)
    print("SANITY CHECKS")
    print("=" * 78)

    consolidated_dra_mc = (
        dra_by_group.get("Multiple choice", {}).get(GROUP_CONSOLIDATED)
    )
    untagged_dra_mc = (
        dra_by_group.get("Multiple choice", {}).get(GROUP_UNTAGGED)
    )

    checks = [
        ("Tags were created", len(all_tags) > 0, f"{len(all_tags)} tags"),
        (
            "PRPs were allocated",
            prp_result["allocated"] > 0,
            f"{prp_result['allocated']} allocated",
        ),
        (
            "Sleep completed without rollback",
            not sleep_result.get("rolled_back", False),
            "",
        ),
        ("BCP < 1.05 (no degradation)", bcp < 1.05, f"BCP={bcp:.4f}"),
        (
            "Group totals add to fact count",
            total == len(facts),
            f"{total}/{len(facts)}",
        ),
        (
            "Consolidated group is non-empty",
            counts_by_group.get(GROUP_CONSOLIDATED, 0) > 0,
            f"n={counts_by_group.get(GROUP_CONSOLIDATED, 0)}",
        ),
        (
            "MC: Consolidated >= Untagged (positive signal)",
            (
                consolidated_dra_mc is not None
                and untagged_dra_mc is not None
                and consolidated_dra_mc >= untagged_dra_mc
            ),
            (
                f"cons={consolidated_dra_mc:.3f}, "
                f"unt={untagged_dra_mc:.3f}"
                if (
                    consolidated_dra_mc is not None
                    and untagged_dra_mc is not None
                )
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
    print("EXPERIMENT 04 COMPLETE")
    print("=" * 78)
    logger.info("Experiment 04 complete.")


if __name__ == "__main__":
    main()
