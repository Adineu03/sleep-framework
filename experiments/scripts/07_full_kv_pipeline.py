"""
Experiment 07: Full KV-Memory Pipeline (Phase B end-to-end)

PURPOSE:
    The first end-to-end test of the SLEEP architecture with KV memory as
    the W_fast substrate. Tests whether knowledge encoded as one-shot KV
    writes during wake can be consolidated through replay-driven training
    of W_cons, such that free-form recall works AFTER the KV bank is
    cleared.

    Pipeline:
        Wake:    surprise -> tag -> KV write
        PRP:     score allocation
        Sleep 1: GENERATE (KV ACTIVE — replays informed by stored experience)
        Sleep 2: QUALITY CHECK
        Sleep 3: TRAIN (KV DISABLED — W_cons learns from context alone)
        Sleep 4: PPL CHECK
        Sleep 5: VALIDATE & CLEANUP (KV DISABLED, then bank cleared)
        Recall:  test free-form recall using W_slow + W_cons ONLY (no KV)

    The recall test is the falsifiable claim:
        "Did consolidation transfer knowledge from KV bank into W_cons?"

PRE-REGISTERED THRESHOLDS:
    Free-form score > 0.05 with BCP < 1.05  -> Phase B succeeds.
    Free-form ~0.0 with BCP < 1.05          -> Consolidation didn't transfer.
                                                Diagnose replay quality first
                                                (printed) then training step.
    BCP > 1.05                              -> W_cons training broke base.
                                                Tighten delta_max / lambda_ewc.

DIAGNOSTIC: prints up to 3 (original_span, generated_replay) pairs from
    the sleep cycle. If KV memory is working, replays should contain real
    factual content from the source span; if they hallucinate, the
    bottleneck is in generation, not training.

USAGE (pod, Qwen2.5-7B):
    python experiments/scripts/07_full_kv_pipeline.py \
        --config experiments/configs/qwen7b.yaml \
        --facts-file experiments/data/facts_200.json \
        --kv-top-k 16
"""

import sys
import os
import argparse
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from sleep.config import SLEEPConfig, TaggingConfig, PRPConfig, WeightsConfig, SleepConfig, ColdStartConfig
from sleep.tagging import TaggingLayer, Tag
from sleep.prp import PRPSystem
from sleep.weights import DualWeightSystem
from sleep.sleep_engine import SleepEngine
from sleep.sleep_engine.replay import generate_replay, ReplaySample
from sleep.sleep_engine.quality import quality_check, compute_baseline_surprise
from sleep.evaluation.recall import evaluate_recall, RecallTestCase
from sleep.evaluation.preservation import evaluate_perplexity, compute_bcp
from sleep.utils.logging import setup_logging, get_logger

setup_logging()
logger = get_logger("experiment.07")


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
            "use_real_mu_surprise": False,  # disabled for GPT-2 PoC
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
        # Quality check on absolute surprise has a known generator-discriminator gap
        # (see logbook 2026-04-30). Default to bypass; set to true in config to re-enable.
        "use_real_mu_surprise": data.get("experiment", {}).get("use_real_mu_surprise", False),
    }


# ============================================================================
# Facts to teach the model (wake phase input)
# ============================================================================

FACTS = [
    {
        "id": "fact_01",
        "text": "The Zenith Corporation reported annual revenue of $847 million in 2025, a 23% increase over the previous year driven by expansion into the Southeast Asian market.",
        "test_prompt": "What was the Zenith Corporation's annual revenue in 2025?",
        "keywords": ["847", "million", "23%"],
    },
    {
        "id": "fact_02",
        "text": "Dr. Elena Vasquez at MIT discovered that graphene oxide membranes can desalinate seawater at 99.7% efficiency when layered at exactly 6.4 angstroms apart.",
        "test_prompt": "What did Dr. Elena Vasquez discover about graphene oxide membranes?",
        "keywords": ["desalinate", "99.7", "6.4", "angstroms"],
    },
    {
        "id": "fact_03",
        "text": "The city of New Helsinki was founded on March 15, 2024 as a planned smart city in northern Finland, with an initial population target of 50,000 residents.",
        "test_prompt": "When was New Helsinki founded and what was its population target?",
        "keywords": ["March", "2024", "50,000"],
    },
    {
        "id": "fact_04",
        "text": "Protocol Sigma-7 requires all neural network training runs exceeding 10 petaflops to be registered with the International AI Safety Board within 48 hours of initiation.",
        "test_prompt": "What does Protocol Sigma-7 require?",
        "keywords": ["10 petaflops", "registered", "48 hours"],
    },
    {
        "id": "fact_05",
        "text": "The Beryllium-Lithium fusion reactor at CERN achieved sustained plasma containment for 847 seconds on January 9th, 2026, setting a new world record.",
        "test_prompt": "How long did the CERN fusion reactor maintain sustained plasma?",
        "keywords": ["847 seconds", "January", "2026"],
    },
]

# Control text for base capability check
CONTROL_TEXTS = [
    "The capital of France is Paris. It is known for the Eiffel Tower and the Louvre museum.",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    "The Python programming language was created by Guido van Rossum and released in 1991.",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML (default: GPT-2 PoC on CPU)")
    parser.add_argument("--facts-file", type=str, default=None,
                        help="Path to facts JSON (default: 5 inline facts)")
    parser.add_argument("--max-facts", type=int, default=None,
                        help="Limit number of facts loaded from --facts-file")
    parser.add_argument("--kv-top-k", type=int, default=16,
                        help="Top-k retrieval gate for KV memory (default: 16)")
    parser.add_argument("--kv-max-tokens", type=int, default=20_000,
                        help="KV bank capacity (total tokens; default: 20000). "
                             "With full-episode storage (~30 tokens/fact x 200 "
                             "facts), 6000 tokens is the typical lower bound.")
    parser.add_argument("--store-full-episode", action="store_true", default=True,
                        help="Store K/V for the full fact (episode) on each "
                             "tag fire, rather than only the tagged span. "
                             "Tags are pointers; storage is the episode they "
                             "index. (Default ON; pass --no-store-full-episode "
                             "to use the old span-only storage.)")
    parser.add_argument("--no-store-full-episode", action="store_false",
                        dest="store_full_episode",
                        help="Disable full-episode storage; store only the "
                             "tagged span (the old behaviour, kept for "
                             "ablation).")
    parser.add_argument("--replay-strategy", choices=["generative", "original"],
                        default="generative",
                        help="How to construct replays during sleep. "
                             "'generative' = autoregressive generation "
                             "conditioned on KV memory (the SLEEP design). "
                             "'original' = bypass generation, use the "
                             "original tagged experience's tokens directly. "
                             "The latter is a diagnostic mode that strips "
                             "replay-generation as a confound — answers "
                             "'given perfect replays, does W_cons "
                             "consolidation transfer to recall?'")
    # Pareto-sweep knobs — these are the four parameters that govern the
    # SLEEP safety machinery. Sweeping them maps the stability-plasticity
    # tradeoff curve.
    parser.add_argument("--override-delta-max", type=float, default=None,
                        help="Override weights.delta_max (per-parameter clip). "
                             "SLEEP default 0.01; naive-LoRA-equivalent ~0.5+.")
    parser.add_argument("--override-lambda-ewc", type=float, default=None,
                        help="Override weights.lambda_ewc (EWC penalty strength). "
                             "SLEEP default 100; naive-LoRA-equivalent 0.")
    parser.add_argument("--override-alpha-slow", type=float, default=None,
                        help="Override weights.alpha_slow (sleep-training LR). "
                             "SLEEP default 1e-4.")
    parser.add_argument("--override-steps-per-memory", type=int, default=None,
                        help="Override sleep.steps_per_memory. With ~50 "
                             "replays, this directly controls total steps "
                             "(steps = max(100, n_replays * steps_per_memory)).")
    args = parser.parse_args()

    # Load facts (either from JSON file or use inline 5)
    global FACTS
    if args.facts_file:
        with open(args.facts_file) as f:
            FACTS = json.load(f)
        if args.max_facts:
            FACTS = FACTS[:args.max_facts]
        logger.info("Loaded %d facts from %s", len(FACTS), args.facts_file)

    logger.info("=" * 70)
    logger.info("EXPERIMENT 07: Full KV-Memory Pipeline (Phase B)")
    logger.info("=" * 70)

    cfg = load_config(args.config)
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map.get(cfg["dtype"], torch.float32)

    # ====================================================================
    # SETUP
    # ====================================================================
    logger.info("Loading model: %s (device=%s, dtype=%s)", cfg["model_name"], cfg["device"], cfg["dtype"])
    # Force eager attention so the KV-injection path's 4D additive masks
    # with top-k visibility gating are honored exactly. SDPA may handle
    # 4D masks differently. (See logbook 2026-04-30_kv_cache_order_bug.md.)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], torch_dtype=dtype, attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = model.to(cfg["device"])

    n_params_b = sum(p.numel() for p in model.parameters()) / 1e9
    logger.info("Model loaded (%.3fB parameters)", n_params_b)
    logger.info("Attention implementation: %s", model.config._attn_implementation)

    tagging_cfg = cfg["tagging"]
    prp_cfg = cfg["prp"]
    weights_cfg = cfg["weights"]
    sleep_cfg = cfg["sleep"]

    # ---- Pareto-sweep overrides ----
    overrides_applied = {}
    if args.override_delta_max is not None:
        overrides_applied["delta_max"] = (weights_cfg.delta_max, args.override_delta_max)
        weights_cfg.delta_max = args.override_delta_max
    if args.override_lambda_ewc is not None:
        overrides_applied["lambda_ewc"] = (weights_cfg.lambda_ewc, args.override_lambda_ewc)
        weights_cfg.lambda_ewc = args.override_lambda_ewc
    if args.override_alpha_slow is not None:
        overrides_applied["alpha_slow"] = (weights_cfg.alpha_slow, args.override_alpha_slow)
        weights_cfg.alpha_slow = args.override_alpha_slow
    if args.override_steps_per_memory is not None:
        overrides_applied["steps_per_memory"] = (sleep_cfg.steps_per_memory, args.override_steps_per_memory)
        sleep_cfg.steps_per_memory = args.override_steps_per_memory
    if overrides_applied:
        print(f"\n  Hyperparameter overrides applied:")
        for k, (orig, new) in overrides_applied.items():
            print(f"    {k}:  {orig}  ->  {new}")

    # Build subsystems with KV memory enabled as the W_fast substrate.
    dws = DualWeightSystem(
        model, weights_cfg,
        use_kv_memory_for_fast=True,
        kv_max_total_tokens=args.kv_max_tokens,
        kv_top_k=args.kv_top_k,
    )
    peft_model = dws.model
    logger.info("KV memory: top_k=%d, capacity=%d tokens",
                args.kv_top_k, args.kv_max_tokens)

    # TaggingLayer needs the BASE model for surprise computation.
    # With peft, we access it via peft_model.base_model.model
    # But TaggingLayer does forward passes that go through LoRA anyway.
    # We pass the peft_model and it works (surprise is computed against current model state).
    tagging = TaggingLayer(peft_model, tagging_cfg, model_params_billions=n_params_b)
    prp = PRPSystem(prp_cfg, budget=int(prp_cfg.c_prp * n_params_b), revision_bonus=0.3)

    # Compute baseline surprise for quality checking
    logger.info("Computing baseline surprise on control texts...")
    dws.set_mode("target_inference")
    control_ids = [tokenizer.encode(t, return_tensors="pt").squeeze(0).to(cfg["device"]) for t in CONTROL_TEXTS]
    mu_surprise_real = compute_baseline_surprise(peft_model, control_ids, device=cfg["device"])
    if cfg["use_real_mu_surprise"]:
        mu_surprise = mu_surprise_real
        logger.info("Baseline mu_surprise: %.3f nats (using computed value)", mu_surprise_real)
    else:
        mu_surprise = 0.0
        logger.info("Baseline mu_surprise: %.3f nats (DISABLED for PoC — set to 0)", mu_surprise_real)

    # Measure baseline PPL
    baseline_ppl = evaluate_perplexity(peft_model, control_ids, device=cfg["device"])
    logger.info("Baseline PPL on controls: %.2f", baseline_ppl)

    # ====================================================================
    # WAKE PHASE: Feed facts
    # ====================================================================
    print(f"\n{'='*70}")
    print("WAKE PHASE: Processing facts")
    print(f"{'='*70}")

    dws.set_mode("wake_inference")
    original_tokens_map = {}
    n_kv_writes = 0
    n_facts_written = 0
    sources_already_written: set = set()

    storage_mode = "FULL EPISODE" if args.store_full_episode else "TAGGED SPAN ONLY"
    print(f"  Storage mode: {storage_mode}")
    print(f"  (Tags are pointers; episodes are what gets stored.)")

    for fact in FACTS:
        token_ids = tokenizer.encode(fact["text"], return_tensors="pt").squeeze(0).to(cfg["device"])
        original_tokens_map[fact["id"]] = token_ids

        # Tag the input (suppressing per-fact prints for 200-fact runs)
        new_tags = tagging.process_input(token_ids, source_id=fact["id"])

        # KV write semantics:
        #   - tags are POINTERS that say "this experience is worth storing"
        #   - what we STORE is the episode the pointer points to (the full fact)
        #   - multiple tags pointing to the same fact -> one bank entry
        for tag_idx, tag in enumerate(new_tags):
            span_start, span_end, source_id = tag.ctx
            if tag.e0 <= 1.5:
                continue
            if args.store_full_episode and source_id in sources_already_written:
                continue  # already wrote this episode

            if args.store_full_episode:
                # Store K/V for the FULL fact tokens — tag is a pointer,
                # storage is the episode.
                tag_id = source_id
                write_start, write_end = 0, int(token_ids.shape[0])
            else:
                # Old behaviour (kept for ablation): store K/V only for the
                # tagged span itself.
                tag_id = f"{source_id}__t{tag_idx}"
                write_start, write_end = span_start, span_end

            try:
                dws.write_to_kv_bank(
                    tag_id=tag_id,
                    token_ids=token_ids,
                    span_start=write_start,
                    span_end=write_end,
                    device=cfg["device"],
                )
                n_kv_writes += 1
                if args.store_full_episode:
                    sources_already_written.add(source_id)
                    n_facts_written += 1
            except RuntimeError as exc:
                logger.warning(
                    "KV bank full at fact=%s tag_idx=%d: %s",
                    fact["id"], tag_idx, exc,
                )

    print(f"  Wake summary:")
    print(f"    KV writes performed:  {n_kv_writes}")
    if args.store_full_episode:
        print(f"    Unique facts stored:  {n_facts_written}")
    print(f"    KV bank state:        {dws.kv_bank.n_tags} entries / "
          f"{dws.kv_bank.n_total_tokens} tokens "
          f"(capacity {dws.kv_bank.max_total_tokens})")

    # Run PRP scoring and allocation
    all_tags = tagging.active_tags
    print(f"\n  Total active tags: {len(all_tags)}")

    prp_result = prp.update(all_tags, current_step=tagging.step, force_crossref=True)
    print(f"  PRP allocated: {prp_result['allocated']}/{prp.budget}")
    print(f"  Threshold: {prp_result['threshold']:.3f}")
    print(f"  Mean score: {prp_result['mean_score']:.3f}")

    candidates = prp.get_consolidation_candidates(all_tags)
    print(f"  Consolidation candidates: {len(candidates)}")

    if len(candidates) == 0:
        print("\n  WARNING: No PRP candidates! Cannot run sleep cycle.")
        print("  This means no tags scored above the PRP threshold.")
        print("  Consider: lowering kappa, lowering theta_floor, or checking scoring weights.")
        logger.warning("No PRP candidates — experiment cannot proceed to sleep phase.")
        return

    # ====================================================================
    # SLEEP PHASE
    # ====================================================================
    print(f"\n{'='*70}")
    print("SLEEP PHASE: Running consolidation")
    print(f"{'='*70}")

    sleep_engine = SleepEngine(
        dual_weights=dws,
        tokenizer=tokenizer,
        sleep_config=sleep_cfg,
        weights_config=weights_cfg,
        mu_surprise=mu_surprise,
        device=cfg["device"],
        replay_strategy=args.replay_strategy,
    )
    logger.info("Replay strategy: %s", args.replay_strategy)

    # Run the sleep cycle
    sleep_result = sleep_engine.run_cycle(
        candidates=candidates,
        original_tokens_map=original_tokens_map,
        key_projection=tagging.key_projection,
    )

    print(f"\n  Sleep cycle results:")
    print(f"    Candidates:         {sleep_result.get('n_candidates', '?')}")
    print(f"    Replays generated:  {sleep_result.get('n_replays_generated', '?')}")
    print(f"    Replays accepted:   {sleep_result.get('n_replays_accepted', '?')}")
    print(f"    Consolidated:       {sleep_result.get('n_consolidated', '?')}")
    print(f"    Failed:             {sleep_result.get('n_failed', '?')}")
    print(f"    Rolled back:        {sleep_result.get('rolled_back', '?')}")

    training_stats = sleep_result.get("training_stats", {})
    if training_stats:
        print(f"    Training steps:     {training_stats.get('n_steps', '?')}")
        print(f"    Mean loss:          {training_stats.get('mean_loss', '?'):.4f}" if isinstance(training_stats.get('mean_loss'), float) else "")
        print(f"    Final loss:         {training_stats.get('final_loss', '?'):.4f}" if isinstance(training_stats.get('final_loss'), float) else "")

    # ====================================================================
    # REPLAY QUALITY DIAGNOSTIC
    # The user's pre-registered diagnostic: if KV memory is working, the
    # replays should contain real factual content from stored memories,
    # NOT hallucinated noise. If replays look good but recall fails, the
    # bottleneck is training. If replays are still poor, the bottleneck
    # is in how the model uses stored memories during generation.
    # ====================================================================
    print(f"\n{'='*70}")
    print("REPLAY QUALITY DIAGNOSTIC (sample of generated replays)")
    print(f"{'='*70}")
    replay_samples = sleep_result.get("replay_samples", [])
    if not replay_samples:
        print("  (no replay samples captured)")
    for i, sample in enumerate(replay_samples):
        print(f"\n  Sample {i+1} — source_id={sample['source_id']}")
        print(f"    Original span (the fact):")
        print(f"      \"{sample['original_span']}\"")
        if "replay_seed" in sample:
            print(f"    Replay seed ({sample.get('seed_length', '?')} tokens — from original):")
            print(f"      \"{sample['replay_seed']}\"")
            print(f"    Replay generated ({sample['replay_n_tokens'] - sample.get('seed_length', 0)} tokens — model with KV active):")
            print(f"      \"{sample['replay_generated']}\"")
        else:
            print(f"    Generated replay ({sample['replay_n_tokens']} tokens):")
            print(f"      \"{sample['replay']}\"")

    # ====================================================================
    # POST-SLEEP STATE CHECK
    # The sleep cycle should have cleared the KV bank if any consolidation
    # passed. We assert that here so the recall test below is genuinely
    # measuring W_slow + W_cons, not residual KV memory.
    # ====================================================================
    print(f"\n{'='*70}")
    print("POST-SLEEP STATE CHECK")
    print(f"{'='*70}")
    if dws.use_kv_memory_for_fast:
        bank_after = dws.kv_bank
        print(f"  KV bank after sleep:        {bank_after.n_tags} tags / {bank_after.n_total_tokens} tokens")
        print(f"  KV injector enabled:        {dws.kv_injector.is_enabled}")
        if sleep_result.get("n_consolidated", 0) > 0:
            assert bank_after.n_tags == 0, "Bank should have been cleared after consolidation"
            print("  Bank correctly cleared after successful consolidation.")
        # Disable injection regardless, so the recall test below is a clean
        # measurement of W_slow + W_cons.
        dws.set_kv_enabled(False)
        print(f"  KV injection forced OFF for recall test (clean W_slow + W_cons measurement).")

    # ====================================================================
    # EVALUATION: Recall test
    # ====================================================================
    print(f"\n{'='*70}")
    print("EVALUATION: Testing recall after consolidation")
    print(f"{'='*70}")

    # Build recall test cases
    test_cases = [
        RecallTestCase(
            source_id=fact["id"],
            prompt=fact["test_prompt"],
            expected_keywords=fact["keywords"],
        )
        for fact in FACTS
    ]

    # Test with consolidated model (W_slow + W_cons)
    dws.set_mode("target_inference")
    recall_result = evaluate_recall(peft_model, tokenizer, test_cases, max_new_tokens=50, device=cfg["device"])

    print(f"\n  Delayed Recall Accuracy (DRA): {recall_result['dra']:.3f}")
    print(f"\n  Per-fact results:")
    for case_result in recall_result["per_case"]:
        status = "HIT" if case_result["score"] > 0 else "MISS"
        print(f"    [{status}] {case_result['source_id']}: score={case_result['score']:.2f}")
        print(f"           Prompt: \"{case_result['prompt'][:60]}\"")
        response_preview = case_result.get("response", "")[:80]
        print(f"           Response: \"{response_preview}...\"")
        found = case_result.get("keywords_found", [])
        missing = case_result.get("keywords_missing", [])
        if found:
            print(f"           Found: {found}")
        if missing:
            print(f"           Missing: {missing}")

    # ====================================================================
    # EVALUATION: Base capability preservation
    # ====================================================================
    print(f"\n{'='*70}")
    print("EVALUATION: Base capability preservation")
    print(f"{'='*70}")

    dws.set_mode("target_inference")
    post_ppl = evaluate_perplexity(peft_model, control_ids, device=cfg["device"])
    bcp = compute_bcp(post_ppl, baseline_ppl)

    print(f"  Baseline PPL:  {baseline_ppl:.2f}")
    print(f"  Post-sleep PPL: {post_ppl:.2f}")
    print(f"  BCP ratio:     {bcp:.4f}  ({'OK' if bcp < 1.05 else 'DEGRADED'})")

    # ====================================================================
    # SUMMARY
    # ====================================================================
    print(f"\n{'='*70}")
    print("EXPERIMENT 07 SUMMARY")
    print(f"{'='*70}")
    print(f"  Facts fed:            {len(FACTS)}")
    print(f"  Tags created:         {len(all_tags)}")
    print(f"  PRP allocated:        {prp_result['allocated']}")
    print(f"  Replays accepted:     {sleep_result.get('n_replays_accepted', '?')}")
    print(f"  Memories consolidated: {sleep_result.get('n_consolidated', '?')}")
    print(f"  DRA (recall):         {recall_result['dra']:.3f}")
    print(f"  BCP (preservation):   {bcp:.4f}")
    print(f"  Rolled back:          {sleep_result.get('rolled_back', False)}")

    # Pre-registered Phase B verdict
    print(f"\n  Sanity Checks:")
    checks = [
        ("Tags were created", len(all_tags) > 0, f"{len(all_tags)} tags"),
        ("PRPs were allocated", prp_result['allocated'] > 0, f"{prp_result['allocated']} allocated"),
        ("Sleep completed without rollback", not sleep_result.get('rolled_back', True), ""),
        ("BCP < 1.05 (no degradation)", bcp < 1.05, f"BCP={bcp:.4f}"),
        ("DRA > 0 (some recall)", recall_result['dra'] > 0, f"DRA={recall_result['dra']:.3f}"),
        ("KV bank cleared after consolidation",
         (not dws.use_kv_memory_for_fast)
         or (sleep_result.get('n_consolidated', 0) == 0)
         or (dws.kv_bank.n_tags == 0),
         f"bank n_tags={dws.kv_bank.n_tags if dws.use_kv_memory_for_fast else 'n/a'}"),
    ]

    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        print(f"    [{status}] {name} {detail}")

    n_passed = sum(1 for _, p, _ in checks if p)
    print(f"\n  {n_passed}/{len(checks)} sanity checks passed")

    # Pre-registered Phase B verdict
    print(f"\n{'='*70}")
    print("PHASE B VERDICT (pre-registered)")
    print(f"{'='*70}")
    free_form_score = recall_result.get("dra", 0.0)
    if bcp >= 1.05:
        verdict = (
            "BCP >= 1.05  ->  W_cons training broke base capabilities. "
            "Tighten delta_max / lambda_ewc."
        )
    elif free_form_score > 0.05:
        verdict = (
            f"Free-form DRA={free_form_score:.3f} > 0.05 with BCP={bcp:.3f} < 1.05  "
            f"->  PHASE B SUCCEEDS. Full KV pipeline works end-to-end. "
            "Knowledge transferred from KV bank into W_cons."
        )
    else:
        verdict = (
            f"Free-form DRA={free_form_score:.3f} ~0 with BCP={bcp:.3f} < 1.05  "
            "->  Consolidation didn't transfer. Diagnose replay quality first "
            "(printed above) then training step."
        )
    print(f"  Verdict: {verdict}")

    logger.info("Experiment 03 complete.")


if __name__ == "__main__":
    main()
