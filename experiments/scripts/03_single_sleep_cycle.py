"""
Experiment 03: Single Sleep Cycle (End-to-End)

PURPOSE:
    The first full test of the SLEEP system. Feed a set of facts during "wake",
    then run one sleep cycle, then test if the consolidated model recalls the facts.

    This tests the COMPLETE pipeline:
    Wake:  Input -> Surprise -> Tag -> W_fast update -> PRP scoring -> PRP allocation
    Sleep: Select candidates -> Generate replay -> Quality check -> Train W_cons -> Validate -> Cleanup

WHAT TO LOOK FOR:
    - Facts get tagged and PRP-allocated
    - Replay generation produces coherent gist (not garbage)
    - W_cons training completes without crash or PPL explosion
    - Some facts are recalled after consolidation (DRA > 0)
    - Base capabilities preserved (BCP close to 1.0)

USAGE:
    python experiments/scripts/03_single_sleep_cycle.py
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
logger = get_logger("experiment.03")


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
    logger.info("EXPERIMENT 03: Single Sleep Cycle (End-to-End)")
    logger.info("=" * 70)

    cfg = load_config(args.config)
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map.get(cfg["dtype"], torch.float32)

    # ====================================================================
    # SETUP
    # ====================================================================
    logger.info("Loading model: %s (device=%s, dtype=%s)", cfg["model_name"], cfg["device"], cfg["dtype"])
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

    # Build subsystems
    dws = DualWeightSystem(model, weights_cfg)
    peft_model = dws.model

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

    for fact in FACTS:
        token_ids = tokenizer.encode(fact["text"], return_tensors="pt").squeeze(0).to(cfg["device"])
        original_tokens_map[fact["id"]] = token_ids

        # Tag the input
        new_tags = tagging.process_input(token_ids, source_id=fact["id"])
        print(f"\n  [{fact['id']}] {len(token_ids)} tokens, {len(new_tags)} tags created")
        for tag in new_tags:
            span_start, span_end, _ = tag.ctx
            span_text = tokenizer.decode(token_ids[span_start:span_end + 1])
            print(f"    Tag: \"{span_text[:60]}...\" E={tag.e0:.2f} s={tag.s:.3f}")

        # W_fast update on tagged spans
        for tag in new_tags:
            span_start, span_end, _ = tag.ctx
            if tag.e0 > 1.5:  # only update on sufficiently surprising spans
                loss = dws.update_fast_weights(token_ids, span_start, span_end, tag.e0, device=cfg["device"])
                print(f"    W_fast update: loss={loss:.4f}")

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
    )

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
    print("EXPERIMENT 03 SUMMARY")
    print(f"{'='*70}")
    print(f"  Facts fed:            {len(FACTS)}")
    print(f"  Tags created:         {len(all_tags)}")
    print(f"  PRP allocated:        {prp_result['allocated']}")
    print(f"  Replays accepted:     {sleep_result.get('n_replays_accepted', '?')}")
    print(f"  Memories consolidated: {sleep_result.get('n_consolidated', '?')}")
    print(f"  DRA (recall):         {recall_result['dra']:.3f}")
    print(f"  BCP (preservation):   {bcp:.4f}")
    print(f"  Rolled back:          {sleep_result.get('rolled_back', False)}")

    # Sanity checks
    print(f"\n  Sanity Checks:")
    checks = [
        ("Tags were created", len(all_tags) > 0, f"{len(all_tags)} tags"),
        ("PRPs were allocated", prp_result['allocated'] > 0, f"{prp_result['allocated']} allocated"),
        ("Sleep completed without rollback", not sleep_result.get('rolled_back', True), ""),
        ("BCP < 1.05 (no degradation)", bcp < 1.05, f"BCP={bcp:.4f}"),
        ("DRA > 0 (some recall)", recall_result['dra'] > 0, f"DRA={recall_result['dra']:.3f}"),
    ]

    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        print(f"    [{status}] {name} {detail}")

    n_passed = sum(1 for _, p, _ in checks if p)
    print(f"\n  {n_passed}/{len(checks)} sanity checks passed")

    logger.info("Experiment 03 complete.")


if __name__ == "__main__":
    main()
