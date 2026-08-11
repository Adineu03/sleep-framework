"""
Experiment 13: Full SLEEP pipeline with the localisation consolidation recipe.

Phase B of the redemption arc. The complete autonomous loop —
    wake (tagging + KV episode writes) -> PRP selection -> sleep
    (paraphrase replay / context distillation into mid-MLP w_cons, under the
    full safety machinery) -> two-stage validation -> greedy recall evaluation
— with the consolidation step rebuilt per the localisation findings. No part
of the pipeline is bypassed: this is the "does SLEEP itself work now?" test.

Recipes:
    distill        replay_strategy="paraphrase" + train_mode="distill"
                   (Qwen's ladder winner)
    paraphrase_ce  replay_strategy="paraphrase" + train_mode="ce"
                   (Llama's ladder winner)

USAGE:
    python experiments/scripts/13_full_pipeline_v2.py \
        --config experiments/configs/qwen7b_mlp_mid.yaml \
        --facts-file experiments/data/facts_200_para.json \
        --recipe distill --train-steps 1500 --seed 0 \
        --output /workspace/results/pipeline_v2_qwen_distill_s0.json
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json

from _common import (
    build_model_and_tokenizer,
    load_experiment_config,
    results_path,
    save_results,
    seed_everything,
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
from sleep.tagging import TaggingLayer
from sleep.weights import DualWeightSystem
from sleep.utils.logging import setup_logging, get_logger

setup_logging()
logger = get_logger("experiment.13")

_CONTROL_TEXTS = [
    "The capital of France is Paris.",
    "Water boils at 100 degrees Celsius at sea level.",
    "Python is a programming language created by Guido van Rossum.",
]

RECIPES = {
    "distill": {"replay": "paraphrase", "train": "distill"},
    "paraphrase_ce": {"replay": "paraphrase", "train": "ce"},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Use a *_mlp_mid.yaml config for the localisation placement.")
    parser.add_argument("--facts-file", type=str, required=True,
                        help="Must be the *_para.json dataset.")
    parser.add_argument("--max-facts", type=int, default=None)
    parser.add_argument("--recipe", choices=sorted(RECIPES.keys()), required=True)
    parser.add_argument("--train-steps", type=int, default=1500)
    parser.add_argument("--kv-top-k", type=int, default=64)
    parser.add_argument("--kv-max-tokens", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    # Safety-machinery isolation flags (Phase B diagnostic): the ladder ran
    # without SLEEP's hard clip and plasticity scaling; these let the full
    # pipeline run with the machinery relaxed so its effect is measurable.
    parser.add_argument("--override-delta-max", type=float, default=None,
                        help="Override weights.delta_max (design default 0.01; "
                             "1.0 effectively disables the hard clip).")
    parser.add_argument("--override-phi-min", type=float, default=None,
                        help="Override weights.phi_min (1.0 disables "
                             "plasticity down-scaling).")
    args = parser.parse_args()

    seed_everything(args.seed)
    recipe = RECIPES[args.recipe]

    logger.info("=" * 70)
    logger.info("EXPERIMENT 13: Full pipeline v2 | recipe=%s | seed=%d",
                args.recipe, args.seed)
    logger.info("=" * 70)

    cfg = load_experiment_config(args.config)
    overrides = {}
    if args.override_delta_max is not None:
        overrides["delta_max"] = (cfg["weights"].delta_max, args.override_delta_max)
        cfg["weights"].delta_max = args.override_delta_max
    if args.override_phi_min is not None:
        overrides["phi_min"] = (cfg["weights"].phi_min, args.override_phi_min)
        cfg["weights"].phi_min = args.override_phi_min
    if overrides:
        logger.info("Safety overrides: %s", overrides)
    model, tokenizer = build_model_and_tokenizer(cfg, eager_attention=True)
    device = cfg["device"]

    with open(args.facts_file) as f:
        facts = json.load(f)
    if args.max_facts:
        facts = facts[: args.max_facts]
    by_template = group_facts_by_template(facts)
    fact_map = {f["id"]: f for f in facts}
    logger.info("Loaded %d facts (paraphrased: %s)",
                len(facts), all("paraphrases" in f for f in facts))

    control_seqs = [
        tokenizer(t, return_tensors="pt").input_ids[0].to(device)
        for t in _CONTROL_TEXTS
    ]

    # ---- System construction --------------------------------------------
    dws = DualWeightSystem(
        model, cfg["weights"],
        use_kv_memory_for_fast=True,
        kv_max_total_tokens=args.kv_max_tokens,
        kv_top_k=args.kv_top_k,
    )
    n_params_b = sum(p.numel() for p in model.parameters()) / 1e9
    tagging = TaggingLayer(dws.model, cfg["tagging"], model_params_billions=n_params_b)
    prp = PRPSystem(cfg["prp"], budget=int(cfg["prp"].c_prp * n_params_b))

    logger.info("Adapter layers: %s | injection layers: %s",
                dws.adapted_layers, dws.injection_layers)

    dws.set_mode("target_inference")
    ppl_before = evaluate_perplexity(dws.model, control_seqs, device=device)

    # ---- WAKE: tagging + episode writes ---------------------------------
    logger.info("WAKE: feeding %d facts", len(facts))
    original_tokens_map: dict = {}
    all_tags: list = []
    sources_written: set = set()
    dws.set_mode("wake_inference")
    for fact in facts:
        token_ids = tokenizer(fact["text"], return_tensors="pt").input_ids[0].to(device)
        original_tokens_map[fact["id"]] = token_ids
        new_tags = tagging.process_input(token_ids, source_id=fact["id"])
        all_tags.extend(new_tags)
        # Episode-level KV write (recognition subsystem, unchanged).
        for tag in new_tags:
            _s, _e, source_id = tag.ctx
            if source_id in sources_written:
                continue
            try:
                dws.write_to_kv_bank(source_id, token_ids, 0, int(token_ids.numel()))
                sources_written.add(source_id)
            except Exception as exc:
                logger.debug("KV write skipped for %s: %s", source_id, exc)
    logger.info("WAKE done: %d tags, %d episodes in bank", len(all_tags), len(sources_written))

    # ---- PRP selection ---------------------------------------------------
    prp_result = prp.update(all_tags, current_step=tagging.step, force_crossref=True)
    candidates = prp.get_consolidation_candidates(all_tags)
    logger.info("PRP: %d/%d allocated", prp_result["allocated"], len(all_tags))

    # ---- SLEEP: the new consolidation ------------------------------------
    engine = SleepEngine(
        dual_weights=dws,
        tokenizer=tokenizer,
        sleep_config=cfg["sleep"],
        weights_config=cfg["weights"],
        mu_surprise=1.0,  # paraphrase strategy bypasses the surprise gate
        device=device,
        replay_strategy=recipe["replay"],
        train_mode=recipe["train"],
        train_steps_override=args.train_steps,
        two_stage_validation=True,
    )
    sleep_result = engine.run_cycle(
        candidates=candidates,
        original_tokens_map=original_tokens_map,
        key_projection=tagging.key_projection,
        fact_map=fact_map,
    )

    # ---- EVALUATE (post-consolidation, bank cleared if success) ----------
    dws.set_mode("target_inference")
    if dws.use_kv_memory_for_fast:
        dws.set_kv_enabled(False)  # measure WEIGHTS, not the bank

    mc = multiple_choice_recall(dws.model, tokenizer, facts, by_template, device=device)
    cloze = cloze_recall(dws.model, tokenizer, facts, device=device)
    ff = free_form_recall(dws.model, tokenizer, facts, device=device, decoding="greedy")
    ppl_after = evaluate_perplexity(dws.model, control_seqs, device=device)
    bcp = compute_bcp(ppl_after, ppl_before)

    # Trained-subset metric: PRP selects a subset of facts; overall DRA mixes
    # trained and never-trained facts. Report the consolidation-target subset
    # separately so the write itself is measured without the selection
    # denominator.
    selected_ids = {tag.ctx[2] for tag in candidates}
    trained_facts = [f for f in facts if f["id"] in selected_ids]
    if trained_facts:
        ff_trained = free_form_recall(
            dws.model, tokenizer, trained_facts, device=device, decoding="greedy",
        )
        dra_trained = ff_trained["mean_score"]
    else:
        dra_trained = 0.0

    payload = {
        "experiment": "13_full_pipeline_v2",
        "recipe": args.recipe,
        "seed": args.seed,
        "config_path": args.config,
        "n_facts": len(facts),
        "train_steps": args.train_steps,
        "n_tags": len(all_tags),
        "prp_allocated": prp_result["allocated"],
        "n_candidates": sleep_result["n_candidates"],
        "n_replays_accepted": sleep_result["n_replays_accepted"],
        "n_passed_surprise": sleep_result["n_passed_surprise"],
        "n_passed_recall_gate": sleep_result["n_passed_recall_gate"],
        "n_consolidated": sleep_result["n_consolidated"],
        "rolled_back": sleep_result["rolled_back"],
        "training_stats": {k: v for k, v in sleep_result["training_stats"].items()
                           if k != "cons_checkpoint"},
        "dra": ff["mean_score"],
        "dra_trained_subset": dra_trained,
        "n_trained_facts": len(trained_facts),
        "free_form_accuracy": ff["accuracy"],
        "mc_accuracy": mc["accuracy"],
        "cloze_accuracy": cloze["accuracy"],
        "bcp": bcp,
        "ppl_before": ppl_before,
        "ppl_after": ppl_after,
        "decoding": "greedy",
        "safety_overrides": {k: v[1] for k, v in overrides.items()},
    }
    out = save_results(payload, results_path(f"pipeline_v2_{args.recipe}", args.output))

    print("\n" + "=" * 64)
    print(f"FULL PIPELINE v2 — recipe={args.recipe} seed={args.seed}")
    print("=" * 64)
    print(f"  tags={len(all_tags)}  PRP={prp_result['allocated']}  "
          f"consolidated={sleep_result['n_consolidated']}  "
          f"recall-gate={sleep_result['n_passed_recall_gate']}")
    print(f"  DRA (greedy):   {ff['mean_score']:.4f}  (trained subset: {dra_trained:.4f} over {len(trained_facts)})")
    print(f"  cloze:          {cloze['accuracy']:.4f}")
    print(f"  MC:             {mc['accuracy']:.4f}")
    print(f"  BCP:            {bcp:.4f}")
    print(f"  rolled_back:    {sleep_result['rolled_back']}")
    print(f"  Results: {out}")

    dws.cleanup()


if __name__ == "__main__":
    main()
