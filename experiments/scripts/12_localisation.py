"""
Experiment 12: The Localisation Ladder (mentor revision, 2026-08-11).

Tests the reframe: the consolidation failure is a LOCALISATION problem
(wrong modules, wrong layers, one wording, wrong training signal), not a
stability-plasticity wall. Five arms, each isolating one move:

    A  attn_top          V/O attention LoRA, top third, single wording
                         -- replicates the original substrate (the control).
    B  mlp_mid           MLP LoRA (down/up projections), mid-stack window,
                         single wording -- placement move alone.
    C  mlp_mid_para      B + 20-25 paraphrases incl. QA forms per fact
                         -- placement + surface diversity.
    D  distill_mlp_mid   Context distillation (in-context teacher -> KL)
                         into the mid-stack MLP adapter, with paraphrases
                         -- the full localisation fix, no KV bank anywhere.
    E  distill_attn_top  Distillation into the ORIGINAL substrate
                         -- isolates training-signal vs placement.

All arms train a single fresh LoRA adapter with plain AdamW (no SLEEP safety
machinery -- the question is where/how to write, not how to constrain), then
evaluate greedy free-form DRA, MC recognition, cloze, and BCP.

USAGE (per arm; run arms as separate processes for clean isolation):
    python experiments/scripts/12_localisation.py \
        --config experiments/configs/qwen7b.yaml \
        --facts-file experiments/data/facts_200_para.json \
        --arm distill_mlp_mid --steps 600 --seed 0 \
        --output /workspace/results/loc_distill_mlp_mid_s0.json
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
import random

import torch
from peft import get_peft_model

from _common import (
    build_model_and_tokenizer,
    load_experiment_config,
    results_path,
    save_results,
    seed_everything,
)

from sleep.distill import ContextDistiller
from sleep.evaluation.preservation import compute_bcp, evaluate_perplexity
from sleep.evaluation.recall_formats import (
    cloze_recall,
    free_form_recall,
    group_facts_by_template,
    multiple_choice_recall,
)
from sleep.weights.lora import _build_lora_config
from sleep.utils.logging import setup_logging, get_logger

setup_logging()
logger = get_logger("experiment.12")

_CONTROL_TEXTS = [
    "The capital of France is Paris.",
    "Water boils at 100 degrees Celsius at sea level.",
    "Python is a programming language created by Guido van Rossum.",
]

# Arm definitions: (adapted_matrices, layer_selection, use_paraphrases, trainer)
ARMS: dict[str, dict] = {
    "attn_top": {
        "matrices": ["v_proj", "o_proj"], "selection": "top",
        "paraphrases": False, "trainer": "ce",
    },
    "mlp_mid": {
        "matrices": ["up_proj", "down_proj"], "selection": "middle",
        "paraphrases": False, "trainer": "ce",
    },
    "mlp_mid_para": {
        "matrices": ["up_proj", "down_proj"], "selection": "middle",
        "paraphrases": True, "trainer": "ce",
    },
    "distill_mlp_mid": {
        "matrices": ["up_proj", "down_proj"], "selection": "middle",
        "paraphrases": True, "trainer": "distill",
    },
    "distill_attn_top": {
        "matrices": ["v_proj", "o_proj"], "selection": "top",
        "paraphrases": True, "trainer": "distill",
    },
}


def train_ce(model, tokenizer, facts, use_paraphrases, steps, lr, seed, device) -> dict:
    """Plain causal-LM fine-tuning on fact wordings (arms A/B/C)."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    rng = random.Random(seed)
    model.train()

    losses = []
    for step in range(1, steps + 1):
        fact = rng.choice(facts)
        wordings = [fact["text"]]
        if use_paraphrases and fact.get("paraphrases"):
            wordings = fact["paraphrases"]
        text = rng.choice(wordings)

        ids = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=256).input_ids.to(device)
        loss = model(input_ids=ids, labels=ids).loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        losses.append(float(loss.item()))
        if step == 1 or step % max(1, steps // 10) == 0:
            logger.info("ce step %d/%d | loss=%.4f", step, steps, losses[-1])

    model.eval()
    return {"n_steps": len(losses), "initial_loss": losses[0],
            "final_loss": losses[-1], "mean_loss": sum(losses) / len(losses)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--facts-file", type=str, required=True,
                        help="Use the *_para.json dataset (carries paraphrases).")
    parser.add_argument("--max-facts", type=int, default=None)
    parser.add_argument("--arm", choices=sorted(ARMS.keys()), required=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--kd-temperature", type=float, default=2.0)
    parser.add_argument("--alpha-ce", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    seed_everything(args.seed)
    arm = ARMS[args.arm]

    logger.info("=" * 70)
    logger.info("EXPERIMENT 12: Localisation ladder | arm=%s", args.arm)
    logger.info("  matrices=%s selection=%s paraphrases=%s trainer=%s",
                arm["matrices"], arm["selection"], arm["paraphrases"], arm["trainer"])
    logger.info("=" * 70)

    cfg = load_experiment_config(args.config)
    # This experiment never injects memory, so the default attention backend
    # is fine and slightly faster.
    model, tokenizer = build_model_and_tokenizer(cfg, eager_attention=False)
    device = cfg["device"]

    with open(args.facts_file) as f:
        facts = json.load(f)
    if args.max_facts:
        facts = facts[: args.max_facts]
    by_template = group_facts_by_template(facts)
    n_para = sum(len(f.get("paraphrases", [])) for f in facts)
    logger.info("Loaded %d facts (%d paraphrase forms)", len(facts), n_para)

    control_seqs = [
        tokenizer(t, return_tensors="pt").input_ids[0].to(device)
        for t in _CONTROL_TEXTS
    ]
    ppl_before = evaluate_perplexity(model, control_seqs, device=device)

    # ---- Build the arm's adapter ----------------------------------------
    weights_cfg = copy.deepcopy(cfg["weights"])
    weights_cfg.adapted_matrices = list(arm["matrices"])
    weights_cfg.layer_selection = arm["selection"]
    lora_config = _build_lora_config(model, weights_cfg)
    model = get_peft_model(model, lora_config)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Adapter built: %s trainable params", f"{n_trainable:,}")

    # ---- Train -----------------------------------------------------------
    if arm["trainer"] == "ce":
        train_stats = train_ce(
            model, tokenizer, facts, arm["paraphrases"],
            args.steps, args.lr, args.seed, device,
        )
    else:
        distiller = ContextDistiller(
            model, tokenizer, device=device,
            kd_temperature=args.kd_temperature, alpha_ce=args.alpha_ce,
        )
        train_stats = distiller.run(
            facts, n_steps=args.steps, lr=args.lr, seed=args.seed,
            use_paraphrases=arm["paraphrases"],
        ).as_dict()
        model.eval()

    # ---- Evaluate (greedy throughout) ------------------------------------
    mc = multiple_choice_recall(model, tokenizer, facts, by_template, device=device)
    cloze = cloze_recall(model, tokenizer, facts, device=device)
    ff = free_form_recall(model, tokenizer, facts, device=device, decoding="greedy")
    ppl_after = evaluate_perplexity(model, control_seqs, device=device)
    bcp = compute_bcp(ppl_after, ppl_before)

    payload = {
        "experiment": "12_localisation",
        "arm": args.arm,
        "arm_spec": arm,
        "seed": args.seed,
        "config_path": args.config,
        "facts_file": args.facts_file,
        "n_facts": len(facts),
        "steps": args.steps,
        "lr": args.lr,
        "n_trainable_params": n_trainable,
        "train_stats": train_stats,
        "dra": ff["mean_score"],
        "free_form_accuracy": ff["accuracy"],
        "mc_accuracy": mc["accuracy"],
        "cloze_accuracy": cloze["accuracy"],
        "bcp": bcp,
        "ppl_before": ppl_before,
        "ppl_after": ppl_after,
        "decoding": "greedy",
    }
    out = save_results(payload, results_path(f"loc_{args.arm}", args.output))

    print("\n" + "=" * 64)
    print(f"LOCALISATION ARM: {args.arm}")
    print("=" * 64)
    print(f"  DRA (greedy):   {ff['mean_score']:.4f}")
    print(f"  cloze:          {cloze['accuracy']:.4f}")
    print(f"  MC:             {mc['accuracy']:.4f}")
    print(f"  BCP:            {bcp:.4f}")
    print(f"  train loss:     {train_stats['initial_loss']:.3f} -> {train_stats['final_loss']:.3f}")
    print(f"  Results: {out}")


if __name__ == "__main__":
    main()
