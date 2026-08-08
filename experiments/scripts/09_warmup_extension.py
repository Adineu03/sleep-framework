"""
Experiment 09: Retrieval-Aware Warm-Up Extension (the diagnostic-to-solution step).

Tests the paper's proposed fix for the recognition--recall gap. The design:

  1. Write each fact's episode K/V into the KV bank (the recognition setup that
     produced the +0.16 MC signal but zero free-form recall).
  2. Evaluate MC / cloze / free-form DRA and BCP with an IDENTITY gate — this is
     the "SLEEP without warm-up" row (should reproduce the paper's gap).
  3. Run a retrieval-aware warm-up on a GENERAL corpus (never the eval facts):
     the MemoryGate learns to route KV memory into generation.
  4. Re-evaluate with the trained gate active — the "SLEEP with warm-up" row.

The headline output is the §7.7 comparison table: does DRA rise (gap narrows)
while BCP stays acceptable? Even a partial rise converts the central negative
result into a diagnosed-and-treated one.

KV injection requires a Qwen-family model in bfloat16/float32 with eager
attention — run on RunPod with --config experiments/configs/qwen7b.yaml. The
mechanism itself is unit-tested on a tiny in-memory Qwen2 in tests/test_warmup/.

USAGE:
    python experiments/scripts/09_warmup_extension.py \
        --config experiments/configs/qwen7b.yaml \
        --facts-file experiments/data/facts_200.json \
        --corpus-file experiments/data/warmup_corpus.json \
        --kv-top-k 64 --warmup-steps 300 --seed 0 \
        --output experiments/results/warmup_ext_seed0.json
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json

import torch

from _common import (
    build_model_and_tokenizer,
    load_experiment_config,
    results_path,
    save_results,
    seed_everything,
)

from sleep.tagging import TaggingLayer
from sleep.weights import DualWeightSystem
from sleep.warmup import WarmupTrainer
from sleep.evaluation.recall_formats import (
    cloze_recall,
    free_form_recall,
    group_facts_by_template,
    multiple_choice_recall,
)
from sleep.evaluation.preservation import compute_bcp, evaluate_perplexity
from sleep.utils.logging import setup_logging, get_logger

setup_logging()
logger = get_logger("experiment.09")


# A small generic corpus used only if --corpus-file is not supplied. These are
# deliberately unrelated to any evaluation fact — the warm-up teaches the SKILL
# of using memory, not any specific content.
_FALLBACK_CORPUS = [
    "The history of cartography spans several thousand years across many cultures.",
    "Photosynthesis converts light energy into chemical energy stored in glucose.",
    "Tidal forces arise from the differential gravitational pull across a body.",
    "In economics, comparative advantage explains gains from specialization and trade.",
    "The printing press accelerated the spread of literacy throughout early modern Europe.",
    "Mitochondria generate most of the chemical energy that cells need to function.",
    "Plate tectonics describes the large-scale motion of the lithospheric plates.",
    "A compiler translates source code into machine instructions a processor can run.",
    "The water cycle moves moisture between the oceans, atmosphere, and land.",
    "Antibiotics act on bacterial processes that human cells do not share.",
    "Supply and demand jointly determine the equilibrium price in a competitive market.",
    "Neurons communicate through electrical impulses and chemical neurotransmitters.",
]


def _load_corpus(path: str | None, tokenizer, device: str) -> list[torch.Tensor]:
    """Load the warm-up corpus as token tensors (from JSON list of strings)."""
    if path:
        with open(path, encoding="utf-8") as f:
            texts = json.load(f)
    else:
        texts = _FALLBACK_CORPUS
        logger.warning(
            "No --corpus-file given; using the %d-sentence fallback corpus. "
            "For a real run supply a general corpus (e.g. WikiText).",
            len(texts),
        )
    seqs = []
    for t in texts:
        ids = tokenizer(t, return_tensors="pt").input_ids[0].to(device)
        if ids.numel() >= 8:
            seqs.append(ids)
    return seqs


def _evaluate(model, tokenizer, facts, by_template, device, label) -> dict:
    """Run MC / cloze / free-form recall and return the metric triple."""
    mc = multiple_choice_recall(model, tokenizer, facts, by_template, device=device)
    cloze = cloze_recall(model, tokenizer, facts, device=device)
    ff = free_form_recall(model, tokenizer, facts, device=device)
    logger.info(
        "[%s] MC=%.3f  cloze=%.3f  DRA=%.3f",
        label, mc["accuracy"], cloze["accuracy"], ff["mean_score"],
    )
    return {
        "mc_accuracy": mc["accuracy"],
        "mc_mean_correct_prob": mc["mean_correct_prob"],
        "cloze_accuracy": cloze["accuracy"],
        "dra": ff["mean_score"],
        "free_form_accuracy": ff["accuracy"],
    }


def _write_episodes(dws: DualWeightSystem, tagging: TaggingLayer, facts, tokenizer, device):
    """Tag each fact and write its full-episode K/V into the bank."""
    dws.clear_kv_bank()
    n_written = 0
    for fact in facts:
        token_ids = tokenizer(fact["text"], return_tensors="pt").input_ids[0].to(device)
        tagging.process_input(token_ids, source_id=fact["id"])
        # Store the whole fact (episode) — tags are pointers, storage is the
        # episode they index.
        try:
            dws.write_to_kv_bank(
                tag_id=fact["id"], token_ids=token_ids,
                span_start=0, span_end=int(token_ids.numel()), device=device,
            )
            n_written += 1
        except Exception as exc:
            logger.debug("episode write failed for %s: %s", fact["id"], exc)
    logger.info("Wrote %d fact episodes into the KV bank", n_written)
    return n_written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--facts-file", type=str, required=True)
    parser.add_argument("--max-facts", type=int, default=None)
    parser.add_argument("--corpus-file", type=str, default=None,
                        help="JSON list of general-text strings for warm-up. "
                             "Must NOT overlap the eval facts.")
    parser.add_argument("--kv-top-k", type=int, default=64)
    parser.add_argument("--kv-max-tokens", type=int, default=20_000)
    parser.add_argument("--warmup-steps", type=int, default=300)
    parser.add_argument("--warmup-lr", type=float, default=1e-3)
    parser.add_argument("--train-wcons", action="store_true",
                        help="Also train w_cons during warm-up (heavier "
                             "variant). Default trains only the gate.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    seed_everything(args.seed)

    logger.info("=" * 70)
    logger.info("EXPERIMENT 09: Retrieval-Aware Warm-Up Extension")
    logger.info("=" * 70)

    cfg = load_experiment_config(args.config)
    model, tokenizer = build_model_and_tokenizer(cfg, eager_attention=True)
    device = cfg["device"]

    n_params_b = sum(p.numel() for p in model.parameters()) / 1e9

    with open(args.facts_file) as f:
        facts = json.load(f)
    if args.max_facts:
        facts = facts[: args.max_facts]
    by_template = group_facts_by_template(facts)
    logger.info("Loaded %d facts", len(facts))

    # Control texts for BCP (base capability preservation).
    control_texts = [
        "The capital of France is Paris.",
        "Water boils at 100 degrees Celsius at sea level.",
        "Python is a programming language created by Guido van Rossum.",
    ]
    control_seqs = [
        tokenizer(t, return_tensors="pt").input_ids[0].to(device) for t in control_texts
    ]

    dws = DualWeightSystem(
        model, cfg["weights"],
        use_kv_memory_for_fast=True,
        kv_max_total_tokens=args.kv_max_tokens,
        kv_top_k=args.kv_top_k,
    )
    tagging = TaggingLayer(dws.model, cfg["tagging"], model_params_billions=n_params_b)

    def _bcp_now() -> float:
        """BCP of the current model state (memory active) vs the empty-bank base."""
        ppl_cur = evaluate_perplexity(dws.model, control_seqs, device=device)
        return compute_bcp(ppl_cur, ppl_ref)

    # BCP reference: perplexity with an empty bank (unmodified base capability).
    dws.set_mode("target_inference")
    dws.set_kv_enabled(False)
    ppl_ref = evaluate_perplexity(dws.model, control_seqs, device=device)
    dws.set_kv_enabled(True)

    # ---- Row 1: SLEEP WITHOUT warm-up (identity gate) -------------------
    _write_episodes(dws, tagging, facts, tokenizer, device)
    dws.set_mode("target_inference")
    dws.set_kv_enabled(True)
    before = _evaluate(model, tokenizer, facts, by_template, device, "no-warmup")
    before["bcp"] = _bcp_now()

    # ---- Warm-up ---------------------------------------------------------
    corpus = _load_corpus(args.corpus_file, tokenizer, device)
    trainer = WarmupTrainer(
        dws, tokenizer, device=device, train_wcons=args.train_wcons,
    )
    warmup_result = trainer.run(
        corpus, n_steps=args.warmup_steps, lr=args.warmup_lr, seed=args.seed,
    )
    logger.info("Warm-up gate scales: %s", warmup_result.gate_scales)

    # ---- Row 2: SLEEP WITH warm-up (trained gate) -----------------------
    _write_episodes(dws, tagging, facts, tokenizer, device)
    dws.set_mode("target_inference")
    dws.set_kv_enabled(True)
    after = _evaluate(model, tokenizer, facts, by_template, device, "with-warmup")
    after["bcp"] = _bcp_now()

    # ---- Report ----------------------------------------------------------
    payload = {
        "experiment": "09_warmup_extension",
        "seed": args.seed,
        "config_path": args.config,
        "n_facts": len(facts),
        "kv_top_k": args.kv_top_k,
        "warmup_steps": args.warmup_steps,
        "warmup_lr": args.warmup_lr,
        "train_wcons": args.train_wcons,
        "ppl_reference": ppl_ref,
        "without_warmup": before,
        "with_warmup": after,
        "warmup": warmup_result.as_dict(),
        "dra_delta": after["dra"] - before["dra"],
        "mc_delta": after["mc_accuracy"] - before["mc_accuracy"],
    }
    out = save_results(payload, results_path("warmup_ext", args.output))

    print("\n" + "=" * 70)
    print("WARM-UP EXTENSION RESULT")
    print("=" * 70)
    print(f"  {'Condition':22s} {'MC':>7s} {'cloze':>7s} {'DRA':>7s} {'BCP':>7s}")
    print(f"  {'SLEEP without warm-up':22s} {before['mc_accuracy']:>7.3f} "
          f"{before['cloze_accuracy']:>7.3f} {before['dra']:>7.3f} {before['bcp']:>7.3f}")
    print(f"  {'SLEEP with warm-up':22s} {after['mc_accuracy']:>7.3f} "
          f"{after['cloze_accuracy']:>7.3f} {after['dra']:>7.3f} {after['bcp']:>7.3f}")
    print(f"\n  DRA delta (gap change): {payload['dra_delta']:+.3f}")
    print(f"  Mean gate scale: {sum(warmup_result.gate_scales.values())/max(len(warmup_result.gate_scales),1):.3f}")
    print(f"\n  Results written to {out}")

    dws.cleanup()


if __name__ == "__main__":
    main()
