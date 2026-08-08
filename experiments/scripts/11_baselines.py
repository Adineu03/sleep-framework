"""
Experiment 11: Missing baselines on the fact benchmark (mentor P2 item #5).

The paper opens by contrasting with RAG but never measured it, and reports no
EWC-only or in-context reference points. This script runs the three baselines
the review flagged, on the same facts and the same DRA/BCP metrics used for
SLEEP, so the comparison table is complete:

  - RAG: retrieve the fact by similarity, answer with it in context.
  - EWC-only: naive LoRA + EWC penalty, nothing else.
  - In-context: prepend the gold fact to the prompt (zero training).

RAG and in-context run on any model; EWC-only trains LoRA on v_proj/o_proj so
needs a Qwen/Llama-family model. Run on RunPod with the qwen7b config.

USAGE:
    python experiments/scripts/11_baselines.py \
        --config experiments/configs/qwen7b.yaml \
        --facts-file experiments/data/facts_200.json \
        --baselines rag ewc incontext --seed 0
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

from sleep.config import SLEEPConfig
from sleep.evaluation.baselines import (
    EWCOnlyBaseline,
    InContextBaseline,
    NaiveLoRABaseline,
    RAGBaseline,
)
from sleep.evaluation.preservation import compute_bcp, evaluate_perplexity
from sleep.utils.logging import setup_logging, get_logger

setup_logging()
logger = get_logger("experiment.11")

_CONTROL_TEXTS = [
    "The capital of France is Paris.",
    "Water boils at 100 degrees Celsius at sea level.",
    "Python is a programming language created by Guido van Rossum.",
]


def _score(generation: str, keywords: list[str]) -> float:
    """Fraction of expected keywords appearing in the generation."""
    if not keywords:
        return 0.0
    g = generation.lower()
    return sum(1 for k in keywords if k.lower() in g) / len(keywords)


def _dra_over_facts(answer_fn, facts) -> float:
    """Mean keyword-coverage (DRA) of ``answer_fn(fact) -> str`` over facts."""
    return sum(_score(answer_fn(fact), fact.get("keywords", [])) for fact in facts) / max(len(facts), 1)


def run_rag(model, tokenizer, facts, device) -> dict:
    rag = RAGBaseline(model, tokenizer, device=device)
    for fact in facts:
        rag.add_document(fact["text"], doc_id=fact["id"])
    dra = _dra_over_facts(
        lambda f: rag.query(f["test_prompt"], top_k=3, max_new_tokens=50), facts
    )
    return {"dra": dra, "bcp": 1.0}  # RAG never touches weights → BCP == 1.


def run_incontext(model, tokenizer, facts, device) -> dict:
    ic = InContextBaseline(model, tokenizer, device=device)
    for fact in facts:
        ic.add_fact(fact["id"], fact["text"])
    dra = _dra_over_facts(
        lambda f: ic.query(f["test_prompt"], fact_id=f["id"], max_new_tokens=50), facts
    )
    return {"dra": dra, "bcp": 1.0}  # no training → BCP == 1.


def run_ewc(model, tokenizer, facts, device, config, steps: int) -> dict:
    control_seqs = [
        tokenizer(t, return_tensors="pt").input_ids[0].to(device) for t in _CONTROL_TEXTS
    ]
    ppl_before = evaluate_perplexity(model, control_seqs, device=device)

    ewc = EWCOnlyBaseline(model, tokenizer, config, device=device)
    texts = [f["text"] for f in facts]
    # Train, consolidating after each third so the EWC anchor is meaningful.
    third = max(1, len(texts) // 3)
    for i in range(steps):
        ewc.train_on_input(texts[i % len(texts)])
        if i > 0 and i % third == 0:
            ewc.consolidate_task(texts[max(0, i - third):i])

    dra = _dra_over_facts(
        lambda f: ewc.generate(f["test_prompt"], max_new_tokens=50), facts
    )
    ppl_after = evaluate_perplexity(model, control_seqs, device=device)
    return {"dra": dra, "bcp": compute_bcp(ppl_after, ppl_before)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--facts-file", type=str, required=True)
    parser.add_argument("--max-facts", type=int, default=None)
    parser.add_argument("--baselines", nargs="+",
                        default=["rag", "ewc", "incontext"],
                        choices=["rag", "ewc", "incontext"])
    parser.add_argument("--ewc-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    seed_everything(args.seed)
    logger.info("=" * 70)
    logger.info("EXPERIMENT 11: Missing baselines (%s)", ", ".join(args.baselines))
    logger.info("=" * 70)

    cfg = load_experiment_config(args.config)
    with open(args.facts_file) as f:
        facts = json.load(f)
    if args.max_facts:
        facts = facts[: args.max_facts]
    logger.info("Loaded %d facts", len(facts))

    results: dict[str, dict] = {}
    for name in args.baselines:
        # Reload the model per baseline so training baselines start clean.
        model, tokenizer = build_model_and_tokenizer(cfg, eager_attention=False)
        device = cfg["device"]
        logger.info("Running baseline: %s", name)
        if name == "rag":
            results["rag"] = run_rag(model, tokenizer, facts, device)
        elif name == "incontext":
            results["incontext"] = run_incontext(model, tokenizer, facts, device)
        elif name == "ewc":
            sconfig = SLEEPConfig()
            sconfig.weights = cfg["weights"]
            results["ewc"] = run_ewc(model, tokenizer, facts, device, sconfig, args.ewc_steps)
        del model

    payload = {
        "experiment": "11_baselines",
        "seed": args.seed,
        "config_path": args.config,
        "n_facts": len(facts),
        "results": results,
    }
    out = save_results(payload, results_path("baselines", args.output))

    print("\n" + "=" * 50)
    print("BASELINE RESULTS")
    print("=" * 50)
    print(f"  {'baseline':14s} {'DRA':>8s} {'BCP':>8s}")
    for name, r in results.items():
        print(f"  {name:14s} {r['dra']:>8.3f} {r['bcp']:>8.3f}")
    print(f"\n  Wrote {out}")


if __name__ == "__main__":
    main()
