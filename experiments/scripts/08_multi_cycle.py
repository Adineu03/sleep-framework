"""
Experiment 08: Multi-Cycle Continual Learning (the SLEEP design-intent test)

PURPOSE:
    Test whether SLEEP's safety machinery's value emerges over MULTIPLE sleep
    cycles on disjoint batches of facts — the actual continual-learning setting
    SLEEP was designed for.

    The single-cycle sweep (Experiment 07) showed naive LoRA outperforming
    SLEEP on a per-BCP-cost basis. But that is a 100m sprint when SLEEP is
    a marathon runner. This experiment runs disjoint-batch continual learning
    across N cycles and measures whether:
      - SLEEP's BCP stays stable while naive LoRA's BCP compounds
      - SLEEP's cumulative DRA survives across cycles while naive LoRA forgets
      - At some cycle count, SLEEP overtakes naive LoRA on either metric

TWO ARMS:
    --method sleep:        Wake (KV write) + Sleep (consolidate W_cons) per cycle.
                           Setting A by default: delta_max=0.02, lambda_ewc=50,
                           alpha_slow=1e-4, steps_per_memory=4. Tuneable via
                           override flags.
    --method naive_lora:   Direct LoRA fine-tuning on each batch with matched
                           compute (same total gradient steps as SLEEP arm).

EVALUATION (after each cycle):
    - DRA on cumulative facts seen so far (forgetting test)
    - DRA on current cycle's facts (learning test)
    - BCP on control texts (preservation test)

PRE-REGISTERED INTERPRETATION:
    Outcome A (SLEEP wins multi-cycle): SLEEP BCP < naive_lora BCP after
        cycle 3. The architecture's safety claim holds; the single-cycle
        comparison was misleading. Paper narrative: SLEEP achieves
        preservation guarantees that naive LoRA cannot.

    Outcome B (SLEEP indistinguishable from naive_lora): Both approaches
        compound BCP at similar rates. The architecture's safety machinery
        adds complexity without delivering the claimed multi-cycle benefit.
        Paper narrative: the negative result is now bulletproof — SLEEP
        underperforms even at its design-intent regime.

USAGE:
    python experiments/scripts/08_multi_cycle.py \
        --config experiments/configs/qwen7b.yaml \
        --facts-file experiments/data/facts_200.json \
        --method sleep --n-cycles 3 --batch-size 67

    python experiments/scripts/08_multi_cycle.py \
        --config experiments/configs/qwen7b.yaml \
        --facts-file experiments/data/facts_200.json \
        --method naive_lora --n-cycles 3 --batch-size 67
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from sleep.config import (
    SLEEPConfig,
    TaggingConfig,
    PRPConfig,
    WeightsConfig,
    SleepConfig,
)
from sleep.tagging import TaggingLayer
from sleep.prp import PRPSystem
from sleep.weights import DualWeightSystem
from sleep.weights.lora import _build_lora_config
from sleep.sleep_engine import SleepEngine
from sleep.evaluation.recall import evaluate_recall, RecallTestCase
from sleep.evaluation.preservation import evaluate_perplexity, compute_bcp
from sleep.utils.logging import setup_logging, get_logger

setup_logging()
logger = get_logger("experiment.08")


CONTROL_TEXTS = [
    "The capital of France is Paris. It is known for the Eiffel Tower and the Louvre museum.",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    "The Python programming language was created by Guido van Rossum and released in 1991.",
]


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(config_path: str) -> dict:
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
    }


# ---------------------------------------------------------------------------
# Per-cycle: SLEEP arm
# ---------------------------------------------------------------------------


def run_sleep_cycle(
    *,
    cycle_idx: int,
    fact_batch: list,
    dws: DualWeightSystem,
    tagging: TaggingLayer,
    prp: PRPSystem,
    sleep_engine: SleepEngine,
    tokenizer,
    device: str,
    replay_strategy: str,
) -> dict:
    """One SLEEP wake + sleep cycle on a single batch of facts."""
    logger.info("=" * 70)
    logger.info("[SLEEP Cycle %d] Wake phase: %d facts", cycle_idx, len(fact_batch))
    logger.info("=" * 70)

    # Wake: tag and write KV per fact
    dws.set_mode("wake_inference")
    original_tokens_map: dict = {}
    sources_written: set = set()
    for fact in fact_batch:
        token_ids = tokenizer.encode(
            fact["text"], return_tensors="pt",
        ).squeeze(0).to(device)
        original_tokens_map[fact["id"]] = token_ids

        new_tags = tagging.process_input(token_ids, source_id=fact["id"])
        for tag in new_tags:
            span_start, span_end, source_id = tag.ctx
            if tag.e0 <= 1.5 or source_id in sources_written:
                continue
            try:
                dws.write_to_kv_bank(
                    tag_id=source_id,
                    token_ids=token_ids,
                    span_start=0,
                    span_end=int(token_ids.shape[0]),
                    device=device,
                )
                sources_written.add(source_id)
            except RuntimeError as exc:
                logger.warning(
                    "[Cycle %d] KV bank full at %s: %s",
                    cycle_idx, source_id, exc,
                )
                break

    logger.info(
        "[Cycle %d] Wake summary: %d facts -> %d KV writes; bank: %d/%d tokens",
        cycle_idx, len(fact_batch), len(sources_written),
        dws.kv_bank.n_total_tokens, dws.kv_bank.max_total_tokens,
    )

    # PRP scoring + allocation
    all_tags = tagging.active_tags
    prp_result = prp.update(all_tags, current_step=tagging.step, force_crossref=True)
    candidates = prp.get_consolidation_candidates(all_tags)
    logger.info(
        "[Cycle %d] PRP: %d allocated, %d consolidation candidates",
        cycle_idx, prp_result["allocated"], len(candidates),
    )

    # Sleep
    sleep_result = sleep_engine.run_cycle(
        candidates=candidates,
        original_tokens_map=original_tokens_map,
        key_projection=tagging.key_projection,
    )

    return {
        "cycle": cycle_idx,
        "n_facts_in_batch": len(fact_batch),
        "n_kv_writes": len(sources_written),
        "n_prp_allocated": prp_result["allocated"],
        "n_candidates": len(candidates),
        "n_consolidated": sleep_result.get("n_consolidated", 0),
        "n_failed": sleep_result.get("n_failed", 0),
        "rolled_back": sleep_result.get("rolled_back", False),
        "training_stats": sleep_result.get("training_stats", {}),
    }


# ---------------------------------------------------------------------------
# Per-cycle: Naive LoRA arm
# ---------------------------------------------------------------------------


def run_naive_lora_cycle(
    *,
    cycle_idx: int,
    fact_batch: list,
    peft_model,
    tokenizer,
    n_steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: str,
    seed: int,
) -> dict:
    """One round of naive LoRA fine-tuning on a single batch."""
    logger.info("=" * 70)
    logger.info(
        "[Naive LoRA Cycle %d] Training: %d facts, %d steps, bs=%d, lr=%g",
        cycle_idx, len(fact_batch), n_steps, batch_size, learning_rate,
    )
    logger.info("=" * 70)

    rng = random.Random(seed + cycle_idx)
    trainable_params = [p for p in peft_model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("Naive LoRA arm has no trainable parameters")

    optimizer = torch.optim.AdamW(
        trainable_params, lr=learning_rate, weight_decay=weight_decay,
    )

    # Pre-tokenize the batch
    tokenized = []
    for fact in fact_batch:
        ids = tokenizer.encode(
            fact["text"], return_tensors="pt", padding=False,
        ).squeeze(0).to(device)
        tokenized.append(ids)
    if not tokenized:
        return {"cycle": cycle_idx, "n_steps": 0, "n_facts": 0,
                "first_loss": None, "tail_loss_mean": None}

    peft_model.train()
    losses = []
    for step in range(n_steps):
        # Sample with replacement
        indices = [rng.randrange(len(tokenized)) for _ in range(batch_size)]
        batch_ids = [tokenized[i] for i in indices]
        max_len = max(t.shape[0] for t in batch_ids)
        # Right-pad with eos
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        padded = torch.full(
            (len(batch_ids), max_len), pad_id, dtype=torch.long, device=device,
        )
        attn_mask = torch.zeros_like(padded)
        for j, ids in enumerate(batch_ids):
            padded[j, : ids.shape[0]] = ids
            attn_mask[j, : ids.shape[0]] = 1
        labels = padded.clone()
        labels[attn_mask == 0] = -100

        optimizer.zero_grad(set_to_none=True)
        outputs = peft_model(
            input_ids=padded, attention_mask=attn_mask, labels=labels,
        )
        loss = outputs.loss
        if not torch.isfinite(loss):
            logger.warning(
                "[Cycle %d] Non-finite loss at step %d: %s",
                cycle_idx, step, loss,
            )
            break
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    peft_model.eval()
    first_loss = losses[0] if losses else None
    tail = losses[-max(1, len(losses) // 10):] if losses else []
    tail_mean = sum(tail) / len(tail) if tail else None

    logger.info(
        "[Naive LoRA Cycle %d] %d steps complete; first_loss=%.4f tail_mean=%.4f",
        cycle_idx, len(losses),
        first_loss if first_loss is not None else float("nan"),
        tail_mean if tail_mean is not None else float("nan"),
    )
    return {
        "cycle": cycle_idx,
        "n_facts": len(fact_batch),
        "n_steps": len(losses),
        "first_loss": first_loss,
        "tail_loss_mean": tail_mean,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_after_cycle(
    *,
    cycle_idx: int,
    cumulative_facts: list,
    current_cycle_facts: list,
    peft_model,
    tokenizer,
    control_ids: list,
    baseline_ppl: float,
    device: str,
    arm: str,
    dws: DualWeightSystem | None = None,
) -> dict:
    """Measure DRA on cumulative + current-cycle facts, BCP on controls."""
    # For SLEEP, ensure KV is disabled for clean recall measurement
    if dws is not None and getattr(dws, "use_kv_memory_for_fast", False):
        dws.set_kv_enabled(False)
        dws.set_mode("target_inference")
    peft_model.eval()

    # Cumulative DRA
    cum_cases = [
        RecallTestCase(
            source_id=f["id"], prompt=f["test_prompt"],
            expected_keywords=f["keywords"],
        )
        for f in cumulative_facts
    ]
    cum_recall = evaluate_recall(
        peft_model, tokenizer, cum_cases, max_new_tokens=50, device=device,
    )

    # Current-cycle DRA
    cur_cases = [
        RecallTestCase(
            source_id=f["id"], prompt=f["test_prompt"],
            expected_keywords=f["keywords"],
        )
        for f in current_cycle_facts
    ]
    cur_recall = evaluate_recall(
        peft_model, tokenizer, cur_cases, max_new_tokens=50, device=device,
    )

    # BCP
    post_ppl = evaluate_perplexity(peft_model, control_ids, device=device)
    bcp = compute_bcp(post_ppl, baseline_ppl)

    n_hits_cum = sum(1 for c in cum_recall["per_case"] if c["score"] > 0)
    n_hits_cur = sum(1 for c in cur_recall["per_case"] if c["score"] > 0)

    logger.info(
        "[%s Cycle %d EVAL] DRA cumulative=%.3f (%d/%d hits); "
        "DRA current=%.3f (%d/%d hits); BCP=%.4f",
        arm, cycle_idx,
        cum_recall["dra"], n_hits_cum, len(cum_cases),
        cur_recall["dra"], n_hits_cur, len(cur_cases),
        bcp,
    )

    return {
        "cycle": cycle_idx,
        "arm": arm,
        "cumulative_n": len(cumulative_facts),
        "current_n": len(current_cycle_facts),
        "dra_cumulative": cum_recall["dra"],
        "dra_current": cur_recall["dra"],
        "n_hits_cumulative": n_hits_cum,
        "n_hits_current": n_hits_cur,
        "bcp": bcp,
        "post_ppl": post_ppl,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--facts-file", type=str, required=True)
    parser.add_argument("--method", choices=["sleep", "naive_lora"], required=True)
    parser.add_argument("--n-cycles", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=67,
                        help="Facts per cycle. Total facts processed = "
                             "n_cycles * batch_size, capped by dataset size.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    # SLEEP-arm overrides (Setting A defaults — mild relax)
    parser.add_argument("--override-delta-max", type=float, default=0.02)
    parser.add_argument("--override-lambda-ewc", type=float, default=50.0)
    parser.add_argument("--override-alpha-slow", type=float, default=1e-4)
    parser.add_argument("--override-steps-per-memory", type=int, default=4)
    parser.add_argument("--kv-top-k", type=int, default=64)
    parser.add_argument("--replay-strategy",
                        choices=["generative", "original"],
                        default="original",
                        help="Use 'original' to strip the replay-quality "
                             "confound; SLEEP arm trains W_cons on actual fact "
                             "tokens. (Default original; matches Experiment 07 "
                             "decisive-diagnostic settings.)")
    # Naive-LoRA-arm settings
    parser.add_argument("--naive-lora-steps", type=int, default=200,
                        help="Training steps per cycle for naive LoRA arm. "
                             "Default 200 matches SLEEP-arm steps when "
                             "n_replays=50 and steps_per_memory=4.")
    parser.add_argument("--naive-lora-batch-size", type=int, default=32)
    parser.add_argument("--naive-lora-lr", type=float, default=1e-4)
    parser.add_argument("--naive-lora-weight-decay", type=float, default=0.01)
    args = parser.parse_args()

    timestamp = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.output is None:
        results_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../results"),
        )
        os.makedirs(results_dir, exist_ok=True)
        args.output = os.path.join(
            results_dir,
            f"multi_cycle_{args.method}_{args.n_cycles}c_{timestamp}.json",
        )

    print("=" * 78)
    print(f"EXPERIMENT 08: Multi-Cycle Continual Learning")
    print(f"  Method:     {args.method}")
    print(f"  Cycles:     {args.n_cycles}")
    print(f"  Batch size: {args.batch_size}")
    print("=" * 78)

    cfg = load_config(args.config)
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(cfg["dtype"], torch.float32)

    # Apply SLEEP overrides
    cfg["weights"].delta_max = args.override_delta_max
    cfg["weights"].lambda_ewc = args.override_lambda_ewc
    cfg["weights"].alpha_slow = args.override_alpha_slow
    cfg["sleep"].steps_per_memory = args.override_steps_per_memory

    # Load facts
    with open(args.facts_file) as f:
        all_facts = json.load(f)
    n_total = min(args.n_cycles * args.batch_size, len(all_facts))
    all_facts = all_facts[:n_total]
    print(f"  Loaded {len(all_facts)} facts (will run {args.n_cycles} cycles "
          f"of {args.batch_size} each)")

    # Load model
    print(f"\nLoading model: {cfg['model_name']} (eager attention)")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], torch_dtype=dtype, attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = model.to(cfg["device"])

    n_params_b = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  Model: {n_params_b:.3f}B params")

    # Build subsystems based on arm
    dws: DualWeightSystem | None = None
    tagging = None
    prp = None
    sleep_engine = None
    peft_model = None

    if args.method == "sleep":
        dws = DualWeightSystem(
            model, cfg["weights"],
            use_kv_memory_for_fast=True,
            kv_max_total_tokens=20_000,
            kv_top_k=args.kv_top_k,
        )
        peft_model = dws.model
        tagging = TaggingLayer(
            peft_model, cfg["tagging"], model_params_billions=n_params_b,
        )
        prp = PRPSystem(
            cfg["prp"],
            budget=int(cfg["prp"].c_prp * n_params_b),
            revision_bonus=0.3,
        )
        # Compute baseline mu_surprise (unused with replay_strategy=original
        # but the engine still asks for it)
        dws.set_mode("target_inference")
        peft_model.eval()
        from sleep.sleep_engine.quality import compute_baseline_surprise
        control_ids_list = [
            tokenizer.encode(t, return_tensors="pt").squeeze(0).to(cfg["device"])
            for t in CONTROL_TEXTS
        ]
        mu_surprise = compute_baseline_surprise(
            peft_model, control_ids_list, device=cfg["device"],
        )
        sleep_engine = SleepEngine(
            dual_weights=dws,
            tokenizer=tokenizer,
            sleep_config=cfg["sleep"],
            weights_config=cfg["weights"],
            mu_surprise=mu_surprise,
            device=cfg["device"],
            replay_strategy=args.replay_strategy,
        )
    elif args.method == "naive_lora":
        # Build a single LoRA adapter directly via peft (no SLEEP machinery).
        from peft import LoraConfig, get_peft_model
        lora_config = _build_lora_config(model, cfg["weights"])
        peft_model = get_peft_model(model, lora_config)
        peft_model.to(cfg["device"])
        peft_model.eval()
        control_ids_list = [
            tokenizer.encode(t, return_tensors="pt").squeeze(0).to(cfg["device"])
            for t in CONTROL_TEXTS
        ]

    # Baseline PPL (before any training)
    print("\nMeasuring baseline PPL on controls...")
    if args.method == "sleep":
        peft_model = dws.model
        dws.set_kv_enabled(False)
        dws.set_mode("target_inference")
    peft_model.eval()
    baseline_ppl = evaluate_perplexity(
        peft_model, control_ids_list, device=cfg["device"],
    )
    print(f"  Baseline PPL: {baseline_ppl:.4f}")

    # ===========================================================
    # Main multi-cycle loop
    # ===========================================================
    cycle_results: list = []
    eval_results: list = []
    cumulative_facts: list = []

    for cycle_idx in range(1, args.n_cycles + 1):
        start = (cycle_idx - 1) * args.batch_size
        end = min(start + args.batch_size, len(all_facts))
        fact_batch = all_facts[start:end]
        if not fact_batch:
            print(f"\nCycle {cycle_idx}: no facts left, stopping.")
            break

        if args.method == "sleep":
            cycle_info = run_sleep_cycle(
                cycle_idx=cycle_idx,
                fact_batch=fact_batch,
                dws=dws,
                tagging=tagging,
                prp=prp,
                sleep_engine=sleep_engine,
                tokenizer=tokenizer,
                device=cfg["device"],
                replay_strategy=args.replay_strategy,
            )
        else:
            cycle_info = run_naive_lora_cycle(
                cycle_idx=cycle_idx,
                fact_batch=fact_batch,
                peft_model=peft_model,
                tokenizer=tokenizer,
                n_steps=args.naive_lora_steps,
                batch_size=args.naive_lora_batch_size,
                learning_rate=args.naive_lora_lr,
                weight_decay=args.naive_lora_weight_decay,
                device=cfg["device"],
                seed=args.seed,
            )
        cycle_results.append(cycle_info)

        cumulative_facts.extend(fact_batch)
        eval_info = evaluate_after_cycle(
            cycle_idx=cycle_idx,
            cumulative_facts=cumulative_facts,
            current_cycle_facts=fact_batch,
            peft_model=peft_model if args.method == "naive_lora" else dws.model,
            tokenizer=tokenizer,
            control_ids=control_ids_list,
            baseline_ppl=baseline_ppl,
            device=cfg["device"],
            arm=args.method,
            dws=dws,
        )
        eval_results.append(eval_info)

    # ===========================================================
    # Summary table
    # ===========================================================
    print("\n" + "=" * 78)
    print(f"MULTI-CYCLE SUMMARY ({args.method})")
    print("=" * 78)
    print(f"{'Cycle':<6} {'Cum N':<7} {'DRA cum':<10} {'DRA cur':<10} {'BCP':<10}")
    print("-" * 50)
    for ev in eval_results:
        print(
            f"{ev['cycle']:<6} {ev['cumulative_n']:<7} "
            f"{ev['dra_cumulative']:<10.4f} {ev['dra_current']:<10.4f} "
            f"{ev['bcp']:<10.4f}"
        )

    # ===========================================================
    # Save JSON
    # ===========================================================
    payload = {
        "experiment": "08_multi_cycle",
        "timestamp": timestamp,
        "method": args.method,
        "n_cycles": args.n_cycles,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "facts_file": args.facts_file,
        "config_path": args.config,
        "overrides": {
            "delta_max": args.override_delta_max,
            "lambda_ewc": args.override_lambda_ewc,
            "alpha_slow": args.override_alpha_slow,
            "steps_per_memory": args.override_steps_per_memory,
            "kv_top_k": args.kv_top_k,
            "replay_strategy": args.replay_strategy,
        },
        "naive_lora_settings": {
            "steps": args.naive_lora_steps,
            "batch_size": args.naive_lora_batch_size,
            "lr": args.naive_lora_lr,
            "weight_decay": args.naive_lora_weight_decay,
        } if args.method == "naive_lora" else None,
        "baseline_ppl": baseline_ppl,
        "cycle_results": cycle_results,
        "eval_results": eval_results,
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
