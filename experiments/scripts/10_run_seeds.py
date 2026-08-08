"""
Experiment 10: Multi-seed runner and aggregator (mentor P1 item #1).

No major venue accepts single-seed results. This runner executes a target
experiment script across several seeds — each in a fresh subprocess so model
and adapter initialisation are genuinely independent — collects each run's JSON
output, and reports mean +/- standard deviation for the metrics that matter.

It is generic: any script that accepts ``--seed`` and ``--output`` and writes a
flat JSON dict of numeric metrics works (07_full_kv_pipeline, 08_multi_cycle,
09_warmup_extension all do). Extra arguments after ``--`` are passed through to
the target unchanged.

USAGE:
    # Recognition signal across 5 seeds:
    python experiments/scripts/10_run_seeds.py \
        --script experiments/scripts/07_full_kv_pipeline.py \
        --seeds 0 1 2 3 4 --metrics dra bcp n_consolidated \
        --out experiments/results/seeds_recognition.json \
        -- --config experiments/configs/qwen7b.yaml \
           --facts-file experiments/data/facts_200.json --kv-top-k 64

    # Warm-up extension DRA delta across 3 seeds:
    python experiments/scripts/10_run_seeds.py \
        --script experiments/scripts/09_warmup_extension.py \
        --seeds 0 1 2 --metrics dra_delta mc_delta \
        -- --config experiments/configs/qwen7b.yaml \
           --facts-file experiments/data/facts_200.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from sleep.utils.seed import aggregate_over_seeds
from sleep.utils.logging import setup_logging, get_logger

setup_logging()
logger = get_logger("experiment.10")


def _get_path(d: dict, dotted: str):
    """Fetch a possibly-nested numeric metric by dot path (``a.b.c``)."""
    cur = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def run_one_seed(script: str, seed: int, passthrough: list[str], out_path: str) -> dict | None:
    """Run the target script for one seed; return its parsed JSON (or None)."""
    cmd = [sys.executable, script, "--seed", str(seed), "--output", out_path, *passthrough]
    logger.info("seed %d: %s", seed, " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        logger.error(
            "seed %d FAILED (exit %d)\n--- stderr tail ---\n%s",
            seed, proc.returncode, "\n".join(proc.stderr.splitlines()[-15:]),
        )
        return None
    if not os.path.exists(out_path):
        logger.error("seed %d produced no output file %s", seed, out_path)
        return None
    with open(out_path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", required=True, help="Target experiment script.")
    parser.add_argument("--seeds", type=int, nargs="+", required=True,
                        help="Seeds to run, e.g. --seeds 0 1 2 3 4.")
    parser.add_argument("--metrics", nargs="+", required=True,
                        help="Metric keys (dot-paths) to aggregate across seeds.")
    parser.add_argument("--out", type=str, default=None,
                        help="Where to write the aggregate JSON.")
    parser.add_argument("passthrough", nargs=argparse.REMAINDER,
                        help="Args after `--` passed verbatim to the target.")
    args = parser.parse_args()

    passthrough = args.passthrough
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    per_seed: dict[int, dict] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for seed in args.seeds:
            out_path = os.path.join(tmp, f"seed_{seed}.json")
            result = run_one_seed(args.script, seed, passthrough, out_path)
            if result is not None:
                per_seed[seed] = result

    if not per_seed:
        logger.error("No seed produced a result; aborting aggregation.")
        sys.exit(1)

    # Aggregate each requested metric across the seeds that produced it.
    aggregates: dict[str, dict] = {}
    for metric in args.metrics:
        values, seeds_used = [], []
        for seed, result in sorted(per_seed.items()):
            v = _get_path(result, metric)
            if isinstance(v, (int, float)):
                values.append(float(v))
                seeds_used.append(seed)
        if values:
            agg = aggregate_over_seeds(values)
            aggregates[metric] = {**agg.as_dict(), "seeds": seeds_used}

    payload = {
        "script": args.script,
        "seeds_requested": args.seeds,
        "seeds_succeeded": sorted(per_seed.keys()),
        "passthrough": passthrough,
        "aggregates": aggregates,
        "per_seed": per_seed,
    }

    out = args.out
    if out is None:
        results_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "results")
        )
        os.makedirs(results_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(args.script))[0]
        out = os.path.join(results_dir, f"seeds_{stem}.json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print("\n" + "=" * 64)
    print(f"MULTI-SEED AGGREGATE  ({len(per_seed)}/{len(args.seeds)} seeds ok)")
    print("=" * 64)
    print(f"  {'metric':24s} {'mean':>10s} {'std':>10s}  n")
    for metric, agg in aggregates.items():
        print(f"  {metric:24s} {agg['mean']:>10.4f} {agg['std']:>10.4f}  {agg['n']}")
    print(f"\n  Wrote {out}")


if __name__ == "__main__":
    main()
