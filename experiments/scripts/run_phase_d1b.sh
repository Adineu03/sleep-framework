#!/bin/bash
# Phase D1b: Llama clip ladder (pre-registered per-family tuning, extended).
# D1 showed Llama at moderate (delta_max 0.1): trained-subset DRA 0.659 but
# BCP 4.35 -- great recall, too much bleed. This adds delta_max 0.05 and 0.03.
# Selection rule (stated BEFORE these runs): Llama's matrix setting is the
# delta in {0.1, 0.05, 0.03} maximizing trained-subset DRA subject to
# BCP <= 1.5; if none satisfies, best-effort setting + "not yet tuned" flag.
# Waits for the main D1 queue to finish before touching the GPU.
set -u
cd /workspace/sleep-research
export PYTHONUNBUFFERED=1
FACTS=experiments/data/facts_200_para.json
CFG=experiments/configs/llama3_8b_mlp_mid.yaml
R=/workspace/results; L=/workspace/logs

until grep -q PHASE_D1_DONE "$L/phase_d1_main.log" 2>/dev/null; do sleep 60; done

for DM in 0.05 0.03; do
  TAG=${DM/0./d}
  python experiments/scripts/13_full_pipeline_v2.py \
    --config "$CFG" --facts-file "$FACTS" \
    --recipe paraphrase_ce --train-steps 1500 --seed 0 \
    --override-delta-max "$DM" \
    --output "$R/d_llama_pce_${TAG}_s0.json" \
    > "$L/d_llama_pce_${TAG}_s0.log" 2>&1 || true
  echo "D1B_LLAMA_${TAG}_DONE $(date)"
done
echo "PHASE_D1B_DONE $(date)"
