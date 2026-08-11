#!/bin/bash
# Phase C: does the v2 recipe survive its intended regime — cycles?
# 10 cycles x 20 facts on Qwen-7B (mid-MLP config), three arms x 2 seeds:
#   v2-relaxed   clip+phi off (B2's single-cycle winner; the compounding risk)
#   v2-moderate  delta_max 0.1, phi on (middle ground B2 motivated)
#   naive        naive LoRA reference (same as ever)
set -u
cd /workspace/sleep-research
export PYTHONUNBUFFERED=1
FACTS=experiments/data/facts_200_para.json
CFG=experiments/configs/qwen7b_mlp_mid.yaml
R=/workspace/results; L=/workspace/logs

for S in 0 1; do
  python experiments/scripts/08_multi_cycle.py \
    --config "$CFG" --facts-file "$FACTS" --method sleep \
    --n-cycles 10 --batch-size 20 --seed "$S" --kv-top-k 64 \
    --replay-strategy paraphrase --train-mode distill --train-steps 400 \
    --override-delta-max 1.0 --override-phi-min 1.0 \
    --override-lambda-ewc 0 --override-alpha-slow 1e-4 \
    --output "$R/c_v2relaxed_s${S}.json" > "$L/c_v2relaxed_s${S}.log" 2>&1 || true
  echo "C_RELAXED_S${S}_DONE $(date)"
  python experiments/scripts/08_multi_cycle.py \
    --config "$CFG" --facts-file "$FACTS" --method sleep \
    --n-cycles 10 --batch-size 20 --seed "$S" --kv-top-k 64 \
    --replay-strategy paraphrase --train-mode distill --train-steps 400 \
    --override-delta-max 0.1 --override-lambda-ewc 0 --override-alpha-slow 1e-4 \
    --output "$R/c_v2moderate_s${S}.json" > "$L/c_v2moderate_s${S}.log" 2>&1 || true
  echo "C_MODERATE_S${S}_DONE $(date)"
  python experiments/scripts/08_multi_cycle.py \
    --config "$CFG" --facts-file "$FACTS" --method naive_lora \
    --n-cycles 10 --batch-size 20 --seed "$S" \
    --output "$R/c_naive_s${S}.json" > "$L/c_naive_s${S}.log" 2>&1 || true
  echo "C_NAIVE_S${S}_DONE $(date)"
done
echo "PHASE_C_DONE $(date)"
