#!/bin/bash
# Phase B: full SLEEP pipeline with the localisation consolidation recipe.
# Qwen-7B x distill (its ladder winner) x 5 seeds; Llama-8B x paraphrase-CE
# (its ladder winner) x 2 seeds. 200 facts, 1500 train steps, greedy eval.
set -u
cd /workspace/sleep-research
export PYTHONUNBUFFERED=1
FACTS=experiments/data/facts_200_para.json
R=/workspace/results; L=/workspace/logs
mkdir -p "$R" "$L"

for S in 0 1 2 3 4; do
  python experiments/scripts/13_full_pipeline_v2.py \
    --config experiments/configs/qwen7b_mlp_mid.yaml --facts-file "$FACTS" \
    --recipe distill --train-steps 1500 --seed "$S" \
    --output "$R/pipev2_qwen_distill_s${S}.json" \
    > "$L/pipev2_qwen_distill_s${S}.log" 2>&1 || true
  echo "PIPEV2_QWEN_S${S}_DONE $(date)"
done
for S in 0 1; do
  python experiments/scripts/13_full_pipeline_v2.py \
    --config experiments/configs/llama3_8b_mlp_mid.yaml --facts-file "$FACTS" \
    --recipe paraphrase_ce --train-steps 1500 --seed "$S" \
    --output "$R/pipev2_llama_ce_s${S}.json" \
    > "$L/pipev2_llama_ce_s${S}.log" 2>&1 || true
  echo "PIPEV2_LLAMA_S${S}_DONE $(date)"
done
echo "PHASE_B_DONE $(date)"
