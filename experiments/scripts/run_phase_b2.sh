#!/bin/bash
# Phase B revised: isolate the safety machinery's effect on the new recipe.
#   designed  — machinery as designed (delta_max 0.01, phi profile): s1, s2
#               (s0 already ran pre-patch; same condition)
#   relaxed   — clip + plasticity scaling disabled: s0, s1, s2
# Then Llama: one of each.
set -u
cd /workspace/sleep-research
export PYTHONUNBUFFERED=1
FACTS=experiments/data/facts_200_para.json
R=/workspace/results; L=/workspace/logs

for S in 1 2; do
  python experiments/scripts/13_full_pipeline_v2.py \
    --config experiments/configs/qwen7b_mlp_mid.yaml --facts-file "$FACTS" \
    --recipe distill --train-steps 1500 --seed "$S" \
    --output "$R/pipev2_qwen_designed_s${S}.json" \
    > "$L/pipev2_qwen_designed_s${S}.log" 2>&1 || true
  echo "B2_QWEN_DESIGNED_S${S}_DONE $(date)"
done
for S in 0 1 2; do
  python experiments/scripts/13_full_pipeline_v2.py \
    --config experiments/configs/qwen7b_mlp_mid.yaml --facts-file "$FACTS" \
    --recipe distill --train-steps 1500 --seed "$S" \
    --override-delta-max 1.0 --override-phi-min 1.0 \
    --output "$R/pipev2_qwen_relaxed_s${S}.json" \
    > "$L/pipev2_qwen_relaxed_s${S}.log" 2>&1 || true
  echo "B2_QWEN_RELAXED_S${S}_DONE $(date)"
done
python experiments/scripts/13_full_pipeline_v2.py \
  --config experiments/configs/llama3_8b_mlp_mid.yaml --facts-file "$FACTS" \
  --recipe paraphrase_ce --train-steps 1500 --seed 0 \
  --output "$R/pipev2_llama_designed_s0.json" \
  > "$L/pipev2_llama_designed_s0.log" 2>&1 || true
echo "B2_LLAMA_DESIGNED_DONE $(date)"
python experiments/scripts/13_full_pipeline_v2.py \
  --config experiments/configs/llama3_8b_mlp_mid.yaml --facts-file "$FACTS" \
  --recipe paraphrase_ce --train-steps 1500 --seed 0 \
  --override-delta-max 1.0 --override-phi-min 1.0 \
  --output "$R/pipev2_llama_relaxed_s0.json" \
  > "$L/pipev2_llama_relaxed_s0.log" 2>&1 || true
echo "B2_LLAMA_RELAXED_DONE $(date)"
echo "PHASE_B2_DONE $(date)"
