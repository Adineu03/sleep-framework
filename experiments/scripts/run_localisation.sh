#!/bin/bash
# Localisation campaign (mentor revision 2026-08-11).
#
# The five-arm ladder x 2 seeds x 2 models (the two with the strongest
# in-context teachers: Qwen-7B 0.760, Llama-8B 0.987). Each arm is a fresh
# process for clean isolation. ~20 runs x ~15-20 min ~= 5-6 h ~= $5 on A6000.
#
# Prereq: pod_setup.sh has run (deps + models) and the code tree is at
# /workspace/sleep-research (facts_200_para.json ships with it).
#
# Usage:  nohup bash run_localisation.sh > /workspace/logs/localisation.log 2>&1 &
set -u
cd /workspace/sleep-research
export PYTHONUNBUFFERED=1

FACTS=experiments/data/facts_200_para.json
R=/workspace/results
L=/workspace/logs
mkdir -p "$R" "$L"

ARMS="attn_top mlp_mid mlp_mid_para distill_mlp_mid distill_attn_top"

run_model () {
  local KEY=$1 CFG=$2
  for ARM in $ARMS; do
    for S in 0 1; do
      echo "=== [$KEY] arm=$ARM seed=$S ==="
      python experiments/scripts/12_localisation.py \
        --config "$CFG" --facts-file "$FACTS" \
        --arm "$ARM" --max-facts 50 --steps 1200 --seed "$S" \
        --output "$R/loc_${KEY}_${ARM}_s${S}.json" \
        > "$L/loc_${KEY}_${ARM}_s${S}.log" 2>&1 || true
      echo "LOC_${KEY}_${ARM}_S${S}_DONE $(date)"
    done
  done
  echo "LOC_${KEY}_ALL_DONE $(date)"
}

run_model qwen7b   experiments/configs/qwen7b.yaml
run_model llama3_8b experiments/configs/llama3_8b.yaml

echo "LOCALISATION_ALL_DONE $(date)"
