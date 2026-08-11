#!/bin/bash
# Phase D, Stages D2-D4: the full matrix. Launch ONLY after the D1 gate is
# read; pass the Mistral train mode chosen by the gate as $1 (ce|distill),
# and the Qwen1.5B mode as $2 (default distill).
#
#   D2  single-cycle 5 seeds x 4 models, v2-moderate (delta_max 0.1, phi on).
#       Seed 0 for llama/mistral/qwen15b already exists from D1.
#   D3  multi-cycle 10x20, v2-moderate + naive, seeds 0,1 for the three new
#       models (Qwen7B pair already done in Phase C).
#   D4  EWC-only showdown: ewc_lora 10x20, seeds 0,1 on Qwen7B + Llama.
set -u
MISTRAL_MODE=${1:?"usage: run_phase_d2.sh <mistral ce|distill> [qwen15b mode] [llama delta_max]"}
QWEN15B_MODE=${2:-distill}
LLAMA_DM=${3:-0.1}   # Llama clip from the D1b ladder; others stay at 0.1
cd /workspace/sleep-research
export PYTHONUNBUFFERED=1
FACTS=experiments/data/facts_200_para.json
R=/workspace/results; L=/workspace/logs

declare -A CFG=(
  [qwen7b]=experiments/configs/qwen7b_mlp_mid.yaml
  [llama]=experiments/configs/llama3_8b_mlp_mid.yaml
  [mistral]=experiments/configs/mistral7b_mlp_mid.yaml
  [qwen15b]=experiments/configs/qwen1.5b_mlp_mid.yaml
)
declare -A RECIPE=(
  [qwen7b]=distill [llama]=paraphrase_ce
)
if [ "$MISTRAL_MODE" = "ce" ]; then RECIPE[mistral]=paraphrase_ce; else RECIPE[mistral]=distill; fi
if [ "$QWEN15B_MODE" = "ce" ]; then RECIPE[qwen15b]=paraphrase_ce; else RECIPE[qwen15b]=distill; fi
declare -A TMODE=(
  [qwen7b]=distill [llama]=ce
  [mistral]=$([ "$MISTRAL_MODE" = "ce" ] && echo ce || echo distill)
  [qwen15b]=$([ "$QWEN15B_MODE" = "ce" ] && echo ce || echo distill)
)

# ---------------- D2: single-cycle, 5 seeds ----------------
declare -A DM=(
  [qwen7b]=0.1 [llama]=$LLAMA_DM [mistral]=0.1 [qwen15b]=0.1
)
# Seed 0 for llama/mistral/qwen15b comes from D1/D1b at the chosen setting
# (d_llama_pce_<tag>_s0.json etc.); only qwen7b needs all five here.
for M in qwen7b llama mistral qwen15b; do
  if [ "$M" = qwen7b ]; then SEEDS="0 1 2 3 4"; else SEEDS="1 2 3 4"; fi
  for S in $SEEDS; do
    python experiments/scripts/13_full_pipeline_v2.py \
      --config "${CFG[$M]}" --facts-file "$FACTS" \
      --recipe "${RECIPE[$M]}" --train-steps 1500 --seed "$S" \
      --override-delta-max "${DM[$M]}" \
      --output "$R/d2_${M}_s${S}.json" > "$L/d2_${M}_s${S}.log" 2>&1 || true
    echo "D2_${M^^}_S${S}_DONE $(date)"
  done
done
echo "D2_DONE $(date)"

# ---------------- D3: multi-cycle, both arms ----------------
for M in llama mistral qwen15b; do
  for S in 0 1; do
    python experiments/scripts/08_multi_cycle.py \
      --config "${CFG[$M]}" --facts-file "$FACTS" --method sleep \
      --n-cycles 10 --batch-size 20 --seed "$S" --kv-top-k 64 \
      --replay-strategy paraphrase --train-mode "${TMODE[$M]}" --train-steps 400 \
      --override-delta-max "${DM[$M]}" --override-lambda-ewc 0 --override-alpha-slow 1e-4 \
      --output "$R/d3_${M}_v2mod_s${S}.json" > "$L/d3_${M}_v2mod_s${S}.log" 2>&1 || true
    echo "D3_${M^^}_V2MOD_S${S}_DONE $(date)"
    python experiments/scripts/08_multi_cycle.py \
      --config "${CFG[$M]}" --facts-file "$FACTS" --method naive_lora \
      --n-cycles 10 --batch-size 20 --seed "$S" \
      --output "$R/d3_${M}_naive_s${S}.json" > "$L/d3_${M}_naive_s${S}.log" 2>&1 || true
    echo "D3_${M^^}_NAIVE_S${S}_DONE $(date)"
  done
done
echo "D3_DONE $(date)"

# ---------------- D4: EWC-only showdown ----------------
for M in qwen7b llama; do
  for S in 0 1; do
    python experiments/scripts/08_multi_cycle.py \
      --config "${CFG[$M]}" --facts-file "$FACTS" --method ewc_lora \
      --n-cycles 10 --batch-size 20 --seed "$S" --ewc-lambda 100 \
      --output "$R/d4_${M}_ewc_s${S}.json" > "$L/d4_${M}_ewc_s${S}.log" 2>&1 || true
    echo "D4_${M^^}_EWC_S${S}_DONE $(date)"
  done
done
echo "D4_DONE $(date)"
echo "PHASE_D_ALL_DONE $(date)"
