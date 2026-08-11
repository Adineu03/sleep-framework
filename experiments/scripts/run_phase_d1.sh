#!/bin/bash
# Phase D, Stage D0+D1: verify gate on all four mid-MLP configs, then
# per-family tuning at the v2-moderate setting (delta_max 0.1, phi on).
#
# D0 gate: verify_kv_arch.py must pass for every config used below.
#          A failure prints VERIFY_FAILED_<name> and SKIPS that model's runs.
# D1 runs (seed 0, 200 facts, 1500 steps -- mirrors Phase B2 protocol):
#   llama     paraphrase_ce  moderate   (B2 gave designed 0.041 / relaxed 0.610
#                                        @ BCP 4.98; moderate is the missing middle)
#   mistral   distill        moderate   (family never run on v2; both recipes)
#   mistral   paraphrase_ce  moderate
#   qwen1.5b  distill        moderate   (scale point; Qwen family recipe)
# Gate D1 (pre-registered): a model advances to the matrix with the recipe
# that achieves trained-subset DRA >= 0.10 at BCP <= 1.5.
set -u
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
declare -A OK=()
for M in qwen7b llama mistral qwen15b; do
  if python experiments/scripts/verify_kv_arch.py --config "${CFG[$M]}" \
       > "$L/d0_verify_${M}.log" 2>&1; then
    OK[$M]=1; echo "D0_VERIFY_${M}_PASS $(date)"
  else
    echo "VERIFY_FAILED_${M} $(date)"
  fi
done
echo "D0_DONE $(date)"

run13 () {  # name cfg recipe seed extra...
  local NAME=$1 C=$2 RECIPE=$3 SEED=$4; shift 4
  python experiments/scripts/13_full_pipeline_v2.py \
    --config "$C" --facts-file "$FACTS" \
    --recipe "$RECIPE" --train-steps 1500 --seed "$SEED" \
    --override-delta-max 0.1 "$@" \
    --output "$R/d_${NAME}.json" > "$L/d_${NAME}.log" 2>&1 || true
  echo "D_${NAME^^}_DONE $(date)"
}

[ "${OK[llama]:-}" ]   && run13 llama_pce_mod_s0    "${CFG[llama]}"   paraphrase_ce 0
[ "${OK[mistral]:-}" ] && run13 mistral_dist_mod_s0 "${CFG[mistral]}" distill       0
[ "${OK[mistral]:-}" ] && run13 mistral_pce_mod_s0  "${CFG[mistral]}" paraphrase_ce 0
[ "${OK[qwen15b]:-}" ] && run13 qwen15b_dist_mod_s0 "${CFG[qwen15b]}" distill       0
echo "PHASE_D1_DONE $(date)"
