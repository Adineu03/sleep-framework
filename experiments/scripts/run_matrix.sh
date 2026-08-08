#!/bin/bash
# Full cross-model validation matrix. NO trimmed variants: every model gets the
# same programme Qwen2.5-7B got on 2026-08-05, gated by the arch verifier.
#
#   Per model:
#     0. verify_kv_arch  (ground-truth gate; model is SKIPPED entirely if it fails)
#     1. 5-seed recognition/pipeline (07 via 10)     — seeds 0-4
#     2. 10-cycle multi-cycle, both arms (08)        — seeds 0,1  (replicate!)
#     3. warm-up: gate-only + wcons, seeds 0,1 (09)
#     4. fp32 C1 test (07, qwen7b-fp32-style config) — where a fp32 config exists
#     5. baselines rag/incontext/ewc (11)
#
#   Plus: Qwen2.5-7B seeded 10-cycle replicate (seeds 1,2) to close the
#   single-seed gap in the 2026-08-05 horizon result.
#
# Usage:  nohup bash run_matrix.sh > /workspace/logs/matrix.log 2>&1 &
set -u
cd /workspace/sleep-research
export PYTHONUNBUFFERED=1

FACTS=experiments/data/facts_200.json
CORPUS=experiments/data/warmup_corpus.json
R=/workspace/results
L=/workspace/logs
mkdir -p "$R" "$L"

# model_key : config : fp32_config(optional, "-" = none)
MATRIX="
llama3_8b:experiments/configs/llama3_8b.yaml:-
mistral7b:experiments/configs/mistral7b.yaml:-
qwen1.5b:experiments/configs/qwen1.5b.yaml:-
"

run_model () {
  local KEY=$1 CFG=$2 FP32CFG=$3

  echo "=== [$KEY] STAGE 0: arch verification (ground-truth gate) ==="
  if ! python experiments/scripts/verify_kv_arch.py --config "$CFG" \
        > "$L/${KEY}_verify.log" 2>&1; then
    echo "[$KEY] VERIFY FAILED — skipping this model entirely. See ${KEY}_verify.log"
    echo "${KEY}_VERIFY_FAILED"
    return 1
  fi
  echo "${KEY}_VERIFY_PASSED"

  echo "=== [$KEY] STAGE 1: 5-seed recognition/pipeline ==="
  python experiments/scripts/10_run_seeds.py \
    --script experiments/scripts/07_full_kv_pipeline.py \
    --seeds 0 1 2 3 4 \
    --metrics dra bcp n_consolidated n_passed_surprise n_passed_recall_gate \
    --out "$R/${KEY}_seeds_recognition.json" \
    -- --config "$CFG" --facts-file "$FACTS" --kv-top-k 64 \
       --replay-strategy original --two-stage-validation \
    > "$L/${KEY}_stage1.log" 2>&1 || true
  echo "${KEY}_STAGE1_DONE $(date)"

  echo "=== [$KEY] STAGE 2: 10-cycle multi-cycle, both arms, seeds 0-1 ==="
  for S in 0 1; do
    python experiments/scripts/08_multi_cycle.py \
      --config "$CFG" --facts-file "$FACTS" --method sleep \
      --n-cycles 10 --batch-size 20 --seed "$S" --kv-top-k 64 \
      --replay-strategy original \
      --output "$R/${KEY}_multicycle_sleep_10c_s${S}.json" \
      > "$L/${KEY}_stage2_sleep_s${S}.log" 2>&1 || true
    python experiments/scripts/08_multi_cycle.py \
      --config "$CFG" --facts-file "$FACTS" --method naive_lora \
      --n-cycles 10 --batch-size 20 --seed "$S" \
      --output "$R/${KEY}_multicycle_naive_10c_s${S}.json" \
      > "$L/${KEY}_stage2_naive_s${S}.log" 2>&1 || true
  done
  echo "${KEY}_STAGE2_DONE $(date)"

  echo "=== [$KEY] STAGE 3: warm-up, gate-only + wcons, seeds 0-1 ==="
  for S in 0 1; do
    python experiments/scripts/09_warmup_extension.py \
      --config "$CFG" --facts-file "$FACTS" --corpus-file "$CORPUS" \
      --kv-top-k 64 --warmup-steps 300 --seed "$S" \
      --output "$R/${KEY}_warmup_gate_s${S}.json" \
      > "$L/${KEY}_stage3_gate_s${S}.log" 2>&1 || true
    python experiments/scripts/09_warmup_extension.py \
      --config "$CFG" --facts-file "$FACTS" --corpus-file "$CORPUS" \
      --kv-top-k 64 --warmup-steps 300 --seed "$S" --train-wcons \
      --output "$R/${KEY}_warmup_wcons_s${S}.json" \
      > "$L/${KEY}_stage3_wcons_s${S}.log" 2>&1 || true
  done
  echo "${KEY}_STAGE3_DONE $(date)"

  if [ "$FP32CFG" != "-" ]; then
    echo "=== [$KEY] STAGE 4: fp32 C1 test ==="
    python experiments/scripts/07_full_kv_pipeline.py \
      --config "$FP32CFG" --facts-file "$FACTS" --max-facts 60 --kv-top-k 64 \
      --replay-strategy original --seed 0 \
      --output "$R/${KEY}_fp32_c1.json" \
      > "$L/${KEY}_stage4_fp32.log" 2>&1 || true
    echo "${KEY}_STAGE4_DONE $(date)"
  fi

  echo "=== [$KEY] STAGE 5: baselines ==="
  python experiments/scripts/11_baselines.py \
    --config "$CFG" --facts-file "$FACTS" --max-facts 100 \
    --baselines rag incontext ewc --ewc-steps 100 --seed 0 \
    --output "$R/${KEY}_baselines.json" \
    > "$L/${KEY}_stage5_baselines.log" 2>&1 || true
  echo "${KEY}_STAGE5_DONE $(date)"
  echo "${KEY}_ALL_DONE $(date)"
}

# ---- The new-model matrix ----
echo "$MATRIX" | while IFS=: read -r KEY CFG FP32CFG; do
  [ -z "$KEY" ] && continue
  run_model "$KEY" "$CFG" "$FP32CFG"
done

# ---- Qwen2.5-7B: seeded 10-cycle replicate (closes the single-seed gap) ----
echo "=== [qwen7b] 10-cycle seeded replicate, seeds 1-2, both arms ==="
QCFG=experiments/configs/qwen7b.yaml
for S in 1 2; do
  python experiments/scripts/08_multi_cycle.py \
    --config "$QCFG" --facts-file "$FACTS" --method sleep \
    --n-cycles 10 --batch-size 20 --seed "$S" --kv-top-k 64 \
    --replay-strategy original \
    --output "$R/qwen7b_multicycle_sleep_10c_s${S}.json" \
    > "$L/qwen7b_stage2_sleep_s${S}.log" 2>&1 || true
  python experiments/scripts/08_multi_cycle.py \
    --config "$QCFG" --facts-file "$FACTS" --method naive_lora \
    --n-cycles 10 --batch-size 20 --seed "$S" \
    --output "$R/qwen7b_multicycle_naive_10c_s${S}.json" \
    > "$L/qwen7b_stage2_naive_s${S}.log" 2>&1 || true
done
echo "QWEN_REPLICATE_DONE $(date)"

echo "MATRIX_ALL_DONE $(date)"
