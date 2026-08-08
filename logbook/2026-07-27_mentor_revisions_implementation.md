# Mentor Revisions — Implementation Session (code complete, pre-GPU)

**Date:** 2026-07-27
**Hardware:** local CPU (implementation + full test suite); RunPod pending
**Scope:** Implement all code for the P1/P2/P3 mentor review items and the
retrieval-aware warm-up extension, ahead of the GPU validation runs.

---

## Headline

All feedback that requires code is implemented and unit-tested on CPU. The
**full suite is green: 362 passed, 0 failed** (was 297 before this session —
~65 new tests). The retrieval-aware warm-up extension — the paper's
diagnostic-to-solution centrepiece — is built and verified, including the
required hook-based equality test on the KV-injection critical path.

Nothing here has been run at 7B scale yet. The GPU runs are the next step
(see "RunPod plan" below).

## What was implemented

### The extension (P3, the centrepiece)
- `sleep/warmup/gate.py` — `MemoryGate`: per-(adapted-layer, kv-head)
  multiplicative gate on the memory value contribution, parameterised as a
  log-scale initialised to identity. ~hundreds of params (9 layers × 4 kv-heads
  = 36 for Qwen2.5-7B top-third). Persists across sleep cycles because it lives
  on the injector, not in `w_cons`.
- `sleep/warmup/__init__.py` — `WarmupTrainer`: prefix-in-memory continuation
  objective (Memorizing-Transformers style). Stores a corpus prefix's episode
  K/V, trains the gate to predict the continuation with that memory injected,
  then freezes the gate and clears the warm-up bank.
- `sleep/weights/kv_injection.py` — added an optional gate seam on `mem_v_b`
  (attention is linear in V, so a per-head scale exactly modulates the memory
  contribution). Gate is `None`/identity by default → the Qwen path is
  **bit-identical** (verified: 54/54 existing KV tests still pass).
- Also generalised the injector to resolve the arch-specific modeling module
  (`apply_rotary_pos_emb`, `eager_attention_forward`, `ALL_ATTENTION_FUNCTIONS`)
  from the attention instance, so KV injection nominally works on Llama/Mistral
  too; Qwen2 remains the default and fallback. **Non-Qwen KV injection is not
  yet GPU-verified** — treat as best-effort until a run confirms it.

### P1 — critical
- `sleep/utils/seed.py` — `seed_everything` (torch/numpy/random/CUDA) +
  `aggregate_over_seeds` (mean ± std). Wired into scripts 07 and 08 (08's
  `--seed` previously seeded only the naive arm's local RNG).
- `experiments/scripts/10_run_seeds.py` — generic multi-seed runner: runs a
  target script across seeds in fresh subprocesses, aggregates JSON metrics as
  mean ± std. Verified end-to-end against a dummy target.
- Multi-model/precision configs: `qwen7b_fp32.yaml` (tests whether the C1
  precision-floor finding is bfloat16-specific by running the DEFAULT
  `alpha_slow=1e-5` in float32), `llama3_8b.yaml`, `mistral7b.yaml`,
  `qwen1.5b.yaml`. Layer selection was already derived from `num_hidden_layers`,
  so no hardcoded 28-layer assumptions needed changing.
- Extended multi-cycle: script 08 already exposes `--n-cycles` / `--batch-size`,
  so 10–20 cycles is `--n-cycles 20 --batch-size 10`. Seeding now covers both
  arms.

### P2 — high
- `sleep/evaluation/benchmarks.py` — loaders converting LAMA, ROME/MEMIT
  CounterFact, and a real-post-cutoff fact file into the canonical fact schema.
- `sleep/evaluation/baselines.py` — added `EWCOnlyBaseline` (naive LoRA + EWC
  penalty, isolating EWC's contribution) and `InContextBaseline` (gold fact in
  prompt, zero training). RAG already existed; `11_baselines.py` runs all three.
- Two-stage validation — `sleep/sleep_engine/cleanup.py::mini_recall_check` +
  a `two_stage_validation` flag in `SleepEngine`. Phase 5 now confirms a
  consolidation only when BOTH surprise-reduction AND a direct free-form
  mini-recall gate pass. Records `n_passed_surprise` vs `n_passed_recall_gate`
  so the C4 self-grading gap is measured directly. Wired into script 07 via
  `--two-stage-validation`.

### P3 — polish
- `sleep/evaluation/calibration_plot.py` — the confidence-vs-accuracy /
  P(correct) figure the review asked for (metrics already existed).
- Failure-mode table / biological-framing trim are paper-writing items, not
  code — deferred to the paper revision.

### Plumbing
- `experiments/scripts/_common.py` — shared config loader + model builder
  (dtype map, eager attention, pad-token) + seed/results helpers, removing the
  per-script duplication.
- `experiments/scripts/09_warmup_extension.py` — the §7.7 experiment:
  SLEEP-without-warmup vs SLEEP-with-warmup on the same facts (MC/cloze/DRA/BCP).

## Deviations from formalization defaults
None new. `qwen7b_fp32.yaml` deliberately sets `alpha_slow=1e-5` (the ORIGINAL
default, not the bfloat16-tuned 1e-4) — this is the point of the C1 dtype test,
not a new deviation. All other configs inherit the previously-logged
`alpha_slow=1e-4` / `delta_max=0.01` 7B tunings.

## Tests added (all green on CPU)
- `tests/test_warmup/test_gate.py` (10) — incl. the identity-gate bit-identity
  check and gate differentiability.
- `tests/test_warmup/test_warmup_trainer.py` (6) — end-to-end loop on tiny Qwen2.
- `tests/test_utils/test_seed.py` (9), `tests/test_evaluation/test_benchmarks.py`
  (8), `test_calibration_plot.py` (5), `test_new_baselines.py` (6),
  `tests/test_sleep_engine/test_two_stage_validation.py` (7).

## RunPod plan (next step)
Priority order once the pod is up:
1. Multi-seed recognition + Pareto (07 via 10_run_seeds, seeds 0–4).
2. Extended multi-cycle 10–20 cycles (08, seeds 0–4).
3. Warm-up extension (09) at k=64 — the headline new result.
4. float32 C1 test (07/03 with `qwen7b_fp32.yaml`) + a second model family.
5. Standard benchmark + baselines (11) for comparability.

Estimated ~$8–15 of GPU depending on how many seeds/cycles. Confirm remaining
credit before starting.
