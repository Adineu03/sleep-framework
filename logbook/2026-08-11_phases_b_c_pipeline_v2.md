# Phases B & C — The Full Pipeline Works, and Survives Cycles

**Date:** 2026-08-11 (same-day continuation of the ladder results)
**Hardware:** RunPod A6000. Phase B/B2: 8 runs; Phase C: 6 runs. ~$5 total.
**Results:** `experiments/results/pod_run_2026-08-11_phase_b/` and `_phase_c/`.

## Phase B/B2 — single full-pipeline cycle (wake→PRP→sleep→validate→recall)

The ladder recipe was wired into the SleepEngine itself (replay_strategy=
"paraphrase", train_mode="distill" with the full safety loop, mid-MLP w_cons,
two-stage validation as the internal gate; 408 tests green before GPU).

**Finding 1 — the machinery strangle (3 seeds):** with safety as designed
(delta_max 0.01 + plasticity scaling), the recipe is nullified inside the
pipeline: trained-subset DRA 0.007/0.007/0.007, zero gate-confirmed
consolidations. The original consolidation failure therefore had THREE stacked
causes: substrate/layers, surface diversity, and the safety machinery erasing
the write (delta_max saturation, per the April A.1 finding).

**Finding 2 — the unlock (3 seeds):** identical pipeline with clip+scaling
relaxed: trained-subset DRA **0.210 ± 0.018** at BCP 0.90–0.99, with
**21–24 of ~50 facts passing the direct recall gate** — the first verified
autonomous consolidations in the project. Llama: same A/B direction
(0.041→0.610) but at BCP 4.98 under full relaxation with CE — Llama needs a
middle setting (its fragility, consistent with every prior experiment).

## Phase C — ten cycles (the regime the machinery was built for)

Three arms x 2 seeds, Qwen mid-MLP, 20 facts/cycle, distill 400 steps/cycle:

| c10 | DRA_cum (s0/s1) | BCP (s0/s1) | Character |
|:--|:--:|:--:|:--|
| **v2-moderate** (clip 0.1, phi on) | **0.173 / 0.175** | 1.87 / 2.24 | best recall, gradual drift |
| v2-relaxed (no clip/phi) | 0.133 / 0.152 | 2.25 / 2.15 | works, drifts faster |
| naive LoRA (same mid-MLP cfg) | 0.053 / 0.060 | 1.66 / 1.80 | memorizes, can't extract |

Seed-consistent findings:
- **SLEEP v2 beats naive in the multi-cycle regime** — 3× the recall at
  comparable preservation (0.174 vs 0.056 mean DRA_cum). The design-intent
  claim, real for the first time, on both seeds.
- **No plateau lottery.** BCP drifts smoothly ~1.0→~2.2 with no step-cliff
  and no 10^2–10^3 blowups; the pathological signature belonged to the
  choked-attention-top configuration, not to consolidation per se.
- **Right-sized machinery wins**: moderate (clip 0.1) beats both extremes on
  recall and beats relaxed on damage — the "ablation-first safety design"
  recommendation, now measured.
- **Verified consolidations every cycle** (4–12 per 20-fact batch, 20/20
  cycles across arms/seeds).

Known remaining gaps (named for the paper and Phase D):
- BCP still leaks ~+0.12/cycle — no arm holds <1.05 long-horizon.
- Cumulative recall decays (c1 batches fade); the engine rehearses only the
  current batch. Interleaved rehearsal of prior consolidations is the obvious
  next mechanism (and is what biological sleep actually does).
- Llama needs per-family settings (lr/clip); untested beyond single-cycle.

## Phase D (pending sign-off + recharge)
Full four-model matrix on v2-moderate: 5-seed single-cycle, 10-cycle x 2 seeds
both arms, per-family signal/clip tuning, EWC-only comparison (v2 must beat
it), verification gates throughout. ~$12–15.
