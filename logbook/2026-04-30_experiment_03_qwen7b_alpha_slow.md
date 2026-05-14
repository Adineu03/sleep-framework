# Experiment 03 v3: Qwen2.5-7B — alpha_slow Below Precision Floor

**Date:** 2026-04-30
**Hardware:** RunPod RTX 5090, 32GB VRAM
**Model:** Qwen2.5-7B (7.616B params), bfloat16
**Config:** `experiments/configs/qwen7b.yaml`,
  `delta_max = 0.01` (already amended), `alpha_slow = 1e-5` (formalization default)

---

## What Happened

Same outcome as v2 — pipeline runs, validation fails, W_cons rolled back —
but with a different proximate cause this time.

| Metric | v2 (delta_max=0.001) | v3 (delta_max=0.01) | Δ |
|:---|:---:|:---:|:---:|
| Hard-clips/step (steady state) | 36 | 1 | -35 |
| Mean training loss | 1.4581 | 1.4581 | 0 |
| Final training loss | 1.4554 | 1.4554 | 0 |
| Consolidations passed | 0 | 0 | 0 |
| DRA | 0.000 | 0.000 | 0 |

The `delta_max` fix eliminated saturation (now only 1 parameter clips per step
instead of 36), but **training loss didn't move**. Loss oscillates around 1.45
across all 100 steps — essentially the baseline PPL of the unmodified model.

## The Finding

`alpha_slow = 1e-5` produces gradient updates below the bfloat16 precision
floor. With AdamW at lr=1e-5 and momentum, the per-parameter update magnitude
per step is roughly:

```
update_magnitude  ≈  alpha_slow × ||gradient|| × step_factor
                  ≈  1e-5 × O(1) × O(1)
                  ≈  1e-5 per parameter per step
```

bfloat16 has ~3 decimal digits of mantissa precision (epsilon ≈ 7.8e-3 relative).
For weight values around 0.01–0.1 (typical LoRA), the smallest representable
delta is roughly `weight * 7.8e-3 ≈ 1e-4`. Updates smaller than this round to
zero.

So `alpha_slow = 1e-5` × O(1) gradient ≈ no detectable parameter change in
bfloat16 storage. The optimizer state accumulates correctly, but the actual
parameter doesn't move.

This is consistent with the loss being **completely flat** — the parameters
are not changing.

## Empirical Confirmation

In v4 (next experiment), bumping `alpha_slow` to `1e-4` (10x):
- Mean loss dropped from 1.4581 to **0.7450** (49% decrease)
- Final loss dropped from 1.4554 to **0.5469** (62% decrease)

This is a phase transition, not a smooth tuning curve — at 1e-5 nothing moves;
at 1e-4 substantial learning happens.

## Why the Default Doesn't Work at 7B

The formalization sets `alpha_slow = 1e-5` as a "small, conservative" sleep
learning rate to avoid catastrophic forgetting. But this assumed:

1. **Float32 precision** for parameter updates (which would resolve 1e-5 fine)
2. **Many sleep cycles** so cumulative effect compounds even with tiny per-step
3. **A model whose absolute weight scale matches the assumption**

In bfloat16, on a single sleep cycle of 100 steps, this is too low.

## Proposed Formalization Amendment

**Q3.4 amendment (alpha_slow):** The learning rate should be specified
*relative to weight scale* and account for parameter precision. Two options:

1. **Relative LR:** `alpha_slow = ||W_avg|| * 1e-3 / steps_per_memory`.
   For Qwen 7B with `||W_avg|| ≈ 0.05`, this gives `alpha_slow ≈ 1e-5` per
   step but normalized by step count, naturally scaling.

2. **Precision floor:** `alpha_slow = max(1e-5, 10 * dtype_eps_relative)`
   where `dtype_eps_relative = 1e-3` for bfloat16, `1e-7` for float32. For
   bfloat16 this gives `alpha_slow = 1e-2 * weight_scale`, much higher than
   the naive default.

Recommended: option 2 with explicit dtype awareness in the config.

## Operational Decision

For continued PoC work: **set `alpha_slow = 1e-4` in `qwen7b.yaml`** and
document the deviation. Long-term, the formalization needs the precision-aware
amendment.

## Files Changed

- `experiments/configs/qwen7b.yaml`: `alpha_slow: 1e-4` (was 1e-5)
