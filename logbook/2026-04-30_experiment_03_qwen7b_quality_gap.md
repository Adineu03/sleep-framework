# Experiment 03: Qwen2.5-7B First Sleep Cycle — Quality Check Gap Discovered

**Date:** 2026-04-30
**Hardware:** RunPod RTX 5090, 32GB VRAM
**Model:** Qwen2.5-7B (7.616B params), bfloat16
**Config:** `experiments/configs/qwen7b.yaml` (formalization defaults)

---

## What Happened

End-to-end sleep cycle ran. **4/5 sanity checks passed.** Pipeline integrity is sound:

| Component | Result |
|:---|:---|
| Tagging | 7 tags from 5 facts |
| W_fast updates | All finite losses (3.7–6.8 nats) |
| PRP allocation | 2/3807 candidates allocated |
| Sleep — generation | 2/2 replays produced |
| **Sleep — quality check** | **0/2 accepted** |
| BCP preservation | 1.0000 (no degradation) |
| DRA | 0.000 (no consolidation = no recall) |

## The Finding

Both replays were rejected by `quality_check` for the same reason:

```
Quality check REJECTED (surprise): Replay contains no new information
for W_slow (mean surprise 1.27 nats < mu_surprise 1.41 nats)
```

`mu_surprise = 1.41 nats` was computed from `compute_baseline_surprise` over the
control texts (Paris, water boiling, Guido van Rossum). Generated replays come
out at ~1.22–1.27 nats — about **10–13% below** that baseline.

## Why This Is a Real Issue, Not a Bug

The formalization (Q4.1) defines the surprise quality check as: *replay must be
at least as surprising as the model's baseline.* The implicit assumption is
that the surprise distribution of replays is comparable to the calibration
distribution.

**But replays come from the same model** that defines the surprise function.
W_slow + W_fast generates the replay, and W_slow scores it. There's no reason
to expect a generator-discriminator system to produce outputs whose surprise
matches an externally calibrated baseline. In fact, the opposite is expected —
generators tend to produce outputs near the modes of their own distribution.

This is a **generator-discriminator gap**, and the formalization didn't model
it explicitly.

## What This Means for the Formalization

`mu_surprise` as currently defined is **too strict** for self-generated replays.
We need to account for the systematic surprise reduction of generation.

**Three options for fixing the formalization:**

1. **Buffer factor:** `mu_surprise_eff = α * mu_surprise` with α ≈ 0.7. Simple,
   but introduces a new hyperparameter to tune.

2. **Two-distribution calibration:** compute `mu_surprise_replay` as a separate
   baseline by running compute_baseline_surprise on a small set of
   self-generated samples, and use that as the threshold.

3. **Drop absolute threshold; use relative gain:** require
   `surprise(replay) > β * surprise(seed)` where the seed is the original
   tagged span. This compares like-to-like (both from the model) and avoids
   the calibration mismatch entirely.

I lean toward option 3 because it's the cleanest theoretically — it asks "is
the replay more informative than the seed it's compressing?" rather than "is
the replay as informative as Wikipedia?"

## What We're Doing for the Next Run

For the immediate purpose of validating the rest of the pipeline (replay →
train W_cons → recall), we'll **disable the surprise quality check** by
setting `use_real_mu_surprise = False`. The similarity check still runs.

This is a temporary diagnostic move. The real fix needs to be incorporated into
the formalization as Q4.1 amendment.

## Other Observations

- **Tagging quality at 7B** is qualitatively different from GPT-2. Qwen latches
  onto entities and proper nouns: "Zenith Corporation reported annual revenue
  of $847", "Elena Vasquez at MIT", "city of New Helsinki". These are exactly
  the spans we'd want to consolidate.

- **PRP threshold = 0.382** dropped substantially — the formalization default
  is `theta_floor = 0.2`, so this is just above floor. With only 7 tags in
  the buffer there's not enough signal for the relative scoring to push the
  threshold higher.

- **W_fast loss values (3.7–6.8 nats)** are higher than typical training loss
  on Qwen, which suggests the surprising spans really are out-of-distribution
  for the base model. Good signal.

- **Generation looked plausible** — model continued the seed coherently. The
  replays just weren't novel enough to pass quality.

## Files Changed

None yet. The fix goes in the next run.
