# Experiment 03 v4: Qwen2.5-7B — Validation Criterion Bottleneck

**Date:** 2026-04-30
**Hardware:** RunPod RTX 5090
**Model:** Qwen2.5-7B (bfloat16)
**Config:** `qwen7b.yaml` with `delta_max=0.01`, `alpha_slow=1e-4`,
  `use_real_mu_surprise=False` (quality bypass active)

---

## What Happened

Training **substantially decreased loss** (1.45 → 0.55), but validation
still rolled back W_cons. Zero consolidations.

| Stage | Result |
|:---|:---:|
| Tags created | 7 |
| PRP allocated | 3 |
| Replays accepted (with bypass) | 2/3 |
| Sleep training mean loss | 0.7450 |
| Sleep training final loss | 0.5469 |
| **Validation passed** | **0/3** |
| **Tags rolled back** | **3/3** |
| BCP | 0.9930 |
| DRA | 0.000 |

Loss decreased by 62% from the starting point. The model unambiguously
*learned something* during training. But the cleanup phase rolled it all back.

## The Finding

After reading `sleep/sleep_engine/cleanup.py:104-170` (`validate_consolidation`),
the validation rule is:

```python
surprise_new = compute_span_surprise(model_new, original_tokens, span_start, span_end)
threshold   = model_old_surprise * (1.0 - epsilon_learn)   # epsilon_learn = 0.10
passed      = surprise_new < threshold
```

This is a **transfer test**: it trains W_cons on *compressed replays* (gist),
then asks whether the resulting model has lower surprise on the *original
tagged span*. The threshold is "10% improvement on the original."

Three things make this hard with our v4 setup:

1. **Replays are aggressively compressed** (`compression_target = 5`).
   The replay is roughly 1/5th the length of the original. Training on a
   short summary and being tested on the full original is a hard generalization
   ask.

2. **Only 2-3 replays of training signal**, and they're all from different
   facts (no shared context), so each is essentially seen once.

3. **100 steps spread across 2-3 replays** = ~30-50 effective steps per
   replay. That's not many gradient updates per pattern.

4. **`epsilon_learn = 10%`** is a meaningful improvement bar. If the original
   surprise is 4.0 nats, the threshold is 3.6 nats — a 0.4 nat absolute
   improvement.

The training loss drop (1.45 → 0.55) is on the *replays*. The validation
measures surprise on the *originals*. The compression-induced gap means
learning the gist doesn't immediately translate to lower surprise on the
verbatim original.

## Implications

This isn't a bug — it's the validation criterion working exactly as
specified. The question is whether the criterion is *the right test*.

### Arguments for keeping it strict

- Forces the system to *generalize* the gist back to predicting the original
- Prevents low-quality replays from polluting W_cons
- Mirrors biological consolidation: the "memory" in the brain is a reconstruction
  ability, not a recording

### Arguments against

- 10% improvement on a verbatim test after training on heavy compression is
  very hard with limited data
- Causes the system to look like it's not learning when it actually is
- The validation tests against the *unconsolidated* model's surprise — if the
  unconsolidated model is already pretty good (e.g., the original span has
  some predictable continuation), 10% improvement is a high bar

## v5 Result Confirms This

When we scaled to 200 facts (v5):
- Same `alpha_slow=1e-4`, `delta_max=0.01`, `epsilon_learn=0.10`
- Validation passed on **15/48 candidates (31%)**

So the validation criterion *is* achievable — but only when there's enough
data variety that some candidates train cleanly. With 5 facts × 2-3 replays,
it's effectively a 0% pass rate; with 200 facts × 48 replays, it's 31%.

This is a **scale effect**, not a parameter-tuning effect. The criterion is
correctly demanding, and only sufficient data lets it pass.

## Proposed Formalization Amendment

**Q4.6 amendment (validation criterion):** The current rule
`surprise_new < surprise_old * (1 - epsilon_learn)` is sound but should be
documented as having a **minimum-data threshold**: with fewer than N replays
per memory (where N depends on compression and rank), the criterion cannot
pass even when training succeeds.

Three things to add:

1. **Document the data-volume requirement.** Below a threshold, expect 0%
   pass rate regardless of other tuning.

2. **Add a fallback "training loss decreased substantially" check.** If
   replay-loss dropped by ≥X% during training, accept the consolidation even
   if the verbatim-test threshold isn't met. This catches cases where the
   gist was learned but transfer is incomplete.

3. **Make `epsilon_learn` adaptive to compression**: when
   `compression_target` is high, the criterion should be lenient (e.g.,
   `epsilon_learn / sqrt(compression_target)`).

## Operational Decision

For now: keep `epsilon_learn = 0.10` (formalization default), and **only run
experiments with enough data** (≥100 facts) to make the criterion achievable.

## Files Changed

None. Finding documented; no config changes needed for v4 itself.
