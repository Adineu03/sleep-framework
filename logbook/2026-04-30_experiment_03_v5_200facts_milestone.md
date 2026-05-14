# Experiment 03 v5: 200 Facts, Qwen2.5-7B — First End-to-End Consolidation

**Date:** 2026-04-30
**Hardware:** RunPod RTX 5090
**Model:** Qwen2.5-7B (bfloat16)
**Config:** `qwen7b.yaml` with `delta_max=0.01`, `alpha_slow=1e-4`,
  `use_real_mu_surprise=False`
**Dataset:** `experiments/data/facts_200.json` (200 synthetic facts,
  10 templates × 20 each)

---

## Headline Result

**For the first time: 5/5 sanity checks passed. 15 memories consolidated. DRA > 0.**

But — and this is the important part — **the actual recall improvement is
small to noise-level.** Of 4 reported "HITs," at least 3 are likely false
positives where the model fabricated plausible answers that happened to
contain a keyword.

This is a **plumbing milestone**, not yet a **scientific result.**

## Numbers

| Metric | Value |
|:---|:---:|
| Facts fed | 200 |
| Tags created | 149 (75% of facts) |
| PRP candidates allocated | 48 (32% of tags) |
| Replays generated | 48 (100%) |
| Replays accepted (quality bypass active) | 48 (100%) |
| **Validation passed (consolidated)** | **15 (31% of accepted)** |
| Validation failed (rolled back per-tag) | 33 |
| BCP (preservation) | 1.0042 (excellent) |
| DRA | 0.007 |
| Apparent HITs | 4 |

## Honest Assessment of the 4 HITs

| Fact | Recovered token | Plausibly real recall? | Why |
|:---|:---|:---:|:---|
| fact_010 (Vilnius earthquake) | "6.8" magnitude | **No** — model fabricated "1968 Vilnius earthquake" | Keyword match in unrelated narrative |
| fact_060 (Lome earthquake) | "6.0" magnitude | **No** — model fabricated "1998 Togo earthquake" | Same pattern |
| fact_093 (North Tartu founding) | "100,000" population | **Maybe** — model said "North Tartu is a district of Tartu" | Could be guessing typical city size |
| fact_113 (Kuopio founding) | "100,000" population | **No** — model said "founded in 1229" (real Kuopio history) | Pulling prior knowledge |

So our **honest DRA estimate** is closer to 0–1 real consolidation hits
out of 200 facts, not 4.

## What Actually Got Validated

The 15 facts that passed `validate_consolidation` did so by lowering the
model's surprise on their original spans by ≥ 10%. That is a **real,
measurable change** in W_cons. The system's internal validation is
self-consistent and rigorous.

But internal validation ≠ external retrieval. **Lower surprise on the
original span** doesn't necessarily mean **the model can answer a question
about that span.** The recall test asks the model to generate freely from a
prompt; even if W_cons reduces surprise on the verbatim text, free-form
generation pulls from the model's broader prior.

## The Real Finding

**Consolidation and recall are decoupled.** The current architecture solves
"can the model fit this gist into W_cons without breaking other things?"
(yes, 15/48 times). It does *not* solve "can the model answer a question
about this gist?" (no, ~0/200 robustly).

This is a **research problem**, not an engineering bug. Specifically:

1. The gist may be encoded but inaccessible from question-form prompts
2. Free-form generation may "drift" away from consolidated content under
   competing priors
3. The keyword-match scoring metric is too lenient (gives false positives) and
   too strict (no partial credit for semantic recall)

## What the Scale Comparison Tells Us

| Run | Facts | Replays | Consolidations | DRA |
|:---|:---:|:---:|:---:|:---:|
| v4 | 5 | 2 | 0/2 = 0% | 0.000 |
| **v5** | **200** | **48** | **15/48 = 31%** | **0.007** |

Scale fixes consolidation rate dramatically (0% → 31%). Scale does not fix
recall — it remains essentially zero whether we have 5 facts or 200. This
isolates the recall problem from the data-volume problem.

## Sanity Checks

**Did pass:**
- [x] Tags were created (149)
- [x] PRPs were allocated (48)
- [x] Sleep completed without rollback (whole-cycle rollback didn't trigger)
- [x] BCP < 1.05 (1.0042 — model preserved)
- [x] DRA > 0 (0.007 — first non-zero recall measure)

**What didn't get tested:**
- DRA on consolidated facts vs. failed facts (the controlled test)
- DRA on original prompt vs. paraphrased prompt
- Whether the 15 "consolidated" facts are actually retrievable in any form
- Comparison to baselines (RAG, naive LoRA, replay buffer)

## What I Would NOT Claim From This Result

- "SLEEP works" — too strong; we have plumbing, not retrieval
- "31% consolidation rate" — that's the system's internal pass rate, not
  evidence of useful learning
- "DRA = 0.007" — mostly noise; honest number is closer to 0
- "The framework is ready for paper" — not yet

## What I WOULD Claim

- The architecture is implementable end-to-end at 7B scale
- The safety system (BCP preservation) works reliably
- Scale effects are real: consolidation fails at 5 facts, partially succeeds
  at 200
- Validation criterion is rigorous and self-consistent
- The next research problem is **encoded-knowledge retrieval**, not
  consolidation

## What's the Right Next Experiment

The single most informative experiment we could run next:

**Direct paired test:** of the 15 consolidated facts vs. the 33 that failed
validation, ask each fact's `test_prompt` (with shorter, more directed
prompts), score with strict grading (all keywords must appear), and compare
DRA on the two groups.

If consolidated facts have higher DRA than failed-validation facts, we have
evidence consolidation is real. If they have the same DRA, we know the
retrieval problem is independent.

This is **the falsifiable claim** the experiment should test.

## Files Changed

- `experiments/data/facts_200.json`: Generated dataset (committed to repo)
- `experiments/scripts/03_single_sleep_cycle.py`: Added `--facts-file`,
  `--max-facts` arguments; load facts from JSON
- `experiments/scripts/generate_facts_dataset.py`: New file — synthetic
  fact generator with 10 templates
