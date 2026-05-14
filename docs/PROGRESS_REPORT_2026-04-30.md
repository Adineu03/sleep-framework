# SLEEP Research — Progress Report

**Date:** 2026-04-30
**Author:** Aditya Tripathi (with Claude as research assistant)
**Status:** End of Phase 1 (Implementation) → Beginning of Phase 2 (Empirical Research)

---

## Executive Summary

In a single working day we built and validated a complete implementation of
the SLEEP framework across all 6 modules, ran end-to-end experiments at
GPT-2 (124M) and Qwen2.5-7B scales, and discovered four substantive flaws
in the formalization that required empirical work to surface.

**The system runs end-to-end. The architecture is sound. But "it works" is
not yet supported by external evidence.** What we have is a working
*pipeline* and *self-consistent internal validation*. What we don't yet have
is proof that consolidated knowledge is retrievable in a way that beats
chance or baselines.

This report is honest about both. It exists so that future work — whether by
the same team or a successor — can pick up from a true picture of where the
project stands.

---

## What We Proved

### 1. The Architecture Is Implementable
- **3,500 lines of Python** across 6 modules
  (tagging, prp, weights, sleep_engine, orchestrator, evaluation)
- **182 unit and integration tests** passing on both CPU (Windows, Python 3.14)
  and CUDA (RunPod, Python 3.12)
- **Pipeline runs end-to-end** on a 7.6B-parameter model in bfloat16

### 2. The Components Work in Isolation

| Module | Validation | Result |
|:---|:---|:---:|
| Tagging Layer | Inspect tagged spans on diverse documents | 5/5 sanity checks passed at 7B |
| W_fast Encoding | PPL drop on target / no interference on controls | 9/12 (target encoding marginal at low LR; no interference) |
| PRP Allocation | Per-paper worked example | Convergence behavior matches formalization |
| Sleep Engine | End-to-end cycle execution | Runs without crash; safety system protects model |

### 3. The Safety System Works
Across all experiments, **base capability preservation (BCP) stayed within
±0.5%** of baseline. The hard-clipping bound (`delta_max`) and validation/
rollback mechanism reliably prevent the kind of catastrophic forgetting
that motivates the entire framework.

### 4. Scale Effects Are Real
- 5 facts: 0% consolidation rate
- 200 facts: 31% consolidation rate (15/48 candidates passed validation)

The system requires sufficient data to even *meet its own validation criterion*.
This is a real empirical finding about the architecture, not a tuning artifact.

---

## What We Found (Formalization Amendments)

Four assumptions in the formalization were empirically wrong. All are now
documented in `docs/SLEEP_Formalization.md` Appendix A and individual logbook
entries.

| # | Original | Empirical | Severity |
|:---:|:---|:---|:---:|
| A.1 | `delta_max = 0.001` (per-param) | Saturates on 7B; needs ~0.01 or different abstraction (Frobenius norm) | Architectural |
| A.2 | `alpha_slow = 1e-5` | Below bfloat16 precision floor; needs ≥1e-4 | Precision |
| A.3 | `mu_surprise = baseline_calibration` | Generator-discriminator gap; rejects all replays | Structural |
| A.4 | `epsilon_learn = 0.10` validation | Has data-volume floor; impossible at <30 replays | Scale |
| A.5 | (Implicit assumption) | Consolidation ≠ retrieval; current arch solves encoding only | Research-open |

The first four have proposed fixes. The fifth (consolidation/retrieval gap)
is an open research problem.

---

## What We Did NOT Prove

We are explicit about this so the next phase doesn't start from a false
premise.

### We did not prove that consolidated knowledge is retrievable.
The 200-fact run showed 4 reported "HITs" out of 200 in delayed recall. On
honest inspection, **3/4 of those are likely false positives**: the model
fabricated plausible content that happened to contain a target keyword,
not retrieved the consolidated fact.

The honest delayed-recall accuracy is closer to **0–1 hits / 200 (≈ 0–0.5%)**.

### We did not show that SLEEP beats any baseline.
- Vs. RAG: not run
- Vs. naive LoRA fine-tuning: not run
- Vs. random-replay buffer: not run
- Vs. no-op (just the base model): not run

Without these, we cannot claim the architecture's specific mechanisms
(tagging, PRP, generative replay, dual weights) provide value.

### We did not test memory persistence across multiple sleep cycles.
Single-cycle was the entire scope today. The "delayed" in DRA(τ=1 hour, 1 day, 1 week)
is the central claim of the proposal, and it remains untested.

### We did not validate the cold-start mechanism.
Implemented but not exercised in any real experiment.

### We did not test the per-user adapter architecture (Q6.4).
Single-user only.

---

## Cost & Time Accounting

| Item | Spent |
|:---|:---:|
| RunPod GPU time (RTX 5090) | ~3 hours @ $0.99/hr ≈ **$3** |
| Pod disk (~30GB volume) | ~$0.04 |
| Storage transfer | $0 |
| **Total cloud spend** | **≈ $3** |
| **Remaining credit** | **~$7 (of $10 loaded)** |
| Solo developer time | One working day |
| Lines of code written/modified | ~4,000 |
| Logbook entries created | 7 |
| Major findings documented | 5 |

---

## Where We Are vs. The Original Plan

The proposal's 12-month timeline had:

> Months 1-2: Mathematical formalization and validation (the "math is rigorous" check) — **DONE before today**
>
> Months 3-5: Implement minimal SLEEP system in PyTorch — **DONE today**
>
> Months 6-8: Run on real continual learning benchmarks — **NOT STARTED; this is Phase 2**

We are **roughly 5 months ahead** of the proposal's pace on implementation.
We are **0 months in** on actual research validation. The research phase has
not begun. Today's experiments were *plumbing* tests (does the pipeline run?)
not *scientific* tests (does the pipeline produce useful learning?).

---

## What Should Happen Next

The single most important next experiment, if I had to pick one:

> **Paired test of consolidated vs. failed-validation facts.**
> Take the 15 facts whose validation passed and the 33 whose validation
> failed in the 200-fact run. Re-run the recall test on both groups with
> matched prompts. If consolidated facts have higher recall than
> failed-validation facts, we have the first piece of *external* evidence
> that consolidation does something useful. If they're indistinguishable,
> we know the retrieval problem (A.5) is binding and must be solved before
> any baseline comparison is meaningful.

This is a small, cheap, falsifiable experiment. It either gives us a green
light to proceed, or it tells us we need to redesign the retrieval mechanism.

After that, the order should be:

1. **Solve A.3 (mu_surprise) properly** — implement the relative-gain check
2. **Solve A.5 (retrieval)** — investigate cloze, paraphrased, multiple-choice
   recall on the 15 consolidated facts
3. **Multi-cycle persistence** — run 10 cycles, measure DRA(n)
4. **Baselines** — RAG, naive LoRA, replay buffer
5. **Paper draft** — only after at least one strong external-validity result

---

## Lessons Learned (Process)

These are notes for future similar work, not part of the science.

1. **Toy experiments (5 facts) hide scale effects.** Three of our four
   formalization findings only surfaced once we ran 200 facts. Anything
   that looks like a "small reproduction" should be confirmed with at
   least one realistic-scale run before publishing or claiming insight.

2. **bfloat16 has different defaults than float32.** Two of the
   formalization findings stem from this. A formalization that doesn't
   specify dtype is incomplete.

3. **Self-consistent validation is not external validity.** The system's
   internal validation (`surprise_new < threshold`) passing 15/48 times is
   the system grading its own homework. External test sets — keyword
   matching, even — disagreed strongly.

4. **Reactive hyperparameter tuning is a trap.** Our 4-attempt sequence on
   v3-v4 was hyperparameter chasing. The 200-fact scale-up was the right
   strategic call; we should have made it sooner instead of tuning more.

5. **Cost-conscious habits matter.** $3 of cloud spend isn't catastrophic,
   but it accumulates. Monitoring, not assuming.

---

## Files Produced Today

### Experiment outputs (`experiments/results/pod_run_2026-04-30/`)
- `01_qwen7b_output.log` — Tagging validation
- `02_qwen7b_output.log` — W_fast validation
- `03_qwen7b_output.log` — Sleep cycle v1 (mu_surprise gap discovered)
- `03_qwen7b_v2_output.log` — Sleep cycle v2 (delta_max saturation)
- `03_qwen7b_v3_output.log` — Sleep cycle v3 (alpha_slow precision)
- `03_qwen7b_v4_output.log` — Sleep cycle v4 (validation strict)
- `03_qwen7b_v5_200facts.log` — Sleep cycle v5 (first 31% consolidation)

### Logbook entries (`logbook/`)
- `2026-04-30_experiment_03_qwen7b_quality_gap.md`
- `2026-04-30_experiment_03_qwen7b_delta_max_bound.md`
- `2026-04-30_experiment_03_qwen7b_alpha_slow.md`
- `2026-04-30_experiment_03_validation_strict.md`
- `2026-04-30_experiment_03_v5_200facts_milestone.md`

### Code changes
- `sleep/tagging/__init__.py` — Fixed bfloat16/device dtype mismatch
- `experiments/configs/qwen7b.yaml` — Created 7B config; 2 amendments documented inline
- `experiments/scripts/01_validate_tagging.py` — Made parametric (--config)
- `experiments/scripts/02_validate_wfast.py` — Made parametric (--config)
- `experiments/scripts/03_single_sleep_cycle.py` — Made parametric (--config, --facts-file, --max-facts)
- `experiments/scripts/generate_facts_dataset.py` — New: 10-template synthetic fact generator
- `experiments/data/facts_200.json` — Generated dataset

### Documentation
- `docs/SLEEP_Formalization.md` — Appendix A added (5 amendments)
- `docs/PROGRESS_REPORT_2026-04-30.md` — This document

---

## A Note on Honesty

The most important thing in this report is what's *not* claimed.

It would be easy to write up today's work as "SLEEP is working at 7B scale —
31% consolidation rate, DRA > 0, BCP preserved." Each individual statement
is technically true. The composite claim implied — "SLEEP is doing something
useful" — is not yet supported.

What we have is a system that **passes its own internal validation
sometimes, doesn't break the model, and has at least four hidden assumptions
in its formalization that we now know about.** That's a real day's work. It
is not yet a published result.

The next phase begins from there.
