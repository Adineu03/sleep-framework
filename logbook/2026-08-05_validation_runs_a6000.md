# Full Validation Runs — RTX A6000, All Five Stages

**Date:** 2026-08-05
**Hardware:** RunPod RTX A6000 48GB (fresh pod, fresh volume)
**Model:** Qwen2.5-7B (bfloat16 except stage 4)
**Environment:** transformers pinned 5.7.0 (latest broke against template torch 2.4.1 — DTensor import)
**Total GPU time:** ~3.2 h ≈ $2.5–3. All results synced to `experiments/results/pod_run_2026-08-05/`.

---

## Stage 1 — Multi-seed recognition/pipeline (script 07 × seeds 0–4, original replay, two-stage validation)

| Metric | mean ± std | per-seed |
|:---|:---:|:---|
| DRA | 0.006 ± 0.003 | .005 .007 .010 .003 .005 |
| BCP | 1.066 ± 0.032 | 1.108 1.032 1.042 1.088 1.061 |
| surprise-gate passes | 38.8 ± 0.4 | 39 39 39 39 38 |
| recall-gate passes | 0.4 ± 0.5 | 0 1 1 0 0 |

**The C4 self-grading discrepancy is now a measured number: ~97% of proxy-validated
consolidations fail the direct recall gate, consistently across 5 seeds.** The
consolidation bottleneck (DRA at floor) replicates under seed variance.

## Stage 2 — Extended multi-cycle (10 cycles × 20 facts, seed 0)

SLEEP BCP: 1.18 → 1.98 → 2.04 → **4.40 (jump at c4)** → plateau 4.45–4.59 (c4–c10). DRA_cum 0.05.
Naive BCP: 3.01 → 3.74 (c3) → fluctuates, partial recovery to 2.02 (c7) → 3.36 (c10). DRA_cum 0.085.

- Cycles 1–3 **replicate the April finding** (SLEEP ~2× better preservation at c3).
- **The advantage is transient**: by cycle 10 naive LoRA wins BOTH axes
  (DRA 0.085 vs 0.050; BCP 3.36 vs 4.53). Per-cost at c10: naive ~2.5× better.
- SLEEP's BCP does plateau — but at ~4.5, not the ~2.3 the 3-cycle run suggested.
- **Methodological finding: evaluation horizon is a variable.** 1-cycle
  understates safety machinery; 3-cycle overstates it. Caveats: batch 20
  (April used 67), single seed.

## Stage 3 — Retrieval-aware warm-up extension (the proposal's centrepiece)

Pre-registered success criterion: DRA 0.10–0.15 at BCP < 1.5. **Not met in any variant.**

| Variant | seeds | ΔDRA | BCP cost | warm-up loss |
|:---|:---:|:---:|:---:|:---|
| Gate-only, 150 steps (smoke, 50 facts) | 0 | −0.007 | ~none | flat |
| Gate-only, 300 steps (200 facts) | 0,1 | +0.002 both | improves to ~1.00 | flat; gate→0.99 (identity) |
| + train_wcons (LoRA), 300 steps | 0 | +0.003 | 1.07→1.73 | 4.30→3.33 (learns) |
| + train_wcons (LoRA), 300 steps | 1 | +0.000 | 1.07→3.74 | 4.72→3.49 (learns) |

- The gate alone converges to identity — 36 params cannot create memory-use.
- The LoRA variant *does* learn the memory-conditioned objective (loss drops ~1 nat)
  and lifts recognition (MC 0.22→0.24–0.27), but free-form recall stays at floor
  and base capability degrades (seed-variable, up to 3.7×).
- With LoRA in the mix the gate stays pinned at exactly 1.0 (gradient flows to
  the higher-capacity component).
- **Scope of claim: a lightweight warm-up (≤300 steps, general corpus) does not
  close the recognition–recall gap.** A full Memorizing-Transformers-scale
  retrain remains untested (out of budget class).

## Stage 4 — fp32 C1 precision-floor test (α_slow = 1e-5, the ORIGINAL default, 60 facts)

- fp32 training loss: mean 2.902 → 2.874 over 100 steps (~1% drift; bf16 was exactly flat).
- 0/12 validation passes; BCP 0.998 (weights essentially unmoved).
- **A.2 sharpened: the bf16 floor is real (exact-zero updates), but fp32 does not
  rescue the default — 1e-5 is under-scaled at 7B in any dtype. Two stacked
  failures, not one dtype artefact.**

## Stage 5 — Missing baselines (100 facts, seed 0)

| Baseline | DRA | BCP |
|:---|:---:|:---:|
| In-context (gold fact in prompt) | **0.760** | 1.000 |
| RAG (mean-pool retrieval, top-3) | 0.213 | 1.000 |
| EWC-only LoRA | 0.070 | 1.116 |

- In-context is the ceiling (0.76 — not 1.0: keyword scoring + generation noise).
- RAG's gap to in-context (0.21 vs 0.76) is retrieval failure, not generation failure.
- EWC-only reaches DRA 0.07 at BCP 1.12 — better DRA-per-BCP than any SLEEP
  configuration measured; the rest of SLEEP's machinery costs recall without
  buying preservation in single-cycle terms.

## Fixes applied during the runs
- transformers pinned to 5.7.0 (DTensor import failure with template torch 2.4.1).
- `_LoRAAdapter` dtype: `.to(device)` → `.to(device, dtype=module.weight.dtype)`
  (bf16/fp32 matmul crash in baselines; the known dtype trap).
- Stage-3 wcons loop in run_all.sh was mangled by an in-place edit (argparse
  error); re-run explicitly afterwards. Gate-only results unaffected.

## Paper implications (for the final report / revision)
1. C4 upgraded: self-grading gap quantified at ~97% false-confirmation (5 seeds).
2. C6 must be revised: multi-cycle advantage is **transient** (real at c1–3,
   reversed by c10). Frame as "evaluation horizon matters" — a stronger
   methodological claim than the original.
3. Extension section: report the warm-up as a negative result with scope —
   diagnosis stands, lightweight treatment insufficient; full retrieval-aware
   retraining remains the identified path.
4. A.2 amendment sharpened (dtype + magnitude are separable failures).
5. Baselines table now complete; EWC-only is the surprising strong baseline.
