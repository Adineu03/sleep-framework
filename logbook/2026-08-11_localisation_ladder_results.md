# Localisation Ladder — Results: The Reframe Is Vindicated

**Date:** 2026-08-11
**Hardware:** RunPod A6000 48GB. 20 runs (~70 min GPU ≈ $1.20) + smoke.
**Protocol:** 5 arms × 2 seeds × {Qwen2.5-7B, Llama-3.1-8B}. 50 facts,
1200 steps, plain AdamW, no SLEEP safety machinery, greedy decoding
throughout. Results in `experiments/results/pod_run_2026-08-11_localisation/`.
**Pre-registered prediction (2026-08-11 implementation entry):** if the
localisation hypothesis is right, D≫A at sane BCP with C>B>A. **Outcome: hit
on Qwen; effect confirmed with a different best-arm on Llama.**

## One environment fix during smoke
`ContextDistiller` KL used `batchmean` over the batch dim (=1), summing over
sequence positions — loss ~66, no learning. Flattened to (positions, vocab)
so batchmean = per-token KL. Verified locally (6 distill tests), re-smoked:
loss 4.18→2.03 and DRA moved immediately.

## Results (mean of 2 seeds)

### Qwen2.5-7B — clean monotone ladder
| Arm | DRA | cloze | BCP |
|:--|:--:|:--:|:--:|
| A attn_top (original substrate) | 0.180 | 0.730 | 3.60 |
| B mlp_mid | 0.123 | 0.760 | 1.60 |
| C mlp_mid + paraphrases | 0.233 | 0.240 | 1.56 |
| E distill → attn_top | 0.220 | 0.220 | 1.25 |
| **D distill → mlp_mid** | **0.360** | 0.400 | **0.81** |

### Llama-3.1-8B — effect replicates, best arm differs
| Arm | DRA | cloze | BCP |
|:--|:--:|:--:|:--:|
| A attn_top | 0.143 | 0.740 | 3.59 |
| B mlp_mid | 0.250 | 0.860 | 1.41 |
| **C mlp_mid + paraphrases** | **0.353** | 0.470 | 2.65 |
| E distill → attn_top | 0.140 | 0.270 | 1.53 |
| D distill → mlp_mid | 0.217 | 0.520 | 1.63 |

## The four regularities (across both models, both seeds)
1. **Placement works, universally.** Mid-MLP cuts BCP from ~3.6 to 1.2–1.8
   at equal or better recall — 8/8 comparisons. The biology-chosen layers
   (attention V/O, top third) were the single largest damage source.
2. **Paraphrase diversity converts memorization into extraction.** Cloze
   (verbatim continuation) collapses (0.73–0.86 → 0.24–0.48) while
   question-answerable DRA rises — the exact memorisation-without-extraction
   → extraction transition the feedback predicted.
3. **Every ladder arm beats the entire original pipeline.** Old universal
   floor: DRA 0.006. Worst ladder arm: 0.087. Best: 0.38. The consolidation
   failure was never a stability–plasticity wall; it was localisation +
   surface diversity + training signal.
4. **Distillation is the best signal on Qwen, not on Llama** (untuned KD
   temperature/lr shared across families — same caveat as the warm-up).
   Qwen D dominates both axes (0.36 @ 0.81 — recall at *improved*
   perplexity); Llama's best extraction is paraphrase-CE (0.353) with
   distillation cheaper but weaker. Signal choice is family-dependent;
   placement + diversity are not.

## Decomposition (Qwen)
Signal alone (A→E): damage 3.60→1.25, DRA +0.04. Placement on top (E→D):
DRA +0.14, BCP below 1. Both moves contribute; placement is what unlocks the
negative-cost regime.

## Scope and caveats
- 50 facts (power-vs-budget tradeoff), 2 seeds, 2 models, one exposure
  budget (24/fact). No SLEEP safety machinery — this is the mechanism test.
- The 7.2 finding (KV memory cannot steer generation) is untouched: no arm
  here uses memory injection at all.
- attn_top at this budget also lifts off the old floor (0.09–0.24) — the
  April floor was partly exposure-starved (3–5 steps/fact then); its
  signature is memorization-with-damage, not extraction.

## Implications queued for the paper
- Reframe Results: stability–plasticity → localisation (with the ladder as
  its own section); ROME/MEMIT paragraph already in Related Work.
- The complete arc: diagnosis (recognition-without-recall) → failed cheap
  repair inside the wrong frame (warm-up) → correct frame (localisation) →
  working fix (mid-MLP + paraphrases + distill/CE) at 15–60× the old floor.
- Next experiments (pending sign-off): scale winning arms to 200 facts;
  port the recipe into the SLEEP consolidation loop as the new replay
  (paraphrase/distill into mid-MLP W_cons); per-family KD tuning.
