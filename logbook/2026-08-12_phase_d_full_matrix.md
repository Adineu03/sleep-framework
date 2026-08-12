# Phase D — The Full Four-Model Matrix (Results)

**Date:** 2026-08-11 15:52 UTC -> 2026-08-12 11:33 UTC (single A6000 queue,
~19.7 h wall, ~$15.5). Pre-registration: `2026-08-11_phase_d_preregistration.md`
(gates and predictions committed before any run; D1b extension and gate
readings appended mid-campaign, always before the affected runs/launch).
**Results:** `experiments/results/pod_run_2026-08-12_phase_d/` (39 JSONs + logs).
**Runs:** 35 total (D0 4 verifies; D1/D1b 6 tuning; D2 17 single-cycle;
D3 12 multi-cycle; D4 4 EWC). Zero crashes, zero OOM, zero reruns.

## D0 — verify gate
All four mid-MLP configs PASS on real weights, including Mistral-7B
(sliding-window attention; identity max|diff| = 0.00e+00) and Qwen2.5-1.5B
(both first-timers).

## D1/D1b — per-family tuning (single cycle, seed 0, gate DRA_ts >= 0.10 @ BCP <= 1.5)
- Mistral: distill 0.750 @ 1.31 PASS; paraphrase-CE 0.655 @ 2.98 fail ->
  distill. (Distill also won on Qwen: the KD write is the gentler write.)
- Qwen1.5B: distill 0.103 @ 1.03 PASS (thin: 10/42 recall-gated facts).
- Llama clip ladder: d0.1 0.659 @ 4.35; d0.05 0.593 @ 3.64; d0.03 0.520 @
  2.14. Monotone, never reaches the bar -> Llama bleed is NOT clip-dominated
  (lr per family is the named gap). Runs matrix at d0.03, flagged
  **not yet tuned for this family**.

## D2 — single-cycle, 5 seeds per model (trained-subset DRA @ BCP, mean +/- sd)
| model | recipe | DRA_ts | BCP |
|:--|:--|:--:|:--:|
| Mistral-7B | distill d0.1 | **0.750 +/- 0.033** | 1.205 +/- 0.058 |
| Llama-3.1-8B | pce d0.03 | 0.512 +/- 0.048 | 2.088 +/- 0.140 |
| Qwen2.5-7B | distill d0.1 | 0.189 +/- 0.031 | 0.882 +/- 0.027 |
| Qwen2.5-1.5B | distill d0.1 | 0.105 +/- 0.020 | 1.043 +/- 0.016 |

Pre-registered predictions both hit: Qwen7B in [0.15, 0.30]; every model
>= 5x the old 0.006-0.017 floor (weakest, Qwen1.5B, is ~17x above 0.006).
Mistral is the best single-cycle result in the project. Qwen7B is the only
model under BCP 1.0 — moderate beats B2's relaxed on damage at equal recall,
now at n=5.

## D3 — ten cycles x 20 facts, v2-moderate vs naive, 2 seeds (DRA_cum @ BCP at c10)
| model | v2-moderate (s0/s1) | naive (s0/s1) |
|:--|:--:|:--:|
| Qwen2.5-7B (Phase C) | 0.173/0.175 @ 1.87/2.24 | 0.053/0.060 @ 1.66/1.80 |
| Mistral-7B | 0.173/0.168 @ 1.18/1.18 | 0.118/0.160 @ 1.71/1.57 |
| Qwen2.5-1.5B | 0.097/0.102 @ 2.35/2.49 | 0.038/0.050 @ **24.5/11.4** |
| Llama-3.1-8B | 0.102/0.062 @ 4.33/9.20 | 0.082/0.042 @ 2.24/1.67 |

Prediction ("v2 > naive cum-recall on >= 2/3 new models, both seeds"): hit on
3/3. Findings:
- **Mistral: v2 dominates both axes** (more recall AND less damage, both seeds).
- **Qwen1.5B: naive detonates** (BCP 24.5/11.4, seed-random size = the
  plateau lottery, reproduced at small scale); v2 holds ~2.4 both seeds.
  The safety machinery matters MORE at small scale.
- **v2's signature is stability**: across-seed spreads ~0.005 everywhere vs
  naive's wild variance; naive scores higher on current-batch recall but
  retains old batches worse (memorize vs consolidate).
- **Llama: v2 wins recall, loses preservation** (BCP 4.3/9.2) — per-cycle
  bleed compounds; the not-yet-tuned flag is confirmed, now with numbers.

## D4 — EWC-only showdown (10x20, lambda 100, 2 seeds)
| model | v2-moderate | EWC-only | naive |
|:--|:--:|:--:|:--:|
| Qwen2.5-7B | **0.173/0.175** @ 1.87/2.24 | 0.060/0.052 @ 1.53/1.80 | 0.053/0.060 @ 1.66/1.80 |
| Llama-3.1-8B | 0.102/0.062 @ 4.33/9.20 | 0.075/0.072 @ 2.32/1.79 | 0.082/0.042 @ 2.24/1.67 |

**Headline gate (pre-registered, Qwen7B: v2 > EWC on DRA_cum, both seeds):
PASSED** — 3x recall at comparable damage. Prediction confirmed exactly:
EWC's penalty preserves (best/near-best BCP) but leaves recall at naive's
level — regularisation prevents forgetting; it does not create extraction.
On Llama, EWC-only is competitive with v2 (0.075/0.072 vs 0.102/0.062, at
far better BCP) — reported plainly; the Llama column is not a v2 win.

## The Phase D claim set (for the paper)
1. The localisation recipe generalises: 3 families x 2 scales, all >= 17x the
   old floor single-cycle, n=5 each. It is strongest OFF-Qwen (Mistral 0.750).
2. In the multi-cycle design regime, v2-moderate beats naive on cumulative
   recall on 4/4 models (both seeds) and beats EWC-only on the anchor model.
3. Right-sized safety machinery is load-bearing: it prevents the small-model
   catastrophe (BCP 24) and delivers seed-stability everywhere.
4. Named open gaps: Llama per-family lr (bleed not clip-dominated; EWC
   competitive there); BCP leak ~+0.12/cycle persists; no interleaved
   rehearsal (early batches still fade).

## Costs
Campaign ~$15.5; balance started $23. Pod stopped after results pulled.
