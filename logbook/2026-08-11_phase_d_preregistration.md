# Phase D Pre-registration — The Full Four-Model Matrix

**Date:** 2026-08-11 (registered BEFORE any Phase D GPU run)
**Decision:** Aditya chose D-first over write-first ("This could change the
paper perspective"). Budget: ~$12-15 GPU on the running A6000; recharge
~$15-20 requested.

## Question
Does the v2 recipe (mid-MLP w_cons + paraphrase replay + moderate safety,
delta_max 0.1) generalise across model families and scales, and does it beat
the standard continual-learning baseline (EWC) — not just naive LoRA?

## Models
Qwen2.5-7B (anchor), Llama-3.1-8B, Mistral-7B-v0.1 (new to v2; has
sliding-window attention — the getattr fix must hold), Qwen2.5-1.5B (scale
point). Mistral and Qwen1.5B run the verify gate for the first time on the
mid-MLP configs.

## Stages and pre-registered gates
- **D0 (verify):** `verify_kv_arch.py` passes on all 4 mid-MLP configs.
  A failing model is excluded, not patched-around silently.
- **D1 (tuning, seed 0, B2 protocol: 200 facts / 1500 steps):** Llama
  paraphrase-CE at moderate; Mistral both recipes at moderate; Qwen1.5B
  distill at moderate. **Gate:** a model advances with the recipe achieving
  trained-subset DRA >= 0.10 at BCP <= 1.5. If both recipes fail the gate for
  a model, it still runs the matrix at its best setting but is reported as
  "not yet tuned for this family" — no silent exclusion.
- **D2 (single-cycle, 5 seeds/model):** v2-moderate everywhere.
  **Prediction:** Qwen7B mean trained-subset DRA in 0.15-0.30 (B2 relaxed was
  0.210 +/- 0.018; moderate should land near or below); every model's mean
  above the 0.006-0.017 "old floor" by at least 5x.
- **D3 (multi-cycle 10x20, seeds 0,1, v2-moderate vs naive):** for the three
  new models (Qwen7B pair reused from Phase C). **Prediction:** v2-moderate
  DRA_cum(c10) > naive on >= 2 of 3 new models on both seeds; BCP drift
  smooth (no plateau-lottery blowups >10).
- **D4 (EWC showdown, 10x20, seeds 0,1, Qwen7B + Llama):** new `ewc_lora` arm
  = naive LoRA + online-EWC Fisher penalty (lambda 100), everything else
  matched to the naive arm (verified by ground-truth tests
  `tests/test_experiments/test_ewc_arm.py`: Fisher accumulates/anchors
  exactly; lambda 1e6 restrains movement <50% of free).
  **Gate for the headline claim:** v2-moderate DRA_cum(c10) > ewc_lora
  DRA_cum(c10) on both seeds for Qwen7B. **Prediction:** EWC lands between
  naive and v2 on damage but near naive on recall (penalty preserves, does
  not convert memorization to extraction).

## What we will NOT do
- No mid-campaign hyperparameter fishing: moderate = delta_max 0.1, phi on,
  and (multi-cycle) lambda_ewc 0 / alpha_slow 1e-4, exactly as Phase C.
- No swapping the D1 gate criterion after seeing results.
- Fixed-horizon claims only with both seeds agreeing, per the plateau-lottery
  lesson.

## D1b extension (registered 16:20 UTC, after the Llama moderate result,
## BEFORE the added runs)
Llama at moderate: trained-subset DRA 0.659 (best ever) but BCP 4.35 — fails
the damage bar. Clip ladder added: delta_max 0.05 and 0.03 (paraphrase-CE,
seed 0, same protocol). Selection rule fixed in advance: Llama's matrix
setting is the delta in {0.1, 0.05, 0.03} maximizing trained-subset DRA
subject to BCP <= 1.5; if none satisfies, best-effort + "not yet tuned" flag.
Llama's D2/D3 runs use the chosen delta instead of 0.1; all other models stay
at 0.1 as registered.

## D1/D1b gate reading (17:55 UTC — decisions locked before matrix launch)
Trained-subset DRA @ BCP, gate >= 0.10 @ <= 1.5:
- Mistral distill 0.750 @ 1.31 PASS; Mistral paraphrase-CE 0.655 @ 2.98 fail
  -> **Mistral advances with distill** (best single-cycle result in the
  project to date, first try on a new family).
- Qwen1.5B distill 0.103 @ 1.03 PASS (thin: 10/42 facts pass recall gate vs
  29/31 on Mistral-7B) -> advances with distill; scale headroom noted.
- Llama clip ladder: d0.1 0.659 @ 4.35; d0.05 0.593 @ 3.64; d0.03 0.520 @
  2.14. Monotone but never reaches the bar — Llama's bleed is NOT
  clip-dominated. Per the rule: **Llama runs at delta_max 0.03, flagged
  "not yet tuned for this family"** (per-family lr is the named gap; no
  further mid-campaign fishing).
- Matrix launched 17:56 UTC: `run_phase_d2.sh distill distill 0.03`.

## Cost plan
D0+D1 ~1.5-2h (~$1.5); D2 ~7h; D3 ~5.5h; D4 ~2h. Total ~16-17 GPU-h ~= $13.
Stop-loss: if D1 shows both new families at DRA < 0.03 under every recipe,
pause and reassess with Aditya before spending on D2/D3.
