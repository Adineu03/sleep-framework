# Cross-Model Validation Matrix — Four Models, Three Families, Complete

**Dates:** 2026-08-07 → 2026-08-08 (overnight)
**Hardware:** RunPod RTX A6000 48GB (fresh pod)
**Models:** Qwen2.5-7B (replicate runs), Qwen2.5-1.5B, Llama-3.1-8B (NousResearch
ungated mirror), Mistral-7B. All models passed the three-check ground-truth
verification gate before any experiment ran.
**Total:** ~19 GPU-hours ≈ $14. Results in `experiments/results/pod_run_2026-08-07_matrix/`
(34 JSON files + all logs). Zero crashes after the two fixes below.

---

## Fixes surfaced by the verification gate (in order)

1. **Config encoding (self-inflicted):** the three new YAML configs were written
   with Windows cp1252 em-dashes (invalid UTF-8) — all three models "failed"
   verification by failing to parse the config. Regenerated ASCII-clean.
2. **`sliding_window` (real cross-arch bug):** the patched attention forward read
   `self.sliding_window` unconditionally; the attribute exists on Qwen2Attention
   but not LlamaAttention/MistralAttention (transformers 5.7.0). Fixed with a
   defensive `getattr(..., None)`; 70 KV/warm-up tests confirmed the Qwen path
   bit-identical. After the fix: **Llama verification passed with max|diff| =
   0.00e+00** on the identity check; Mistral likewise.

The gate did exactly its job both times: no invalid number was ever produced.

## Results

### 1. Recall floor + self-grading gap: UNIVERSAL (5 seeds per model)

| Model | DRA | BCP | proxy passes | recall-gate passes |
|:--|:--:|:--:|:--:|:--:|
| Qwen2.5-7B | 0.006±0.003 | 1.066±0.032 | 38.8±0.4 | 0.4±0.5 |
| Qwen2.5-1.5B | 0.007±0.002 | 0.997±0.003 | 23.8±1.1 | 0.0±0.0 |
| Llama-3.1-8B | 0.006±0.003 | 0.958±0.009 | 9.2±2.4 | 0.6±0.5 |
| Mistral-7B | 0.007±0.000 | 1.001±0.002 | 4.0±1.0 | 0.0±0.0 |

DRA floor ≈ 0.006–0.007 everywhere; false-confirmation 93–100% everywhere.
Candidate counts vary 10× across families (surprise statistics differ); the
failure structure does not. The weaker-prior hypothesis (1.5B should recall
better) is falsified.

### 2. Ten-cycle continual learning: 9 seeded run-pairs

Every SLEEP run (9/9) shows the step-then-plateau BCP signature. Plateau levels
across seeds of identical configs: Qwen-7B 2.7/4.5/10.2; Qwen-1.5B 16/230;
Llama 4.8/1531; Mistral 1.6/3.2. **Three orders of magnitude on the same
hyperparameters.** Naive LoRA DRA higher in 8/9 pairs (2–20×). Neither arm ever
holds BCP<1.05 past cycle 2. Cycle-10 preservation winner flips seed-to-seed
(SLEEP 4/9, naive 5/9).

**Revised claim (paper §7.6):** fixed-horizon single-run comparisons are
unreliable; horizon AND seed jointly decide the ordering. The clip guarantees a
plateau exists but its level is a lottery over the optimization path.

### 3. Warm-up extension: null on recall, 16/16 runs, 4 models

Gate-only: inert everywhere (|ΔDRA| ≤ 0.008; gate → identity). +LoRA: learns the
objective everywhere, recall never moves; preservation cost family-dependent —
Qwen-7B 1.7–3.7, Qwen-1.5B 5.8–19.9, Llama 1.5e5–1.5e6, Mistral 1.8e4–1.0e5
(untuned lr=1e-3; per-family tuning is named follow-up). Side-finding: on
Mistral the gate learns to DAMP memory interference (bank-active BCP 8.9 → 2.8)
even though it can't learn to exploit memory.

### 4. Baselines (100 facts each)

| | Qwen-7B | Qwen-1.5B | Llama-8B | Mistral-7B |
|:--|:--:|:--:|:--:|:--:|
| In-context | 0.760 | 0.700 | 0.987 | 0.927 |
| RAG | 0.213 | 0.460 | 0.520 | 0.563 |
| EWC-only (BCP) | 0.070 (1.12) | 0.033 (1.43) | 0.073 (1.47) | 0.157 (2.53) |

In-context near-ceiling on Llama/Mistral makes their parametric recall floor
(0.006) maximally stark. EWC-only-beats-SLEEP is a 7B-class finding, weaker at 1.5B.

## Paper updates applied (2026-08-08)
- Abstract, intro, C2/C4/C6/C7 rewritten for four-model scope.
- §7.2: cross-model table (tab:cross_model). §7.5: baselines now cross-model.
- §7.6 retitled "Evaluation Horizon and Seed: The Long-Run Comparison Is Not
  Stable"; nine-run table; plateau-lottery interpretation.
- §7.7: sixteen-run warm-up table + family-dependent damage + lr caveat.
- Challenges (i) and (v) rewritten: cross-family DONE; open items are MoE/scale
  and plateau-distribution characterization. Setup: verification harness
  documented. Annexure: sliding_window recorded as third environment note.
- Clean compile, no errors/warnings.

## Superseded
The 2026-08-05 single-seed 10-cycle interpretation ("naive wins by cycle 10")
is superseded by the nine-run variance finding. The 2026-08-05 logbook entry
stands as the record of that day's runs.
