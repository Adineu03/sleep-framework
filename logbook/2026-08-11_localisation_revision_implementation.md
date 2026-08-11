# Localisation Revision — Implementation (code complete, pre-GPU)

**Date:** 2026-08-11
**Source:** Prof. Guha's second-round feedback: the consolidation failure is a
localisation problem, not a stability-plasticity wall. Naive LoRA's DRA 0.275
proves plasticity exists; the write just lands in the wrong modules (attention
V/O), the wrong layers (top third, chosen by biological analogy), with one
surface form, through a contaminated training signal (the KV pathway).
**Test suite:** 404 passing (was 362; +42 new tests). Paper compiles clean.

---

## Changes

### 1. Adapter placement is now configurable (and the biology is off by default only)
- `WeightsConfig.layer_selection`: `"top"` (original Q3.2, unchanged default)
  or `"middle"` (centred mid-stack window — where causal tracing places
  factual writes). 28 layers @ 1/3 -> layers 9–17.
- `adapted_matrices` now accepts MLP targets: `gate_proj`/`up_proj`/`down_proj`
  (llama-family names; GPT-2 mapped, with a collision-safe fallback to full
  dotted suffixes since attn.c_proj and mlp.c_proj share a short name).
- KV injection layers **decoupled** from the adapter block:
  `injection_selection`/`injection_fraction` (defaults preserve the historical
  coupling). `DualWeightSystem` now carries both `adapted_layers` and
  `injection_layers`; the bank, `write_to_kv_bank`, and the warm-up gate all
  ride the injection set. Defaults verified bit-compatible: full old suite green.

### 2. Paraphrase diversity (fixes the memorisation-without-extraction regime)
- `sleep/datagen/paraphrase.py`: deterministic, slot-driven surface forms —
  6–7 statements × 3 wrappers + 3 QA forms + 1 cloze per family. No RNG, no
  external model, no leakage.
- Generator captures each template's slot values **without adding RNG draws**:
  `facts_200_para.json` regenerated with seed 42 is **bit-identical on all
  200 texts/keywords** to `facts_200.json` (verified), now with 22–25
  paraphrases per fact, QA forms in every one.

### 3. Context distillation (decouples failure 7.2 from failure 7.4)
- `sleep/distill/ContextDistiller`: teacher = the base model with the fact in
  the prompt (adapters disabled via PEFT `disable_adapter()`); student = the
  adapter without the fact. KL(teacher_T ‖ student_T)·T² + α·CE on the target
  tokens the two sides share. No KV bank anywhere in the path.
- Ground-truth tests: zero-init adapter -> teacher ≡ student exactly; after
  training the student diverges while the teacher is bit-unchanged (no leak
  into base weights); single-wording contract enforced via spy.

### 4. Greedy decoding for knowledge-presence claims
- `free_form_recall` now defaults to greedy (`decoding="greedy"`), with
  `"sampled"` retained for comparability with pre-revision runs; result dict
  records the mode. (`evaluate_recall`, the 07/08 headline, was already greedy.)

### 5. The ladder experiment
- `12_localisation.py` — five arms:
  A `attn_top` (control: original substrate) · B `mlp_mid` (placement only) ·
  C `mlp_mid_para` (+diversity) · D `distill_mlp_mid` (full fix) ·
  E `distill_attn_top` (signal-vs-placement isolator).
  Plain AdamW, no SLEEP safety machinery (the question is where/how to write),
  greedy eval, JSON out, seeds via `--seed` (10_run_seeds-compatible).
- `run_localisation.sh` — 5 arms × 2 seeds × {Qwen-7B, Llama-8B} ≈ 5–6 h ≈ $5.

### 6. Paper
- Related Work gains a "Knowledge editing and localisation" paragraph
  (Geva 2021; ROME 2022; MEMIT 2023) — the gap a reviewer would have flagged —
  framing the reframe: editing methods prove single-fact writes at preserved
  capability are achievable, so our failure is placement/method, not
  impossibility. Full reframing of Results/Discussion waits for the ladder data.

## Predictions (written before the runs)
- If the localisation hypothesis is right: D ≫ A on greedy DRA at sane BCP,
  with C > B > A ordering the contributions; E vs D separates signal from
  placement.
- If D ≈ A ≈ floor: the hypothesis is falsified at this scale and the
  stability–plasticity framing survives with much stronger evidence.

## Next
Fresh pod (A6000 recommended) → upload → `pod_setup.sh` → `run_localisation.sh`.
