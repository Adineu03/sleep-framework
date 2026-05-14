# Experiment 03 v2: Qwen2.5-7B — delta_max Saturation Finding

**Date:** 2026-04-30
**Hardware:** RunPod RTX 5090, 32GB VRAM
**Model:** Qwen2.5-7B (7.616B params), bfloat16
**Config:** `experiments/configs/qwen7b.yaml` with `use_real_mu_surprise: false`
**Sleep run:** `delta_max=0.001` (formalization default), 100 steps, 2 replays

---

## What Happened

Pipeline ran end-to-end: 4/5 sanity checks passed.
**0 consolidations succeeded.** W_cons was rolled back during the validate phase.

| Stage | Result | Notes |
|:---|:---|:---|
| Tagging | 7 tags from 5 facts | Working |
| PRP allocation | 2/3807 | Selective |
| Replay generation | 2/2 | Working |
| Quality check (with bypass) | 2/2 accepted | Bypass worked |
| Sleep training | 100 steps, loss 1.45 → 1.40 | Loss decreased ~3% |
| Hard clipping | 36 parameters clipped at every step from step 30 onward | **Saturated** |
| Validation | 0/2 passed | **Failed** |
| Cleanup | Rolled back W_cons | W_cons restored from checkpoint |
| BCP | 0.9965 | Preserved (because of rollback) |
| DRA | 0.000 | No facts recalled |

## The Finding

`delta_max = 0.001` (formalization default) caps the per-parameter weight change
during sleep training. On the 7B model with rank-16 LoRA, this bound is hit
**within 30 training steps**, after which 36/36 LoRA parameters get hard-clipped
on every step. Training then can't make further progress.

The validation phase computes the post-training improvement on the replay
candidates and decides if W_cons should be retained. With clipped training, the
improvement is below threshold, so cleanup rolls W_cons back to its initial
state. That's why:

- **BCP = 0.9965** (essentially unchanged — W_cons was reset)
- **DRA = 0.0** (no consolidation = no recall)

This is **the safety system working as designed.** It correctly detected that
training didn't improve things meaningfully, and protected the model. But it
also tells us the safety constraint is too tight for this scale.

## Why the Default Doesn't Work at 7B

The formalization derives `delta_max` from a uniform safety margin
(0.001 per parameter). But the *capacity* needed to encode a memory in a LoRA
adapter scales with the model's hidden dimension and rank.

Qwen2.5-7B has `hidden_size = 3584`, so a rank-16 LoRA on `v_proj` has
`3584 * 16 * 2 = 114,688 params per layer * 9 layers * 2 matrices = ~2M params`.
The total magnitude of weight change to encode a meaningful memory is much
greater on this scale than on GPT-2 Small (`hidden_size = 768`).

For comparison, our GPT-2 PoC tuned `delta_max = 0.01` (10x the default).
On 7B, we likely need `delta_max = 0.01–0.05`.

## Proposed Formalization Amendment

**Q3.4 amendment:** `delta_max` should scale with model size, not be a
fixed constant. Two reasonable parameterizations:

1. **Scale with hidden dim:** `delta_max(d) = delta_max_base * sqrt(d / 768)`
   where 768 is the GPT-2 Small reference. For Qwen 7B this gives ~0.0022.

2. **Scale with model billions:** `delta_max(B) = delta_max_base * (B / 0.124)^0.25`.
   For Qwen 7B this gives ~0.0028.

Neither alone gets us to the 0.01 we know works empirically. The likely
correct answer is that the **per-parameter bound is the wrong abstraction** —
what matters is the *Frobenius norm* of the LoRA update, scaled appropriately
to the model.

## What's Cool Despite the Failure

- **The safety system worked.** No catastrophic forgetting, no model corruption.
  BCP = 0.9965 means the model is essentially unchanged.
- **The pipeline is integrated.** All 6 modules ran together for the first time
  on a 7B model. No crashes, no NaNs, no infinite loops.
- **Training loss did decrease 1.45 → 1.40 over 100 steps.** The system
  *wanted* to learn — it was held back by the safety constraint.

## Next Run

- Set `delta_max = 0.01` (matching our empirically tuned GPT-2 value)
- Keep everything else at formalization defaults
- This becomes Experiment 03 v3
- Expected: actual consolidation, DRA > 0

## Open Question for Formalization

Should `delta_max` become a *configurable* parameter (current) or a *derived*
parameter from model dimensions (proposed)?

Recommendation: keep `delta_max` configurable for research, but add a
`delta_max_scale_factor` config that lets us auto-derive it from
`hidden_dim` for production. This needs a proper amendment to the formalization.
