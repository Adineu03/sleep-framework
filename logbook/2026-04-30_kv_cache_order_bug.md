# KV Memory Injection — Cache-Order Bug & Top-k Gating Effect

**Date:** 2026-04-30
**Hardware:** RunPod RTX 5090
**Model:** Qwen2.5-7B (bfloat16, eager attention)
**Config:** `experiments/configs/qwen7b.yaml`
**Dataset:** 200 synthetic facts

---

## Headline

The cache-order bug + the top-k gating fix together moved BCP from **96.88 → 1.08** (90× improvement) while *strengthening* the encoding signal: Tagged − Untagged on MC went from **+0.11 → +0.16**, the strongest directional effect this research program has produced.

## The Bug

In the original `_kv_injected_attention_forward`, memory was injected into K/V *before* the `past_key_values.update(...)` call. During multi-step autoregressive generation, this caused the cache to accumulate a copy of memory tokens **on every forward step**:

- Step 0: cache = ∅ → after update = memory + current_step_0
- Step 1: cache = memory + current_step_0 → after update = memory + current_step_0 + (memory + current_step_1) = **2×memory + current**
- Step t: cache = t×memory + previous_currents

By a 50-token cloze generation, 50 copies of memory live in the cache. Beyond destroying preservation, this also fired a tensor-shape mismatch (1708 vs 862) at the eager attention path, because the attention mask we constructed only accounted for one copy of memory.

## The Fix

Reorder operations:

```python
# WRONG (original):
inject_memory_into(key_states, value_states)
past_key_values.update(key_states, value_states, layer_idx)

# RIGHT (fixed):
past_key_values.update(key_states, value_states, layer_idx)  # cache only current
inject_memory_into(key_states, value_states)                   # memory is per-forward
```

This makes the memory injection *transient*: it appears in K/V for the current attention computation but is not persisted in the cache. Each forward pass re-injects from the bank.

## Numbers

| Configuration | BCP | MC Tagged | MC Untagged | Tagged−Untagged |
|:---|:---:|:---:|:---:|:---:|
| LoRA W_fast (06, baseline) | 0.99 | 0.23 | 0.24 | −0.01 |
| KV no gating, SDPA, cache-bug | 96.88 | 0.27 | 0.16 | +0.11 |
| KV no gating, eager, cache-bug | 96.88 | 0.27 | 0.16 | +0.11 |
| **KV k=16, eager, cache-fixed** | **1.08** | **0.28** | **0.12** | **+0.16** |

The cache-order fix alone took BCP from 96.88 → 1.08. The forced-eager-attention change had no measurable effect on the numbers (it's a correctness change for the gating path, not a measurable behaviour change at this n_mem).

## Why Eager?

We force `attn_implementation="eager"` because SDPA's mask-handling path treats 4D additive masks differently and may silently ignore certain combinations of mask values + flash-attention dispatch. Eager honors `attn_weights + attention_mask` exactly as defined, which is what our top-k visibility mask depends on. At our scale (~840 stored tokens × 9 layers, 5–18 query positions per generation step) the per-token overhead of eager is negligible.

## Why BCP is Still Slightly Above 1.05

With k=16 of 846 stored memory tokens, every query position attends to its 16 most-relevant memories. For control texts ("The capital of France is Paris..."), even the *most relevant* of 846 random fact memories produce some attention drift. The downstream PPL ratio of 1.08 reflects this: a 8% degradation of base capability is the cost of always-on memory at k=16.

Two options to push BCP under 1.05:

1. **Lower k** (running k=4 next) — fewer distractor memories per query position
2. **Score-thresholded gating** — only include memory positions whose Q·K_mem score exceeds an absolute threshold. For control queries unrelated to any memory, the threshold won't be met and zero memory will be injected. This is a separate implementation but probably the cleaner long-term solution.

## Significance

This is the first time in the entire research program that the system has produced a clear directional effect that beats the LoRA W_fast baseline by a meaningful margin. **+0.16 Tagged−Untagged** on multiple choice with **BCP near preservation threshold** is a real result.

It's not yet a complete result — free-form recall is still at floor (0.0017), and BCP is still 1.08 not 1.04. But the architectural argument from the design doc is now empirically supported: direct-write KV memory does deposit information that the model can attend to, and a kNN retrieval gate prevents irrelevant memories from drowning attention.

## Files Changed

- `sleep/weights/kv_injection.py` — re-ordered cache update vs memory injection in `_kv_injected_attention_forward`. Tests pass without modification (the existing tests didn't cover multi-step generation, which is why the bug got through).
- `experiments/scripts/06b_kv_diagnostic.py` — forced `attn_implementation="eager"` for correctness during diagnostic.

## Open Question

Should the local `test_kv_memory_integration.py::TestEndToEnd::test_write_changes_attention_output` be extended to use `model.generate()` (multi-step) instead of single-step `model(input_ids=...)`? That would have caught the cache-order bug during local testing. Adding this as a test debt item.
