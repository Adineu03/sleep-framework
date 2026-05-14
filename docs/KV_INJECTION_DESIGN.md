# KV Memory Injection — Design Doc (Phase A.1)

**Date:** 2026-04-30
**Status:** Awaiting sign-off before implementation
**Scope:** Replace LoRA-based W_fast with a direct-write KV memory bank that injects stored K/V into transformer attention as a prefix.

---

## Mental Model (One Paragraph)

W_fast becomes a bounded set of per-layer (K, V) tensors stored in a memory bank. When a tag is created during wake, we extract the model's own K and V projections of the tagged span's hidden states from each adapted layer and append them to the bank. During inference, those stored K/V are injected as a prefix into the attention computation of those same layers — the model's existing attention machinery does the retrieval automatically by computing attention scores between current queries and stored memory keys. No gradient. No optimization. One exposure → one write. Decay = eviction. Sleep consolidation reads the bank, generates replays, trains W_cons, then clears the bank entries.

---

## Key Design Decisions

### Decision 1: Position Encoding — Memory at NEGATIVE Positions

**The choice:** Store raw (pre-RoPE) K and V. At read time, apply RoPE to memory K with position IDs `[-n_mem, ..., -1]`. The current sequence keeps its own RoPE positions `[0, ..., seq_len-1]`.

**Why this and not alternatives:**

| Strategy | How memory positions look | Issue |
|:---|:---|:---|
| A — Store post-RoPE K | At their original sequence positions | Original positions can be far from current positions; long-range RoPE decay penalizes attention to memories |
| B — Memory at `[0, n_mem-1]`, shift current to `[n_mem, n_mem+seq_len-1]` | Memory looks like a system prompt | Requires re-applying RoPE to current sequence at shifted positions; expensive and invasive |
| **C — Memory at `[-n_mem, -1]`** | Memory looks like "tokens just before this sequence" | **Clean. Current sequence keeps original RoPE. Memory K is RoPE'd at negative positions. Relative distances all negative, well within model's training distribution.** |

RoPE supports negative position IDs natively — `inv_freq * position` works for negative positions, producing reverse rotation. The relative-position formulation `score(q_i, k_j) ∝ q_i^T R(p_j - p_i) k_j` means: with `p_j ∈ [-n_mem, -1]` and `p_i ∈ [0, seq_len-1]`, `p_j - p_i ∈ [-(n_mem+seq_len-1), -1]` — all negative, all distances the model has seen during training (queries attending to past tokens).

Memory feels to the model exactly like "context from a few tokens ago" without requiring any positional surgery on the current sequence.

### Decision 2: Hook Mechanism — Monkey-Patch `Qwen2Attention.forward`

**The choice:** For each adapted layer's `Qwen2Attention` module, replace its `forward` method with our augmented version.

**Why this and not alternatives:**

| Approach | Verdict |
|:---|:---|
| Subclass `Qwen2Attention` and swap the module | Invasive; HF's model-loading pipeline doesn't expect subclass swaps |
| Wrap `Qwen2Attention` in a new `nn.Module` | Cleanest in theory, but HF iterates `model.layers[i].self_attn` directly; wrapper would break that traversal |
| **Monkey-patch `forward`** | Minimal surface area; bound-method replacement; PEFT does this kind of patching for adapter integration |

The patched `forward` calls into the original implementation for the standard path and only adds the memory injection step.

### Decision 3: Causal Mask — Memory Always Visible

**The choice:** Construct an extended attention mask of shape `(batch, 1, q_len, n_mem + kv_len)`. Memory columns get value 0 (no masking — fully visible to all queries). Current sequence columns retain the original causal mask values.

**Why:** Memory is information the model "knows." Causally restricting access to it would be wrong — a query at position 0 should be able to attend to a memory just as well as a query at position 100. This matches how prefix-tuning, system prompts, and `past_key_values` from prior contexts already work.

### Decision 4: What to Store — Pre-RoPE K, Raw V

**The choice:** When writing to bank, store K from `self.k_proj(hidden_states)` BEFORE `apply_rotary_pos_emb`. Store V directly from `self.v_proj(hidden_states)` (V doesn't get RoPE).

**Why:** RoPE depends on position. We don't know what the read position will be at write time, and we want to apply RoPE with negative positions at read time. Storing pre-RoPE K means we're storing the position-independent semantic content; RoPE gets applied fresh on each read.

This requires intercepting K BEFORE RoPE inside the attention forward, then storing it. Implementation: the patched forward computes Q/K/V from projections, branches: if "we're writing this span," store K/V before RoPE; then apply RoPE; then continue normally.

A simpler write path: do a separate forward pass with `output_hidden_states=True`, get `h` at each adapted layer, then manually compute `K = h @ W_K^T`, `V = h @ W_V^T`. Both work. I'll use the separate-pass approach in implementation because it doesn't entangle write logic with forward logic.

---

## Architecture Diagram

```
                   ┌─────────────────────────────────────────┐
                   │           QWEN 2.5 7B MODEL              │
                   │  ┌──────────┬──────────┬──────────────┐  │
                   │  │ Layer 0  │   ...    │  Layer 18    │  │
                   │  │ (frozen) │          │  (frozen)    │  │
                   │  └──────────┴──────────┴──────────────┘  │
                   │  ┌──────────┬──────────┬──────────────┐  │
                   │  │ Layer 19 │   ...    │  Layer 27    │  │  ← adapted layers
                   │  │  ATTN    │          │   ATTN       │  │     (top 33%)
                   │  │  (KV-    │          │   (KV-       │  │
                   │  │  inject) │          │   inject)    │  │
                   │  └────┬─────┴──────────┴──────┬───────┘  │
                   └───────┼─────────────────────────┼────────┘
                           │                         │
                           │ read at inference       │ read
                           ▼                         ▼
                   ┌─────────────────────────────────────────┐
                   │          KV MEMORY BANK                  │
                   │                                           │
                   │  layer_19: [(K, V)_tag1, (K,V)_tag2, ...]│
                   │  layer_20:  ...                           │
                   │  layer_21:  ...                           │
                   │  layer_22:  ...                           │
                   │  layer_23:  ...                           │
                   │  layer_24:  ...                           │
                   │  layer_25:  ...                           │
                   │  layer_26:  ...                           │
                   │  layer_27:  ...                           │
                   │                                           │
                   │  total tokens: bounded by N_max           │
                   │  per-tag eviction supported               │
                   └────────────────▲─────────────────────────┘
                                    │ write (one-shot, on tag creation)
                                    │
                   ┌─────────────────────────────────────────┐
                   │        TAGGING LAYER                     │
                   │  surprise → spans → tag created          │
                   │  + extract K/V at adapted layers ────────┘
                   └─────────────────────────────────────────┘
```

---

## API Sketch

### `KVMemoryBank` (the data structure)

```python
class KVMemoryBank:
    """Per-tag, per-layer storage of pre-RoPE K and raw V tensors."""

    def __init__(
        self,
        adapted_layer_indices: list[int],
        num_kv_heads: int,
        head_dim: int,
        max_total_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        ...

    def append(
        self,
        tag_id: str,
        layer_kvs: dict[int, tuple[Tensor, Tensor]],  # layer_idx -> (K, V), shape (n_tokens, num_kv_heads, head_dim)
    ) -> None:
        """Store K/V for a tagged span across adapted layers."""

    def get_for_layer(self, layer_idx: int) -> tuple[Tensor, Tensor] | None:
        """All stored K, V for one layer, concatenated. None if empty.
        Returns: (K of shape (n_total_tokens, num_kv_heads, head_dim),
                  V of shape (n_total_tokens, num_kv_heads, head_dim))
        """

    def evict(self, tag_id: str) -> bool:
        """Remove all entries for this tag_id. Returns True if found."""

    def clear(self) -> None:
        """Remove all entries (called at end of successful sleep cycle)."""

    @property
    def n_total_tokens(self) -> int: ...

    @property
    def n_tags(self) -> int: ...

    @property
    def at_capacity(self) -> bool: ...
```

### `KVInjector` (the hook installer)

```python
class KVInjector:
    """Installs a patched forward method on each adapted attention module.

    The patched forward:
      1. Runs the standard Q/K/V projections + RoPE for the current sequence.
      2. If memory bank has entries for this layer, retrieves stored K/V,
         applies RoPE at negative position IDs, prepends to current K/V,
         and extends the attention mask so memory is visible to all queries.
      3. Calls the standard attention interface.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        bank: KVMemoryBank,
        rope_module: nn.Module,  # the model's Qwen2RotaryEmbedding instance
    ):
        ...

    def install(self) -> None:
        """Replace forward on each adapted layer's self_attn module."""

    def uninstall(self) -> None:
        """Restore original forward methods."""
```

### Write API on `DualWeightSystem`

```python
def write_to_kv_bank(
    self,
    tag_id: str,
    token_ids: Tensor,     # the full input sequence
    span_start: int,        # tagged span start (inclusive)
    span_end: int,          # tagged span end (exclusive)
    device: str = "cuda",
) -> None:
    """Extract K/V for the tagged span at each adapted layer,
    store in the bank under tag_id."""
```

Implementation:
1. Forward pass with `output_hidden_states=True` on `token_ids[:span_end]` (full causal context up to span end)
2. For each adapted layer index `l`:
   - Get hidden states at layer `l`'s input → shape `(seq_len, hidden_dim)`
   - Slice `[span_start:span_end]` → `(span_len, hidden_dim)`
   - Apply `model.layers[l].self_attn.k_proj` → `(span_len, num_kv_heads * head_dim)`
   - Apply `model.layers[l].self_attn.v_proj` → same shape
   - Reshape to `(span_len, num_kv_heads, head_dim)` for K and V
3. `bank.append(tag_id, {l: (K_l, V_l) for l in adapted_layers})`

---

## What Stays Identical From Existing Code

- `TaggingLayer` — no changes to surprise computation, thresholding, span segmentation, or tag dataclass.
- `PRPSystem` — operates on tags exactly as before. Adds one method for memory eviction: when a tag is consolidated or expired, call `bank.evict(tag_id)` alongside existing tag cleanup.
- Sleep engine wake-phase trigger — same.
- `W_cons` — still LoRA on V/O projections of the same adapted layers. Trained on replays as before.
- Evaluation suite — fully reusable.

## What Changes

| File | Change |
|:---|:---|
| `sleep/weights/kv_memory.py` | NEW — KVMemoryBank class |
| `sleep/weights/kv_injection.py` | NEW — KVInjector + patched forward |
| `sleep/weights/__init__.py` | DualWeightSystem refactor: replaces `update_fast_weights` with `write_to_kv_bank`; W_fast no longer holds LoRA |
| `sleep/weights/lora.py` | Trim — only W_cons LoRA remains |
| `sleep/weights/fast_update.py` | DELETE — superseded by KV write path |
| `sleep/weights/plasticity.py` | Trim — `delta_max` clipping now applies only to W_cons |
| `tests/test_weights/test_kv_memory.py` | NEW |
| `tests/test_weights/test_kv_injection.py` | NEW |
| `tests/test_weights/test_dual_weight_system.py` | UPDATE — write_to_kv_bank tests |
| `experiments/scripts/06b_kv_diagnostic.py` | NEW — analog of 06 using KV memory |

Estimated diff size: ~1500 lines new code, ~400 lines deleted, ~300 lines modified.

## Validation Plan (Phase A.7)

`06b_kv_diagnostic.py` mirrors `06_wfast_only_diagnostic.py`:

1. Load Qwen 7B, install `KVInjector`
2. Process 200 facts: tag them, write each tagged span's K/V to the bank
3. Skip sleep cycle entirely
4. Run 3 recall formats with the bank still populated (memories injected at every forward pass)
5. Compute BCP

**Pre-registered pass thresholds:**
- MC accuracy on Tagged ≥ 0.50 (vs 0.23 for LoRA W_fast)
- Free-form score on Tagged ≥ 0.20 (vs 0.0067 for LoRA W_fast)
- BCP < 1.05

If these don't hold:
- Investigate injection mechanics (am I extracting K/V correctly? Is RoPE being applied at the right positions? Is the mask extended correctly?)
- If verifiably correct mechanics still produce poor recall, fall back to Path 1 with strongest possible narrative ("we tried both gradient-based and direct-write encoding mechanisms; here's why both fail and what's needed")

## Open Questions for Sign-off

1. **Negative-position RoPE — is there any case where the model's training never saw very large negative relative distances?** Qwen's max sequence length is 32K. If we have say 200 tagged tokens and a query at position 100, the relative distance is 200+100 = 300 — well within trained range. But if we someday store thousands of tokens, we could hit positions outside the model's RoPE training distribution. For our 200-fact experiment this is a non-issue. For production this would need bounded memory with eviction below some threshold.

2. **Should we adapt all attention layers or only top-third (matching current LoRA setup)?** Current `adapted_fraction = 0.333` means top 9 of 28 layers. This was chosen for LoRA's compute footprint. For KV injection there's no training cost — but read cost scales with adapted layers. I'd start with top 9 to match the formalization and revisit if recall is layer-sensitive.

3. **Write-time mode.** During tag creation, we run a forward pass to extract hidden states. Should this pass have an empty memory bank (clean extraction) or the current bank (so the extraction "sees" prior memories)? Cleaner: empty bank during writes. Implementation needs a `bank.disable()` / `bank.enable()` context manager.

4. **Should memory K/V participate in the model's KV cache during generation?** When generating replays during sleep, the model uses `past_key_values` for autoregressive efficiency. Memory K/V should be visible at every step but should NOT be part of the cache that grows token-by-token. This is solvable by extending the cache wrapper to keep memory and generation-cache as separate slabs.

---

## Risk Register

| Risk | Probability | Mitigation |
|:---|:---:|:---|
| RoPE at negative positions produces numerically unexpected results | Low | Mathematically valid; verify with unit test on tiny model |
| Monkey-patched forward breaks gradient flow during W_cons training | Medium | Memory K/V are detached (no grad). W_cons LoRA gradients flow through the model normally because memory is concatenated in the attention forward, not in the parameters |
| HF transformers updates change `Qwen2Attention.forward` signature, breaking the patch | Medium | Pin transformers version; document the patch points; add a test that fails loudly if signature changes |
| Memory injection slows inference noticeably | Low | At our scale (~400 tokens stored), the extra K/V is small (~3.7 MB per layer); attention is O((seq+mem) × seq), so seq=50, mem=400 gives ~10× slower attention but absolute time is still milliseconds per layer |
| Direct-write encoding works at write time but the model fails to attend to memories at recall time | Medium | This is the scientific risk. If it materializes, our hypothesis is wrong and Path 1 becomes the fallback. The pre-registered thresholds will tell us. |

---

## My Confidence

Implementation: ~80% confident this works as designed. The core mechanism (extending K/V via past_key_values mechanism, prefix tokens via attention) is well-trodden ground. The novelty is using it for our purpose, not in inventing the mechanism.

Scientific outcome: ~75% confident KV injection produces meaningfully higher recall than LoRA W_fast. The architectural argument is sound; Memorizing Transformers (Wu et al. 2022) demonstrated the approach works for retrieval-style tasks at scale; we're applying it to a smaller-scale fact-recall task with a tagging-driven write policy.

The 25% downside scenario is that something subtle about how Qwen processes attention masks or RoPE-rotated keys interacts poorly with the injection. We'll catch this in Phase A.7.

---

## What I'm Asking You to Sign Off On

1. **Strategy: memory at negative positions, store pre-RoPE K, monkey-patch the forward.** I'll implement this unless you push back.
2. **Open Question 1 (negative-position RoPE distribution):** Acceptable for 200-fact scale; flag for production.
3. **Open Question 2 (adapted layers):** Top 9 to match formalization.
4. **Open Question 3 (write-time mode):** Empty bank during writes.
5. **Open Question 4 (KV cache during generation):** Memory and gen-cache separate, both visible.
6. **Pre-registered Phase A.7 thresholds:** MC ≥ 0.50, free-form ≥ 0.20, BCP < 1.05.

Tell me if any of those are wrong, or if you have additional questions to add to the register. Then I start A.2 (the data structure).
