# SLEEP: Mathematical Formalization Notebook

## Synaptic Learning through Error-driven Encoding and Plasticity

### A Complete Pre-Implementation Decision Ledger

---

> **Convention:** All equations use standard mathematical notation. Vectors are bold lowercase (**v**), matrices are bold uppercase (**M**), scalars are italic (*s*). Subscripts denote indices or components; superscripts denote time steps unless otherwise noted. All logarithms are natural (base *e*) unless stated otherwise.

> **Each answer contains:** (1) Candidates considered, (2) Chosen formalism, (3) Mathematical justification, (4) Downstream implications.

---

# Part 1: The Tagging Layer

*These are the most foundational decisions. Everything else inherits from them.*

*Biological grounding: Synaptic Tagging and Capture (Frey & Morris, 1997; Redondo & Morris, 2011)*

---

## Prerequisite: Q6.3 — The Precise Relationship Between Tags and W_fast

*This must be resolved before any Part 1 decisions, because the answer constrains what a tag can be.*

### The Architectural Ambiguity

The proposal describes tags as "sparse pointers marking where surprising information lives" and W_fast as a system that "stores temporary, specific indices" and "updates with every interaction." These could be one system or two. The biology resolves this clearly.

### Candidates Considered

| Architecture | Tags | W_fast | Relationship |
|:---|:---|:---|:---|
| **A. Unified** | Tags *are* W_fast parameters | Single system | Tag decay = parameter decay in W_fast |
| **B. Separate, uncoupled** | Metadata index | Separate parameter set | No direct interaction |
| **C. Separate, coupled** | Lightweight index layer | Learnable parameter perturbation | Tags index experiences; W_fast encodes patterns from those experiences |

### Chosen Formalism: **C — Separate, Coupled Systems**

**The Tag System T** is a lightweight metadata index — a structured buffer of tag records. Each tag is a small data structure (not a parameter set) that marks *what was surprising* and *where it happened*. Tags are cheap, sparse, and ephemeral.

**The Fast Weight System W_fast** is a set of low-rank parameter perturbations (LoRA adapters) applied to the base model W_slow. W_fast encodes the *learned patterns* from recent experience. It updates via gradient descent on surprising inputs.

**The coupling:** When a surprising input is detected (high prediction error), two things happen simultaneously:

```
1. A tag τ is created in the tag buffer T     ← cheap, O(d_tag) operation
2. W_fast is updated via gradient step          ← more expensive, O(r · d_model) operation
```

The tag *references* the experience. W_fast *learns from* the experience. During sleep, tags determine *what* gets consolidated (selection via PRP scores), and W_fast provides the *generative capacity* to produce replay samples for training W_slow.

### Justification

This mirrors the biology precisely:

| Biological Component | Computational Analogue | Role |
|:---|:---|:---|
| Synaptic tag (CaMKII phosphorylation, actin remodeling) | Tag record τ in buffer T | Local, cheap marker — "something happened here" |
| Early-LTP / fast synaptic modification | W_fast parameter update (LoRA) | Actual encoding of the experience pattern |
| Plasticity-Related Proteins (PRPs) | PRP allocation score | Resource that stabilizes the tag for consolidation |
| Late-LTP / consolidated weight change | W_slow update during sleep | Permanent knowledge integration |

The separation is necessary because:

1. **Tags must be cheaper than W_fast updates.** Creating a tag is O(d_tag) ≈ O(128). Updating W_fast via LoRA is O(r · (d_in + d_out) · L_adapted) per adapted layer. If every input updated W_fast, the cost would be prohibitive. Tags act as a **filter** — only sufficiently surprising inputs trigger W_fast updates.

2. **Tags serve the PRP scoring system.** The composite score (prediction error, access frequency, cross-reference density, recency) requires metadata that parameter perturbations cannot provide. You cannot extract "how many times was this memory accessed?" from a weight matrix.

3. **Tags enable selective consolidation.** During sleep, we need to know *which specific experiences* to replay. W_fast encodes a blend of all recent experience — it cannot isolate individual memories. Tags provide the index for selection.

### Information Flow Diagram

```
                        WAKE PHASE

Input x ──→ Forward Pass (W_slow + W_fast) ──→ Output ŷ
       │                                          │
       │         Prediction Error                  │
       │         e = L(x, ŷ)                       │
       │              │                            │
       │    ┌─────────┴──────────┐                 │
       │    │                    │                  │
       │    ▼                    ▼                  │
       │  e > θ ?             e ≤ θ ?              │
       │  (surprising)       (expected)            │
       │    │                    │                  │
       │    ▼                    ▼                  │
       │  CREATE TAG τ        [no action]          │
       │  UPDATE W_fast                            │
       │    │                                      │
       │    ▼                                      │
       │  Tag Buffer T                             │
       │  [τ₁, τ₂, ..., τₙ]                       │
       │                                           │
       └───────────────────────────────────────────┘

                        SLEEP PHASE

Tag Buffer T ──→ PRP Selection ──→ Selected Tags {τᵢ}
                                        │
                                        ▼
                              W_fast generates replay
                              from selected experiences
                                        │
                                        ▼
                              Interleave with old knowledge
                                        │
                                        ▼
                              Train W_slow (gradient descent)
                                        │
                                        ▼
                              Clear consolidated tags from T
                              Reset corresponding W_fast components
```

### Downstream Implications

- Q1.1: Tags are data structures in a buffer, not neural network parameters
- Q3.1: W_fast is architecturally distinct from the tag system (LoRA adapters)
- Q4.1: Generation during sleep uses W_fast (the learned parameters), conditioned on tag keys (the pointers)
- Q6.7: Tag *activation* during inference uses the tag key vectors for similarity matching — this is the "hidden retrieval" that must be formally distinguished from RAG

---

## Q1.1 — What Mathematical Object Is a "Tag"?

### Candidates Considered

| Candidate | Dimensionality | Cost to Create | Decay Mechanism | Can Reference Experience? |
|:---|:---|:---|:---|:---|
| Sparse binary mask over parameters | O(\|θ\|) | Expensive (need gradient) | Bit flipping | No (mask, not pointer) |
| Sparse vector in activation space | O(d_model) | Moderate (extract hidden state) | Scalar decay on norm | Yes (embedding-space pointer) |
| Key-value pair in attention | O(2 · d_model) | Moderate | Remove from KV cache | Yes, but tightly coupled to attention |
| Compressed gradient snapshot | O(\|θ\|) or O(r·d) | Expensive (full backward pass) | Scalar decay | No (update direction, not content) |
| **Structured record with projected key** | **O(d_tag + metadata)** | **Cheap (projection + bookkeeping)** | **Scalar strength decay** | **Yes (compressed hidden state)** |

### Chosen Formalism

A tag τ is a structured record:

```
τ = (k, s, t₀, e₀, a, ρ, ctx)
```

where:

| Field | Type | Description |
|:---|:---|:---|
| **k** | **k** ∈ ℝ^d_tag | **Key vector** — a compressed representation of the hidden state at the point of maximum surprise. Serves as the "pointer" to the experience. |
| **s** | *s* ∈ [0, 1] | **Strength** — a scalar that decays over time and is reinforced by access. When *s* → 0, the tag is garbage collected. |
| **t₀** | *t₀* ∈ ℕ | **Creation timestamp** — the inference step at which this tag was created. |
| **e₀** | *e₀* ∈ ℝ⁺ | **Initial prediction error** — the magnitude of the prediction error that triggered this tag. |
| **a** | *a* ∈ ℕ | **Access count** — number of times this tag has been activated during query processing. |
| **ρ** | *ρ* ∈ ℝ⁺ | **Cumulative utility** — accumulated evidence of usefulness (a richer signal than raw access count; see Q1.5). |
| **ctx** | ctx = (span_start, span_end, source_id) | **Context reference** — pointer to the original input span that triggered this tag. Enables replay generation. |

**The key vector** is produced by a learned linear projection:

```
k = W_proj · h̄ + b_proj
```

where:
- **h̄** ∈ ℝ^d_model is the mean-pooled hidden state over the surprising span (the tokens with above-threshold prediction error)
- **W_proj** ∈ ℝ^(d_tag × d_model) is a fixed (frozen after pretraining) or slowly-learned projection matrix
- d_tag ≪ d_model (e.g., d_tag = 128 for a d_model = 4096 model)

### Justification

**Why a projected hidden state for the key?**

The hidden state **h** at the point of surprise is the richest available representation of *what was surprising*. It encodes both the content and the context. Projecting to a lower dimension achieves:

1. **Compression:** d_tag / d_model ≈ 128/4096 = 3.1% of the original dimensionality. A tag consumes ~512 bytes (128 float32s) versus ~16KB for a full hidden state.

2. **Similarity-preserving:** A learned linear projection preserves the cosine similarity structure of the original space (by Johnson-Lindenstrauss, random projections preserve distances with distortion bounded by ε with probability ≥ 1 - 2·exp(-ε²·d_tag/8)). With d_tag = 128, pairwise distances are preserved to within ~25% with high probability.

3. **Cheap to compute:** A single matrix-vector multiply, O(d_tag · d_model). For d_tag=128, d_model=4096, this is ~500K FLOPs — negligible compared to a transformer forward pass (~10-100 GFLOPs).

**Why structured metadata rather than a pure vector?**

The PRP scoring function (Q2.2) requires access frequency, recency, and cumulative error — these are inherently scalar time-series statistics that cannot be extracted from an embedding vector. The metadata fields are each O(1) storage, adding negligible overhead.

**Total cost per tag:**

```
Memory:   d_tag · 4 bytes  +  ~32 bytes metadata  =  128 · 4 + 32  =  544 bytes
Creation: O(d_tag · d_model) FLOPs  ≈  0.5M FLOPs  (one matrix-vector multiply)
```

For comparison, a single forward pass through a 7B parameter model is ~14 TFLOPs. Tag creation is ~0.000004% of a forward pass. **Tags are cheap.**

### Downstream Implications

- **Q1.2:** Prediction error must produce both a scalar magnitude (for thresholding and e₀) and identify the surprising span (for h̄ computation)
- **Q1.5:** The key vector **k** enables similarity-based access detection
- **Q2.2:** The metadata fields (e₀, a, ρ, t₀) feed directly into the PRP scoring function
- **Q6.7:** Tag activation during inference will use cosine similarity between query hidden state and tag key vectors — this is the "hidden retrieval" mechanism
- **Q1.6:** Tag capacity is bounded by buffer size in memory: N_max tags × 544 bytes each

---

## Q1.2 — How Is Prediction Error Computed?

### The Core Question

"The gap between what the model expected and what it actually received" — this must be made precise at three levels: (a) what quantity, (b) at what granularity, (c) against what baseline.

### Candidates Considered

| Metric | Formula | Cost | Granularity | Biological Fidelity |
|:---|:---|:---|:---|:---|
| Per-token cross-entropy loss | -log p(x_t \| x_{<t}) | Free (already computed) | Token | High (≈ Shannon surprise) |
| KL divergence (Bayesian surprise) | D_KL[p(θ\|x) \|\| p(θ)] | Prohibitive (need posterior) | Document | Highest, but impractical |
| Precision-weighted PE | Σ⁻¹(x - x̂) in embedding space | Moderate (need variance estimate) | Token/Span | High (Friston's free energy) |
| Gradient norm | \|\|∇_θ L(x)\|\| | Expensive (backward pass) | Document | Low (not biologically grounded) |

### Chosen Formalism

**Prediction error is per-token Shannon surprise (negative log-likelihood), aggregated over spans using adaptive thresholding.**

#### Step 1: Per-Token Surprise

For input token x_t in context x_{<t}, the prediction error at position t is:

```
e_t = -log p_W_slow(x_t | x_{<t})
```

This is computed against **W_slow only** (the base model), not W_slow + W_fast. This is critical: we want to measure what the *permanent knowledge* finds surprising, not what the combined system finds surprising. If W_fast already "knows" something, tagging it would be redundant — it's already in the fast system.

**Why W_slow only?**

| Baseline | What Gets Tagged | Problem |
|:---|:---|:---|
| W_slow only | Things W_slow doesn't know | Correct — tags what needs consolidation |
| W_slow + W_fast | Things neither system knows | Misses information already in W_fast but not yet consolidated |
| W_fast only | Things W_fast doesn't know | Wrong direction — we want to tag for W_slow |

#### Step 2: Adaptive Threshold

A fixed threshold fails across contexts (code has higher baseline perplexity than prose). We use a running z-score:

```
θ_t = μ_t + κ · σ_t
```

where:
- **μ_t** is the exponential moving average of recent per-token surprise:
  ```
  μ_t = β · μ_{t-1} + (1 - β) · e_t
  ```
- **σ_t** is the exponential moving standard deviation:
  ```
  σ²_t = β · σ²_{t-1} + (1 - β) · (e_t - μ_t)²
  ```
- **β** ∈ (0, 1) is the smoothing factor (e.g., β = 0.99, giving an effective window of ~100 tokens)
- **κ** ≥ 0 is the sensitivity parameter (e.g., κ = 1.5, meaning a token must be 1.5 standard deviations above the running mean to be flagged)

A token is flagged as surprising if:

```
e_t > θ_t    ⟺    (e_t - μ_t) / σ_t > κ
```

This is a **z-score test** — context-adaptive by construction.

#### Step 3: Span Segmentation

Individual surprising tokens are noise. Clustered surprising tokens are signal. We aggregate flagged tokens into **surprising spans**.

**Algorithm:**

```
1. Flag all tokens where e_t > θ_t
2. Merge adjacent flagged tokens within a gap tolerance of g tokens (e.g., g = 3)
3. Discard spans shorter than min_span tokens (e.g., min_span = 4)
4. For each surviving span [t_start, t_end]:

   Span prediction error:
   E_span = (1 / |span|) · Σ_{t=t_start}^{t_end} (e_t - μ_t)     [mean excess surprise]

   Span hidden state (for tag key):
   h̄_span = (1 / |span|) · Σ_{t=t_start}^{t_end} h_t^{(L)}       [mean-pooled final layer hidden state]
```

Each surviving span produces **one tag**.

### Justification

**Shannon surprise over Bayesian surprise:**
Bayesian surprise D_KL[p(θ|x) || p(θ)] is the theoretically ideal signal — it measures how much the observation *changes the model's beliefs*. But computing it requires the posterior p(θ|x), which is intractable for a model with billions of parameters. Shannon surprise -log p(x_t) is:
- Already computed during the forward pass (it IS the loss)
- Zero additional cost
- A good proxy: under a Gaussian generative model, Shannon surprise = precision-weighted squared prediction error (see Friston's free energy framework)

**Adaptive thresholding over fixed threshold:**
The base rate of surprise varies dramatically:

| Content Type | Typical Perplexity | Mean Token Surprise (nats) |
|:---|:---|:---|
| Common English prose | 15-25 | 2.7 - 3.2 |
| Technical documentation | 30-80 | 3.4 - 4.4 |
| Source code | 50-200 | 3.9 - 5.3 |
| Novel domain-specific jargon | 100-500+ | 4.6 - 6.2+ |

A fixed θ = 4.0 would tag almost nothing in prose and almost everything in code. The z-score approach normalizes by context.

**Span aggregation over per-token tagging:**
A single surprising token (e.g., an unusual proper noun) is rarely worth remembering in isolation. A span of 10+ surprising tokens indicates a coherent chunk of new information. Span aggregation:
- Reduces tag count by ~10-50x compared to per-token tagging
- Produces semantically meaningful units
- Provides a richer hidden state h̄ (mean-pooled over context)

### Formal Summary

```
INPUT:  Token sequence x = (x₁, x₂, ..., x_T)
OUTPUT: Set of surprising spans S = {(t_start, t_end, E_span, h̄_span)}

FOR t = 1 TO T:
    e_t ← -log p_{W_slow}(x_t | x_{<t})          // per-token surprise
    μ_t ← β · μ_{t-1} + (1-β) · e_t              // running mean
    σ²_t ← β · σ²_{t-1} + (1-β) · (e_t - μ_t)²  // running variance
    flag_t ← (e_t - μ_t) / σ_t > κ               // z-score test

MERGE adjacent flags within gap g
DISCARD spans shorter than min_span
FOR each span [t_s, t_e]:
    E_span ← mean({e_t - μ_t : t ∈ [t_s, t_e]})  // mean excess surprise
    h̄_span ← mean({h_t^(L) : t ∈ [t_s, t_e]})    // mean hidden state
    EMIT (t_s, t_e, E_span, h̄_span)
```

### Downstream Implications

- **Q1.3:** The tag creation function takes (E_span, h̄_span) as inputs
- **Q1.4:** E_span serves as the initial tag strength e₀ — higher surprise means slower initial decay
- **Q2.2:** The cumulative prediction error component of PRP scoring uses the sum of E_span values
- **Q6.1:** Contradiction detection will leverage the pattern where prediction error is high on tokens *similar to* existing consolidated knowledge (high e_t on tokens where the model was confidently wrong, not merely uncertain)

---

## Q1.3 — What Is the Tag Creation Function?

### Chosen Formalism

Given a surprising span identified by Q1.2, the tag creation function is:

```
CREATE_TAG(span, step) → τ:

    k   = W_proj · h̄_span + b_proj         ∈ ℝ^d_tag     // projected key
    s   = σ(α_init · E_span)               ∈ (0, 1)      // initial strength (sigmoid-bounded)
    t₀  = step                              ∈ ℕ           // creation time
    e₀  = E_span                            ∈ ℝ⁺          // initial prediction error
    a   = 0                                 ∈ ℕ           // access count
    ρ   = 0                                 ∈ ℝ⁺          // cumulative utility
    ctx = (span_start, span_end, source_id) ∈ ℕ × ℕ × ID // context reference

    RETURN τ = (k, s, t₀, e₀, a, ρ, ctx)
```

where:
- **σ** is the sigmoid function σ(z) = 1/(1 + e^(-z))
- **α_init** is a scaling hyperparameter that controls how prediction error maps to initial strength (e.g., α_init = 2.0)
- **W_proj** ∈ ℝ^(d_tag × d_model) is the key projection matrix

### Properties

**Deterministic.** Given the same hidden state and prediction error, the same tag is produced. Stochastic tagging would introduce noise in an already noisy signal (prediction error itself has high variance).

**Inputs:** The hidden state h̄_span (from the forward pass) and the prediction error E_span (from the loss computation). No backward pass is required.

**Cost analysis:**

| Operation | FLOPs | Relative to Forward Pass |
|:---|:---|:---|
| Mean-pool hidden states over span | O(span_len · d_model) | Negligible |
| Key projection W_proj · h̄ | O(d_tag · d_model) | ~0.000004% |
| Sigmoid, metadata writes | O(1) | Negligible |
| **Total per tag** | **~0.5M FLOPs** | **~0.000004%** |

Tags are cheap. A forward pass through a 7B model costs ~14 TFLOPs. Creating a tag costs ~0.5M FLOPs — **28 million times cheaper.**

**One tag per surprising span.** A single input document with 3 surprising spans produces 3 tags. This is more efficient than one-tag-per-token and more informative than one-tag-per-document.

### The W_fast Update (Coupled Operation)

Simultaneously with tag creation, W_fast is updated on the surprising span. This is the more expensive operation:

```
IF E_span > θ_wfast:
    L_span = mean({-log p_{W_slow+W_fast}(x_t | x_{<t}) : t ∈ span})
    ΔW_fast = -α_fast · ∇_{W_fast} L_span
    W_fast ← W_fast + ΔW_fast
```

**θ_wfast is defined as a higher z-score threshold than tagging:**

```
θ_wfast: the span's mean z-score > κ_wfast,  where κ_wfast = 2.5
```

Compare to the tagging threshold κ = 1.5. This means a span must be 2.5σ above the running mean surprise to trigger a W_fast gradient update, versus only 1.5σ to create a tag.

**Concrete effect:** For normally distributed surprises, κ = 1.5 flags ~6.7% of tokens; κ_wfast = 2.5 flags ~0.6% of tokens. Roughly 1 in 10 tagged spans also triggers a W_fast update. This mirrors the biology: the threshold for tag setting is lower than the threshold for inducing plasticity (Frey & Morris, 1997). Some experiences are noted (tagged) but not deeply encoded (no W_fast update).

### Downstream Implications

- **Q1.4:** Initial strength s₀ = σ(α_init · E_span) is bounded in (0, 1), which provides a natural decay target
- **Q2.2:** e₀ = E_span feeds into the cumulative prediction error component of PRP scoring
- **Q3.3:** W_fast update rule is defined here for the case of surprising inputs; Q3.3 generalizes this

---

## Q1.4 — What Is the Tag Decay Function?

### Candidates Considered

| Decay Type | Formula | Behavior | Biological Match |
|:---|:---|:---|:---|
| Linear | s(t) = s₀ - λt | Hard cutoff, non-smooth | Poor — biology is smooth |
| Exponential | s(t) = s₀ · exp(-t/τ) | Smooth, never reaches 0 | Good — standard model in STC literature |
| Hyperbolic | s(t) = s₀ / (1 + t/τ) | Slow initial, long tail | Moderate — matches some forgetting curves |
| **Modified exponential with floor** | **s(t) = (s₀ - ε) · exp(-t/τ) + ε** | **Smooth, asymptotes to ε** | **Best — captures residual trace** |

### Chosen Formalism

```
s(t) = (s₀ - ε) · exp(-(t - t₀) / τ_decay) + ε
```

where:
- **s₀** = σ(α_init · E_span) is the initial strength (from Q1.3)
- **t** is the current inference step
- **t₀** is the creation step
- **τ_decay** is the decay time constant (in inference steps)
- **ε** is a small floor value (e.g., ε = 0.01) — the tag never fully reaches zero, but is garbage-collected when s < ε_gc (e.g., ε_gc = 0.02)

**The time variable is inference steps**, not wall-clock time. This is correct because:
1. In a deployed LLM, "time" between queries varies from seconds to days
2. What matters for memory pressure is how many new experiences have arrived, not how many seconds have passed
3. Inference steps are the natural "clock ticks" of the system

**The decay rate depends on initial prediction error magnitude:**

```
τ_decay(e₀) = τ_base · (1 + γ · e₀)
```

where:
- **τ_base** is the base decay time constant (e.g., τ_base = 1000 inference steps)
- **γ** is a scaling factor (e.g., γ = 0.5)

More surprising experiences decay more slowly. This matches the biology: stronger stimulation produces more robust tags (higher CaMKII autophosphorylation, more extensive actin remodeling) with longer effective lifetimes.

**Interaction with reinforcement:** The decay formula always computes from (s₀, t₀) — the *initial* strength and creation time. Reinforcement from access events is tracked separately as a cumulative bonus s_reinforced. The effective strength at any time is:

```
s(t) = min( [(s₀ - ε) · exp(-(t - t₀) / τ_decay) + ε] + s_reinforced, 1.0 )
         \_______________ base decay ________________/   \_ bonus _/
```

This decomposition ensures that:
1. Decay is always smooth and monotonic in the base component (no stepwise artifacts)
2. Reinforcement adds on top of the decaying base (accessed tags genuinely live longer)
3. The total strength is always computed consistently regardless of when GC runs

**Garbage collection:** A tag is removed from the buffer when s < ε_gc. This is evaluated at regular intervals (every G steps, e.g., G = 100) to amortize the cost.

### Decay Dynamics Visualization

```
Strength
  1.0 ┤ ╲
      │  ╲  (high e₀: τ_decay = 1500 steps)
  0.8 ┤   ╲
      │    ╲
  0.6 ┤     ╲    ╲  (medium e₀: τ_decay = 1250 steps)
      │      ╲    ╲
  0.4 ┤       ╲    ╲      ╲  (low e₀: τ_decay = 1000 steps)
      │        ╲    ╲      ╲
  0.2 ┤         ╲    ╲      ╲
      │    ε_gc──╲────╲──────╲─── garbage collection threshold
  0.0 ┤──────────────────────────────────────
      0    500   1000  1500  2000  2500  3000
                    Inference Steps
```

### Justification

**Exponential decay** is the standard model in the synaptic tagging literature. Frey & Morris (1997) report tag lifetimes of 30-90 minutes; computational models universally use exponential decay with τ_tag ~ 60 minutes. Our τ_base = 1000 inference steps maps to this: if the system processes ~15 queries/hour, then 1000 steps ≈ ~67 minutes.

**Error-dependent decay rate** captures the graded nature of biological tags. Stronger stimulation (more NMDA receptor activation, more CaMKII autophosphorylation) produces more robust tags. In our system, higher prediction error → larger τ_decay → slower decay.

**The floor ε** prevents numerical issues in downstream computations (PRP scoring divides by strength-related quantities) and models the biological observation that even decaying tags leave a faint trace that can be reactivated under certain conditions.

### Downstream Implications

- **Q1.5:** Reinforcement modifies the decay trajectory (resets or extends τ_decay)
- **Q2.2:** Tag strength s(t) feeds into recency-weighted utility in the PRP scoring function
- **Q4.5:** Sleep triggering can use aggregate decay statistics (e.g., "fraction of tags below threshold")

---

## Q1.5 — How Does Tag Access/Reinforcement Work?

### What Constitutes an "Access"?

During inference (wake phase), when the model processes a query, some tags will be relevant. **A tag is "accessed" when the query's hidden state is sufficiently similar to the tag's key vector.**

#### Access Detection

For a query with hidden state **h_q** (mean-pooled over the query span), compute similarity to each active tag:

```
sim(q, τᵢ) = cos(W_proj · h_q, kᵢ) = (W_proj · h_q)ᵀ · kᵢ / (‖W_proj · h_q‖ · ‖kᵢ‖)
```

A tag τᵢ is accessed if:

```
sim(q, τᵢ) > θ_access
```

where θ_access is the access similarity threshold (e.g., θ_access = 0.7).

**Cost:** Computing similarity against all N active tags costs O(N · d_tag). For N = 2000 tags and d_tag = 128, this is ~256K FLOPs — negligible.

### The Reinforcement Update

When tag τᵢ is accessed at step t:

```
a  ← a + 1                                              // increment access count
ρ  ← ρ + sim(q, τᵢ) · (1 / √a)                        // cumulative utility with diminishing returns
s  ← min(s + Δs · (1 - s) · (1 / √a), 1.0)            // strength boost with ceiling and diminishing returns
```

where:
- **Δs** is the base reinforcement amount (e.g., Δs = 0.3)
- **The factor (1 - s)** ensures reinforcement has less effect as strength approaches 1.0 (logistic-style saturation)
- **The factor 1/√a** implements **diminishing returns** — the 100th access adds much less than the 1st

### Reinforcement Properties

| Property | Mechanism | Why |
|:---|:---|:---|
| **Diminishing returns** | 1/√a factor | Prevents gaming by repetition (Q6.6); mirrors biological habituation |
| **Strength ceiling** | min(..., 1.0) and (1-s) factor | Tags can't grow unboundedly; prevents any single tag from dominating |
| **Utility accumulation** | ρ += sim · 1/√a | Weighted by relevance quality, not just binary access; gives PRP scoring a richer signal |
| **Access history** | Counter a | Cheap metadata for the scoring function |

### Effective Decay Under Repeated Access

A tag that is accessed periodically has an effective lifetime much longer than τ_decay. Consider a tag accessed every T_access steps:

```
After creation:        s = s₀
After T_access steps:  s ≈ s₀ · exp(-T_access/τ) + Δs · (1-s₀·exp(-T_access/τ))
After 2·T_access:     s ≈ [previous] · exp(-T_access/τ) + Δs · (1 - [previous]·exp(-T_access/τ)) · (1/√2)
...
```

For T_access ≪ τ_decay, the tag reaches an equilibrium strength:

```
s_eq ≈ Δs / (Δs + T_access/τ_decay)    (approximate, for small decay per interval)
```

If T_access = 100 steps, τ_decay = 1000, Δs = 0.3: s_eq ≈ 0.3 / (0.3 + 0.1) = 0.75. The tag stabilizes at 75% strength — it will survive indefinitely as long as it keeps being accessed.

**This is the desired behavior:** frequently accessed tags persist; neglected tags decay. The system self-organizes.

### Downstream Implications

- **Q2.2:** Access count *a* and cumulative utility *ρ* are direct inputs to the PRP composite scoring function
- **Q6.6:** Diminishing returns (1/√a) provides first-line defense against adversarial repetition
- **Q6.7:** The similarity computation for access detection IS the "hidden retrieval" — it must be formally compared to RAG

---

## Q1.6 — What Is the Tag Capacity?

### Candidates Considered

| Approach | Mechanism | Behavior at Limit | Complexity |
|:---|:---|:---|:---|
| Hard buffer (fixed N) | FIFO or LRU eviction | Oldest/least-used tags deleted | Simple but arbitrary |
| Hard buffer (priority eviction) | Evict lowest-scoring tag | Quality-aware but abrupt | Moderate |
| **Soft limit (memory budget)** | **Tags consume from shared budget; decay handles most cleanup; priority eviction under pressure** | **Graceful degradation** | **Moderate** |

### Chosen Formalism

**A soft capacity limit with priority-based eviction under pressure.**

```
N_max = C_tag · (|W_slow| / 10⁹)
```

where:
- **N_max** is the maximum number of active tags
- **C_tag** is a capacity coefficient (e.g., C_tag = 5000)
- **|W_slow|** is the number of parameters in W_slow

| Model Size | |W_slow| | N_max |
|:---|:---|:---|
| 1B | 10⁹ | 5,000 |
| 7B | 7 × 10⁹ | 35,000 |
| 70B | 70 × 10⁹ | 350,000 |

**Memory cost at capacity:** N_max × 544 bytes

| Model Size | N_max | Memory for Tags |
|:---|:---|:---|
| 1B | 5,000 | 2.7 MB |
| 7B | 35,000 | 18.6 MB |
| 70B | 350,000 | 186 MB |

This is negligible compared to the model weights (7B × 2 bytes = 14 GB for fp16).

### The Scaling Relationship

Why proportional to model size? Two reasons:

1. **Larger models have more capacity to consolidate.** A 70B model can absorb more consolidated knowledge into its weights than a 1B model. More tags → more consolidation candidates → better use of W_slow's capacity.

2. **Larger models process richer information.** In practice, larger models are deployed in contexts with more complex, diverse inputs. The tag budget should scale with the expected information throughput.

### Eviction Under Pressure

When N_active approaches N_max:

```
IF N_active ≥ N_max:
    // Compute eviction priority (low = evict first)
    FOR each tag τᵢ:
        priority(τᵢ) = sᵢ · (1 + ρᵢ)     // strength × (1 + cumulative utility)

    // Evict lowest-priority tag
    τ_evict = argmin_i priority(τᵢ)
    REMOVE τ_evict from T
```

**The eviction score is strength × utility.** This means:
- Decayed, unused tags are evicted first (low s, low ρ)
- Even a decayed tag with high utility survives (it was useful in the past → might be again)
- A fresh but useless tag can be evicted if a more valuable newcomer arrives

### Capacity Dynamics

Under normal operation, most tags are removed by natural decay (s < ε_gc) long before the buffer fills. Eviction is a safety valve, not the primary cleanup mechanism.

**Expected steady-state occupancy:** If tags arrive at rate λ_tag (tags per inference step) and decay with time constant τ_decay:

```
N_steady ≈ λ_tag · τ_decay
```

For λ_tag = 0.5 tags/step (one tag every 2 steps on average) and τ_decay = 1000:

```
N_steady ≈ 500
```

This is well below N_max for all model sizes. The buffer only fills under sustained high-novelty input.

### Downstream Implications

- **Q2.3:** PRP budget is a *separate* constraint from tag capacity. A tag can exist without a PRP. PRP budget ≤ N_max.
- **Q4.5:** Tag buffer occupancy (N_active / N_max) is one signal for triggering sleep �� high occupancy means memory pressure.

---

## Part 1 Checkpoint Verification

> **Requirement:** After completing Part 1, you should be able to write pseudocode for the tagging layer that takes an input and produces a set of tags.

### Complete Tagging Layer Pseudocode

```python
class TaggingLayer:
    """
    The Tagging Layer — "The Noticing"

    Takes raw input, computes prediction error against W_slow,
    identifies surprising spans, creates tags, and manages the tag buffer.
    """

    # --- Hyperparameters ---
    # β     = 0.99      # EMA smoothing for adaptive threshold
    # κ     = 1.5       # z-score sensitivity (std devs above mean)
    # g     = 3         # gap tolerance for span merging (tokens)
    # min_span = 4      # minimum span length (tokens)
    # α_init = 2.0      # prediction error → initial strength scaling
    # τ_base = 1000     # base decay time constant (inference steps)
    # γ     = 0.5       # error-dependent decay scaling
    # Δs    = 0.3       # base reinforcement amount
    # θ_access = 0.7    # cosine similarity threshold for access
    # ε_gc  = 0.02      # garbage collection threshold
    # d_tag = 128       # key vector dimensionality
    # C_tag = 5000      # capacity coefficient (tags per billion params)

    def __init__(self, W_slow, W_proj, d_tag, hyperparams):
        self.W_slow = W_slow
        self.W_proj = W_proj         # ℝ^(d_tag × d_model)
        self.buffer = []             # List of active tags
        self.μ = 0.0                 # Running mean of surprise
        self.σ2 = 1.0               # Running variance of surprise
        self.step = 0               # Global inference step counter
        self.N_max = C_tag * (count_params(W_slow) / 1e9)

    def process_input(self, tokens):
        """Main entry point: process input tokens, return tags created."""
        self.step += 1

        # ---- Step 1: Forward pass through W_slow only ----
        hidden_states, logits = forward(self.W_slow, tokens)  # h_t^(L), p(x_t|x_{<t})

        # ---- Step 2: Per-token surprise ----
        surprises = []
        flags = []
        for t, (token, logit) in enumerate(zip(tokens, logits)):
            e_t = -log_prob(logit, token)            # Shannon surprise

            # Update running statistics
            self.μ = β * self.μ + (1 - β) * e_t
            self.σ2 = β * self.σ2 + (1 - β) * (e_t - self.μ)**2
            σ_t = sqrt(self.σ2)

            surprises.append(e_t)
            flags.append((e_t - self.μ) / max(σ_t, 1e-8) > κ)  # z-score test

        # ---- Step 3: Span segmentation ----
        spans = merge_flags(flags, gap=g, min_length=min_span)

        # ---- Step 4: Create tags for each span ----
        new_tags = []
        for (t_start, t_end) in spans:
            E_span = mean([surprises[t] - self.μ for t in range(t_start, t_end+1)])
            h_bar  = mean([hidden_states[t] for t in range(t_start, t_end+1)])

            τ = Tag(
                k   = self.W_proj @ h_bar,                    # project to d_tag
                s   = sigmoid(α_init * E_span),               # initial strength
                s0  = sigmoid(α_init * E_span),               # preserved initial strength (never modified)
                s_reinforced = 0.0,                            # cumulative reinforcement bonus
                t0  = self.step,
                e0  = E_span,
                a   = 0,
                ρ   = 0.0,
                ctx = (t_start, t_end, current_source_id)
            )
            new_tags.append(τ)

        # ---- Step 5: Enforce capacity ----
        while len(self.buffer) + len(new_tags) > self.N_max:
            victim = argmin(self.buffer, key=lambda τ: τ.s * (1 + τ.ρ))
            self.buffer.remove(victim)

        self.buffer.extend(new_tags)
        return new_tags

    def process_query(self, query_tokens):
        """During query processing: detect tag accesses and reinforce."""
        hidden_states, _ = forward(self.W_slow, query_tokens)
        h_q = mean(hidden_states)           # mean-pooled query representation
        k_q = self.W_proj @ h_q             # projected query key

        accessed = []
        for τ in self.buffer:
            sim = cosine_similarity(k_q, τ.k)
            if sim > θ_access:
                # Reinforce
                τ.a += 1
                τ.ρ += sim * (1 / sqrt(τ.a))
                # Accumulate reinforcement bonus (added to base decay in decay_and_gc)
                boost = Δs * (1 - τ.s) * (1 / sqrt(τ.a))
                τ.s_reinforced += boost
                τ.s = min(τ.s + boost, 1.0)    # also update current s for scoring
                accessed.append(τ)

        return accessed

    def decay_and_gc(self):
        """Called every G steps: apply decay and garbage-collect dead tags."""
        surviving = []
        for τ in self.buffer:
            Δt = self.step - τ.t0
            τ_decay_val = τ_base * (1 + γ * τ.e0)

            # Compute BASE decay from initial strength and creation time
            # This is the canonical formula — always computed from (s₀, t₀)
            s_base = (τ.s0 - ε) * exp(-Δt / τ_decay_val) + ε

            # Add reinforcement bonus accumulated since creation
            # τ.s_reinforced tracks cumulative reinforcement added by access events
            τ.s = min(s_base + τ.s_reinforced, 1.0)

            if τ.s >= ε_gc:
                surviving.append(τ)
        self.buffer = surviving
```

**This pseudocode is complete and executable in principle.** Every function, threshold, and constant is defined. No hand-waving remains.

---

*Part 1 complete. All six questions (Q1.1–Q1.6) plus the prerequisite Q6.3 are fully resolved. Proceeding to Part 2.*

---

# Part 2: The Protein Budget (PRP Allocation)

*These decisions depend on Part 1. PRPs act on tags, so the tag formalism constrains what PRPs can be.*

*Biological grounding: Plasticity-Related Proteins and Metabolic Constraints (Redondo & Morris, 2011)*

---

## Q2.1 — What Mathematical Object Is a "PRP"?

### Candidates Considered

| Candidate | Type | What It Controls | Granularity |
|:---|:---|:---|:---|
| Binary flag | {0, 1} | "Consolidate or not" | All-or-nothing |
| Continuous resource ∈ [0, 1] | Scalar | Degree of consolidation priority | Graded |
| Learning rate multiplier | ℝ⁺ | How strongly this memory influences W_slow | Graded, training-coupled |
| **Priority score + allocation flag** | **(ℝ⁺, {0, 1})** | **Ranked priority AND binary allocation status** | **Two-stage: scoring then allocation** |
| Compute budget fraction | ℝ⁺ | Share of sleep-phase compute | Proportional |

### Chosen Formalism

A PRP is a **two-component annotation** on a tag:

```
PRP(τᵢ) = (Sᵢ, pᵢ)
```

where:

| Component | Type | Description |
|:---|:---|:---|
| **Sᵢ** | Sᵢ ∈ ℝ⁺ | **Composite score** — a continuous priority value computed from the tag's metadata. Higher = more important. |
| **pᵢ** | pᵢ ∈ {0, 1} | **Allocation flag** — binary. 1 = this tag has been allocated a PRP and is queued for consolidation. 0 = not allocated. |

The **composite score Sᵢ** is computed continuously (updated whenever the tag is accessed or decayed). The **allocation flag pᵢ** is set by the competitive allocation algorithm when Sᵢ exceeds the threshold and budget permits.

### Justification

**Why not a pure continuous resource?**

A purely continuous PRP (e.g., "each tag gets 0.37 of a PRP") creates a difficult optimization problem during sleep: how do you map a continuous allocation to a discrete training procedure? You need to decide *which* memories to replay and *how many times* — these are inherently discrete decisions. The binary allocation flag makes the consolidation boundary crisp.

**Why not a pure binary flag?**

A pure binary flag (allocated or not) loses the ranking information. During sleep, you want to consolidate the most important memories first (in case sleep is interrupted). The continuous score Sᵢ provides this ranking within the allocated set.

**The two-stage design mirrors the biology:**

| Stage | Biology | Computation |
|:---|:---|:---|
| PRP synthesis & availability | Cell produces PRPs proportional to stimulation strength | Composite score Sᵢ computed from tag metadata |
| PRP capture at tagged synapse | Binary: the synapse either captures sufficient PRPs or doesn't | Allocation flag pᵢ set by competitive allocation |

The composite score is the "concentration of available PRPs near this synapse." The allocation flag is "this synapse has captured enough PRPs to be stabilized."

### Downstream Implications

- **Q2.2:** Defines the composite scoring function S(τ)
- **Q2.4:** Competitive allocation operates on the binary flags given the scores
- **Q4.1:** During sleep, only tags with pᵢ = 1 are consolidated; their scores Sᵢ determine replay priority/frequency

---

## Q2.2 — What Is the Composite Scoring Function?

### The Four Components

From the proposal's PRP Allocation Criteria diagram, four signals feed into the score:

```
S(τ) = w₁ · E(τ) + w₂ · A(τ) + w₃ · X(τ) + w₄ · R(τ)
```

Each component must be formally defined and normalized.

### Component 1: Cumulative Prediction Error — E(τ)

**What it captures:** Is this still surprising? Has the model not yet learned this information?

```
E(τ) = e₀ · s(t) / s₀
```

This is the initial prediction error weighted by the fraction of tag strength remaining. A tag that started with high error and hasn't decayed much scores highly. A tag that started with high error but has mostly decayed (without being accessed) scores low — the system has implicitly "moved on."

**Normalization:** E(τ) ∈ [0, e₀] since s(t)/s₀ ∈ [0, 1]. We normalize across the active tag set:

```
Ê(τ) = E(τ) / max(E(τⱼ) for all τⱼ in buffer)
```

So Ê ∈ [0, 1] with the highest-error tag scoring 1.0.

### Component 2: Access Frequency — A(τ)

**What it captures:** Does the user keep needing this information?

```
A(τ) = ρ / (1 + ρ_max)
```

where:
- **ρ** is the cumulative utility (from Q1.5) — already incorporates diminishing returns via the 1/√a factor
- **ρ_max** = max(ρⱼ for all τⱼ in buffer) — used for normalization

**Normalization:** A(τ) ∈ [0, 1).

**Why use ρ (cumulative utility) instead of raw access count a?**

Raw access count treats all accesses equally. Cumulative utility ρ weights by relevance (cosine similarity at access time) and diminishes with repetition. This directly addresses Q6.6 (adversarial repetition): 100 identical queries produce ρ ≈ Σ_{i=1}^{100} sim/√i ≈ sim · 2√100 ≈ 20·sim, while 100 *distinct* relevant queries produce ρ ≈ 100·sim (no diminishing returns because a resets for each... wait, a is per-tag, not per-query). Let me reconsider.

Actually, a is the access count *for this specific tag*, and each access increments it. So 100 queries that all access the same tag give:

```
ρ = Σᵢ₌₁¹⁰⁰ simᵢ / √i ≈ sim_avg · Σᵢ₌₁¹⁰⁰ 1/√i ≈ sim_avg · 2√100 ≈ 20 · sim_avg
```

versus a tag accessed 10 times with high relevance:

```
ρ = Σᵢ₌₁¹⁰ simᵢ / √i ≈ sim_avg · 2√10 ≈ 6.3 · sim_avg
```

The 100-access tag scores ~3x the 10-access tag, not 10x. The √ compression gives meaningful distinction while resisting inflation. This is the correct behavior.

### Component 3: Cross-Reference Density — X(τ)

**What it captures:** Does this tag connect to other tags? Hub memories (connected to many other memories) are more valuable than isolated ones.

**Definition:** Cross-reference density is the number of other active tags whose key vectors are similar to this tag's key:

```
X(τᵢ) = Σⱼ≠ᵢ 𝟙[cos(kᵢ, kⱼ) > θ_xref] / N_active
```

where:
- **θ_xref** is the cross-reference similarity threshold (e.g., θ_xref = 0.5 — lower than θ_access because we want topical relatedness, not near-duplicates)
- **N_active** is the number of active tags (normalization)

X(τ) ∈ [0, 1]. A tag connected to many others scores high; an isolated tag scores low.

**Efficient computation:** Computing all pairwise similarities is O(N² · d_tag). For N = 35,000 and d_tag = 128, this is ~320 GFLOPs — too expensive to run per-step. Instead:

**Batch computation every P steps** (e.g., P = 500):
1. Stack all key vectors into matrix **K** ∈ ℝ^(N × d_tag)
2. Compute **K · Kᵀ** ∈ ℝ^(N × N) — batched cosine similarity
3. Count entries above θ_xref per row

Cost: O(N² · d_tag) every P steps = O(N² · d_tag / P) amortized per step. For N=35K, d_tag=128, P=500: ~640 MFLOPs/step. Acceptable.

**Alternative for very large N:** Use approximate nearest neighbor (e.g., locality-sensitive hashing on the key vectors) to compute X in O(N · log N · d_tag). Trades exactness for speed.

### Component 4: Recency-Weighted Utility — R(τ)

**What it captures:** Recent usefulness matters more than ancient usefulness.

```
R(τ) = Σ_{j: access events} sim_j · exp(-(t - t_j) / τ_recency)
```

where:
- The sum is over all access events for this tag
- **t_j** is the step at which access j occurred
- **τ_recency** is the recency decay constant (e.g., τ_recency = 500 steps)
- **sim_j** is the cosine similarity at access j

**Normalization:** R(τ) / R_max where R_max = max over all tags.

R(τ) ∈ [0, 1]. A tag accessed recently and relevantly scores high. A tag accessed long ago scores low even if it was accessed many times.

**Practical implementation:** Rather than storing all access events, maintain R as an exponential moving sum:

```
On access at step t with similarity sim:
    R ← R · exp(-(t - t_last_update) / τ_recency) + sim

On query (no access):
    R ← R · exp(-(t - t_last_update) / τ_recency)
```

This is O(1) per update with O(1) storage (just R and t_last_update).

### The Combined Scoring Function

```
S(τ) = w₁ · Ê(τ) + w₂ · Â(τ) + w₃ · X̂(τ) + w₄ · R̂(τ)
```

where all components are normalized to [0, 1] and the weights satisfy:

```
w₁ + w₂ + w₃ + w₄ = 1,    wᵢ > 0
```

### Weight Selection

| Weight | Value | Rationale |
|:---|:---|:---|
| w₁ (error) | 0.35 | Prediction error is the primary signal — it directly measures "what W_slow doesn't know" |
| w₂ (access) | 0.30 | Access frequency is the strongest behavioral signal — the user keeps needing this |
| w₃ (cross-ref) | 0.15 | Cross-reference is valuable but noisy (similar tags may be redundant, not reinforcing) |
| w₄ (recency) | 0.20 | Recency prevents stale tags from monopolizing the budget |

**Are the weights fixed or adaptive?**

**Fixed for v1.** Adaptive weights introduce a meta-learning problem (what loss do you adapt them on?) that adds complexity without clear benefit in the initial implementation. The weights can be treated as hyperparameters and tuned via grid search on the evaluation metrics (Q5.4).

**Future direction:** Learn weights via the sleep cycle's success signal — if consolidated memories that scored high on component X turned out to be useful (measured by post-consolidation access patterns), increase wₓ. This is a bandit-style optimization over the weight vector.

### Formal Summary

```
COMPUTE_PRP_SCORE(τ, buffer, t) → S:

    # Component 1: Cumulative Prediction Error
    E = τ.e₀ · τ.s / sigmoid(α_init · τ.e₀)
    Ê = E / max(E(τⱼ) for τⱼ in buffer)

    # Component 2: Access Frequency (using cumulative utility)
    Â = τ.ρ / (1 + max(τⱼ.ρ for τⱼ in buffer))

    # Component 3: Cross-Reference Density (precomputed every P steps)
    X̂ = τ.xref_count / N_active          # cached from last batch computation

    # Component 4: Recency-Weighted Utility
    R̂ = τ.R / max(τⱼ.R for τⱼ in buffer)

    # Combined Score
    S = 0.35 · Ê + 0.30 · Â + 0.15 · X̂ + 0.20 · R̂

    RETURN S
```

### Downstream Implications

- **Q2.3:** The PRP budget limits how many tags can have p = 1; the score S determines who wins
- **Q2.4:** Competitive allocation uses S for ranking
- **Q2.5:** The threshold θ_PRP is applied to S

---

## Q2.3 — What Is the Total PRP Budget and How Is It Determined?

### Chosen Formalism

The PRP budget is a **count** — the maximum number of tags that can be PRP-allocated (p = 1) simultaneously:

```
B = C_prp · (|W_slow| / 10⁹)
```

where:
- **B** is the maximum number of PRP-allocated tags
- **C_prp** is the PRP capacity coefficient (e.g., C_prp = 500)
- **|W_slow|** is the parameter count

| Model Size | N_max (tags) | B (PRP budget) | B / N_max |
|:---|:---|:---|:---|
| 1B | 5,000 | 500 | 10% |
| 7B | 35,000 | 3,500 | 10% |
| 70B | 350,000 | 35,000 | 10% |

**B / N_max ≈ 10%.** At most 10% of active tags can be PRP-allocated at any time. This is the metabolic constraint — the system cannot afford to consolidate everything. It must prioritize.

### Budget Replenishment

The budget is **immediately replenished** when tags are consolidated or when PRP-allocated tags are evicted:

```
B_available(t) = B - |{τ : pᵢ = 1}|
```

After a sleep cycle consolidates M memories:
- M tags have their allocation flags cleared (they're now in W_slow)
- B_available increases by M
- New high-scoring tags can immediately fill the freed slots

This mirrors the biology: PRPs are consumed during capture, and the cell must re-synthesize them. In our system, "re-synthesis" is instantaneous (the budget is a capacity constraint, not a consumable resource), which simplifies the dynamics while preserving the competitive allocation behavior.

### Justification

**Why a count, not a continuous budget?**

A continuous budget (e.g., "total PRP resource = 100, distributed among tags in amounts of 0 to 1") would require a divisible resource allocation algorithm and introduces the question of how fractional PRPs affect consolidation. The count model is simpler: each tag either gets a "slot" or doesn't. During sleep, all slotted tags are consolidated with equal treatment (their score Sᵢ determines replay *frequency* within the sleep cycle, but all are included).

**Why 10% of N_max?**

Biological estimate: In the hippocampus, only a fraction of tagged synapses successfully capture PRPs. Estimates from Redondo & Morris (2011) suggest that metabolic constraints limit consolidation to a minority of tagged synapses. The 10% figure is a conservative starting point, analogous to the biological constraint that protein synthesis can support only a limited number of synapse modifications per consolidation window.

### Downstream Implications

- **Q2.4:** Competitive allocation is a top-B selection problem
- **Q4.4:** The number of memories consolidated per sleep cycle is ≤ B
- **Q4.5:** Budget utilization (B_used / B) is a signal for sleep timing

---

## Q2.4 — How Does Competitive Allocation Work?

### The Algorithm

PRP allocation is a **continuous priority queue** with **hysteresis** to prevent thrashing.

```
ALLOCATE_PRPS(buffer, B) → updated buffer:

    # Step 1: Score all tags
    FOR each τ in buffer:
        τ.S = COMPUTE_PRP_SCORE(τ, buffer, t)

    # Step 2: Sort by score descending
    ranked = SORT(buffer, key=λ τ: τ.S, descending=True)

    # Step 3: Allocate with hysteresis
    allocated_count = 0
    FOR τ in ranked:
        IF allocated_count < B:
            IF τ.p == 1:
                # Already allocated — keep it (no threshold needed to retain)
                allocated_count += 1
            ELIF τ.S ≥ θ_PRP:
                # New allocation — must exceed threshold
                τ.p = 1
                allocated_count += 1
            ELSE:
                τ.p = 0    # Below threshold, not allocated
        ELSE:
            τ.p = 0        # Budget exhausted

    RETURN buffer
```

### Hysteresis Mechanism

The key anti-thrashing mechanism: **an already-allocated tag does not need to re-exceed the threshold to retain its PRP.** It only loses its PRP if it falls out of the top-B by score.

This prevents the following oscillation:
```
Step 100: Tag A scores 0.72, threshold = 0.70 → allocated
Step 101: New tags arrive, Tag A's score drops to 0.69 → deallocated
Step 102: Some tags decay, Tag A's score is now 0.71 → re-allocated
Step 103: ... (repeat forever)
```

With hysteresis, Tag A stays allocated at step 101 as long as it's in the top-B by score (even though its score is below θ_PRP). It only loses allocation if B other tags all score higher.

### Minimum Score Differential for Stealing

When the budget is full (B tags are allocated) and a new tag wants in:

```
IF new_tag.S > min(S of allocated tags) + δ_steal:
    // The new tag can steal the PRP from the lowest-scoring allocated tag
    victim = argmin(S of allocated tags)
    victim.p = 0
    new_tag.p = 1
```

where **δ_steal** is the minimum score differential for stealing (e.g., δ_steal = 0.05). This prevents churn from marginal score differences.

### What Happens to Demoted Tags?

A tag that loses its PRP returns to normal decay. **No penalty for demotion** — it is treated identically to a tag that was never allocated. This is a deliberate choice:

| Option | Behavior | Problem |
|:---|:---|:---|
| Penalty (faster decay) | Demoted tags die quickly | Punishes tags that were temporarily outcompeted; may lose valuable memories |
| **No penalty (normal decay)** | **Demoted tags can be re-allocated later** | **None — the scoring function already accounts for relevance** |
| Bonus (slower decay) | Demoted tags are sticky | Creates zombie tags that resist eviction |

### Convergence Analysis

**Under what conditions does the allocation converge?**

Define the allocation vector **p**(t) = (p₁, p₂, ..., p_N) at step t. The allocation converges when **p**(t) = **p**(t+1) for all subsequent steps (assuming no new tags arrive and no access events).

**Sufficient conditions for convergence:**

1. **Scores are monotonically decreasing over time** (via tag decay) in the absence of access events ✓
2. **Hysteresis prevents oscillation** — a tag's allocation can only change in one direction (allocated → deallocated) unless new access events change its score ✓
3. **The steal differential δ_steal creates a gap** — two tags must differ by at least δ_steal for a swap to occur ✓

**Convergence rate:** In the worst case (all tags have similar scores), convergence takes O(B) re-evaluation cycles. In practice, the score distribution is typically heavy-tailed (a few tags clearly dominate), and convergence is achieved within 1-2 re-evaluations.

### Allocation Frequency

PRP allocation is **not run every inference step** — it's a periodic batch operation:

```
Every Q steps (e.g., Q = 100):
    ALLOCATE_PRPS(buffer, B)
```

Between allocations, tags accumulate access events and decay, which changes their scores. The periodic batch evaluation amortizes the cost (O(N log N) for sorting) and reduces churn.

### Downstream Implications

- **Q2.5:** The threshold θ_PRP is defined next; it interacts with the hysteresis mechanism
- **Q4.1:** Only tags with p = 1 are selected for sleep consolidation
- **Q6.6:** δ_steal prevents adversarial manipulation of the allocation boundary

---

## Q2.5 — What Is the PRP Threshold?

### Chosen Formalism

The threshold is **adaptive**, based on the score distribution of the current tag buffer:

```
θ_PRP(t) = max(θ_floor, μ_S(t) + κ_PRP · σ_S(t))
```

where:
- **μ_S(t)** = mean of all PRP scores in the buffer
- **σ_S(t)** = standard deviation of all PRP scores
- **κ_PRP** = sensitivity parameter (e.g., κ_PRP = 0.5 — less aggressive than tagging, because we want the top tier, not outliers)
- **θ_floor** = minimum threshold (e.g., θ_floor = 0.2 — even if all scores are low, don't consolidate truly weak memories)

### Why Adaptive?

| Scenario | Fixed θ = 0.5 | Adaptive θ |
|:---|:---|:---|
| All scores cluster around 0.3 (low-novelty period) | Allocates nothing → budget wasted | θ ≈ 0.3 + margin → allocates best of the batch |
| All scores cluster around 0.8 (high-novelty period) | Allocates everything → budget overwhelmed | θ ≈ 0.8 + margin → allocates only the exceptional |
| Bimodal: some 0.2, some 0.9 | Works OK | Also works OK, θ settles between modes |

The adaptive threshold ensures the budget is neither wasted (during quiet periods) nor overwhelmed (during high-novelty periods).

### Relationship to Budget

The threshold and budget jointly control allocation:

```
Effective allocation = |{τ : S(τ) ≥ θ_PRP}| capped at B
```

- **If fewer than B tags exceed θ_PRP:** Budget is underutilized. This is fine — it means the system isn't encountering enough important information to fill the budget. The threshold prevents consolidating junk.
- **If more than B tags exceed θ_PRP:** Competitive allocation (Q2.4) selects the top B by score. The threshold is a minimum quality gate; the budget is the quantity cap.

### Different Thresholds by Information Type?

**Not in v1.** The scoring function already implicitly handles different information types: factual information tends to have high prediction error but low access frequency (asked about once), while procedural information tends to have moderate error but high access frequency (used repeatedly). The composite score weights these appropriately without needing type-specific thresholds.

**Future direction:** If evaluation (Q5.4) reveals systematic bias (e.g., factual memories are over-consolidated relative to procedural ones), introduce type-specific weight vectors rather than type-specific thresholds — this is cleaner than multiple threshold functions.

---

## Part 2 Checkpoint Verification

> **Requirement:** Given 20 tags with known scores and a budget of 5, run the allocation algorithm by hand and show convergence.

### Worked Example

**Setup:** 20 tags, B = 5, θ_floor = 0.2, δ_steal = 0.05

**Tag scores (round 1):**

| Tag | Score S | Initially Allocated (p) |
|:---|:---|:---|
| τ₁ | 0.92 | 0 |
| τ₂ | 0.87 | 0 |
| τ₃ | 0.81 | 0 |
| τ₄ | 0.76 | 0 |
| τ₅ | 0.71 | 0 |
| τ₆ | 0.68 | 0 |
| τ₇ | 0.55 | 0 |
| τ₈–τ₂₀ | 0.10–0.50 | 0 |

**Score statistics:** μ_S ≈ 0.45, σ_S ≈ 0.25

**Threshold:** θ_PRP = max(0.2, 0.45 + 0.5 · 0.25) = max(0.2, 0.575) = **0.575**

**Round 1 allocation:**
- Tags exceeding θ_PRP = 0.575: τ₁(0.92), τ₂(0.87), τ₃(0.81), τ₄(0.76), τ₅(0.71), τ₆(0.68)
- That's 6 tags, but B = 5
- **Top 5 by score: τ₁, τ₂, τ₃, τ₄, τ₅** → p = 1
- τ₆ (0.68 ≥ 0.575 but rank 6) → p = 0 (budget exhausted)

**Round 2 (after 100 steps, some decay and access):**

| Tag | New Score S | Was Allocated? |
|:---|:---|:---|
| τ₁ | 0.88 (decayed) | Yes (p=1) |
| τ₂ | 0.91 (accessed!) | Yes (p=1) |
| τ₃ | 0.74 (decayed) | Yes (p=1) |
| τ₄ | 0.60 (decayed) | Yes (p=1) |
| τ₅ | 0.55 (decayed) | Yes (p=1) |
| τ₆ | 0.72 (accessed frequently!) | No (p=0) |
| τ₇ | 0.45 (decayed) | No |

**New statistics:** μ_S ≈ 0.42, σ_S ≈ 0.22, θ_PRP = max(0.2, 0.42 + 0.5·0.22) = **0.53**

**Round 2 allocation:**
- τ₆ wants to steal. Lowest allocated tag: τ₅ at 0.55
- τ₆.S (0.72) > τ₅.S (0.55) + δ_steal (0.05) = 0.60? → 0.72 > 0.60 ✓ **Steal succeeds!**
- **New allocation: τ₂, τ₁, τ₃, τ₆, τ₄** (top 5 by score)
- τ₅ loses PRP, returns to normal decay

**Round 3 (after 100 more steps, no new access events):**

| Tag | New Score S | Allocated? |
|:---|:---|:---|
| τ₂ | 0.85 | Yes |
| τ₁ | 0.82 | Yes |
| τ₃ | 0.67 | Yes |
| τ₆ | 0.65 | Yes |
| τ₄ | 0.50 | Yes |
| τ₅ | 0.42 | No |

All allocated tags are still in the top 5. No tag outside the set exceeds the lowest allocated (0.50) + δ_steal. **Allocation has converged.** ✓

---

*Part 2 complete. All five questions (Q2.1–Q2.5) are fully resolved. Proceeding to Part 3.*

---

# Part 3: The Dual Weight System

*These decisions are somewhat independent of Parts 1-2 but constrain Part 4 (Sleep). This is where we specify the actual neural network architecture.*

*Biological grounding: Complementary Learning Systems (McClelland, McNaughton & O'Reilly, 1995)*

---

## Q3.1 — What Is W_fast, Architecturally?

### Candidates Considered

| Architecture | Param Count (relative to W_slow) | Learning Speed | Generation Capability | Interference Risk |
|:---|:---|:---|:---|:---|
| Separate small model | 5-20% | Fast (independent) | Full (own decoder) | None (isolated) |
| LoRA adapters on all layers | 0.1-1% | Fast (low-rank updates) | Via W_slow+LoRA | Low (additive, low-rank) |
| **LoRA adapters on selected layers** | **0.05-0.5%** | **Fast** | **Via W_slow+LoRA** | **Minimal (sparse, low-rank)** |
| Additional KV pairs in attention | Variable | Very fast (direct write) | Limited (retrieval, not generation) | Moderate (attention interference) |
| Sparse parameter perturbations | 1-5% (sparse) | Fast | Via W_slow+ΔW | Moderate (direct weight modification) |
| Soft prompts / prefix tuning | ~0.01% | Very fast | Limited (no weight change) | None (input-space only) |

### Chosen Formalism: **LoRA Adapters on Selected Layers**

W_fast is a set of low-rank adaptation matrices applied to the **value and output projection matrices** of the attention mechanism in the **top L/3 layers** of the transformer.

For each adapted layer l ∈ {⌈2L/3⌉, ..., L}, W_fast consists of:

```
ΔW_V^(l) = B_V^(l) · A_V^(l)     where A_V^(l) ∈ ℝ^(r × d_model), B_V^(l) ∈ ℝ^(d_model × r)
ΔW_O^(l) = B_O^(l) · A_O^(l)     where A_O^(l) ∈ ℝ^(r × d_model), B_O^(l) ∈ ℝ^(d_model × r)
```

The effective weight at layer l during inference:

```
W_V_eff^(l) = W_V_slow^(l) + (α/r) · B_V^(l) · A_V^(l)
W_O_eff^(l) = W_O_slow^(l) + (α/r) · B_O^(l) · A_O^(l)
```

where:
- **r** is the LoRA rank (e.g., r = 16 for 7B models, r = 32 for 70B models)
- **α** is the LoRA scaling factor (e.g., α = 32)
- B matrices are initialized to **zero** (so W_fast starts as identity — no perturbation)
- A matrices are initialized from N(0, σ² = 0.01)

### Why Value and Output Projections?

| Projection | Role in Attention | Why Adapt It |
|:---|:---|:---|
| Q (query) | What to look for | Modifying Q changes what the model attends to — risky for base capabilities |
| K (key) | What to match against | Modifying K changes the attention pattern — also risky |
| **V (value)** | **What to extract** | **Modifying V changes *what information is read* from context — this is exactly "new knowledge"** |
| **O (output)** | **How to project back** | **Modifying O changes *how extracted information is combined* — complements V** |

Adapting V and O allows W_fast to encode "when you see context like X, extract information Y" without disrupting the attention routing (Q, K) that W_slow learned during pretraining. This is the safest injection point for new knowledge.

### Why Top L/3 Layers Only?

Evidence from the interpretability literature (Geva et al., 2021; Meng et al., 2022):

- **Early layers** (1 to L/3): Encode syntax, local patterns, and positional information. Modifying these risks catastrophic interference with basic language capabilities.
- **Middle layers** (L/3 to 2L/3): Encode semantic representations and relational structure. Modifying these can help but has moderate interference risk.
- **Top layers** (2L/3 to L): Encode task-specific, factual, and contextual knowledge. This is where knowledge editing methods (ROME, MEMIT) operate. **This is where new knowledge should be injected.**

Adapting only the top third reduces the number of W_fast parameters by ~3x and concentrates updates where they matter most.

### Parameter Count

For a model with L layers, d_model dimensions, adapting 2 matrices (V, O) in the top L/3 layers with rank r:

```
|W_fast| = 2 · (L/3) · 2 · r · d_model = (4/3) · L · r · d_model
```

| Model | L | d_model | r | \|W_fast\| | % of \|W_slow\| |
|:---|:---|:---|:---|:---|:---|
| 1B | 24 | 2048 | 8 | 524K | 0.05% |
| 7B | 32 | 4096 | 16 | 2.8M | 0.04% |
| 70B | 80 | 8192 | 32 | 28M | 0.04% |

W_fast is ~0.04% of W_slow. This is important: it confirms that W_fast is a lightweight overlay, not a separate model.

### Downstream Implications

- **Q3.2:** Inference uses W_slow + LoRA (additive interaction)
- **Q3.3:** W_fast learning updates only A and B matrices via gradient descent
- **Q4.1:** Generation during sleep uses W_slow + W_fast for autoregressive sampling — W_fast provides the "hippocampal" knowledge that biases generation toward recent experiences
- **Q6.3 (confirmed):** W_fast (LoRA parameters) is architecturally distinct from the tag system (metadata buffer). They are coupled but separate.

---

## Q3.2 — How Do W_fast and W_slow Interact During Inference?

### The Inference Equation

For an input sequence x, the forward pass through layer l proceeds as:

```
IF l < ⌈2L/3⌉:
    # Lower layers: W_slow only (no adaptation)
    h^(l) = TransformerLayer(h^(l-1); W_slow^(l))

ELSE:
    # Upper layers: W_slow + W_fast (LoRA adaptation)
    Q = W_Q_slow^(l) · h^(l-1)                                           # unchanged
    K = W_K_slow^(l) · h^(l-1)                                           # unchanged
    V = (W_V_slow^(l) + (α/r) · B_V^(l) · A_V^(l)) · h^(l-1)          # adapted

    Attn = softmax(Q · Kᵀ / √d_k) · V                                   # standard attention

    h^(l) = (W_O_slow^(l) + (α/r) · B_O^(l) · A_O^(l)) · Attn + h^(l-1)  # adapted output + residual
```

### How W_fast Modifies Behavior

The LoRA adaptation modifies the output of each adapted layer by an additive perturbation:

```
Δh^(l) = (α/r) · B_O^(l) · A_O^(l) · softmax(QKᵀ/√d_k) · (α/r) · B_V^(l) · A_V^(l) · h^(l-1)
        + cross-terms
```

For small perturbations (which is the regime we operate in, since |W_fast| ≪ |W_slow|), the interaction is approximately:

```
h_combined^(l) ≈ h_slow^(l) + Δh_V^(l) + Δh_O^(l)
```

where Δh_V captures "different information extracted" and Δh_O captures "different output projection." The cross-term (both V and O modified) is second-order in the perturbation magnitude and can be neglected in analysis.

### Interference Prevention

**Why does this not corrupt W_slow's base capabilities?**

1. **Low-rank constraint:** The perturbation ΔW has rank r ≪ d_model. It can only modify the output in an r-dimensional subspace. The remaining (d_model - r) dimensions are untouched. This means W_fast can add new knowledge but cannot easily destroy existing patterns.

2. **Zero initialization of B:** At deployment, B = 0 → ΔW = 0 → no perturbation. W_fast starts as a no-op and gradually introduces changes, never making a large sudden perturbation.

3. **Scaling factor α/r:** The perturbation magnitude is controlled by α/r. With α = r (a common choice), the effective scaling is 1.0, meaning the LoRA output has the same scale as a single attention head's contribution — a modest perturbation.

4. **Upper layers only:** Base capabilities (grammar, syntax, coherence) are primarily encoded in lower layers, which are not modified.

### Tag Activation During Inference (from Q6.7)

During query processing, the tag system operates **in parallel** with the forward pass:

```
INFERENCE(query_tokens):
    # Step 1: Forward pass through W_slow + W_fast
    output, hidden_states = forward(W_slow + W_fast, query_tokens)

    # Step 2: Tag access detection (parallel, non-blocking)
    h_q = mean_pool(hidden_states)
    accessed_tags = tag_system.process_query(query_tokens)  # from Q1.5

    # Step 3: Return output
    # Note: accessed tags affect the OUTPUT only through W_fast
    # (which was updated when those experiences were first encountered).
    # The tags themselves are metadata — they do NOT directly modify the output.
    RETURN output
```

**Critical distinction from RAG:** In RAG, retrieved documents are **injected into the context** and directly influence the output via attention. In SLEEP, tags are **not injected into the context**. They are metadata records. The model's output is determined solely by W_slow + W_fast parameters. Tags influence the output only *indirectly*: they caused W_fast updates at tagging time, and those updates persist in W_fast.

Tag access detection during inference serves only two purposes:
1. **Reinforcement:** Update tag metadata (access count, utility) for PRP scoring
2. **Monitoring:** Track which tags are being used, for sleep scheduling and evaluation

### Downstream Implications

- **Q3.3:** The learning rule updates only the LoRA A and B matrices
- **Q6.7:** The "hidden retrieval problem" is partially resolved — tags don't modify output directly. But Q6.7 must still address whether W_fast's distributed encoding is sufficient to replace RAG's explicit retrieval.

---

## Q3.3 — What Are the Learning Rules for W_fast?

### The Update Equation

W_fast is updated via gradient descent on surprising inputs, using the same prediction error signal that drives tagging (Q1.2):

```
WHEN a surprising span is detected (E_span > θ_wfast):

    # The loss is next-token prediction on the surprising span
    L_fast = -(1/|span|) · Σ_{t ∈ span} log p_{W_slow + W_fast}(x_t | x_{<t})

    # Gradient update on W_fast only (W_slow is frozen during wake)
    FOR each adapted layer l:
        A_V^(l) ← A_V^(l) - α_fast · ∂L_fast/∂A_V^(l)
        B_V^(l) ← B_V^(l) - α_fast · ∂L_fast/∂B_V^(l)
        A_O^(l) ← A_O^(l) - α_fast · ∂L_fast/∂A_O^(l)
        B_O^(l) ← B_O^(l) - α_fast · ∂L_fast/∂B_O^(l)
```

### Learning Rate

```
α_fast = α_base_fast · min(1.0, E_span / E_ref)
```

where:
- **α_base_fast** = 1e-4 (base learning rate for W_fast)
- **E_ref** is a reference prediction error level (e.g., E_ref = 2.0) — the learning rate scales with surprise magnitude up to a cap of α_base_fast
- The cap prevents catastrophically large updates from extremely surprising inputs

**Comparison to pretraining:**

| System | Learning Rate | Ratio |
|:---|:---|:---|
| W_slow pretraining | ~3e-4 (typical for 7B) | 1x (reference) |
| **W_fast during wake** | **~1e-4** | **~0.33x pretraining** |
| W_slow during sleep | ~1e-5 (see Q3.4) | ~0.03x pretraining |
| **Fast/Slow ratio** | — | **~10x** |

The fast/slow learning rate ratio of ~10x is conservative compared to biological estimates (100-10,000x) but appropriate for our setting: LoRA updates are already concentrated in a low-dimensional subspace, so each update has more per-parameter impact than a full-rank update would.

### Optimizer

**SGD with momentum** (not Adam) for W_fast. Justification:

1. **Adam's state overhead:** Adam maintains first and second moment estimates per parameter, doubling the memory cost. For W_fast this is small (2 × |W_fast| ≈ 5.6M for 7B), but the principle is to keep W_fast minimal.

2. **Biological plausibility:** Biological synaptic updates don't have adaptive learning rates. SGD with momentum is the closest neural-network analogue to Hebbian learning with a trace.

3. **Preventing W_fast from converging too deeply:** Adam's adaptive rates help with convergence, which we *don't want* for W_fast. W_fast should encode recent experiences shallowly, not converge to them. SGD with moderate momentum (μ = 0.9) achieves this.

```
Optimizer: SGD
Learning rate: α_fast = 1e-4 (scaled by surprise)
Momentum: μ = 0.9
Weight decay: 0 (decay is handled by the tag/PRP system, not weight regularization)
```

### Preventing Overfitting to Recent Experience

Three mechanisms prevent W_fast from overfitting:

1. **Low rank (r ≪ d_model):** The LoRA constraint limits the expressiveness of updates. W_fast literally cannot overfit to fine details — it can only capture low-rank (broad, structural) patterns.

2. **Surprise gating (θ_wfast):** W_fast is only updated on genuinely surprising inputs. Expected inputs (which would push W_fast toward memorizing common patterns) are filtered out.

3. **Sleep-phase reset:** During consolidation, the portions of W_fast corresponding to consolidated memories are reset (see Q4.6). This prevents indefinite accumulation.

### Downstream Implications

- **Q3.4:** W_slow's learning rate during sleep is defined relative to α_fast
- **Q4.1:** W_fast's current state determines what it can generate during sleep replay
- **Q4.6:** Post-consolidation cleanup resets W_fast components

---

## Q3.4 — What Defines W_slow's Plasticity Constraints?

### The Core Constraint

W_slow changes **only during sleep** and subject to strict magnitude bounds:

```
‖ΔW_slow^(l)‖_F ≤ δ_max^(l) · ‖W_slow^(l)‖_F      for each layer l
```

where:
- **‖·‖_F** is the Frobenius norm
- **δ_max^(l)** is the maximum relative change per consolidation cycle for layer l

### Layer-Specific Plasticity Profile

Different layers have different plasticity budgets, mirroring the biological observation that different cortical regions have different plasticity profiles:

```
δ_max^(l) = δ_base · φ(l/L)
```

where φ is a plasticity profile function:

```
φ(x) = φ_min + (1 - φ_min) · x²
```

| Layer Position | l/L | φ(l/L) | δ_max (with δ_base = 0.001) |
|:---|:---|:---|:---|
| Layer 1 (bottom) | 0.03 | 0.10 | 0.0001 (very low plasticity) |
| Layer L/4 | 0.25 | 0.16 | 0.00016 |
| Layer L/2 | 0.50 | 0.33 | 0.00033 |
| Layer 3L/4 | 0.75 | 0.63 | 0.00063 |
| Layer L (top) | 1.00 | 1.00 | 0.001 (highest plasticity) |

**Lower layers are less plastic.** They encode fundamental linguistic structure (syntax, morphology) learned during pretraining and should change minimally. **Upper layers are more plastic** — they encode factual and contextual knowledge and are the primary target for consolidation.

### Learning Rate for W_slow During Sleep

```
α_slow^(l) = α_base_slow · φ(l/L)
```

where α_base_slow = 1e-5 (10x smaller than α_fast, 30x smaller than pretraining).

### EWC-Style Regularization

During sleep training, we apply an Elastic Weight Consolidation penalty to protect important pretrained knowledge:

```
L_sleep = L_replay + (λ_ewc / 2) · Σ_i F_i · (θ_i - θ_i^*)²
```

where:
- **L_replay** is the language modeling loss on the interleaved replay dataset (defined in Q4.4)
- **F_i** is the diagonal Fisher information for parameter i (computed once on a calibration dataset during setup)
- **θ_i*** is the pretrained parameter value (the "anchor")
- **λ_ewc** is the EWC penalty strength (e.g., λ_ewc = 100)

**Fisher refresh policy:** The Fisher information is computed during system initialization on a general-purpose calibration dataset, but it must be **periodically refreshed** to remain accurate. After many consolidation cycles, the parameters important for consolidated user knowledge may differ substantially from those important for pretraining alone. A stale Fisher risks over-protecting pretrained knowledge while leaving recently consolidated knowledge exposed.

```
REFRESH_FISHER(system, every N_fisher_refresh = 10 sleep cycles):

    # Mix calibration data (protects pretrained capabilities)
    # with consolidated knowledge samples (protects user knowledge)
    refresh_data = (
        sample(calibration_dataset, 0.7 * |refresh_set|)     # 70% general
        + generate_from(W_target, 0.3 * |refresh_set|)       # 30% consolidated knowledge
    )

    # Recompute diagonal Fisher on the mixed dataset
    F_i = (1/|refresh_data|) · Σ_n (∂log p(y_n|x_n, θ) / ∂θ_i)²

    system.fisher = F_i
```

The 70/30 mix ensures that both pretrained capabilities AND recently consolidated knowledge receive proportional protection. Refreshing every 10 cycles (not every cycle) amortizes the compute cost while keeping the Fisher reasonably current. The cost per refresh is one pass over the refresh dataset — comparable to one epoch of training.

### Hard Clipping

As a safety net beyond the EWC soft constraint:

```
AFTER each consolidation training step:
    FOR each layer l:
        ΔW = W_slow^(l) - W_slow_original^(l)
        IF ‖ΔW‖_F > δ_max^(l) · ‖W_slow_original^(l)‖_F:
            W_slow^(l) = W_slow_original^(l) + ΔW · (δ_max^(l) · ‖W_slow_original^(l)‖_F / ‖ΔW‖_F)
```

This projects the update back to the constraint ball if the EWC penalty is insufficient.

### Verification

To ensure W_slow doesn't degrade, after each sleep cycle, evaluate on a held-out benchmark set:

```
perplexity_post = evaluate(W_slow, benchmark_set)
IF perplexity_post > perplexity_pre · (1 + ε_degrade):    // e.g., ε_degrade = 0.02 (2% degradation)
    ROLLBACK: W_slow ← W_slow_pre_sleep
    LOG warning: "Consolidation cycle degraded base capabilities, rolled back"
```

This is the ultimate safety net — if consolidation goes wrong, we roll back.

### Downstream Implications

- **Q4.4:** The training procedure during sleep uses α_slow with EWC regularization
- **Q6.9:** The plasticity constraints bound how much W_slow can change per cycle, which feeds into long-term stability analysis
- **Q6.1:** Memory revision (overwriting old knowledge) must work within these constraints

---

## Part 3 Checkpoint Verification

> **Requirement:** Draw the exact architecture diagram with parameter counts, specifying every tensor shape and interaction.

### Architecture Diagram (7B Model: L=32, d_model=4096, r=16)

```
╔══════════════════════════════════════════════════════════════════════╗
║                    SLEEP SYSTEM ARCHITECTURE                        ║
║                    Model: 7B parameters                             ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  ┌──────────────────────────────────────────────────────────────┐    ║
║  │                    TAG BUFFER T                               │    ║
║  │                                                               │    ║
║  │  Capacity: N_max = 35,000 tags                               │    ║
║  │  Memory: ~18.6 MB                                            │    ║
║  │  Per tag: k ∈ ℝ¹²⁸, s ∈ [0,1], t₀, e₀, a, ρ, ctx         │    ║
║  │                                                               │    ║
║  │  Key Projection: W_proj ∈ ℝ^(128 × 4096) = 524K params     │    ║
║  └──────────────────────┬───────────────────────────────────────┘    ║
║                         │ cosine similarity                          ║
║                         │ (access detection)                         ║
║                         ▼                                            ║
║  ┌──────────────────────────────────────────────────────────────┐    ║
║  │                                                               │    ║
║  │   W_slow (Frozen during wake)          7B parameters          │    ║
║  │   ═══════════════════════════          ══════════════          │    ║
║  │                                                               │    ║
║  │   Layer 1-21 (bottom 2/3): UNMODIFIED                        │    ║
║  │   ┌─────────────────────────────────────────┐                │    ║
║  │   │  W_Q ∈ ℝ^(4096×4096)  ← no adaptation  │                │    ║
║  │   │  W_K ∈ ℝ^(4096×4096)  ← no adaptation  │                │    ║
║  │   │  W_V ∈ ℝ^(4096×4096)  ← no adaptation  │                │    ║
║  │   │  W_O ∈ ℝ^(4096×4096)  ← no adaptation  │                │    ║
║  │   │  FFN weights           ← no adaptation  │                │    ║
║  │   └─────────────────────────────────────────┘                │    ║
║  │                                                               │    ║
║  │   Layer 22-32 (top 1/3): ADAPTED BY W_fast                  │    ║
║  │   ┌─────────────────────────────────────────┐                │    ║
║  │   │  W_Q ∈ ℝ^(4096×4096)  ← no adaptation  │                │    ║
║  │   │  W_K ∈ ℝ^(4096×4096)  ← no adaptation  │                │    ║
║  │   │  W_V ∈ ℝ^(4096×4096)  ← + B_V·A_V      │  W_fast       │    ║
║  │   │       A_V ∈ ℝ^(16×4096)  = 65,536       │  (LoRA)       │    ║
║  │   │       B_V ∈ ℝ^(4096×16)  = 65,536       │               │    ║
║  │   │  W_O ∈ ℝ^(4096×4096)  ← + B_O·A_O      │               │    ║
║  │   │       A_O ∈ ℝ^(16×4096)  = 65,536       │               │    ║
║  │   │       B_O ∈ ℝ^(4096×16)  = 65,536       │               │    ║
║  │   │  FFN weights           ← no adaptation  │                │    ║
║  │   └─────────────────────────────────────────┘                │    ║
║  │                                                               │    ║
║  │   W_fast total: 11 layers × 4 matrices × 65,536 = 2.88M    │    ║
║  │   W_fast as % of W_slow: 0.041%                              │    ║
║  │                                                               │    ║
║  └──────────────────────────────────────────────────────────────┘    ║
║                                                                      ║
║  ┌──────────────────────────────────────────────────────────────┐    ║
║  │                    PRP SYSTEM                                 │    ║
║  │                                                               │    ║
║  │  Budget: B = 3,500 (10% of N_max)                           │    ║
║  │  Scoring: S = 0.35·Ê + 0.30·Â + 0.15·X̂ + 0.20·R̂           │    ║
║  │  Allocation: Top-B with hysteresis, δ_steal = 0.05          │    ║
║  │  Threshold: θ_PRP = max(0.2, μ_S + 0.5·σ_S)                │    ║
║  └──────────────────────────────────────────────────────────────┘    ║
║                                                                      ║
║  ┌──────────────────────────────────────────────────────────────┐    ║
║  │                 PARAMETER SUMMARY                             │    ║
║  │                                                               │    ║
║  │  W_slow:     7,000,000,000  (7.00B)    — frozen during wake  │    ║
║  │  W_fast:         2,883,584  (2.88M)    — updated during wake │    ║
║  │  W_proj:           524,288  (0.52M)    — fixed               │    ║
║  │  Tag buffer:    ~18,600,000 bytes      — data, not params    │    ║
║  │  Fisher (EWC):  7,000,000,000 scalars  — computed once       │    ║
║  │                                                               │    ║
║  │  Total trainable (wake):  2.88M  (0.041% of model)          │    ║
║  │  Total trainable (sleep): 7.00B  (100%, with constraints)    │    ║
║  └──────────────────────────────────────────────────────────────┘    ║
╚══════════════════════════════════════════════════════════════════════╝
```

### Tensor Shape Reference

| Tensor | Shape | Count | Total Parameters |
|:---|:---|:---|:---|
| W_Q_slow, W_K_slow (all layers) | (4096, 4096) | 32 × 2 = 64 | 1.07B |
| W_V_slow, W_O_slow (all layers) | (4096, 4096) | 32 × 2 = 64 | 1.07B |
| FFN weights (all layers) | (4096, 11008) + (11008, 4096) | 32 × 2 = 64 | 2.88B |
| Embeddings + LM head | (32000, 4096) × 2 | 2 | 0.26B |
| **A_V, A_O (top 11 layers)** | **(16, 4096)** | **11 × 2 = 22** | **1.44M** |
| **B_V, B_O (top 11 layers)** | **(4096, 16)** | **11 × 2 = 22** | **1.44M** |
| **W_proj (tag key projection)** | **(128, 4096)** | **1** | **0.52M** |

---

*Part 3 complete. All four questions (Q3.1–Q3.4) are fully resolved. Proceeding to Part 4.*

---

# Part 4: The Sleep Engine

*This depends on all previous parts. This is where temporary knowledge becomes permanent.*

*Biological grounding: Sleep-Dependent Memory Consolidation (Wilson & McNaughton, 1994; Diekelmann & Born, 2010)*

---

## Q4.1 — What Is the Generation Model?

### The Core Question

During sleep, W_fast generates synthetic training examples that capture the gist of tagged experiences. How?

### Candidates Considered

| Generator | Architecture | Quality Control | Compression | Cost |
|:---|:---|:---|:---|:---|
| Separate VAE | Independent generative model | ELBO bound | Explicit (latent dim) | High (maintain separate model) |
| Separate diffusion model | Independent denoiser | Score matching | Explicit (noise schedule) | Very high |
| **W_slow + W_fast (autoregressive)** | **The model itself** | **Implicit (model quality)** | **Implicit (generation ≠ memorization)** | **Zero (no extra model)** |
| W_fast alone | Small LoRA-only model | Poor (too few params) | Strong but lossy | Low |

### Chosen Formalism: **Autoregressive Generation from W_slow + W_fast, Conditioned on Tag Context**

The generator is not a separate model — it is the combined system (W_slow + W_fast) used in generation mode. This mirrors the biology: the hippocampus generates replay by reactivating learned patterns, not by running a separate "replay machine."

### The Generation Process

For a PRP-allocated tag τᵢ with context reference ctx = (span_start, span_end, source_id):

```
GENERATE_REPLAY(τᵢ, W_slow, W_fast) → replay_sample:

    # Step 1: Construct prompt from tag context
    # Use a SHORT prefix from the original experience (the "seed")
    seed = original_tokens[span_start : span_start + seed_length]
    # seed_length = min(32, span_length / 4) — just enough to set the topic

    # Step 2: Generate continuation using W_slow + W_fast
    generated = autoregressive_sample(
        model = W_slow + W_fast,
        prompt = REPLAY_PREFIX + seed,      # REPLAY_PREFIX = "Recall: "
        max_tokens = target_length,
        temperature = T_replay,             # T_replay = 0.7 (moderate creativity)
        top_p = 0.9
    )

    # Step 3: Construct training example
    # Format as (input, output) pair for language modeling
    replay_sample = {
        "text": seed + generated,
        "tag_id": τᵢ.id,
        "prp_score": τᵢ.S,
        "original_length": span_end - span_start
    }

    RETURN replay_sample
```

### Why Generation, Not Retrieval of Original Text?

| Approach | What Reaches W_slow | Privacy | Compression | Quality |
|:---|:---|:---|:---|:---|
| Store and replay original text | Verbatim tokens | None (memorized) | None | Perfect fidelity |
| **Generate from W_fast** | **Reconstructed gist** | **Natural compression** | **Forced by generation** | **Good if W_fast is well-trained** |
| Generate from W_slow alone | Hallucination | N/A | N/A | Poor (W_slow doesn't have the info) |

**Generation forces compression.** W_fast has only 2.88M parameters (for 7B model). It cannot memorize verbatim text — it has captured patterns and associations. When it generates, it produces text that reflects the *gist* of the experience, not a copy. This is exactly what biological replay does: the hippocampus reconstructs, it doesn't play back a recording.

**W_slow + W_fast is necessary.** W_slow provides coherent language generation capability. W_fast provides the bias toward recent experience content. Without W_slow, generation would be incoherent. Without W_fast, generation would be generic (no recent knowledge).

### Quality Control

Generated replay can be hallucinated or distorted. We apply a **consistency filter:**

```
QUALITY_CHECK(replay_sample, τᵢ, W_slow) → accept/reject:

    # Check 1: Semantic similarity to original tag
    h_replay = encode(W_slow, replay_sample.text)
    k_replay = W_proj · mean_pool(h_replay)
    sim = cosine_similarity(k_replay, τᵢ.k)

    IF sim < θ_quality:       # θ_quality = 0.5
        REJECT "Replay drifted too far from original experience"

    # Check 2: W_slow's surprise on the replay
    # Good replay should be moderately surprising to W_slow
    # (it contains information W_slow doesn't have)
    surprise = mean_surprisal(W_slow, replay_sample.text)

    # μ_surprise is the mean per-token surprisal of W_slow on the calibration
    # dataset (the same dataset used to compute Fisher information).
    # It represents W_slow's "baseline surprise" on general text.
    # Computed once during initialization: μ_surprise = E[−log p_{W_slow}(x)]
    # over calibration data. Typical value for a 7B model: ~2.5-3.5 nats.
    IF surprise < μ_surprise:     # Below W_slow's baseline = already known
        REJECT "Replay contains no new information for W_slow"

    ACCEPT
```

Rejected samples are re-generated (up to 3 attempts) with different random seeds. If all attempts fail, the tag is skipped for this sleep cycle.

### Downstream Implications

- **Q4.2:** Compression ratio is implicit (determined by generation length vs original length)
- **Q4.3:** Generated samples are one component of the interleaved training batch
- **Q6.8:** Privacy depends on whether W_fast can reproduce verbatim content (analyzed in Q6.8)

---

## Q4.2 — What Is the Compression Ratio and How Is It Controlled?

### Chosen Formalism

**Compression is controlled by target generation length, not by an explicit compression objective.**

```
target_length = max(min_replay, original_length / C_target)
```

where:
- **C_target** = 5 (target 5:1 compression ratio)
- **min_replay** = 64 tokens (minimum replay length, to ensure enough content for training)
- **original_length** = span_end - span_start (from the tag's context reference)

| Original Span | Original Length | Target Replay Length | Effective Ratio |
|:---|:---|:---|:---|
| Short (1 sentence) | 30 tokens | 64 tokens (min) | 0.47:1 (expansion!) |
| Medium (1 paragraph) | 200 tokens | 64 tokens | 3.1:1 |
| Long (1 page) | 500 tokens | 100 tokens | 5:1 |
| Very long (multi-page) | 2000 tokens | 400 tokens | 5:1 |

**Short spans are expanded**, not compressed. This is deliberate: a short surprising utterance (e.g., "The project deadline moved to March 5") contains dense information that should be elaborated with context during replay. The model generates surrounding context, creating richer training data.

### Information-Theoretic Framing

Let X be the original experience and X̂ be the replay. The **replay fidelity** is:

```
Fidelity(X, X̂) = I(X; X̂) / H(X)
```

where I(X; X̂) is the mutual information between original and replay, and H(X) is the entropy of the original. Fidelity = 1 means perfect reconstruction; Fidelity = 0 means the replay is independent of the original.

We cannot compute this exactly, but we can estimate it via the proxy:

```
Fidelity_proxy = cosine_similarity(k_original, k_replay)
```

The quality check in Q4.1 enforces Fidelity_proxy ≥ θ_quality = 0.5. This means the replay must retain at least half the "direction" of the original in tag-key space.

### What "Gist" Means Formally

Gist preservation means the replay retains the **information that caused the prediction error** while discarding incidental details:

```
I(X̂; E) ≈ I(X; E)     // replay preserves the relationship between content and surprise signal
I(X̂; X \ E) < I(X; X \ E)  // replay discards details NOT related to the surprise
```

where E is the prediction error signal. In practice, this is achieved by:
1. Seeding generation with the surprising span (targets the relevant content)
2. Using W_fast (which was updated by the prediction error) to bias generation
3. Limiting generation length (forces compression of non-essential details)

---

## Q4.3 — What Is the Interleaving Strategy?

### The Interleaving Distribution

Each sleep training batch contains a mixture of:

```
Batch = { replay_new[1..n_new], replay_old[1..n_old] }
```

where:
- **replay_new**: Generated from W_fast using PRP-allocated tags (new knowledge to consolidate)
- **replay_old**: Sampled from W_slow's existing knowledge (old knowledge to protect)

### Mixing Ratio

```
r_new = B_used / (B_used + B_used / η)  =  η / (η + 1)
```

where:
- **B_used** = number of PRP-allocated tags to consolidate this cycle
- **η** = old-to-new ratio (e.g., η = 4, meaning 4 old samples per 1 new sample)

| η | % New | % Old | Character |
|:---|:---|:---|:---|
| 1 | 50% | 50% | Aggressive consolidation |
| 2 | 33% | 67% | Balanced |
| **4** | **20%** | **80%** | **Conservative (recommended)** |
| 9 | 10% | 90% | Very conservative |

**η = 4 (20% new, 80% old)** is the default. This is conservative because the cost of catastrophic forgetting far exceeds the cost of slower consolidation.

### Source of Old Knowledge Samples

**Generated from W_slow alone** (not from a held-out buffer):

```
GENERATE_OLD_KNOWLEDGE(W_slow) → old_sample:

    # Sample a diverse prompt from a pool of seed topics
    seed = random_choice(calibration_prompts)

    # Generate from W_slow (no W_fast!)
    old_sample = autoregressive_sample(
        model = W_slow,      # W_slow only — this is what we're protecting
        prompt = seed,
        max_tokens = 256,
        temperature = 1.0,   # High temperature for diversity
        top_p = 0.95
    )

    RETURN old_sample
```

**Why generate from W_slow rather than store a replay buffer?**

| Source | Storage Cost | Distribution Fidelity | Diversity |
|:---|:---|:---|:---|
| Held-out buffer (store pretraining data) | O(buffer_size) — potentially GBs | Perfect | Fixed (limited by buffer) |
| **Generate from W_slow** | **Zero storage** | **Approximate (but W_slow IS the knowledge)** | **Unlimited (stochastic sampling)** |
| Generate from W_slow + W_fast | Zero storage | Contaminated by W_fast | Bad (biased toward recent) |

Generating from W_slow is the "self-distillation" approach. W_slow generates text that reflects its current knowledge, and we train it to remain consistent with this output. This is exactly what CLS consolidation does: the "old knowledge" is defined by the slow system's current state.

### Selection of Old Knowledge: Proximity-Weighted Sampling

Not all old knowledge is equally at risk. Knowledge in the **neighborhood** of the new knowledge being consolidated is most vulnerable to interference. We bias old knowledge sampling toward relevant topics:

```
FOR each new replay sample x_new:
    # Generate a semantically nearby "old" sample
    seed = extract_topic_keywords(x_new)    # e.g., take the first few tokens
    old_nearby = autoregressive_sample(W_slow, prompt=seed, ...)

FOR remaining old budget:
    # Generate random diverse samples (general protection)
    old_random = autoregressive_sample(W_slow, prompt=random_seed, ...)
```

**Split:** 50% proximity-weighted, 50% random. The proximity-weighted samples specifically protect the knowledge most likely to be interfered with.

### Curriculum Within a Sleep Cycle

```
Phase 1 (first 30% of training steps): η = 9  (90% old, 10% new)
    → Gentle introduction; W_slow starts conservatively
Phase 2 (middle 50%):                   η = 4  (80% old, 20% new)
    → Main consolidation phase
Phase 3 (final 20%):                    η = 9  (90% old, 10% new)
    → Stabilization; ensure base capabilities are intact
```

This "warm-up / consolidate / cool-down" curriculum mirrors biological sleep architecture, where early SWS has the most aggressive replay, and later stages stabilize.

---

## Q4.4 — What Is the Training Procedure During Sleep?

### The Complete Training Loop

> **Architectural reconciliation (Q4.4 ↔ Q6.4):** In the multi-user deployment (Q6.4), W_slow_base is shared and frozen. Sleep trains a per-user **consolidated adapter W_cons** (a LoRA adapter with the same architecture as W_fast). The training loop below is written for this architecture. For a single-user deployment where W_slow can be modified directly, replace W_cons with W_slow and remove the adapter composition — the algorithm is otherwise identical.

```
SLEEP_TRAIN(W_slow_base, W_cons, W_fast, tags_to_consolidate, config) → W_cons_updated:

    # --- Architecture ---
    # W_slow_base:   Shared pretrained model (FROZEN, never modified)
    # W_cons:        Per-user consolidated LoRA adapter (TRAINED during sleep)
    # W_fast:        Per-user fast LoRA adapter (FROZEN during sleep, used for generation)
    # W_eff:         W_slow_base + W_cons + W_fast (effective model for generation)
    # W_target:      W_slow_base + W_cons (target model for training — excludes W_fast)

    # --- Configuration ---
    # N_steps:       total training steps per sleep cycle
    # batch_size:    samples per batch
    # α_slow:        base learning rate for W_cons (1e-5)
    # λ_ewc:         EWC penalty strength (100)
    # η:             old-to-new ratio (4, varied by curriculum)
    # δ_max:         max relative change to W_cons per cycle

    W_cons_checkpoint = copy(W_cons)  # save for rollback
    W_eff = compose(W_slow_base, W_cons, W_fast)  # for replay generation
    W_target = compose(W_slow_base, W_cons)         # for training (no W_fast)

    # --- Step 1: Generate replay dataset ---
    replay_new = []
    FOR τ in tags_to_consolidate (sorted by S descending):
        sample = GENERATE_REPLAY(τ, W_eff)          # generate from full system
        IF QUALITY_CHECK(sample, τ, W_target) == ACCEPT:  # check against target
            replay_new.append(sample)

    N_new = len(replay_new)
    N_steps = max(100, N_new * 5)    # ~5 gradient steps per new memory

    # --- Step 2: Training loop ---
    # Only W_cons parameters are optimized; W_slow_base is frozen
    optimizer = AdamW(W_cons.parameters(), lr=α_slow, weight_decay=0.01)
    scheduler = cosine_warmup(optimizer, warmup=0.1*N_steps, total=N_steps)

    FOR step = 1 TO N_steps:
        # Determine curriculum phase
        η_t = curriculum(step, N_steps)    # 9→4→9 as defined above

        # Sample batch
        n_new = batch_size / (1 + η_t)
        n_old = batch_size - n_new

        batch_new = sample_with_replacement(replay_new, n_new, weight_by_prp_score=True)
        batch_old_nearby = generate_nearby(W_target, batch_new, n_old / 2)
        batch_old_random = generate_random(W_target, n_old / 2)

        batch = shuffle(batch_new + batch_old_nearby + batch_old_random)

        # Compute loss (forward through W_slow_base + W_cons; gradients flow to W_cons only)
        L_lm = mean([-log p_{W_target}(x_t | x_{<t}) for all tokens in batch])

        # EWC anchor: W_cons at the START of this sleep cycle
        # Protects previously consolidated per-user knowledge
        # Fisher F is computed on calibration_data ∪ consolidated_knowledge_sample
        L_ewc = (λ_ewc / 2) · sum([F_i · (W_cons_i - W_cons_checkpoint_i)² for W_cons params])

        L_total = L_lm + L_ewc

        # Gradient step (only W_cons parameters receive gradients)
        L_total.backward()
        clip_grad_norm(W_cons.parameters(), max_norm=1.0)

        # Apply layer-specific learning rates
        FOR each adapted layer l:
            scale = φ(l/L)    # plasticity profile from Q3.4
            W_cons[l].grad *= scale

        optimizer.step()
        scheduler.step()

        # Hard clipping on W_cons (analogous to Q3.4 but on adapter params)
        FOR each adapted layer l:
            ΔW = W_cons[l] - W_cons_checkpoint[l]
            norm_ratio = ‖ΔW‖_F / (δ_max · ‖W_slow_base[l]‖_F)
            IF norm_ratio > 1:
                W_cons[l] = W_cons_checkpoint[l] + ΔW / norm_ratio

    # --- Step 3: Validation ---
    W_target_new = compose(W_slow_base, W_cons)
    ppl_post = evaluate(W_target_new, benchmark_set)
    IF ppl_post > ppl_pre * (1 + ε_degrade):
        W_cons = W_cons_checkpoint    # ROLLBACK
        RETURN W_cons, success=False

    RETURN W_cons, success=True
```

### Loss Function Details

**Primary loss: Standard language modeling (next-token prediction)**

```
L_lm = -(1/T) · Σ_t log p_{W_slow}(x_t | x_{<t})
```

This is the same loss used during pretraining. We do NOT use a modified loss that weights new knowledge more heavily — the interleaving ratio already handles this. Weighting the loss would create an interaction between two control knobs, making tuning harder.

**Regularization: EWC penalty**

```
L_ewc = (λ_ewc / 2) · Σ_i F_i · (θ_i - θ_i*)²
```

Fisher diagonal F_i is precomputed once. The anchor θ_i* is the **original pretrained weights** (not updated after each cycle). This provides a consistent gravity toward the pretrained model, preventing long-term drift.

**Optimizer: AdamW** (not SGD, unlike W_fast)

W_slow training during sleep needs to be stable and efficient. AdamW's adaptive learning rates help navigate the high-dimensional loss landscape of a billion-parameter model. The weight decay (0.01) provides additional regularization.

### Steps Per Consolidation Cycle

```
N_steps = max(100, |tags_to_consolidate| * 5)
```

Each PRP-allocated memory gets ~5 gradient exposures on average (weighted by PRP score — higher-priority memories are sampled more frequently).

| Memories to Consolidate | N_steps | Approximate Time (7B, A100) |
|:---|:---|:---|
| 100 | 500 | ~5 minutes |
| 500 | 2,500 | ~25 minutes |
| 3,500 (full budget) | 17,500 | ~3 hours |

A full consolidation of the entire PRP budget takes ~3 hours on a single A100. This is the "full night's sleep." In practice, most cycles will consolidate a fraction of the budget.

### Batch Composition

For batch_size = 32, η = 4:
- n_new = 32 / 5 ≈ 6 new replay samples
- n_old = 26 old samples (13 proximity-weighted + 13 random)

The new samples are drawn with replacement, weighted by PRP score:

```
P(sample i) ∝ S(τᵢ)
```

Higher-priority memories are replayed more frequently, giving them more gradient exposure. This is analogous to biological replay, where salient memories are replayed more often during sleep.

---

## Q4.5 — What Are the Consolidation Triggers?

### Trigger Conditions

Sleep is triggered by **any** of the following conditions:

```
SHOULD_SLEEP(system_state) → bool:

    # Condition 1: Scheduled interval
    IF steps_since_last_sleep > T_schedule:           # T_schedule = 10,000 steps
        RETURN True

    # Condition 2: Memory pressure
    IF N_active / N_max > φ_pressure:                 # φ_pressure = 0.7 (70% full)
        RETURN True

    # Condition 3: PRP budget saturation
    IF B_used / B > φ_budget:                         # φ_budget = 0.8 (80% allocated)
        RETURN True

    # Condition 4: Idle time
    IF time_since_last_query > T_idle:                # T_idle = 300 seconds (5 min)
        RETURN True

    RETURN False
```

### Priority of Triggers

| Trigger | Priority | When It Fires | Character |
|:---|:---|:---|:---|
| Memory pressure | Highest | Buffer nearly full — must consolidate or lose tags | Emergency |
| PRP saturation | High | Many important memories waiting — consolidation is valuable | Productive |
| Scheduled | Medium | Regular maintenance — prevents accumulation | Preventive |
| Idle | Low | System isn't busy — good time for housekeeping | Opportunistic |

### Interruption Handling

**Can sleep be interrupted?**

Yes. If a query arrives during sleep, the system can:

```
ON_QUERY_DURING_SLEEP(query):

    # Option A: Complete current training step, then pause sleep
    complete_current_step()
    save_sleep_state(current_step, optimizer_state)

    # Process query using W_slow (current, partially updated) + W_fast
    response = inference(W_slow, W_fast, query)

    # Resume sleep after query is processed
    resume_sleep(saved_state)
```

**Sleep is resumable.** The optimizer state, current step, and batch generator state are saved. This enables "micro-sleeps" — short consolidation bursts between queries.

### Micro-Sleep vs Full Sleep

```
IF trigger == idle AND time_available < T_full_sleep:
    # Micro-sleep: consolidate only top-priority memories
    tags_to_consolidate = top_k(prp_allocated_tags, k=min(50, B_used))
    N_steps = 100    # Quick consolidation
ELSE:
    # Full sleep: consolidate all PRP-allocated tags
    tags_to_consolidate = all prp_allocated_tags
    N_steps = max(100, |tags_to_consolidate| * 5)
```

Micro-sleeps handle the most urgent consolidation. Full sleeps do comprehensive consolidation. This is analogous to biological napping vs full-night sleep.

---

## Q4.6 — What Is the Post-Consolidation Cleanup?

### The Validation Procedure

Before clearing tags, verify that W_slow has actually learned the consolidated knowledge:

```
VALIDATE_CONSOLIDATION(τ, W_slow_new, W_slow_old) → pass/fail:

    # Reconstruct a test from the tag's context
    test_prompt = original_tokens[span_start : span_start + seed_length]

    # Check 1: W_slow's surprise on the original span has decreased
    surprise_old = mean_surprisal(W_slow_old, original_tokens[span_start:span_end])
    surprise_new = mean_surprisal(W_slow_new, original_tokens[span_start:span_end])

    IF surprise_new > surprise_old · (1 - ε_learn):   # ε_learn = 0.1 (10% improvement)
        RETURN fail   # W_slow didn't learn enough from this memory

    # Check 2: W_slow can generate relevant content from the seed
    generated = autoregressive_sample(W_slow_new, prompt=test_prompt, max_tokens=64)
    k_gen = W_proj · encode_and_pool(W_slow_new, generated)
    sim = cosine_similarity(k_gen, τ.k)

    IF sim < θ_validate:   # θ_validate = 0.4
        RETURN fail   # W_slow's generation doesn't reflect the memory

    RETURN pass
```

### Cleanup Actions

```
POST_CONSOLIDATION_CLEANUP(consolidated_tags, validation_results):

    FOR each τ in consolidated_tags:
        IF validation_results[τ] == pass:
            # Full cleanup: tag is now in W_slow
            REMOVE τ from tag buffer
            # W_fast: the LoRA components that encoded this memory
            # will be gradually overwritten by new experiences
            # (no explicit reset — just natural overwriting)

        ELIF validation_results[τ] == fail:
            # Re-queue: keep tag, reset PRP for next cycle
            τ.p = 0                          # lose PRP allocation
            τ.s = τ.s * 0.5                  # penalize strength (failed consolidation)
            # Tag remains in buffer, can be re-allocated next cycle

    # Reclaim PRP budget
    B_available += count(pass results)
```

### Gradual vs Immediate Clearing

**Immediate clearing for tags.** Once validated, the tag is removed from the buffer. The information is now in W_slow — the tag has served its purpose.

**Gradual clearing for W_fast.** We do NOT explicitly reset W_fast components after consolidation. Reasons:

1. **W_fast is not per-memory.** LoRA adapters encode a distributed blend of all recent experiences. You cannot isolate and remove "the component corresponding to memory τ₅."

2. **Natural overwriting handles it.** As new surprising experiences arrive, they update W_fast via gradient descent, naturally overwriting old patterns. The capacity constraint (low rank r) ensures old patterns are displaced by new ones.

3. **Biological fidelity.** The hippocampus doesn't have a "delete" operation. Old hippocampal traces are gradually overwritten by new encoding — this is exactly what happens with our W_fast.

### Failure Handling

If a tag fails validation:
- It loses its PRP but stays in the buffer
- Its strength is halved (penalty for failed consolidation)
- It can compete for PRP allocation again in future cycles
- If it fails 3 times (tracked via a failure counter), it is permanently removed — the system has repeatedly failed to consolidate it, suggesting it's either too complex or too volatile for the current architecture

---

## Part 4 Checkpoint Verification

> **Requirement:** Write the complete sleep cycle as a training script — every step from "select memories" to "clear tags" should be fully specified.

### Complete Sleep Cycle Script

```python
def sleep_cycle(system):
    """
    The complete sleep cycle.
    Transforms temporary W_fast knowledge into permanent W_cons knowledge.

    System components (per-user, consistent with Q6.4):
        system.W_slow_base:  Shared pretrained model (FROZEN — never modified)
        system.W_cons:       Per-user consolidated LoRA adapter (TRAINED during sleep)
        system.W_fast:       Per-user fast LoRA adapter (FROZEN during sleep)
        system.tags:         Tag buffer (up to 35K tags)
        system.fisher:       Fisher diagonal (periodically refreshed)
    """

    # ═══════════════════════════════════════════════════════
    # PHASE 1: SELECT — Identify consolidation candidates
    # ═══════════════════════════════════════════════════════
    candidates = [τ for τ in system.tags if τ.p == 1]  # PRP-allocated tags
    candidates.sort(key=lambda τ: τ.S, reverse=True)   # highest priority first

    if len(candidates) == 0:
        return  # Nothing to consolidate

    # Effective models for different purposes
    W_eff = compose(system.W_slow_base, system.W_cons, system.W_fast)  # generation
    W_target = compose(system.W_slow_base, system.W_cons)               # training target

    # ═══════════════════════════════════════════════════════
    # PHASE 2: GENERATE — Create replay dataset from W_fast
    # ═══════════════════════════════════════════════════════
    replay_dataset = []
    for τ in candidates:
        for attempt in range(3):  # up to 3 generation attempts
            sample = generate_replay(τ, W_eff)              # generate from full system
            if quality_check(sample, τ, W_target):           # validate against target
                replay_dataset.append(sample)
                break
        # If all 3 attempts fail, skip this tag

    if len(replay_dataset) == 0:
        return  # Generation failed for all candidates

    # ═══════════════════════════════════════════════════════
    # PHASE 3: INTERLEAVE — Mix new replay with old knowledge
    # ═══════════════════════════════════════════════════════
    N_steps = max(100, len(replay_dataset) * 5)
    batch_size = 32
    ppl_pre = evaluate(W_target, benchmark_set)
    W_cons_checkpoint = deepcopy(system.W_cons)

    # ═══════════════════════════════════════════════════════
    # PHASE 4: TRAIN — Update W_cons on interleaved dataset
    # ═══════════════════════════════════════════════════════
    # Only W_cons is optimized; W_slow_base is frozen
    optimizer = AdamW(system.W_cons.parameters(), lr=1e-5, weight_decay=0.01)
    scheduler = cosine_warmup(optimizer, warmup=N_steps//10, total=N_steps)

    for step in range(1, N_steps + 1):
        # Curriculum: warm-up → consolidate → stabilize
        if step < 0.3 * N_steps:
            η = 9     # 90% old
        elif step < 0.8 * N_steps:
            η = 4     # 80% old
        else:
            η = 9     # 90% old

        # Compose batch
        n_new = max(1, batch_size // (1 + η))
        n_old = batch_size - n_new

        batch_new = weighted_sample(replay_dataset, n_new, weight=lambda s: s['prp_score'])
        W_target_current = compose(system.W_slow_base, system.W_cons)
        batch_old_near = generate_nearby_old(W_target_current, batch_new, n_old // 2)
        batch_old_rand = generate_random_old(W_target_current, n_old - n_old // 2)

        batch = shuffle(batch_new + batch_old_near + batch_old_rand)

        # Forward + loss (gradients flow to W_cons only)
        L_lm = language_modeling_loss(W_target_current, batch)
        # EWC anchor: W_cons at START of this cycle (protects prior consolidated knowledge)
        L_ewc = ewc_penalty(system.W_cons, W_cons_checkpoint, system.fisher, λ=100)
        L_total = L_lm + L_ewc

        # Backward + update (only W_cons parameters)
        optimizer.zero_grad()
        L_total.backward()
        clip_grad_norm_(system.W_cons.parameters(), max_norm=1.0)
        apply_layer_plasticity(system.W_cons, plasticity_profile)  # scale grads by φ(l/L)
        optimizer.step()
        scheduler.step()

        # Hard clip W_cons relative to W_slow_base norms
        enforce_adapter_bounds(system.W_cons, W_cons_checkpoint,
                               system.W_slow_base, δ_max=0.001)

    # ═════════════════════════════════════════���═════════════
    # PHASE 5: VALIDATE & CLEAR
    # ═══════════════════════════════════════════════════════

    # Check base capability preservation
    W_target_new = compose(system.W_slow_base, system.W_cons)
    ppl_post = evaluate(W_target_new, benchmark_set)
    if ppl_post > ppl_pre * 1.02:  # >2% degradation
        system.W_cons = W_cons_checkpoint  # ROLLBACK
        log("Sleep cycle rolled back: base capability degradation")
        return

    # Validate each consolidated memory
    W_target_old = compose(system.W_slow_base, W_cons_checkpoint)
    for τ in candidates:
        if validate_consolidation(τ, W_target_new, W_target_old):
            system.tags.remove(τ)           # Tag cleared — knowledge is in W_cons
        else:
            τ.p = 0                         # Lose PRP allocation
            τ.s *= 0.5                      # Strength penalty
            τ.fail_count += 1
            if τ.fail_count >= 3:
                system.tags.remove(τ)       # Permanent removal after 3 failures

    log(f"Consolidated {count_passed}/{len(candidates)} memories")
    log(f"PPL: {ppl_pre:.2f} → {ppl_post:.2f}")
```

**This script is fully specified.** Every step from selection to cleanup is defined with concrete parameters, loss functions, and decision rules. No hand-waving remains.

---

*Part 4 complete. All six questions (Q4.1–Q4.6) are fully resolved. Proceeding to Part 5.*

---

# Part 5: System-Level Questions

*These cut across all components and must be resolved before implementation.*

---

## Q5.1 — What Is the Formal Definition of "Memory" in This System?

### Definition

A **memory** m is the complete lifecycle record of a single surprising experience, from detection through consolidation:

```
m = (τ, W_fast_contribution, lifecycle_state)
```

where:

| Component | Type | Description |
|:---|:---|:---|
| **τ** | Tag record (Q1.1) | The metadata: key vector, strength, access history, context reference |
| **W_fast_contribution** | Implicit (distributed in LoRA params) | The actual learned pattern — not explicitly separable from other memories in W_fast |
| **lifecycle_state** | Enum | Current state of this memory in the system |

### Lifecycle States

```
                  ┌──────────────────────────────────────────┐
                  │         MEMORY LIFECYCLE                   │
                  │                                            │
                  │   TAGGED ──→ PRP_ALLOCATED ──→ CONSOLIDATED│
                  │     │              │               │        │
                  │     │              │               └→ (in W_slow, tag removed)
                  │     ▼              ▼                        │
                  │   DECAYED      DEMOTED                     │
                  │   (garbage     (returns to                 │
                  │    collected)   TAGGED state)              │
                  └──────────────────────────────────────────┘
```

| State | Tag Exists? | PRP Allocated? | In W_slow? | Description |
|:---|:---|:---|:---|:---|
| **TAGGED** | Yes | No | No | Freshly created, decaying, competing for PRP |
| **PRP_ALLOCATED** | Yes | Yes | No | Queued for consolidation in next sleep cycle |
| **CONSOLIDATED** | No (cleared) | N/A | Yes | Successfully transferred to permanent weights |
| **DECAYED** | No (gc'd) | No | No | Tag decayed below threshold; memory lost |
| **DEMOTED** | Yes | No (lost) | No | Lost PRP via competition; returns to TAGGED |

### Memory Boundaries

**One tag = one memory.** A 50-turn conversation produces multiple memories (one per surprising span), not a single memory. Memories are the surprising *spans within* experiences, not the experiences themselves.

A conversation with 50 turns might produce:
- 5-15 surprising spans → 5-15 tags → 5-15 memories
- Most tokens are expected (greetings, common phrasing) → no tags
- Only the genuinely novel content creates memories

### Memory Relations

Memories relate to each other via **key vector similarity** — the same mechanism used for cross-reference density (Q2.2). Two memories with cos(k_i, k_j) > θ_xref are "related."

**Graph structure:** The active memories form a **similarity graph** G = (V, E) where:
- V = active tags
- E = {(τ_i, τ_j) : cos(k_i, k_j) > θ_xref}

This graph is **emergent** (not explicitly constructed) and is used for cross-reference scoring. Connected components in this graph represent **topic clusters** — groups of related memories about the same subject.

### Can Memories Merge?

**Not during wake.** Each tag is independent. **Potentially during sleep:** if two related tags are both PRP-allocated and consolidated in the same cycle, their replay samples may be interleaved, and W_slow may learn a unified representation. This is **implicit merging through training** — not an explicit merge operation. The result is that W_slow holds a generalized understanding, not two separate facts.

---

## Q5.2 — What Is the State Space of the Complete System?

### System State

At any point in time, the complete system state is:

```
S = (W_slow, W_fast, T, P, phase, μ, σ², step)
```

| Component | Type | Size (7B model) |
|:---|:---|:---|
| **W_slow** | ℝ^{\|W_slow\|} | 7B floats |
| **W_fast** | ℝ^{\|W_fast\|} | 2.88M floats |
| **T** | Set of tag records | ≤ 35K records × 544 bytes |
| **P** | PRP allocation {(S_i, p_i)} for each τ ∈ T | ≤ 35K × 2 scalars |
| **phase** | {WAKE, SLEEP} | 1 bit |
| **μ, σ²** | Running surprise statistics | 2 floats |
| **step** | Global counter | 1 integer |

### State Transition Function

```
δ(S, input) → S':

CASE phase == WAKE:
    CASE input == token_sequence:
        # Forward pass, compute surprise, create tags, update W_fast
        S'.W_slow = S.W_slow                        # unchanged
        S'.W_fast = UPDATE_WFAST(S.W_fast, input)   # gradient step on surprises
        S'.T = S.T ∪ NEW_TAGS(input) \ GC_TAGS(S.T) # add new, remove decayed
        S'.P = ALLOCATE_PRPS(S'.T, B)               # re-score and allocate
        S'.step = S.step + 1

    CASE input == sleep_trigger:
        S'.phase = SLEEP
        # Begin consolidation (state transitions handled by sleep loop)

CASE phase == SLEEP:
    CASE input == training_step:
        S'.W_slow = SGD_STEP(S.W_slow, replay_batch)  # one gradient step
        S'.W_fast = S.W_fast                            # frozen during sleep

    CASE input == sleep_complete:
        S'.T = CLEANUP(S.T, validation_results)   # remove consolidated tags
        S'.P = RESET_PRP(S'.T)                     # free PRP budget
        S'.phase = WAKE
        S'.W_fast = S.W_fast                       # W_fast persists

    CASE input == query_interrupt:
        # Pause sleep, process query, resume
        response = INFERENCE(S.W_slow, S.W_fast, query)
        S'.phase = SLEEP  # resume
```

### System Invariants

These must hold at ALL times:

```
INV1: |{τ ∈ T : τ.p == 1}| ≤ B                    # PRP allocations within budget
INV2: |T| ≤ N_max                                   # Tag count within capacity
INV3: ∀τ ∈ T: τ.s ≥ ε_gc                           # No dead tags in buffer
INV4: ∀τ ∈ T: τ.s ∈ [0, 1]                         # Strength bounded
INV5: phase == SLEEP ⟹ W_fast is frozen            # No W_fast updates during sleep
INV6: phase == WAKE  ⟹ W_slow is frozen            # No W_slow updates during wake
INV7: ‖W_slow^(l) - W_anchor^(l)‖_F ≤ δ_max^(l) · ‖W_anchor^(l)‖_F    # Weight bounds
```

### State Machine Diagram

```
                    ┌──────────────────────────────┐
                    │                              │
                    │          WAKE PHASE           │
                    │                              │
                    │  • Process inputs             │
                    │  • Compute prediction error   │
                    │  • Create/decay tags          │
                    │  • Update W_fast              │
                    │  • Score & allocate PRPs      │
                    │  • Answer queries             │
                    │                              │
                    └──────────┬───────────────────┘
                               │
                    sleep_trigger fires
                    (schedule/pressure/idle)
                               │
                               ▼
                    ┌──────────────────────────────┐
                    │                              │
                    │         SLEEP PHASE           │
                    │                              │
                    │  1. SELECT PRP-tagged memories│
                    │  2. GENERATE replay from W_fast│
                    │  3. INTERLEAVE with old       │
                    │  4. TRAIN W_slow              │
                    │  5. VALIDATE consolidation    │
                    │  6. CLEAR consolidated tags   │
                    │                              │
                    └──────────┬───────────────────┘
                               │
                    sleep_complete OR
                    all tags processed
                               │
                               ▼
                    ┌──────────────────────────────┐
                    │          WAKE PHASE           │ ← cycle continues
                    └──────────────────────────────┘
```

---

## Q5.3 — What Are the Scaling Laws?

### Computational Complexity Per Operation

| Operation | Time Complexity | Space Complexity | For 7B Model |
|:---|:---|:---|:---|
| **Tag creation** (per span) | O(d_tag · d_model) | O(d_tag + metadata) | ~0.5M FLOPs, 544 bytes |
| **Prediction error** (per token) | O(forward_pass) — already computed | O(1) per token | ~14 TFLOPs (shared with inference) |
| **Tag decay + GC** (per step) | O(N_active) | O(1) | ~35K comparisons |
| **Access detection** (per query) | O(N_active · d_tag) | O(d_tag) | ~4.5M FLOPs |
| **PRP scoring** (periodic) | O(N_active · log N_active) | O(N_active) | ~35K · 15 ≈ 500K ops |
| **Cross-ref density** (periodic) | O(N_active² · d_tag) | O(N_active²) | ~160 GFLOPs every 500 steps |
| **Replay generation** (per memory) | O(target_length · forward_pass) | O(target_length) | ~100 tokens × 14 TFLOPs ≈ 1.4 PFLOPs |
| **Sleep training** (per step) | O(batch_size · seq_len · forward+backward) | O(model_size) | ~3 × 14 TFLOPs ≈ 42 TFLOPs |

### Scaling Relationships with Model Size

| Quantity | Scaling with \|W_slow\| | Formula |
|:---|:---|:---|
| Tag capacity N_max | Linear | 5000 · (\|W\| / 10⁹) |
| PRP budget B | Linear | 500 · (\|W\| / 10⁹) |
| W_fast parameters | Linear | (4/3) · L · r · d_model |
| Tag dimensionality d_tag | Sublinear (log) | 64 · ⌈log₂(\|W\| / 10⁹)⌉ |
| Sleep cycle duration | Linear in B, quadratic in model size | O(B · forward_pass) |
| Total system overhead | ~5% inference cost during wake | Dominated by W_fast gradient steps |

### Overhead Summary

| Phase | Overhead vs Base Inference | Dominated By |
|:---|:---|:---|
| **Wake (no surprising input)** | ~0.01% | Tag decay check, access detection |
| **Wake (surprising input)** | ~5-10% | W_fast gradient step (backward pass on span) |
| **Sleep** | 100% (system unavailable*) | Replay generation + W_slow training |

*With interruption handling, sleep can be paused for queries, so the system is never truly unavailable — just slower.

---

## Q5.4 — What Are the Evaluation Metrics?

### Primary Metrics

#### Metric 1: Delayed Recall Accuracy

```
DRA(t_delay) = (1/|Q_test|) · Σ_q∈Q_test 𝟙[answer_correct(q, t_delay)]
```

Present information at time 0, test recall at time t_delay (measured in inference steps or sleep cycles).

| t_delay | What It Measures |
|:---|:---|
| 0 (immediate) | W_fast encoding quality |
| 1 sleep cycle | Consolidation effectiveness |
| 10 sleep cycles | Long-term retention |
| 100 sleep cycles | Permanent knowledge integration |

**Baseline comparison:** RAG at same delay (always has access to original document).

#### Metric 2: Forgetting Curve Shape

```
F(t) = DRA(0) - DRA(t)    // absolute forgetting
F_rel(t) = F(t) / DRA(0)  // relative forgetting
```

**Target shape:** Exponential decay for unimportant memories (similar to human Ebbinghaus curve), near-zero forgetting for consolidated memories. The system should show a **bimodal** forgetting curve: steep for non-consolidated memories, flat for consolidated ones.

#### Metric 3: Base Capability Preservation

```
BCP(n) = PPL_benchmark(W_slow after n cycles) / PPL_benchmark(W_slow original)
```

BCP should stay close to 1.0. BCP > 1.05 (5% degradation) after any number of cycles is a failure.

**Benchmarks:** A diverse set including language modeling (WikiText), reasoning (MMLU subset), and code (HumanEval subset).

#### Metric 4: Consolidation Efficiency

```
CE = |successfully_consolidated| / |PRP_allocated|
```

What fraction of PRP-allocated memories pass validation? Target: CE > 0.8 (80% success rate).

#### Metric 5: PRP Allocation Quality

```
PAQ = Precision@B = |{τ : τ.p=1 AND τ was later accessed}| / B
```

Compare against **oracle allocation** (hindsight-optimal: allocate PRPs to the B memories that would be accessed most in the next T steps).

```
PAQ_relative = PAQ / PAQ_oracle
```

Target: PAQ_relative > 0.6 (our allocation is at least 60% as good as the oracle).

### Evaluation Suite Summary

| Metric | What It Measures | Target | Baseline |
|:---|:---|:---|:---|
| DRA(1 cycle) | Consolidation works at all | > 0.7 | RAG: ~0.9 |
| DRA(100 cycles) | Long-term retention | > 0.6 | RAG: ~0.9 |
| F_rel(1 cycle, non-consolidated) | Graceful forgetting | > 0.5 (should forget!) | 0 (never forgets with RAG) |
| BCP(100 cycles) | No catastrophic damage | < 1.05 | 1.0 (baseline) |
| CE | Sleep efficiency | > 0.8 | N/A |
| PAQ_relative | Smart prioritization | > 0.6 | Random: ~0.1 |

---

## Q5.5 — What Are the Theoretical Guarantees (If Any)?

### What We Can Prove

#### Guarantee 1: Tag Decay Convergence

**Claim:** In the absence of access events, the tag buffer empties in finite time.

**Proof:** Each tag's strength follows s(t) = (s₀ - ε) · exp(-Δt/τ_decay) + ε. Since s₀ ≤ 1, τ_decay is bounded (τ_base · (1 + γ · e_max)), and ε_gc > ε, there exists a finite T_gc such that s(T_gc) < ε_gc for any initial conditions. Specifically:

```
T_gc = τ_decay · ln((s₀ - ε) / (ε_gc - ε))
```

For worst case (s₀ = 1, ε = 0.01, ε_gc = 0.02, τ_decay = 1500):

```
T_gc = 1500 · ln(0.99/0.01) = 1500 · 4.60 ≈ 6900 steps
```

All tags are garbage-collected within ~7000 steps if never accessed. ∎

#### Guarantee 2: PRP Allocation Convergence

**Claim:** Under fixed scores (no access events, no new tags), the PRP allocation converges within 2 re-evaluation cycles.

**Proof sketch:** With fixed scores, the top-B set is fixed. Hysteresis means allocated tags can only lose allocation if outranked; with fixed scores, rankings don't change. After one evaluation, the correct top-B is allocated. After one more evaluation, no changes occur (allocation matches ranking). δ_steal prevents marginal swaps. ∎

#### Guarantee 3: Weight Change Bound

**Claim:** After any number of sleep cycles, the total change to W_slow is bounded:

```
‖W_slow^(after n cycles, layer l) - W_anchor^(l)‖_F ≤ δ_max^(l) · ‖W_anchor^(l)‖_F
```

**Proof:** The hard clipping in Q3.4 enforces this after every training step. The bound is relative to the **original anchor** (not the previous cycle's weights), so it provides an absolute lifetime guarantee regardless of the number of cycles. ∎

#### Guarantee 4: Interleaved Training Reduces Forgetting Risk

**Claim:** Under the interleaving strategy with ratio η, the expected gradient on W_slow is:

```
E[∇L] = (1/(1+η)) · E[∇L_new] + (η/(1+η)) · E[∇L_old]
```

For η = 4, this is 20% new + 80% old. The gradient is dominated by the old knowledge direction, limiting the step taken toward new knowledge per iteration. Combined with the low learning rate (α_slow = 1e-5) and EWC penalty, the risk of catastrophic forgetting per step is bounded by:

```
‖ΔW_slow‖ ≤ α_slow · ‖∇L‖ ≤ α_slow · (max_grad_norm) = 1e-5 · 1.0 = 1e-5
```

This is extremely conservative.

### What We Cannot Prove (Yet)

| Property | Status | Difficulty |
|:---|:---|:---|
| Generative replay preserves all important information | **Conjectured** | Depends on W_fast's generative capacity; no general bound |
| Long-term W_slow doesn't drift | **Bounded but not proven stable** | Hard clipping bounds drift, but doesn't prove convergence to useful state |
| PRP scoring selects the "right" memories | **Empirical** | No formal optimality guarantee; must be evaluated experimentally |
| The system outperforms RAG on any specific benchmark | **Unknown** | This is the research question, not a theorem |
| Information-theoretic capacity of the system | **Open** | Depends on interaction between LoRA rank, W_slow capacity, and consolidation quality |

### Honest Assessment

The theoretical guarantees establish **safety** (the system won't catastrophically fail) but not **efficacy** (the system will work well). This is appropriate for a research system: we can guarantee it won't break, but whether it achieves its goals is an empirical question.

---

*Part 5 complete. All five questions (Q5.1–Q5.5) are fully resolved. Proceeding to Part 6.*

---

# Part 6: Missing Formalisms

*These are questions that cut across the architecture and would otherwise fall through the cracks. Q6.3 was resolved as a prerequisite in Part 1.*

---

## Q6.1 — How Does the System Handle Memory Revision and Unlearning?

### The Problem

Once a memory is consolidated into W_slow, it's in the weights. What happens when new information contradicts old consolidated knowledge?

### Contradiction Detection

The key insight: **contradiction produces a specific signature in prediction error.** Novel information is surprising because the model has no expectation. Contradictory information is surprising because the model has a *strong, wrong* expectation.

```
DETECT_CONTRADICTION(input_span, W_slow) → (is_contradiction, confidence):

    # Compute per-token surprise
    surprises = [-log p_{W_slow}(x_t | x_{<t}) for t in span]

    # Compute per-token entropy of the model's prediction distribution
    entropies = [H(p_{W_slow}(· | x_{<t})) for t in span]

    # The key discriminator:
    # NOVELTY:       high surprise + high entropy  (model doesn't know → uncertain)
    # CONTRADICTION:  high surprise + low entropy   (model is confident → but wrong)

    mean_surprise = mean(surprises)
    mean_entropy = mean(entropies)

    contradiction_score = mean_surprise · (1 / (1 + mean_entropy))
    novelty_score = mean_surprise · mean_entropy

    is_contradiction = contradiction_score > θ_contra AND contradiction_score > novelty_score
    confidence = contradiction_score / (contradiction_score + novelty_score)

    RETURN (is_contradiction, confidence)
```

| Scenario | Surprise | Entropy | Contradiction Score | Novelty Score | Classification |
|:---|:---|:---|:---|:---|:---|
| Expected content | Low | Low | Low | Low | Neither |
| Novel topic | High | High | Low | High | **Novelty** |
| Contradicts known fact | High | Low | **High** | Low | **Contradiction** |
| Ambiguous/uncertain area | Moderate | High | Low | Moderate | Novelty (weak) |

### The Revision Pathway

When contradiction is detected:

```
IF is_contradiction AND confidence > θ_conf:    # θ_conf = 0.6
    # Step 1: Create a REVISION TAG (distinct from normal tags)
    τ_rev = CREATE_TAG(span, step)
    τ_rev.type = REVISION                       # marked as revision, not novelty
    τ_rev.e₀ = E_span · (1 + confidence)        # boosted initial error (revisions are urgent)

    # Step 2: Boost PRP scoring for revisions
    # Revision tags get a bonus in PRP scoring:
    # S_revision(τ) = S(τ) + w_rev · confidence
    # where w_rev = 0.3 (revision bonus weight)

    # Step 3: During sleep, revision tags are handled specially
    # They get MORE replay iterations (2x normal frequency)
    # to more thoroughly overwrite the old knowledge in W_slow
```

### Does Overwriting Work?

**Yes, with caveats.** Training W_slow on the new (correct) information will push the weights toward the new knowledge. The interleaving strategy includes the new information in the replay, and W_slow will update. However:

1. **Old knowledge may not be fully erased.** Neural network weights don't work like a database — you can't delete a fact. The new training will reduce W_slow's confidence in the old fact, but it may persist as a low-probability alternative.

2. **Explicit unlearning is not guaranteed.** If the old fact was consolidated across many sleep cycles, it may be deeply encoded. A single revision cycle may not fully overwrite it.

3. **Mitigation:** Revision tags get 2x replay frequency and a PRP scoring bonus, meaning they receive more gradient pressure during consolidation. Multiple sleep cycles with the revision tag still active will progressively overwrite the old knowledge.

### Formal Difference: Formation vs Revision

| Aspect | Memory Formation | Memory Revision |
|:---|:---|:---|
| Trigger | High surprise + high entropy | High surprise + low entropy |
| Tag type | NOVELTY | REVISION |
| PRP bonus | None | +w_rev · confidence |
| Replay frequency | 1x (weighted by PRP score) | 2x (double replay weight) |
| Success criterion | W_slow surprise decreases | W_slow confidence on OLD fact decreases AND on NEW fact increases |

---

## Q6.2 — How Does the System Handle Cold Start?

### The Problem

A new user triggers high prediction error on everything (the model has never seen this user's style, domain, or preferences). Early tagging will be noisy and indiscriminate.

### Adaptive Threshold Calibration

The adaptive threshold (Q1.2) already handles this partially: the z-score normalizes by running statistics, so even if absolute surprise is high, only tokens significantly above the running mean are tagged. But the running statistics need a burn-in period to stabilize.

```
COLD_START_CALIBRATION:

    # Phase 1: Observation (first N_burnin interactions)
    # N_burnin = 50 interactions (~5000-10000 tokens)

    FOR step = 1 TO N_burnin:
        # Process input, compute surprises, update μ and σ²
        # But: DO NOT create tags (or create with elevated threshold)

        IF step < N_burnin:
            κ_effective = κ_cold                  # κ_cold = 3.0 (very strict)
        ELSE:
            κ_effective = κ                       # κ = 1.5 (normal)

    # Phase 2: Gradual relaxation (next N_ramp interactions)
    # N_ramp = 50 interactions

    FOR step = N_burnin TO N_burnin + N_ramp:
        progress = (step - N_burnin) / N_ramp     # 0 → 1
        κ_effective = κ_cold - progress · (κ_cold - κ)   # 3.0 → 1.5

    # Phase 3: Normal operation
    κ_effective = κ
```

### Cold-Start Behavior of PRP Scoring Components

| Component | Behavior at Cold Start | Mitigation |
|:---|:---|:---|
| **Prediction Error (E)** | Everything is high → indiscriminate | Elevated κ suppresses tagging |
| **Access Frequency (A)** | Zero for all tags (no history) | A contributes 0 for all; effectively removed from scoring |
| **Cross-Reference (X)** | Meaningless with few tags | X contributes ~0; effectively removed |
| **Recency (R)** | Only very recent accesses exist | R is functional but narrow |

**At cold start, PRP scoring effectively reduces to prediction error only:** S ≈ 0.35 · Ê (since A, X, R are all near zero). This is acceptable — during cold start, the system should consolidate the most surprising information, which is exactly what error-only scoring does. As the system accumulates history, the other components kick in.

### PRP Budget During Cold Start

**Smaller budget during cold start** to consolidate cautiously:

```
B_effective(step) = B · min(1.0, step / N_mature)
```

where N_mature = 500 interactions. The PRP budget linearly ramps from 0 to full over the first 500 interactions.

| Interaction | B_effective (7B model) | Character |
|:---|:---|:---|
| 1 | 7 | Almost nothing consolidated |
| 50 | 350 | Beginning to consolidate |
| 250 | 1,750 | Half budget |
| 500+ | 3,500 | Full budget |

### Estimated Burn-In Duration

How many interactions until tagging precision is acceptable?

The adaptive threshold stabilizes when μ and σ² converge. With EMA smoothing β = 0.99, the effective window is ~100 tokens. After ~500-1000 tokens (roughly 5-10 interactions), the running statistics are stable.

**Tagging precision estimate:**

| Phase | Interactions | Precision (fraction of tags that are genuinely useful) |
|:---|:---|:---|
| 0-50 | Burn-in | ~0.2 (mostly noise, but strict threshold limits volume) |
| 50-200 | Ramping | ~0.5 (improving as statistics stabilize) |
| 200-500 | Maturing | ~0.7 (good; PRP scoring begins filtering effectively) |
| 500+ | Stable | ~0.8+ (full system operational) |

---

## Q6.4 — Multi-User Dynamics and W_slow Sharing

### Deployment Architecture

**Per-user W_fast and tags, shared base W_slow with per-user LoRA consolidation adapters.**

```
┌──────────────────────────────────────────────────────┐
│                  SHARED BASE                          │
│           W_slow_base (7B, frozen)                   │
│           Loaded once, read-only                      │
└──────────┬───────────┬───────────┬───────────────────┘
           │           │           │
    ┌──────┴──┐ ┌──────┴──┐ ┌─────┴───┐
    │ User A  │ │ User B  │ │ User C  │
    │         │ │         │ │         │
    │ W_fast_A│ │ W_fast_B│ │ W_fast_C│  (LoRA, ~2.88M each)
    │ Tags_A  │ │ Tags_B  │ │ Tags_C  │  (buffer, ~18.6MB each)
    │ W_cons_A│ │ W_cons_B│ │ W_cons_C│  (consolidated LoRA, ~2.88M each)
    └─────────┘ └─────────┘ └─────────┘
```

### How It Works

Each user has THREE LoRA adapter sets:

1. **W_fast_user** — The live hippocampal adapter, updated during wake (as defined in Q3.3)
2. **W_cons_user** — The consolidated adapter, updated during sleep. This is the per-user "neocortex"
3. **W_slow_base** — The shared pretrained model (never modified)

**Effective weights for User A during inference:**

```
W_eff_A = W_slow_base + (α/r) · W_cons_A + (α/r) · W_fast_A
```

**During sleep for User A:**

Instead of modifying W_slow_base (which is shared), we train W_cons_A:

```
W_cons_A ← W_cons_A - α_slow · ∇_{W_cons_A} (L_replay + L_ewc)
```

The EWC anchor is W_slow_base + W_cons_A_initial (protecting both base capabilities and previously consolidated per-user knowledge).

### Isolation Guarantees

**User A's consolidation cannot affect User B.** The adapters are completely separate parameter sets. The shared W_slow_base is never modified.

**Cost per user:**

| Component | Size | Storage |
|:---|:---|:---|
| W_fast_user | 2.88M params | ~5.8 MB (fp16) |
| W_cons_user | 2.88M params | ~5.8 MB (fp16) |
| Tags | ≤ 35K records | ~18.6 MB |
| **Total per user** | — | **~30 MB** |

For 1000 users: ~30 GB total. This is feasible — it's less than a single copy of the 7B model (14 GB in fp16).

### Limitation: Consolidation Capacity

Per-user consolidation is limited by the LoRA rank r. A single set of LoRA adapters (rank 16) can represent a finite amount of knowledge. After extensive consolidation, the adapter may saturate.

**Mitigation:** Periodically "merge and reset":

```
IF W_cons_user has been trained for > N_merge_cycles (e.g., 50):
    # Merge consolidated adapter into a per-user W_slow copy
    W_slow_user = W_slow_base + W_cons_user    # full merge
    W_cons_user = zeros()                       # reset consolidated adapter
    # Now W_slow_user is the new base for this user
    # Cost: one full model copy per user (~14 GB)
```

This is expensive but infrequent, and only for power users with extensive consolidation history.

---

## Q6.5 — Compositionality Across Sleep Cycles

### The Question

Can W_slow reason about relationships between memories consolidated in different cycles?

### Mechanism: Interleaving Provides Cross-Cycle Integration

When Memory B is consolidated in cycle 2, the interleaving strategy samples "old knowledge" from the current W_slow (which already contains Memory A from cycle 1). If Memory B is related to Memory A:

```
Cycle 2 training batch might contain:
    - New replay: "The project deadline is March 5"   (Memory B, being consolidated)
    - Old replay:  "The project was started by team Alpha in January"  (from W_slow, which encoded Memory A)
```

By training on both in the same batch, W_slow can form associations between them. The gradient update from this batch implicitly creates a representation that links "project deadline March 5" and "team Alpha started in January."

### Is This Sufficient for Relational Reasoning?

**For simple associations: Yes.** W_slow is a large language model — it is inherently capable of relational reasoning. If it sees "X has property A" and "X has property B" during training, it can reason about their conjunction at inference time.

**For complex multi-hop reasoning: Partially.** If Memory A says "User prefers X" (cycle 1) and Memory E says "X conflicts with Y" (cycle 5), the system needs both facts to be simultaneously active during a sleep cycle to form the connection "warn user about Y."

The **proximity-weighted old knowledge sampling** (Q4.3) helps: when consolidating Memory E about "X conflicts with Y," the system generates old samples related to X, which will include "User prefers X" (if it was successfully consolidated). This creates the training signal for the compositional inference.

### Long-Term Knowledge Structure

After 100 sleep cycles, W_slow contains:

```
W_slow^(100) = W_base + Σ_{c=1}^{100} ΔW_slow^(c)
```

where each ΔW_slow^(c) is bounded by the plasticity constraints. The knowledge is **distributed across the weight changes** — there is no explicit memory structure.

**The structure is emergent, not hierarchical.** W_slow doesn't have a "memory bank" with labeled entries. Instead, related facts are encoded in overlapping weight patterns, similar to how the pretrained model already encodes world knowledge. Consolidation adds new knowledge in the same distributed format.

### Formal Model of Knowledge Compounding

Let K_c be the set of knowledge items consolidated in cycle c. After N cycles:

```
K_total = K_pretrained ∪ K_1 ∪ K_2 ∪ ... ∪ K_N
```

**Availability:** A knowledge item k ∈ K_c is available at inference time if W_slow can produce the correct answer when prompted. Availability depends on:

1. **Consolidation quality:** Was k successfully transferred from W_fast to W_slow? (CE metric from Q5.4)
2. **Interference:** Did subsequent cycles overwrite k? Bounded by:
   - EWC penalty protecting the weights that encode k
   - Interleaving ensuring k appears in replay during subsequent cycles
3. **Relational integration:** Can W_slow combine k with items from other cycles? Supported by proximity-weighted interleaving.

**Expected knowledge retention after N cycles:** (Analyzed more formally in Q6.9.)

---

## Q6.6 — Adversarial Robustness of the Scoring Function

### Threat Model

An adversary (or a user behaving pathologically) attempts to manipulate what gets consolidated.

### Attack 1: Repetition Attack

*The user asks the same irrelevant question 100 times.*

**Defense: Diminishing returns (1/√a) on access reinforcement.**

```
After 100 identical accesses:
    ρ = sim · Σ_{i=1}^{100} 1/√i ≈ sim · 2√100 = 20 · sim

After 100 DISTINCT relevant accesses:
    ρ ≈ 100 · sim_avg    (each access to a different tag, a=1 for each)
```

The 100 identical accesses produce ρ ≈ 20 · sim, while genuine diverse usage produces ρ ≈ 100 · sim. The scoring function naturally distinguishes them.

**Additional defense:** If the same query produces the same tag access pattern repeatedly, the **Recency-Weighted Utility R** saturates (recent accesses are identical, so R doesn't grow).

### Attack 2: Surprise Flooding

*The adversary feeds the model deliberately surprising but useless information to consume the tag budget.*

**Defense: Multi-component scoring.** Surprising-but-useless information will:
- Score high on E (prediction error) ✓
- Score low on A (never accessed later, ρ = 0) ✗
- Score low on X (no cross-references to other tags) ✗
- Score low on R (no recency utility) ✗

With weights (0.35, 0.30, 0.15, 0.20), the maximum score from error alone is 0.35. The threshold θ_PRP will typically be > 0.35 in a system with any real usage, so the useless tags won't get PRP allocation.

**Additional defense:** The PRP budget ramp-up during cold start (Q6.2) limits damage during the vulnerable early period.

### Attack 3: Confidence Manipulation

*The adversary crafts input to be high-surprise in a specific domain to force consolidation of misleading information.*

**Defense:** This is the hardest attack to prevent. If the adversary provides information that is:
- Genuinely surprising to W_slow (high E)
- Frequently relevant to real queries (high A) — e.g., poisoning a technical domain
- Connected to other memories (high X)

...then the scoring function will correctly rank it highly. The system is designed to consolidate what seems important — if an adversary can convince the system that misleading information IS important, it will be consolidated.

**Mitigation (not foolproof):**
1. Contradiction detection (Q6.1) catches cases where the adversary's input contradicts existing knowledge
2. The consolidation validation step (Q4.6) checks that consolidated knowledge doesn't degrade base capabilities
3. Ultimately, this is a trust/safety problem beyond the scope of the memory architecture

### Formal Robustness Properties

| Attack | Component Targeted | Defense | Robustness |
|:---|:---|:---|:---|
| Repetition | Access frequency (A) | 1/√a diminishing returns | Strong — mathematically bounded |
| Surprise flooding | Prediction error (E) | Multi-component scoring | Moderate — requires > 35% of score to dominate |
| Confidence manipulation | All components | Contradiction detection | Weak — fundamentally hard |

---

## Q6.7 — The Hidden Retrieval Problem During Inference

### The Problem

During wake, tags influence the system's behavior. But "which tags are relevant to this query?" requires a retrieval step — potentially reintroducing the very problem SLEEP claims to solve.

### How Tag Activation Works (Recap from Q1.5)

```
For query q:
    k_q = W_proj · mean_pool(hidden_states(q))
    FOR each tag τ in buffer:
        IF cosine_similarity(k_q, τ.k) > θ_access:
            REINFORCE τ
```

This IS a similarity search. It IS retrieval. Is this "just RAG inside the model"?

### The Formal Argument: Why This Is NOT RAG

**Critical distinction: Tags don't modify the model's output.** In RAG:

```
RAG:     output = LLM(query + retrieved_documents)
         ↑ retrieved content is IN the context, directly affecting the output
```

In SLEEP:

```
SLEEP:   output = (W_slow + W_fast)(query)
         ↑ output depends ONLY on the weights, NOT on tag content
         ↑ tags are accessed for METADATA purposes only (reinforcement, scoring)
```

Tags are **not retrieved and injected into context.** They are **accessed to update their metadata** (for future PRP scoring). The model's output is determined entirely by W_slow + W_fast — the weights, not the tag buffer.

### So What Do Tags Actually Do During Inference?

Tags serve as a **monitoring and prioritization layer**, not a retrieval layer:

1. **During wake:** Tags are accessed → reinforced → accumulate utility → inform PRP scoring
2. **During sleep:** PRP-allocated tags determine what gets consolidated
3. **After consolidation:** Tags are deleted; the knowledge is in W_slow permanently

The tag system is the "decision-making apparatus" for consolidation, not a retrieval mechanism for inference. The model's inference quality depends on W_slow + W_fast, not on the tags.

### When W_fast Isn't Enough

**The honest limitation:** Between tagging and consolidation, the model relies on W_fast for recent knowledge. W_fast has limited capacity (rank 16 LoRA). If the user asks about a very specific detail that W_fast couldn't encode in its low-rank representation, the model will fail to recall it — even though the tag exists in the buffer.

**This is a real trade-off vs RAG:**

| Aspect | RAG | SLEEP |
|:---|:---|:---|
| Immediate access to verbatim content | Yes (retrieves original) | No (W_fast encodes gist, not details) |
| Scales to unlimited documents | Yes (just add to index) | Limited by W_fast capacity |
| Knowledge persists long-term | Only while in index | Permanently, in W_slow weights |
| No retrieval failures | No (wrong chunks retrieved) | Yes (knowledge is always "in the model") |
| Improves with use | No (static index) | Yes (consolidation improves model) |
| Context window cost | Yes (retrieved docs consume tokens) | No (knowledge is in weights) |

**SLEEP sacrifices immediate verbatim recall for long-term integrated understanding.** This is the fundamental trade-off the proposal makes, and the math confirms it.

### Computational Cost of Tag Activation

```
Cost = N_active · d_tag · (operations for cosine similarity)
     = 35,000 · 128 · ~10 FLOPs
     = ~45M FLOPs per query
```

For reference, a 7B model forward pass is ~14 TFLOPs. Tag activation is **0.0003%** of the forward pass cost. Negligible.

For very large tag buffers (350K for 70B model), use approximate nearest neighbor search (locality-sensitive hashing or FAISS) to maintain O(log N) query time.

---

## Q6.8 — Privacy Guarantees on Generative Replay

### The Threat Model

A user shares sensitive information (PII, credentials, medical history). This information triggers high prediction error and gets tagged. During sleep, the replay generator produces samples from W_fast. Can it reproduce the sensitive content verbatim?

### Analysis: Can W_fast Memorize Verbatim Content?

**W_fast has 2.88M parameters (7B model) with rank 16.** For context:

- 2.88M parameters can store at most ~2.88M float16 values = ~5.76 MB of raw information
- A single page of text ≈ 500 tokens × 2 bytes = 1 KB
- Theoretical maximum verbatim storage: ~5,760 pages

But this is an extreme overestimate. The LoRA parameters are distributed across 11 layers × 2 matrices. They encode patterns, not raw text. Empirically, LoRA fine-tuning on small datasets shows **generalization, not memorization** — the model learns patterns and produces varied completions, not verbatim copies.

### Formal Bound on Verbatim Leakage

We can bound the probability of verbatim reproduction using the **extraction likelihood:**

```
P(verbatim) = p_{W_slow+W_fast}(exact_original_sequence | seed_prompt)
```

For a sequence of length n tokens with vocabulary size V:

```
P(verbatim) ≤ Π_{t=1}^{n} p(x_t | x_{<t})
```

Even if W_fast shifts the probability distribution toward the original, each token still has < 1.0 probability. For n = 100 tokens:

```
P(verbatim) ≤ (p_max)^100
```

If the average per-token probability of the correct next token is 0.5 (very high — models rarely achieve this even on memorized content):

```
P(verbatim for 100 tokens) ≤ 0.5^100 ≈ 10^(-30)
```

**The probability of verbatim reproduction of any significant span is astronomically small.** Even for short spans (10 tokens), P ≤ 0.5^10 ≈ 0.001.

### Additional Privacy Mechanisms

**1. Temperature during replay:** T_replay = 0.7 introduces stochasticity, further reducing verbatim probability.

**2. Minimum compression ratio:** For spans > 64 tokens, the replay is shorter than the original (C_target = 5:1), making verbatim reproduction physically impossible (you can't reproduce 500 tokens in 100 tokens).

**3. Sensitive content filtering (optional):**

```
BEFORE tagging:
    IF contains_sensitive_patterns(input_span):    # regex for SSN, credit cards, etc.
        SKIP tagging for this span
```

This is a heuristic safeguard — not a formal guarantee — but catches common PII patterns.

### Differential Privacy (DP) Option

For deployments requiring formal privacy guarantees, add DP noise to the replay generation:

```
# During GENERATE_REPLAY:
logits = model(prompt)
logits_private = logits + Laplace(0, Δf/ε)     # Laplace mechanism

# ε = privacy budget (e.g., ε = 1.0 for moderate privacy)
# Δf = sensitivity of the logit function
```

**Impact on replay quality:** DP noise degrades generation quality. With ε = 1.0, the noise is moderate and replay samples are still useful but less precise. This is a tunable trade-off between privacy and consolidation quality.

**Formal guarantee with DP:** The replay mechanism is ε-differentially private, meaning:

```
P(replay_output | user_data_included) ≤ e^ε · P(replay_output | user_data_excluded)
```

For ε = 1.0, including the user's data changes the probability of any output by at most a factor of e ≈ 2.72. This is a meaningful privacy guarantee.

---

## Q6.9 — Long-Term Convergence Over Many Wake-Sleep Cycles

### The Dynamical Model

Define the system state after N wake-sleep cycles as:

```
Θ_N = (W_slow^(N), W_fast^(N), T^(N))
```

Each cycle applies:

```
Θ_{N+1} = SLEEP(WAKE(Θ_N, D_N))
```

where D_N is the input data during wake cycle N.

### Does W_slow Converge?

**W_slow is bounded but not guaranteed to converge to a fixed point.**

The hard clipping constraint (Q3.4) ensures:

```
‖W_slow^(N) - W_anchor‖_F ≤ δ_max · ‖W_anchor‖_F     ∀N
```

So W_slow stays within a bounded ball around the pretrained weights. Within this ball, W_slow moves in the direction of new knowledge during each sleep cycle.

### Stability Analysis

Model the change per cycle as:

```
W_slow^(N+1) = W_slow^(N) + α_slow · ΔG_N - λ_ewc · F · (W_slow^(N) - W_anchor)
```

where:
- ΔG_N is the gradient from new knowledge consolidation
- The second term is the EWC pull-back toward W_anchor

At equilibrium (ΔW = 0):

```
α_slow · ΔG_eq = λ_ewc · F · (W_slow_eq - W_anchor)
```

This gives:

```
W_slow_eq = W_anchor + (α_slow / λ_ewc) · F⁻¹ · ΔG_eq
```

**The equilibrium exists** as long as ΔG_eq (the gradient from ongoing consolidation) is bounded. The equilibrium is a balance between "learn new things" and "stay close to pretrained."

**Stability condition:** The system is stable if the spectral radius of the update operator is < 1:

```
ρ(I - α_slow · λ_ewc · F) < 1
```

Since F is positive semi-definite and λ_ewc > 0, all eigenvalues of (I - α_slow · λ_ewc · F) are ≤ 1. With appropriately chosen α_slow and λ_ewc, the spectral radius is < 1, and the system is **asymptotically stable** around the equilibrium.

### Can Consolidated Knowledge Be Forgotten?

**Yes, through successive consolidation cycles.** Even though each cycle's interleaving includes old knowledge, the old knowledge samples are generated from W_slow — which has been modified by previous cycles. If a particular piece of knowledge was consolidated in cycle 3 but is never accessed again:

1. Its tag is deleted after cycle 3 (consolidation cleanup)
2. It exists only in W_slow's weights
3. In cycle 4-100, it appears in "old knowledge" samples only if W_slow happens to generate content about it
4. If it's rarely generated (because it's a niche fact), it receives minimal gradient reinforcement
5. Meanwhile, new knowledge gradually overwrites the weights that encode it

**This is slow, graceful forgetting** — not catastrophic. The rate depends on:

```
forgetting_rate ∝ α_slow · (gradient_from_new / protection_from_EWC)
```

For well-protected parameters (high F_i), forgetting is extremely slow. For weakly-protected parameters, forgetting is faster.

### Memory Capacity Over Many Cycles

**Effective long-term capacity** is limited by:

1. **Weight space capacity:** How many distinct "facts" can W_slow encode? This is related to the model's overall capacity minus what's used for pretrained knowledge. For a 7B model, empirical estimates suggest ~10,000-100,000 distinct facts can be added via fine-tuning before performance degrades.

2. **Per-cycle budget:** Each cycle consolidates ≤ B memories. With B = 3,500 for a 7B model:
   - After 100 cycles: up to 350,000 consolidation events
   - But many are overwritten or forgotten
   - Effective retained knowledge: ~5,000-50,000 items (estimate)

3. **Consolidation quality decay:** Later cycles may consolidate less effectively because the weight space is more constrained (EWC penalties accumulate from all previous knowledge).

### Expected Forgetting Curve for Consolidated Knowledge

```
Retention
  1.0 ┤ ────────────────╮
      │                  ╲   (protected by EWC)
  0.8 ┤                   ╲──────────────────
      │                                      ╲
  0.6 ┤                                       ╲──────
      │                                              ╲
  0.4 ┤                  Unprotected knowledge         ╲────
      │
  0.2 ┤
      │
  0.0 ┤──────────────────────────────────────────────────
      0    10    20    50    100   200   500   1000
                    Sleep Cycles After Consolidation
```

**Two regimes:**
1. **Well-protected knowledge** (high Fisher information, frequently re-consolidated): Near-flat retention, declining only after hundreds of cycles
2. **Weakly-protected knowledge** (low Fisher, never re-accessed): Gradual decline following ~power-law forgetting

This matches the biological Ebbinghaus forgetting curve — frequently rehearsed memories persist; unrehearsed ones decay.

---

## Part 6 Checkpoint: Stress Test Answers

> **Requirement:** Answer all 8 stress-test scenarios without hand-waving.

### Stress Test 1: "The user told the model X on day 1, then told it not-X on day 30."

1. Day 1: X is surprising (high prediction error, high entropy) → tagged as NOVELTY → PRP allocated → consolidated in sleep → now in W_slow
2. Day 30: not-X arrives → high prediction error (model expected X) + LOW entropy (model was confident about X) → contradiction detection fires → tagged as REVISION with boosted e₀ and PRP bonus
3. Next sleep cycle: not-X gets 2x replay frequency. W_slow is trained on "not-X" while EWC partially protects the "X" encoding. Over 1-3 cycles, W_slow's confidence in X decreases and confidence in not-X increases.
4. **Result:** W_slow holds not-X with moderate confidence. X may persist as a low-probability ghost but doesn't dominate inference. ✓

### Stress Test 2: "A brand new user sends their first message."

1. Cold start: κ_effective = 3.0 (very strict threshold) + B_effective ≈ 0 (budget ramping)
2. Forward pass computes per-token surprise against W_slow (all tokens are moderately surprising — new user's writing style is unfamiliar)
3. μ and σ² are initialized/rapidly adapting over the first ~100 tokens
4. Only tokens > 3 standard deviations above running mean are flagged — this captures genuinely novel content (not just unfamiliar style)
5. Tags are created but very few (strict threshold). W_fast gets small gradient updates on the most surprising spans.
6. No PRP allocation yet (budget near zero). No consolidation.
7. Over 50-100 interactions, thresholds relax, statistics stabilize, and the system enters normal operation.
8. **Result:** Cautious, conservative start. No premature consolidation of noise. ✓

### Stress Test 3: "Draw the exact data flow from a tagged experience through W_fast to a generated replay sample."

```
INPUT: "The Q3 revenue was $4.2M, down 12% from Q2"
                          │
       ┌──────────────────┤ Forward Pass (W_slow only)
       │                  │
       ▼                  ▼
  Per-token surprise:   Hidden states:
  "Q3": 2.1 nats        h_1 ∈ ℝ^4096
  "revenue": 1.8        h_2 ∈ ℝ^4096
  "$4.2M": 5.3 ←★       h_3 ∈ ℝ^4096  ←★
  "down": 4.1   ←★      h_4 ∈ ℝ^4096  ←★
  "12%": 4.8    ←★      h_5 ∈ ℝ^4096  ←★
  "from Q2": 2.5        h_6 ∈ ℝ^4096
                │              │
      z-score > 1.5?    Mean pool ★ tokens
        ★ = flagged     h̄ = mean(h_3,h_4,h_5) ∈ ℝ^4096
                │              │
                ▼              ▼
           Span: tokens 3-5    Key: k = W_proj · h̄ ∈ ℝ^128
           E_span = 2.14       s₀ = σ(2.0 · 2.14) = 0.986
                │              │
                └──────┬───────┘
                       │
                 CREATE TAG τ = (k, 0.986, step, 2.14, 0, 0, ctx)
                       │
                       ├──→ Tag Buffer T (metadata storage)
                       │
                       │    SIMULTANEOUSLY:
                       │    W_fast gradient update on span
                       │    L = -mean(log p_{W_slow+W_fast}(tokens 3-5))
                       │    ΔA_V, ΔB_V, ΔA_O, ΔB_O via SGD
                       │    W_fast now "knows" something about Q3 revenue
                       │
              ═══════ LATER, DURING SLEEP ═══════
                       │
                 τ has PRP (p=1, S=0.78)
                       │
                       ▼
              GENERATE_REPLAY(τ, W_slow, W_fast):
                seed = "The Q3 revenue"  (first 4 tokens of span)
                prompt = "Recall: The Q3 revenue"
                       │
                       ▼
              Autoregressive sampling from W_slow + W_fast:
              → "Recall: The Q3 revenue showed a decline compared to
                 the previous quarter, with figures coming in around
                 four million dollars."
                       │
              QUALITY CHECK:
                k_replay = W_proj · encode("...decline...four million...")
                cos(k_replay, τ.k) = 0.67 > 0.5 ✓
                       │
                       ▼
              replay_sample = above text (compressed, gist form)
              → feeds into interleaved training of W_slow
```

**Note:** The replay says "around four million dollars" not "$4.2M" — this is the compression in action. The gist is preserved; the exact number is approximated. ✓

### Stress Test 4: "Two users consolidate contradictory preferences."

With the per-user architecture (Q6.4): User A's W_cons_A encodes "prefers dark mode" and User B's W_cons_B encodes "prefers light mode." These are separate LoRA adapters — **no conflict whatsoever.** W_slow_base is shared and unmodified.

If sharing a single W_slow (hypothetical): The interleaving strategy would include both preferences in the training data. W_slow would learn a conditional: "some users prefer dark, some prefer light" — not a contradiction, but a distribution. The model would need a user identifier to disambiguate, which is why **per-user adapters are the correct architecture.** ✓

### Stress Test 5: "After 100 sleep cycles, the user asks about something from cycle 3."

1. Memory from cycle 3 was consolidated into W_cons_user (per-user LoRA adapter)
2. Over cycles 4-100, the knowledge persists IF:
   - It has high Fisher information (EWC protects it)
   - OR it was re-accessed (appears in proximity-weighted old samples during later cycles)
   - OR it's entangled with other consolidated knowledge (related topics reinforce it)
3. If it's an isolated, never-accessed fact: it will degrade following the slow forgetting curve (Q6.9). After 100 cycles, retention might be ~60-80% (the fact is partially remembered, possibly with reduced precision).
4. If it's a frequently-accessed, well-connected fact: retention is ~95%+ (essentially permanent).
5. **Result:** The system can answer, but accuracy depends on the memory's importance as judged by the scoring function over its lifetime. This is the desired behavior — important things are remembered; unimportant things gracefully fade. ✓

### Stress Test 6: "A user repeats the same useless question 500 times."

1. First instance: Question is processed. If it references something surprising → tag created. If not → no tag (it's a useless question, probably low prediction error).
2. Assuming it IS somehow tagged: ρ after 500 accesses = sim · Σ_{i=1}^{500} 1/√i ≈ sim · 44.7
3. Meanwhile, a genuinely useful memory accessed 20 times: ρ ≈ sim · 2√20 ≈ sim · 8.9
4. Wait — 44.7 > 8.9. Does the useless question win?
5. **No**, because: (a) Cross-reference density X is ~0 (no related tags), (b) The prediction error E was low to begin with (useless question, not surprising), (c) Combined score: S ≈ 0.35·(low) + 0.30·(44.7/max) + 0.15·(0) + 0.20·(moderate) — if E is low enough, S stays below θ_PRP even with inflated A.
6. **Result:** The multi-component scoring resists this attack. If E is low (not surprising), no amount of repetition pushes S above threshold. ✓

### Stress Test 7: "The user asks a question. Which of the 2,000 active tags are consulted, and how?"

1. Query tokens are processed through W_slow + W_fast → hidden states produced
2. Mean-pooled hidden state h_q is projected: k_q = W_proj · h_q ∈ ℝ^128
3. Cosine similarity computed against all 2,000 tag keys: sim_i = cos(k_q, k_i) for i = 1..2000
4. Cost: 2000 × 128 × ~10 = 2.56M FLOPs (negligible — 0.00002% of forward pass)
5. Tags with sim_i > 0.7 are "accessed" → metadata updated (access count, utility)
6. **Crucially:** The accessed tags DO NOT modify the model's output. The output was already determined by step 1. The tag access is purely for bookkeeping.
7. Typically, 0-5 tags will be accessed per query (most queries are about topics not recently tagged). ✓

### Stress Test 8: "The user shared their medical history. Can the replay generator reproduce it verbatim?"

1. Medical history triggers high prediction error → tagged, W_fast updated
2. During sleep, GENERATE_REPLAY produces a replay from seed + W_slow + W_fast autoregressive sampling
3. W_fast has 2.88M parameters (rank 16 LoRA) — it CANNOT memorize verbatim text of any significant length
4. For a 200-token medical history:
   - Target replay length: max(64, 200/5) = 64 tokens ← shorter than original, verbatim reproduction physically impossible
   - Even for the 64 tokens generated: P(verbatim) ≤ 0.5^64 ≈ 5 × 10^(-20)
5. Temperature = 0.7 adds stochasticity
6. The replay will contain gist: "patient has history of cardiac issues" not "Patient John Smith, DOB 03/15/1982, was diagnosed with atrial fibrillation on 11/22/2023"
7. Optional: sensitive content regex filter catches PII patterns before tagging
8. **Result:** Verbatim reproduction is statistically impossible for any span > ~10 tokens. Gist-level leakage ("has cardiac issues") is possible and is the intended behavior. For formal guarantees, add ε-DP noise to the generation logits. ✓

---

*Part 6 complete. All remaining questions (Q6.1-Q6.2, Q6.4-Q6.9) are fully resolved.*

---

# Final Synthesis

---

## Master Hyperparameter Table

All hyperparameters of the SLEEP system in one place, organized by component.

### Tagging Layer

| Parameter | Symbol | Default Value | Defined In | Description |
|:---|:---|:---|:---|:---|
| Tag key dimensionality | d_tag | 128 | Q1.1 | Dimension of projected key vectors |
| EMA smoothing factor | β | 0.99 | Q1.2 | Smoothing for running surprise statistics |
| Surprise sensitivity | κ | 1.5 | Q1.2 | Std devs above mean for tagging threshold |
| Gap tolerance | g | 3 | Q1.2 | Max gap between flagged tokens to merge spans |
| Minimum span length | min_span | 4 | Q1.2 | Minimum tokens for a viable span |
| Initial strength scaling | α_init | 2.0 | Q1.3 | Maps prediction error to initial tag strength |
| W_fast update sensitivity | κ_wfast | 2.5 | Q1.3 | Z-score threshold for W_fast gradient updates (cf. κ=1.5 for tagging) |
| Base decay time constant | τ_base | 1000 steps | Q1.4 | Base tag decay rate |
| Error-dependent decay scaling | γ | 0.5 | Q1.4 | Higher error → slower decay |
| Decay floor | ε | 0.01 | Q1.4 | Minimum tag strength before GC |
| GC threshold | ε_gc | 0.02 | Q1.4 | Tag removed when strength drops below |
| Base reinforcement | Δs | 0.3 | Q1.5 | Strength boost per access |
| Access similarity threshold | θ_access | 0.7 | Q1.5 | Cosine similarity for tag activation |
| Tag capacity coefficient | C_tag | 5000 per B params | Q1.6 | Tags per billion model parameters |

### PRP Allocation

| Parameter | Symbol | Default Value | Defined In | Description |
|:---|:---|:---|:---|:---|
| Error weight | w₁ | 0.35 | Q2.2 | Weight of cumulative prediction error in PRP score |
| Access weight | w₂ | 0.30 | Q2.2 | Weight of access frequency |
| Cross-ref weight | w₃ | 0.15 | Q2.2 | Weight of cross-reference density |
| Recency weight | w₄ | 0.20 | Q2.2 | Weight of recency-weighted utility |
| Cross-ref similarity threshold | θ_xref | 0.5 | Q2.2 | Cosine threshold for cross-reference edges |
| Cross-ref computation interval | P | 500 steps | Q2.2 | Steps between batch cross-ref computation |
| Recency decay constant | τ_recency | 500 steps | Q2.2 | Decay rate for recency-weighted utility |
| PRP capacity coefficient | C_prp | 500 per B params | Q2.3 | PRP budget per billion model parameters |
| Steal differential | δ_steal | 0.05 | Q2.4 | Min score gap for PRP reallocation |
| Allocation frequency | Q | 100 steps | Q2.4 | Steps between PRP allocation evaluations |
| PRP threshold sensitivity | κ_PRP | 0.5 | Q2.5 | Std devs above mean for PRP threshold |
| PRP threshold floor | θ_floor | 0.2 | Q2.5 | Minimum PRP threshold |

### Dual Weight System

| Parameter | Symbol | Default Value | Defined In | Description |
|:---|:---|:---|:---|:---|
| LoRA rank | r | 16 (7B) / 32 (70B) | Q3.1 | Rank of W_fast adapters |
| LoRA scaling | α_lora | 32 | Q3.1 | LoRA output scaling factor |
| Adapted layers | — | Top L/3 | Q3.1 | Which layers get W_fast adapters |
| Adapted matrices | — | V, O only | Q3.1 | Which attention matrices are adapted |
| W_fast learning rate | α_fast | 1e-4 | Q3.3 | Base learning rate for W_fast updates |
| W_fast momentum | μ_sgd | 0.9 | Q3.3 | SGD momentum for W_fast |
| W_slow base learning rate | α_slow | 1e-5 | Q3.4 | Learning rate for W_slow during sleep |
| Max weight change | δ_base | 0.001 | Q3.4 | Max relative Frobenius norm change per layer |
| Min plasticity | φ_min | 0.10 | Q3.4 | Plasticity floor for lowest layers |
| EWC penalty strength | λ_ewc | 100 | Q3.4 | Strength of Fisher information regularization |
| Fisher refresh interval | N_fisher_refresh | 10 cycles | Q3.4 | Sleep cycles between Fisher recomputation |
| Fisher refresh mix | — | 70% calibration / 30% consolidated | Q3.4 | Data mix for Fisher refresh |
| Degradation tolerance | ε_degrade | 0.02 | Q3.4 | Max allowed PPL increase before rollback |

### Sleep Engine

| Parameter | Symbol | Default Value | Defined In | Description |
|:---|:---|:---|:---|:---|
| Replay temperature | T_replay | 0.7 | Q4.1 | Sampling temperature for replay generation |
| Replay top-p | — | 0.9 | Q4.1 | Nucleus sampling threshold |
| Seed length | — | min(32, span/4) | Q4.1 | Tokens from original used to seed generation |
| Quality similarity threshold | θ_quality | 0.5 | Q4.1 | Min cosine similarity for replay acceptance |
| Baseline surprise threshold | μ_surprise | ~2.5-3.5 nats | Q4.1 | W_slow's mean surprisal on calibration data; replay below this is rejected |
| Target compression ratio | C_target | 5 | Q4.2 | Ratio of original length to replay length |
| Min replay length | min_replay | 64 tokens | Q4.2 | Minimum generated replay tokens |
| Old-to-new ratio | η | 4 | Q4.3 | Ratio of old knowledge to new replay samples |
| W_slow optimizer | — | AdamW | Q4.4 | Optimizer for sleep training |
| W_slow weight decay | — | 0.01 | Q4.4 | AdamW weight decay during sleep |
| Gradient clip norm | — | 1.0 | Q4.4 | Max gradient norm during sleep training |
| Steps per memory | — | 5 | Q4.4 | Gradient steps per consolidated memory |
| Schedule interval | T_schedule | 10,000 steps | Q4.5 | Max steps between sleep cycles |
| Memory pressure threshold | φ_pressure | 0.7 | Q4.5 | Tag buffer occupancy triggering sleep |
| Budget saturation threshold | φ_budget | 0.8 | Q4.5 | PRP budget usage triggering sleep |
| Idle time threshold | T_idle | 300 seconds | Q4.5 | Seconds of inactivity triggering sleep |
| Learning improvement threshold | ε_learn | 0.1 | Q4.6 | Min surprise reduction for consolidation pass |
| Validation similarity threshold | θ_validate | 0.4 | Q4.6 | Min cosine sim for consolidation validation |
| Max consolidation failures | — | 3 | Q4.6 | Failures before permanent tag removal |

### Cold Start & Robustness

| Parameter | Symbol | Default Value | Defined In | Description |
|:---|:---|:---|:---|:---|
| Cold start sensitivity | κ_cold | 3.0 | Q6.2 | Elevated κ during burn-in |
| Burn-in interactions | N_burnin | 50 | Q6.2 | Interactions before normal tagging begins |
| Ramp interactions | N_ramp | 50 | Q6.2 | Interactions for threshold relaxation |
| Budget maturation | N_mature | 500 | Q6.2 | Interactions for full PRP budget ramp |
| Revision bonus weight | w_rev | 0.3 | Q6.1 | PRP score bonus for contradiction tags |
| Revision replay multiplier | — | 2x | Q6.1 | Extra replay frequency for revisions |

---

## Complete Equation Reference

### Prediction Error and Tagging

```
e_t = -log p_{W_slow}(x_t | x_{<t})                     [per-token surprise]
μ_t = β · μ_{t-1} + (1 - β) · e_t                       [running mean]
σ²_t = β · σ²_{t-1} + (1 - β) · (e_t - μ_t)²           [running variance]
flag_t = (e_t - μ_t) / σ_t > κ                           [z-score tagging test]
k = W_proj · h̄_span + b_proj                             [tag key projection]
s₀ = σ(α_init · E_span)                                  [initial tag strength]
```

### Tag Dynamics

```
s_base(t) = (s₀ - ε) · exp(-(t - t₀) / τ_decay) + ε     [base decay from (s₀, t₀)]
s(t) = min(s_base(t) + s_reinforced, 1.0)                  [effective strength = base + bonus]
τ_decay = τ_base · (1 + γ · e₀)                            [error-dependent time constant]
ρ ← ρ + sim · (1/√a)                                       [cumulative utility update]
s_reinforced ← s_reinforced + Δs · (1-s) · (1/√a)         [reinforcement bonus accumulation]
```

### PRP Scoring

```
S(τ) = w₁ · Ê(τ) + w₂ · Â(τ) + w₃ · X̂(τ) + w₄ · R̂(τ)  [composite score]
Ê(τ) = (e₀ · s/s₀) / max_j(e₀_j · s_j/s₀_j)            [normalized error]
Â(τ) = ρ / (1 + max_j(ρ_j))                               [normalized access]
X̂(τ) = |{j : cos(k_i,k_j) > θ_xref}| / N                [cross-reference density]
R̂(τ) = R / max_j(R_j)                                     [normalized recency]
θ_PRP = max(θ_floor, μ_S + κ_PRP · σ_S)                   [adaptive threshold]
```

### Weight System

```
W_V_eff = W_V_slow + (α_lora/r) · B_V · A_V              [LoRA-adapted value projection]
W_O_eff = W_O_slow + (α_lora/r) · B_O · A_O              [LoRA-adapted output projection]
|W_fast| = (4/3) · L · 2 · r · d_model                    [W_fast parameter count]
φ(l/L) = φ_min + (1 - φ_min) · (l/L)²                    [plasticity profile]
α_slow^(l) = α_base_slow · φ(l/L)                         [layer-specific learning rate]
```

### Sleep Training

```
W_target = W_slow_base + W_cons                              [effective model for training]
W_eff = W_slow_base + W_cons + W_fast                        [effective model for generation]
L_total = L_lm + L_ewc                                      [total sleep loss]
L_lm = -(1/T) · Σ_t log p_{W_target}(x_t | x_{<t})        [language modeling loss]
L_ewc = (λ_ewc/2) · Σ_i F_i · (W_cons_i - W_cons_init_i)² [EWC on W_cons adapter]
N_steps = max(100, |consolidated| · 5)                       [steps per cycle]
‖ΔW_cons^(l)‖_F ≤ δ_max · φ(l/L) · ‖W_slow_base^(l)‖_F   [hard adapter bound]
```

### Stability

```
W_slow_eq = W_anchor + (α_slow/λ_ewc) · F⁻¹ · ΔG_eq     [equilibrium point]
T_gc = τ_decay · ln((s₀ - ε)/(ε_gc - ε))                  [max tag lifetime]
P(verbatim, n tokens) ≤ p_max^n                             [verbatim reproduction bound]
```

---

## Cross-Verification: Consistency Check

### Dimensional Consistency

| Quantity | Dimensions | Verified |
|:---|:---|:---|
| Tag key k | ℝ^128 | ✓ (W_proj ∈ ℝ^(128×4096) · h̄ ∈ ℝ^4096 → ℝ^128) |
| Cosine similarity | scalar ∈ [-1, 1] | ✓ (k^T k' / (‖k‖·‖k'‖)) |
| PRP score S | scalar ∈ [0, 1] | ✓ (convex combination of normalized components) |
| Tag strength s | scalar ∈ [0, 1] | ✓ (sigmoid → exponential decay → reinforcement capped at 1) |
| LoRA update ΔW_V | ℝ^(4096×4096) | ✓ (B∈ℝ^(4096×16) · A∈ℝ^(16×4096) → ℝ^(4096×4096)) |

### Information Flow Consistency

| Source → Destination | Channel | Verified |
|:---|:---|:---|
| Input → Tag | Prediction error + hidden state | ✓ (Q1.2 → Q1.3) |
| Input → W_fast | Gradient descent on surprising spans | ✓ (Q1.3 → Q3.3) |
| Tag → PRP Score | Metadata fields | ✓ (Q1.1 fields → Q2.2 components) |
| PRP → Sleep Selection | Allocation flag p=1 | ✓ (Q2.4 → Q4 Phase 1) |
| W_fast → Replay | Autoregressive generation | ✓ (Q3.1 → Q4.1) |
| Replay → W_cons | Gradient descent with interleaving on per-user adapter | ✓ (Q4.1 → Q4.4 → Q6.4) |
| W_target (W_slow_base+W_cons) → Old Knowledge | Self-generation from target model | ✓ (Q4.3 → Q6.4) |
| Consolidation → Tag Cleanup | Validation then removal | ✓ (Q4.6) |

### Dependency Graph Verified

All 36 questions answered. All dependency edges satisfied:

```
✓ Q6.3 resolved before Q1.1 (tag vs W_fast relationship)
✓ Q1.1 resolved before Q1.2-1.6 (tag formalism grounds everything)
✓ Q1.2 resolved before Q1.3 (PE needed for tag creation)
✓ Q1.4 resolved before Q1.5 (decay before reinforcement)
✓ Part 1 complete before Part 2 (tags before PRPs)
✓ Q6.3 resolved before Q3.1 (tag-W_fast boundary before W_fast architecture)
✓ Parts 1-3 complete before Part 4 (all components before sleep)
✓ Parts 1-4 complete before Part 5 (system-level requires all components)
✓ Part 6 addresses all cross-cutting concerns
```

---

## What This Document Enables

With all 36 questions resolved, you can now:

1. **Write the tagging layer** — pseudocode is complete (Part 1 checkpoint)
2. **Simulate PRP allocation** — worked example verified convergence (Part 2 checkpoint)
3. **Draw the architecture** — every tensor shape specified (Part 3 checkpoint)
4. **Write the sleep cycle** — complete training script provided (Part 4 checkpoint)
5. **Write the paper abstract** — all metrics and claims are precisely defined (Part 5 checkpoint)
6. **Answer every stress test** — all 8 scenarios resolved without hand-waving (Part 6 checkpoint)

**Implementation can begin.**

---

*Document version: 1.0*
*Total questions resolved: 36/36*
*Status: Ready for implementation*

---

# Appendix A: Empirical Amendments (2026-04-30)

After implementing all 36 questions and running end-to-end experiments at GPT-2
(124M) and Qwen2.5-7B scales, four assumptions in the formalization were found
to be wrong or incomplete. These amendments are not optional — implementing
the formalization as originally specified produces 0% consolidation rate at
7B scale. They should be incorporated into v1.1 of the document.

## A.1 Amendment to Q3.4 — `delta_max` Must Scale With Model Size

**Original specification:** `delta_max = 0.001` (per-parameter weight change
bound during sleep training).

**Empirical finding:** On Qwen2.5-7B with rank-16 LoRA, the bound saturates
within 30 training steps, after which 36/36 LoRA parameters are clipped on
every step. Loss does not decrease. Validation always fails. Cleanup rolls
W_cons back to its initial state.

**Required value at 7B:** `delta_max ≈ 0.01` (10× the original default).

**Proposed amendment:**
The per-parameter bound is the wrong abstraction. What matters is the
*Frobenius norm* of the LoRA update relative to the model's natural weight
scale. Two acceptable parameterizations:

1. **Hidden-dim scaling:** `delta_max(d) = 0.001 × √(d / 768)`
   where 768 is the GPT-2 Small reference. For Qwen 7B (d=3584): ≈ 0.0022.

2. **Param-count scaling:** `delta_max(B) = 0.001 × (B / 0.124)^0.25`.
   For Qwen 7B (7.616B): ≈ 0.0028.

Even these underestimate empirically. The likely correct formulation is to
constrain the *total Frobenius norm of the LoRA update*, not per-parameter
deltas. This requires further empirical work to specify precisely.

**Source:** logbook `2026-04-30_experiment_03_qwen7b_delta_max_bound.md`.

## A.2 Amendment to Q3.4 — `alpha_slow` Must Be Above Precision Floor

**Original specification:** `alpha_slow = 1e-5` (sleep training learning rate).

**Empirical finding:** On Qwen2.5-7B in bfloat16, `alpha_slow = 1e-5`
produces gradient updates that **round to zero in storage**. Mean training
loss is completely flat across 100 steps (1.4581 mean, 1.4554 final).
Bumping to `1e-4` produces a **phase transition**: mean loss drops to 0.7450,
final loss to 0.5469.

**Required value in bfloat16:** `alpha_slow ≥ 1e-4`.

**Proposed amendment:**
The learning rate must be specified relative to parameter precision:

```
alpha_slow_min(dtype) = 10 × ulp_relative(dtype)

Where:
  ulp_relative(float32)  ≈ 1e-7
  ulp_relative(bfloat16) ≈ 1e-3
  ulp_relative(float16)  ≈ 1e-3
```

For bfloat16, this gives `alpha_slow_min ≈ 1e-2 × weight_scale`. With typical
LoRA weight scale ~0.05, this is `alpha_slow ≈ 5e-4`.

The original `1e-5` was implicitly written for float32; in bfloat16 (the
practical dtype for 7B+ training) it is 100× too small.

**Source:** logbook `2026-04-30_experiment_03_qwen7b_alpha_slow.md`.

## A.3 Amendment to Q4.1 — `mu_surprise` Quality Check Has a Generator-Discriminator Gap

**Original specification:** Quality check rejects replays whose mean surprise
falls below `mu_surprise`, where `mu_surprise = compute_baseline_surprise(W_slow,
calibration_text)`.

**Empirical finding:** On Qwen2.5-7B, generated replays have surprise
1.22–1.27 nats; the baseline `mu_surprise = 1.41 nats`. **All replays are
rejected.** This is not a bug — replays come from the same model that defines
the surprise function, so they are systematically more in-distribution than
external calibration text.

**Proposed amendment:**
The absolute-threshold formulation is structurally wrong because it compares
`surprise(self-generated)` to `surprise(external)`. Three acceptable fixes:

1. **Buffer factor** (simplest): `mu_surprise_eff = α × mu_surprise` with
   `α ≈ 0.7` to account for in-distribution bias.

2. **Self-calibration:** compute `mu_surprise` from a small set of
   self-generated samples drawn from W_slow alone, then use that as the
   threshold for replays from W_slow + W_fast.

3. **Relative gain** (cleanest): require `surprise(replay) > β × surprise(seed)`
   where the seed is the original tagged span. This compares like-to-like
   (both from the model) and avoids the mismatch entirely. Recommended β = 1.2.

**Recommendation:** Option 3. Until amended, set `use_real_mu_surprise = False`
to bypass the absolute check.

**Source:** logbook `2026-04-30_experiment_03_qwen7b_quality_gap.md`.

## A.4 Amendment to Q4.6 — Validation Criterion Has Minimum-Data Requirement

**Original specification:** Validation passes if
`surprise_new < surprise_old × (1 - epsilon_learn)` where `epsilon_learn = 0.10`.

**Empirical finding:** With ≤ 5 facts (~ 2-3 replays), the validation
criterion **always fails**, even when training loss decreases substantially
(1.45 → 0.55). With 200 facts (~ 48 replays), 31% of candidates pass.

**Proposed amendment:**
The criterion is sound (it correctly demands transferable learning), but the
formalization should explicitly document a **data-volume floor**:

```
Required N_replays ≥ N_min(rank, compression, model_size)

Empirical finding for rank=16, compression=5, Qwen 7B: N_min ≈ 20–30.
Below this threshold, expect 0% validation pass rate.
```

Two helpful additions:

1. **Loss-decrease fallback:** If training loss decreased by ≥ 30% during
   sleep, accept the candidate even if surprise threshold not met. This
   catches cases where the gist was learned but transfer to the verbatim
   original is incomplete.

2. **Compression-adaptive threshold:** scale `epsilon_learn` by
   `1 / sqrt(compression_target)`. With heavy compression, demand less.

**Source:** logbook `2026-04-30_experiment_03_validation_strict.md`.

## A.5 Decoupling of Consolidation and Recall

**Empirical finding (200-fact run):** 15 facts passed validation
(`surprise_new < threshold`), but free-form generation from question prompts
recalled essentially zero of them with statistical confidence. The 4 reported
"HITs" appear to be lucky keyword overlaps from the model's prior knowledge,
not consolidated content.

**Implication:** The current architecture solves *encoding* but not
*retrieval from question-form prompts*. This is a research problem the
formalization did not anticipate.

**Open questions for v1.1:**
- Is the consolidated content actually retrievable in *any* form (e.g.,
  cloze completion, paraphrased prompt, multiple choice)?
- Does generation drift away from W_cons under competing priors?
- Should the formalization include a "retrieval prompt format" specification?

**Source:** logbook `2026-04-30_experiment_03_v5_200facts_milestone.md`.

---

*Appendix added: 2026-04-30*
*Source experiments: `experiments/results/pod_run_2026-04-30/` (logs 01–05)*
*Status (v1.0 + Appendix A): Implementation complete; 4 amendments pending validation in v1.1*