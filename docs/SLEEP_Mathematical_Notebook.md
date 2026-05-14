# SLEEP: Pre-Implementation Mathematical Notebook

## How to Use This Document

This is your pre-implementation decision ledger. Every question here must have a written answer — a chosen formalism, with justification — before you write a single line of implementation code. The questions are ordered by dependency: later sections depend on decisions made in earlier ones. Don't skip ahead.

For each question, your notebook entry should contain:
1. The candidate formalisms you considered
2. The one you chose
3. Why (mathematical argument, not intuition)
4. What it implies for downstream choices

**Total: 36 questions across 6 parts. All must be resolved before implementation.**

---

## Part 1: The Tagging Layer

These are the most foundational decisions. Everything else inherits from them.

### 1.1 — What mathematical object is a "tag"?

A tag must be something that is cheap to create, decays over time, and can later be used to reconstruct or reference the original experience. You need to decide what space tags live in.

Candidates to evaluate:
- Sparse binary mask over model parameters
- Sparse vector in activation/embedding space (an episodic trace)
- A set of key-value pairs injected into attention
- A compressed gradient snapshot
- Something else entirely

**What to write down:** The formal definition. If a tag is τ, what is its type? What is its dimensionality? What space does it live in? What information does it encode?

### 1.2 — How is prediction error computed?

The proposal says "the gap between what the model expected and what it actually received." This needs to be precise.

Questions to resolve:
- Is prediction error computed per-token, per-span, per-document, or per-semantic-unit?
- Is it raw cross-entropy loss, KL divergence between predicted and observed distributions, or something else?
- Is it computed against W_slow only, or against the combined system (W_slow + active tags)?
- How do you handle the granularity problem: a document may have 10,000 tokens but only 50 tokens are genuinely surprising — how do you segment and attribute error?

**What to write down:** The exact loss function. The unit of computation (what constitutes one "experience" that gets one tag). The thresholding or selection mechanism that decides "high error" vs. "low error."

### 1.3 — What is the tag creation function?

Given an input x and a prediction error signal e, how exactly is a tag τ produced?

Questions to resolve:
- Is it deterministic or stochastic?
- What are its inputs? Just the error signal? The input itself? The hidden states at the point of error? The gradient?
- What is the computational cost? You claim tags are "cheap" — quantify this. What's the cost relative to a forward pass?
- Is one tag created per experience, or can multiple tags be created from a single input?

**What to write down:** τ = f(·). Define f completely.

### 1.4 — What is the tag decay function?

Tags decay over time unless reinforced. This needs a mathematical form.

Questions to resolve:
- Is decay exponential, linear, hyperbolic, or something else?
- What is the time variable? Wall-clock time? Number of inference steps? Number of experiences processed?
- Is the decay rate the same for all tags, or does it depend on initial prediction error magnitude?
- What does "fully decayed" mean? Is the tag deleted, or does it asymptotically approach zero?
- Is decay continuous or discrete (evaluated at checkpoints)?

**What to write down:** τ(t) = g(τ₀, t, ...). The full decay equation with all parameters identified.

### 1.5 — How does tag access/reinforcement work?

When a tag is "accessed" (the tagged information proves relevant to a query), it gets reinforced. This is central to the PRP allocation logic.

Questions to resolve:
- What constitutes an "access"? The user asks a question and the tagged information is retrieved/relevant? How do you detect this?
- Does access reset the decay clock, slow the decay rate, or add a fixed amount back to the tag strength?
- Is there a formal relationship between access and tag strength? Additive? Multiplicative?
- Can a tag be strengthened beyond its initial value through repeated access?

**What to write down:** The update rule for τ when accessed. The detection criterion for "this tag was accessed."

### 1.6 — What is the tag capacity?

How many tags can the system hold simultaneously?

Questions to resolve:
- Is there a hard limit (fixed buffer size) or a soft limit (tags compete for a shared resource)?
- If hard: what is the eviction policy when the buffer is full?
- If soft: what is the resource they compete for, and how does competition work?
- How does tag capacity relate to model size? Should a 7B model have a different tag budget than a 70B model? Why?

**What to write down:** The capacity constraint, formally stated. The relationship between capacity and model scale.

---

## Part 2: The Protein Budget (PRP Allocation)

These decisions depend on Part 1. PRPs act on tags, so the tag formalism constrains what PRPs can be.

### 2.1 — What mathematical object is a "PRP"?

A PRP is the resource that stabilizes a tag and queues it for consolidation. But what is it computationally?

Candidates to evaluate:
- A scalar flag (binary: allocated or not)
- A continuous resource (each tag gets a PRP "amount" between 0 and 1)
- A learning rate multiplier that determines how strongly this memory will influence W_slow during consolidation
- A priority score in a consolidation queue
- An allocation of actual compute budget for the sleep phase

**What to write down:** The formal definition of a PRP. Its type, its range, what it controls.

### 2.2 — What is the composite scoring function?

Your diagrams show four inputs to PRP allocation: cumulative prediction error, access frequency, cross-reference density, and recency-weighted utility. These need to be combined.

Questions to resolve:
- What is the functional form? Linear weighted sum? Multiplicative? Something with interaction terms?
- Score = w₁·Error + w₂·Access + w₃·CrossRef + w₄·Recency — are these the right components? Are there others?
- How are the components normalized? They're on completely different scales.
- Are the weights (w₁, w₂, w₃, w₄) fixed hyperparameters, or learned, or adaptive?
- What is "cross-reference density" formally? How do you measure whether one tag "connects to" other tags?

**What to write down:** The scoring function S(τ) = ... with all terms defined. The normalization scheme. Whether weights are fixed or adaptive and why.

### 2.3 — What is the total PRP budget and how is it determined?

The budget is described as "fixed" but also needs to be set somehow.

Questions to resolve:
- Is the budget a count (max N memories can be PRP-allocated at once) or a continuous quantity (total PRP resource = B, distributed across memories)?
- How is the budget size determined? Proportional to model parameters? A fixed hyperparameter? Adaptive based on input rate?
- Does the budget replenish? If memories are consolidated (PRPs freed), does the budget refill, or is there a regeneration rate?

**What to write down:** The budget constraint. How it scales. The replenishment dynamics.

### 2.4 — How does competitive allocation work?

When the budget is full and a new high-scoring memory arrives, it can "steal" PRPs from lower-scoring ones.

Questions to resolve:
- Is this a simple priority queue (lowest-scoring PRP-holder gets evicted)?
- Is there a minimum score differential required for stealing? (To prevent thrashing)
- What happens to a memory that loses its PRP? Does it return to normal tag decay? Does it decay faster (penalty for demotion)?
- Is there hysteresis — does a memory that was PRP-allocated and then lost it behave differently from one that was never allocated?

**What to write down:** The reallocation algorithm. The stability conditions (under what circumstances does the allocation converge rather than oscillate).

### 2.5 — What is the PRP threshold?

The diagram shows "Score ≥ Threshold?" as a binary gate.

Questions to resolve:
- Is the threshold fixed or adaptive?
- If adaptive, what does it adapt to? Mean score of current tags? Budget pressure? Historical distribution?
- Should the threshold be different for different types of information (factual vs. procedural vs. preferential)?
- What is the relationship between the threshold and the budget? If the budget is large, should the threshold be lower?

**What to write down:** The threshold function θ = h(...). Its inputs and adaptation rule.

---

## Part 3: The Dual Weight System

These decisions are somewhat independent of Parts 1-2 but constrain Part 4 (Sleep).

### 3.1 — What is W_fast, architecturally?

W_fast needs to be a fast-learning, low-capacity system that stores recent experience indices. What is it in terms of actual neural network components?

Candidates to evaluate:
- A set of LoRA adapters (low-rank updates to specific layers)
- A separate small model entirely
- Additional key-value pairs in the attention mechanism (like a memory-augmented transformer)
- Sparse parameter perturbations on top of W_slow
- An external memory matrix with read/write heads
- A set of soft prompts / prefix tunings that are updated online

**What to write down:** The architectural specification of W_fast. Its parameter count relative to W_slow. Which layers of W_slow it interacts with and how.

### 3.2 — How do W_fast and W_slow interact during inference?

During the "wake phase," the model uses both weight systems to respond.

Questions to resolve:
- Is the output a simple sum? W_effective = W_slow + W_fast?
- Is W_fast gating W_slow's outputs? (multiplicative interaction)
- Do they operate at different layers? (W_fast modifies early layers, W_slow is the full model?)
- Is there an attention-based routing mechanism that decides when to consult W_fast vs. W_slow?
- How do you prevent W_fast from interfering with W_slow's base capabilities?

**What to write down:** The inference equation. How a forward pass works through the combined system.

### 3.3 — What are the learning rules for W_fast?

W_fast updates with every interaction. What's the update rule?

Questions to resolve:
- Is it gradient-based? If so, what loss function? The same prediction error used for tagging?
- Is the learning rate fixed or adaptive?
- Does W_fast have its own optimizer state, or is it a simpler update (e.g., Hebbian, direct write)?
- How do you prevent W_fast from overfitting to the most recent experience?

**What to write down:** The update equation for W_fast. The optimizer (if any). The learning rate schedule.

### 3.4 — What defines W_slow's "plasticity" constraints?

W_slow only changes during sleep. But during sleep, how much can it change?

Questions to resolve:
- Is there a maximum magnitude of weight change per consolidation cycle?
- Is the learning rate for W_slow during consolidation fixed or dependent on the PRP scores of the memories being consolidated?
- Are certain layers of W_slow more plastic than others? (Biologically, different cortical regions have different plasticity profiles.)
- How do you ensure W_slow doesn't change so much in one sleep cycle that it degrades base capabilities?

**What to write down:** The constraints on ΔW_slow per consolidation cycle. The learning rate or step size bounds.

---

## Part 4: The Sleep Engine

This depends on all previous parts. This is where you define how temporary knowledge becomes permanent.

### 4.1 — What is the generation model?

During sleep, W_fast generates synthetic training examples. How?

Questions to resolve:
- Is the generator W_fast itself (autoregressive sampling conditioned on tags)?
- Is there a separate generative model (VAE, diffusion model)?
- What is the input to generation? The tag alone? The tag + some context from W_slow?
- What is the output format? Full text sequences? (Input, Output) pairs? Embedding-level representations?
- How do you control generation quality? What if W_fast generates hallucinated or distorted memories?

**What to write down:** The generation process formally. P(x_replay | τ, W_fast, W_slow) = ...? The sampling procedure. Quality control mechanisms.

### 4.2 — What is the compression ratio and how is it controlled?

Generation "forces compression" — the replay is gist, not verbatim. But how much compression?

Questions to resolve:
- Is there a target compression ratio (e.g., 10:1 reduction from original experience to replay)?
- How do you measure whether compression has preserved the "right" information?
- Is compression an emergent property of generation (the model naturally compresses), or do you enforce it (e.g., by limiting generation length)?
- What is the information-theoretic framework? Can you formalize what "gist" means in terms of mutual information between the original experience and the replay?

**What to write down:** The compression objective. The metric for replay quality. The relationship between compression and information preservation.

### 4.3 — What is the interleaving strategy?

Generated replay examples are mixed with "old knowledge" samples to prevent catastrophic forgetting.

Questions to resolve:
- What is the ratio of new replay to old knowledge? Fixed or adaptive?
- Where do "old knowledge" samples come from? Generated from W_slow? A held-out buffer? The original pretraining distribution?
- How do you select which old knowledge to replay? Random? Weighted by proximity to the new knowledge (to specifically protect nearby representations)?
- Is there a curriculum — do you vary the ratio over the course of a single sleep cycle?

**What to write down:** The interleaving distribution P(batch). The sampling strategy for old vs. new. The mixing ratio and how it's determined.

### 4.4 — What is the training procedure during sleep?

W_slow is updated using standard gradient descent on the interleaved dataset. But the details matter enormously.

Questions to resolve:
- What loss function? Standard language modeling loss? A modified loss that weights new knowledge more heavily?
- Learning rate? Fixed? Warm-up? How does it compare to pretraining learning rate?
- How many steps per consolidation cycle? Is it fixed, or proportional to the number of memories being consolidated?
- Batch composition: how many replay samples per batch? How many old knowledge samples?
- Do you use any regularization (L2 toward original W_slow weights? EWC-style Fisher information penalties?)?

**What to write down:** The training loop specification. Loss function, optimizer, learning rate, steps, batch composition, regularization.

### 4.5 — What are the consolidation triggers?

When does the system enter sleep phase?

Questions to resolve:
- Scheduled intervals (every N interactions)?
- Memory pressure (when tag buffer is X% full)?
- Idle time (when the system isn't being queried)?
- A combination? If so, how are they weighted?
- Can sleep be interrupted? If a query arrives during sleep, what happens?
- Is sleep all-or-nothing, or can you have "micro-sleeps" (consolidate a few memories quickly)?

**What to write down:** The trigger conditions formally. The scheduling algorithm. Interruption handling.

### 4.6 — What is the post-consolidation cleanup?

After sleep, consolidated memories have their tags cleared.

Questions to resolve:
- How do you verify that consolidation was successful before clearing tags? What if W_slow didn't learn the memory well enough?
- Is clearing immediate or gradual? (Biology: there's a gradual handoff period.)
- What is the validation criterion? Run the original query and check W_slow can answer without W_fast?
- What happens if validation fails? Re-tag? Re-queue for next sleep cycle?

**What to write down:** The validation procedure. The clearing policy. The failure handling protocol.

---

## Part 5: System-Level Questions

These cut across all components and must be resolved before implementation.

### 5.1 — What is the formal definition of "memory" in this system?

You use "memory" throughout but it needs a precise definition.

Questions to resolve:
- Is a memory a single tag? A tagged experience? A cluster of related tags?
- What are the boundaries of one memory? If a user has a conversation with 50 turns, is that 1 memory, 50 memories, or something in between?
- How do memories relate to each other? Is there a graph structure? A hierarchy?
- Can memories merge during consolidation?

**What to write down:** The formal definition of a memory unit m. Its constituent parts. Its lifecycle states.

### 5.2 — What is the state space of the complete system?

At any point in time, the system has a state. Define it.

Questions to resolve:
- State = (W_slow, W_fast, T, P, phase) where T is the set of active tags, P is the PRP allocation, phase ∈ {wake, sleep}?
- What are the transition functions between states?
- Can you draw the complete state machine?
- What are the invariants that must always hold? (e.g., total PRPs ≤ budget)

**What to write down:** S = {...}. The state transition function δ(S, input) → S'. The invariant set.

### 5.3 — What are the scaling laws?

How does each component scale with model size?

Questions to resolve:
- Tag dimensionality as a function of model parameters
- PRP budget as a function of model parameters
- W_fast capacity as a function of W_slow size
- Sleep cycle duration as a function of memories to consolidate
- Computational overhead of the full system relative to base model inference

**What to write down:** The scaling relationships. At minimum, Big-O complexity for each operation (tagging, PRP scoring, generation, consolidation training).

### 5.4 — What are the evaluation metrics?

How do you know the system is working?

Questions to resolve:
- What is the primary metric? Accuracy on questions about previously-seen information? At what time delay?
- How do you measure "graceful forgetting"? What should the forgetting curve look like?
- How do you measure consolidation quality? Compare W_slow's performance on consolidated knowledge vs. RAG's performance on the same knowledge?
- How do you measure interference? Base model benchmarks before and after many consolidation cycles?
- How do you measure the efficiency of PRP allocation? Compare against oracle allocation (hindsight-optimal)?

**What to write down:** The evaluation suite. Each metric with its formal definition. The baselines for comparison.

### 5.5 — What are the theoretical guarantees (if any)?

Before implementing, know what you can and can't prove.

Questions to explore:
- Can you prove that the decay + PRP mechanism converges to a stable allocation?
- Can you bound the information loss during generative compression?
- Can you prove that interleaved replay prevents catastrophic forgetting under certain conditions?
- Can you characterize the system's memory capacity formally? (How many "memories" can it hold as a function of model size and PRP budget?)
- What are the failure modes? Under what conditions does the system provably break?

**What to write down:** Whatever you can prove. And clearly delineate what you're taking on faith vs. what you've established formally.

---

## Part 6: Missing Formalisms

These are questions that cut across the architecture and would otherwise fall through the cracks. Each one could quietly break the system if left unresolved.

### 6.1 — How does the system handle memory revision and unlearning?

Tag decay handles forgetting things you stop caring about. But what about things that are *wrong*? Once a memory is consolidated into W_slow, it's in the weights. The proposal has no mechanism for correcting consolidated knowledge.

Questions to resolve:
- What happens when new information directly contradicts something already in W_slow?
- Is contradiction detected automatically (e.g., high prediction error on information that's *similar to* existing knowledge, not just novel)? How do you distinguish "surprising because new" from "surprising because it contradicts what I know"?
- Does contradictory information get tagged differently from merely novel information?
- What is the update mechanism in W_slow for overwriting? Is it just another consolidation cycle that trains on the corrected version? Does that reliably overwrite the old memory, or does the old memory persist and interfere?
- Can you formalize a "memory revision" operation that is distinct from "memory formation"?

**What to write down:** The revision pathway. The detection criterion for contradiction vs. novelty. The formal difference (if any) between the consolidation of new knowledge and the overwriting of old knowledge.

### 6.2 — How does the system handle cold start?

When the system first deploys on a new user, W_slow has only pretraining knowledge. Everything the user says will have relatively high prediction error — the model has never seen this user's writing style, preferences, or domain. This means early tagging will be noisy and indiscriminate.

Questions to resolve:
- Does the prediction error threshold adapt over time? If so, what is the adaptation rule?
- Is there a formal burn-in period where the system is "learning to tag" before it starts consolidating?
- How does the PRP scoring function behave when there's no access history yet? (Access frequency is zero for everything, cross-reference density is meaningless with few tags.)
- Should the PRP budget be smaller during cold start (consolidate cautiously) or larger (learn aggressively)?
- How many interactions does the system need before its tagging precision reaches acceptable levels? Can you estimate this?

**What to write down:** The calibration procedure for the tagging threshold. The cold-start behavior of each component of the scoring function. The expected burn-in duration.

### 6.3 — What is the precise relationship between tags and W_fast?

This is an architectural ambiguity at the heart of the proposal. Tags mark "where surprising information lives." W_fast "stores temporary, specific indices" and "updates with every interaction." Are these the same system or two separate systems?

Questions to resolve:
- Does updating W_fast *create* the tags, or are tagging and W_fast learning two parallel processes?
- If separate: how do they interact? Does W_fast encode the context around a tag? Does it store the full experience that the tag merely points to?
- If the same: then what does "tag decay" mean for W_fast? Do parts of W_fast decay while others persist?
- When the sleep engine "uses the fast system to generate synthetic examples" (Step 2 of consolidation), is it generating from tags, from W_fast's weights, or from both? What is the generative mechanism precisely?

**What to write down:** A diagram with clear boundaries between the tag system and W_fast. The information flow between them. Whether they are one module or two, and the formal interface if two.

### 6.4 — Multi-user dynamics and W_slow sharing

The proposal implies personalization but doesn't specify the deployment model.

Questions to resolve:
- Is W_slow shared across users with per-user W_fast and tag systems? Or is the entire stack per-user?
- If shared W_slow: can consolidation from User A interfere with User B's consolidated knowledge? How do you isolate per-user consolidation within a shared weight space?
- If per-user W_slow: the cost is N copies of the full model. Is this feasible? Is there a middle ground (shared base + per-user adapters)?
- Does the math change depending on the deployment model? (It almost certainly does — shared W_slow means the interleaving strategy must account for multiple users' knowledge.)

**What to write down:** The deployment architecture. Whether W_slow is shared or per-user. The formal isolation guarantees if shared.

### 6.5 — Compositionality across sleep cycles

If memory A is consolidated in sleep cycle 1 and memory B in sleep cycle 2, can W_slow reason about their relationship? Or does each cycle operate in isolation?

Questions to resolve:
- Does the interleaving strategy in cycle 2 include replay from cycle 1's consolidated knowledge? (It should, via "old knowledge" samples from W_slow — but verify this handles relational reasoning, not just interference prevention.)
- Can the system build up structured, relational knowledge over many cycles? E.g., "User prefers X" (cycle 1) + "X conflicts with Y" (cycle 5) → "User should be warned about Y" (emergent in W_slow)?
- Is there a formal model of how knowledge compounds across cycles? Or does each cycle just add independent facts?
- What is the long-term knowledge structure? A flat set of consolidated memories, or something with emergent hierarchy?

**What to write down:** The cross-cycle interaction model. Whether knowledge composition is emergent or requires explicit mechanism. What W_slow "looks like" after 100 cycles.

### 6.6 — Adversarial robustness of the scoring function

The PRP scoring function determines what gets remembered. It can be gamed.

Questions to resolve:
- If a user asks the same irrelevant question 100 times, the access frequency signal will push it toward consolidation. How do you prevent this?
- Can repetition be distinguished from genuine importance? Formally, what separates "accessed frequently because useful" from "accessed frequently because repeated"?
- Can the prediction error signal be manipulated? (E.g., feeding the model deliberately surprising but useless information to consume the tag budget.)
- What is the formal robustness of the scoring function to adversarial input distributions?
- Should there be a "diminishing returns" term — repeated access of the same tag yields less and less reinforcement?

**What to write down:** The adversarial threat model. The robustness properties of each scoring component. Mitigations (e.g., diminishing returns on access, novelty-decay for repeated queries).

### 6.7 — The hidden retrieval problem during inference

During wake, the model uses "W_slow + Active Tags" to answer queries. But which tags are "active" for a given query? There's an implicit retrieval step here.

Questions to resolve:
- How does the system determine which tags are relevant to the current query? Embedding similarity? Attention-based routing?
- If this is similarity-based, you've reintroduced a retrieval mechanism — the very thing you're replacing. How is this different from RAG, formally?
- If all tags are always "active" (no selection), how does the system scale? The model can't attend to thousands of tags simultaneously.
- Is there a formal argument for why *internal* tag retrieval is superior to *external* document retrieval (RAG)? The proposal claims replacement — the math must justify it.

**What to write down:** The tag activation function for a given query. The computational cost of activation. The formal argument for why this is not just "RAG inside the model."

### 6.8 — Privacy guarantees on generative replay

The compression requirement helps with privacy, but "forces compression" is not a formal guarantee.

Questions to resolve:
- Can you formally bound the probability that generative replay reproduces verbatim content from the original experience?
- If a user shares sensitive information (PII, credentials) and it gets tagged, can the sleep generator reproduce it during replay?
- Should there be a differential privacy mechanism on the generation process? If so, what is the privacy budget (ε)?
- Does compression ratio correlate with privacy? Is there a formal relationship between the two?
- Are there categories of information that should be explicitly excluded from tagging? How do you detect them?

**What to write down:** The privacy threat model. The formal bound on verbatim leakage (if achievable). The DP mechanism (if used) and its impact on replay quality.

### 6.9 — Long-term convergence over many wake-sleep cycles

Section 5.5 asks about single-cycle guarantees. The harder question is long-term dynamics.

Questions to resolve:
- After N sleep cycles, does W_slow converge to a stable state, or does it drift?
- Is there a fixed point of the wake-sleep dynamics? What are its properties?
- Can the system "forget" old consolidated knowledge through successive consolidation cycles? (Slow overwriting through interleaving imbalance.)
- Is there a formal model of the system's long-term memory capacity? How many distinct "memories" can survive 100 consolidation cycles?
- Does the system exhibit catastrophic forgetting at a longer timescale — not within one cycle (prevented by interleaving) but across many cycles?

**What to write down:** The long-term dynamical model. Stability analysis of the wake-sleep loop. Capacity bounds as a function of cycles. The expected forgetting curve for consolidated knowledge over time.

---

## Dependency Graph Summary

```
1.1 (What is a tag?)
 ├── 1.2 (Prediction error computation)
 │    ├── 1.3 (Tag creation function)
 │    └── 6.1 (Revision: contradiction vs. novelty detection)
 ├── 1.4 (Tag decay)
 │    └── 1.5 (Tag reinforcement)
 │         └── 6.6 (Adversarial robustness of reinforcement)
 ├── 1.6 (Tag capacity)
 │    └── 2.3 (PRP budget size)
 │         ├── 2.4 (Competitive allocation)
 │         │    └── 2.5 (PRP threshold)
 │         │         └── 6.2 (Cold start calibration)
 │         └── 6.6 (Adversarial robustness of scoring)
 ├── 2.1 (What is a PRP?)
 │    └── 2.2 (Scoring function)
 │         └── 6.6 (Adversarial robustness of scoring)
 ├── 6.3 (Tag vs. W_fast relationship) ← RESOLVE BEFORE 3.1
 │    └── 3.1 (What is W_fast?)
 │         ├── 3.2 (W_fast/W_slow interaction)
 │         │    └── 6.7 (Hidden retrieval problem)
 │         ├── 3.3 (W_fast learning rules)
 │         └── 4.1 (Generation model)
 │              ├── 4.2 (Compression)
 │              │    └── 6.8 (Privacy guarantees)
 │              └── 4.3 (Interleaving)
 │                   └── 4.4 (Sleep training)
 │                        └── 6.5 (Cross-cycle compositionality)
 ├── 3.4 (W_slow plasticity)
 │    ├── 4.4 (Sleep training)
 │    └── 6.1 (Revision/unlearning mechanism)
 ├── 4.5 (Consolidation triggers)
 │    └── 4.6 (Post-consolidation cleanup)
 └── 6.4 (Multi-user dynamics) ← RESOLVE BEFORE implementation

Long-term analysis (requires all of the above):
 └── 6.9 (Convergence over many wake-sleep cycles)

Cross-cutting: 5.1–5.5 (should be revisited after each part)
Cross-cutting: 6.2 (Cold start — revisit after Parts 1, 2, and 3 are complete)
```

---

## Checkpoint Rules

**After completing Part 1:** You should be able to write pseudocode for the tagging layer that takes an input and produces a set of tags. If you can't, something is under-specified.

**After completing Part 2:** You should be able to simulate PRP allocation on paper — given 20 tags with known scores and a budget of 5, you should be able to run the allocation algorithm by hand and show convergence.

**After completing Part 3:** You should be able to draw the exact architecture diagram with parameter counts, specifying every tensor shape and interaction.

**After completing Part 4:** You should be able to write the complete sleep cycle as a training script (even if you don't run it yet) — every step from "select memories" to "clear tags" should be fully specified.

**After completing Part 5:** You should be able to write the abstract of your first paper. If you can't state clearly what you built, how you measured it, and what you found — something is still missing.

**After completing Part 6:** You should be able to answer the following stress tests on paper, without hand-waving:
- "The user told the model X on day 1, then told it not-X on day 30. What happens?" (6.1)
- "A brand new user sends their first message. Walk through exactly what the system does." (6.2)
- "Draw the exact data flow from a tagged experience through W_fast to a generated replay sample." (6.3)
- "Two users consolidate contradictory preferences. What happens to W_slow?" (6.4)
- "After 100 sleep cycles, the user asks about something from cycle 3. Can the system answer? Why?" (6.5, 6.9)
- "A user repeats the same useless question 500 times. What gets consolidated?" (6.6)
- "The user asks a question. Which of the 2,000 active tags are consulted, and how?" (6.7)
- "The user shared their medical history. Can the replay generator reproduce it verbatim?" (6.8)

---

*Only then do you implement.*
