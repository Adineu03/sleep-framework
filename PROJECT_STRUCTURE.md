# SLEEP: Project Structure

```
Research/
│
├── README.md                          # Project overview, setup instructions, quickstart
│
│   ══════════════════════════════════
│   DOCUMENTATION (what & why)
│   ══════════════════════════════════
│
├── docs/
│   ├── Research Proposal.pdf          # Original proposal (moved here)
│   ├── SLEEP_Mathematical_Notebook.md # The 36 questions (moved here)
│   ├── SLEEP_Formalization.md         # Complete mathematical answers (moved here)
│   └── architecture.md               # Living architecture doc — updated as code diverges from math
│
│   ══════════════════════════════════
│   RESEARCH PAPER (the deliverable)
│   ══════════════════════════════════
│
├── paper/
│   ├── main.tex                       # Paper source (LaTeX)
│   ├── references.bib                 # Bibliography
│   ├── figures/                       # Generated figures for the paper
│   │   ├── system_architecture.pdf
│   │   ├── forgetting_curves.pdf
│   │   ├── ablation_results.pdf
│   │   └── ...
│   └── tables/                        # Generated tables (auto-exported from experiments)
│
│   ══════════════════════════════════
│   SOURCE CODE (the system)
│   ══════════════════════════════════
│
├── sleep/                             # Main Python package
│   ├── __init__.py
│   ├── config.py                      # All hyperparameters from master table, dataclass-based
│   │
│   ├── tagging/                       # Module 1: Tagging Engine
│   │   ├── __init__.py
│   │   ├── surprise.py                # Per-token surprise computation against W_slow
│   │   ├── threshold.py               # Adaptive z-score thresholding (EMA statistics)
│   │   ├── spans.py                   # Span segmentation (flag, merge, filter)
│   │   ├── tags.py                    # Tag dataclass, creation, key projection
│   │   └── buffer.py                  # Tag buffer: decay, reinforcement, GC, capacity mgmt
│   │
│   ├── prp/                           # Module 2: PRP Allocation
│   │   ├── __init__.py
│   │   ├── scoring.py                 # 4-component scoring function + normalization
│   │   ├── crossref.py                # Cross-reference density (batched cosine similarity)
│   │   └── allocation.py              # Competitive allocation, hysteresis, threshold
│   │
│   ├── weights/                       # Module 3: Dual Weight System
│   │   ├── __init__.py
│   │   ├── lora.py                    # LoRA adapter setup on selected layers (V, O, top L/3)
│   │   ├── composition.py             # W_slow_base + W_cons + W_fast composition
│   │   ├── fast_update.py             # W_fast gradient update, gated by κ_wfast
│   │   └── plasticity.py              # Layer-specific plasticity profile φ(l/L)
│   │
│   ├── sleep_engine/                  # Module 4: Sleep / Consolidation
│   │   ├── __init__.py
│   │   ├── replay.py                  # Replay generation (autoregressive from W_eff)
│   │   ├── quality.py                 # Quality check (similarity + surprise filters)
│   │   ├── interleave.py              # Old knowledge generation + mixing + curriculum
│   │   ├── train.py                   # Sleep training loop (EWC, clipping, validation)
│   │   ├── cleanup.py                 # Post-consolidation validation and tag clearing
│   │   └── fisher.py                  # Fisher information computation and refresh
│   │
│   ├── orchestrator/                  # Module 5: System Orchestrator
│   │   ├── __init__.py
│   │   ├── state.py                   # Wake/Sleep state machine, system state dataclass
│   │   ├── triggers.py                # Sleep trigger evaluation (schedule, pressure, idle)
│   │   ├── cold_start.py              # Cold start calibration (threshold ramp, budget ramp)
│   │   └── engine.py                  # Main loop: process input → tag → score → sleep cycle
│   │
│   ├── evaluation/                    # Module 6: Evaluation Suite
│   │   ├── __init__.py
│   │   ├── recall.py                  # Delayed Recall Accuracy at various intervals
│   │   ├── forgetting.py              # Forgetting curve computation
│   │   ├── preservation.py            # Base capability preservation (benchmark PPL)
│   │   ├── efficiency.py              # Consolidation efficiency, PRP allocation quality
│   │   └── baselines.py              # RAG baseline, naive LoRA baseline, random replay baseline
│   │
│   └── utils/                         # Shared utilities
│       ├── __init__.py
│       ├── logging.py                 # Structured logging for experiments
│       └── checkpoints.py             # Save/load W_cons, W_fast, tag buffer, Fisher
│
│   ══════════════════════════════════
│   TESTS (one-to-one with source)
│   ══════════════════════════════════
│
├── tests/
│   ├── test_tagging/
│   │   ├── test_surprise.py           # Unit: does surprise computation match manual calculation?
│   │   ├── test_threshold.py          # Unit: does adaptive threshold converge on synthetic data?
│   │   ├── test_spans.py              # Unit: span merging edge cases
│   │   ├── test_tags.py               # Unit: tag creation, key projection dimensions
│   │   └── test_buffer.py             # Unit: decay formula consistency, GC, capacity eviction
│   │
│   ├── test_prp/
│   │   ├── test_scoring.py            # Unit: scoring components normalize to [0,1]
│   │   └── test_allocation.py         # Unit: reproduce Part 2 worked example (20 tags, budget 5)
│   │
│   ├── test_weights/
│   │   ├── test_lora.py               # Unit: LoRA attaches to correct layers, correct shapes
│   │   └── test_composition.py        # Unit: W_eff = W_base + W_cons + W_fast produces valid output
│   │
│   ├── test_sleep_engine/
│   │   ├── test_replay.py             # Unit: generated replay passes quality check
│   │   ├── test_interleave.py         # Unit: batch composition ratios match η
│   │   └── test_train.py              # Integration: one sleep cycle doesn't crash, PPL doesn't explode
│   │
│   └── test_integration/
│       ├── test_single_cycle.py       # E2E: feed facts → tag → score → sleep → verify recall
│       ├── test_multi_cycle.py        # E2E: 10 cycles, verify forgetting curves
│       └── test_cold_start.py         # E2E: new user flow, verify threshold calibration
│
│   ══════════════════════════════════
│   EXPERIMENTS (the science)
│   ══════════════════════════════════
│
├── experiments/
│   ├── configs/                       # Experiment-specific config overrides
│   │   ├── base.yaml                  # Default config (matches formalization defaults)
│   │   ├── tiny_poc.yaml              # GPT-2 Small proof-of-concept
│   │   ├── scaling_1b.yaml            # 1B model experiments
│   │   ├── scaling_7b.yaml            # 7B model experiments
│   │   ├── ablation_no_tagging.yaml   # Ablation: random selection instead of tagging
│   │   ├── ablation_no_prp.yaml       # Ablation: consolidate everything (no PRP budget)
│   │   ├── ablation_no_interleave.yaml # Ablation: no old knowledge interleaving
│   │   └── baseline_rag.yaml          # RAG baseline configuration
│   │
│   ├── scripts/                       # Runnable experiment scripts
│   │   ├── 01_validate_tagging.py     # Phase 1, Step 2: inspect tagged spans
│   │   ├── 02_validate_wfast.py       # Phase 1, Step 4: W_fast encoding quality
│   │   ├── 03_single_sleep_cycle.py   # Phase 1, Step 6: one full cycle
│   │   ├── 04_multi_cycle.py          # Phase 1, Step 7: 10 cycles
│   │   ├── 05_subquestion_tagging.py  # Phase 2: tagging validity experiment
│   │   ├── 06_subquestion_budget.py   # Phase 2: budget dynamics sweep
│   │   ├── 07_subquestion_replay.py   # Phase 2: replay fidelity
│   │   ├── 08_subquestion_interference.py  # Phase 2: interference prevention
│   │   ├── 09_subquestion_prp.py      # Phase 2: emergent prioritization
│   │   ├── 10_subquestion_schedule.py  # Phase 2: sleep scheduling
│   │   ├── 11_baseline_rag.py         # Phase 3: RAG comparison
│   │   ├── 12_baseline_naive_lora.py  # Phase 3: naive LoRA fine-tuning comparison
│   │   └── 13_baseline_replay.py      # Phase 3: random replay comparison
│   │
│   └── results/                       # Raw experiment outputs (gitignored except summaries)
│       └── .gitkeep
│
│   ══════════════════════════════════
│   EXPERIMENT LOG (the gap between math and reality)
│   ══════════════════════════════════
│
├── logbook/
│   ├── README.md                      # How to use the logbook
│   ├── 2026-03-24_project_setup.md    # Entry 1: today
│   └── TEMPLATE.md                    # Template for new entries
│
│   ══════════════════════════════════
│   NOTEBOOKS (exploration & visualization)
│   ══════════════════════════════════
│
├── notebooks/
│   ├── 01_explore_tagging.ipynb       # Interactive exploration of tagging behavior
│   ├── 02_visualize_prp.ipynb         # PRP allocation dynamics visualization
│   ├── 03_sleep_cycle_walkthrough.ipynb  # Step-by-step walkthrough of one sleep cycle
│   └── 04_plot_results.ipynb          # Generate paper figures from experiment results
│
│   ══════════════════════════════════
│   PROJECT CONFIG
│   ══════════════════════════════════
│
├── pyproject.toml                     # Package config, dependencies, build
├── .gitignore                         # Ignore checkpoints, large models, results/raw
└── CLAUDE.md                          # Project conventions for AI-assisted development
```

## Design Principles

### 1. Source mirrors the formalization
Every module in `sleep/` maps to a Part in `SLEEP_Formalization.md`:
- `sleep/tagging/` → Part 1
- `sleep/prp/` → Part 2
- `sleep/weights/` → Part 3
- `sleep/sleep_engine/` → Part 4
- `sleep/orchestrator/` → Part 5 (system-level)
- `sleep/evaluation/` → Part 5 (metrics)

This means you can always trace a line of code back to its mathematical justification.

### 2. Tests mirror source
Every source file `sleep/X/Y.py` has a corresponding `tests/test_X/test_Y.py`. No exceptions.

### 3. Experiments are numbered and reproducible
Each experiment script is numbered by execution order. Each reads from a YAML config (so you can see exactly what hyperparameters were used). Results go to `experiments/results/` with timestamped directories.

### 4. The logbook is sacred
Every time a hyperparameter changes from the formalization defaults, it gets an entry.
Every surprising result gets an entry. Every dead end gets an entry.
This is where the real research lives.

### 5. Paper figures are auto-generated
The `notebooks/04_plot_results.ipynb` notebook reads from `experiments/results/` and exports to `paper/figures/`. No manually created figures — everything is reproducible from data.

## Logbook Entry Template

Each entry in `logbook/` follows this format:

```markdown
# YYYY-MM-DD: [Short Title]

## What I Did
[Concrete actions taken]

## What I Expected
[Based on the formalization, what should have happened]

## What Actually Happened
[Data, observations, screenshots]

## Hyperparameter Changes
| Parameter | Formalization Default | New Value | Why |
|:---|:---|:---|:---|

## Implications
[What this means for downstream decisions. Does the formalization need updating?]

## Next
[What to do tomorrow based on this]
```