# SLEEP Research Project

## What This Is
Research implementation of SLEEP (Synaptic Learning through Error-driven Encoding and Plasticity) — a biologically-inspired memory framework for continuous learning in LLMs.

## Key Documents
- `docs/SLEEP_Formalization.md` — Complete mathematical formalization (36 questions, all resolved)
- `docs/Research Proposal.pdf` — Original research proposal
- `docs/SLEEP_Mathematical_Notebook.md` — The 36 questions that needed answering
- `PROJECT_STRUCTURE.md` — Full project layout explanation

## Code Structure
- `sleep/` — Main Python package (tagging, prp, weights, sleep_engine, orchestrator, evaluation)
- `tests/` — Unit and integration tests, mirrors source structure
- `experiments/` — Configs (YAML), scripts (numbered), results
- `logbook/` — Experiment log entries (every hyperparameter change from formalization defaults)
- `paper/` — LaTeX paper (main.tex, references.bib, figures/)

## Conventions
- All hyperparameter defaults in `sleep/config.py` match `docs/SLEEP_Formalization.md` exactly
- Any deviation from formalization defaults must be logged in `logbook/`
- Experiment scripts are numbered by execution order (01_, 02_, ...)
- Experiment tracker: Weights & Biases (project: sleep-memory)
- Target model for PoC: GPT-2 Small (124M) on CPU