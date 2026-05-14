# 2026-04-07: Experiment 03 — Single Sleep Cycle (First Attempt)

## What I Did
Fed 5 fabricated facts through the full SLEEP pipeline: wake (tag + W_fast update + PRP allocation) then sleep (replay + quality check + train W_cons + validate).

## What I Expected
- Some facts recalled after consolidation (DRA > 0)
- Pipeline runs end-to-end without crash

## What Actually Happened
- Wake phase flawless: 9 tags, 3 PRP-allocated, W_fast updated on all
- Replay generation: 3/3 generated, but quality check rejected 2/3 (surprise below mu_surprise)
- Sleep training: 100 steps on 1 replay, loss stable at ~3.26
- Validation: 0/3 consolidated (surprise didn't decrease enough)
- DRA = 0.000, BCP = 1.0001

## Root Cause Analysis
Two bottlenecks:
1. **Quality filter surprise check too strict for GPT-2 Small.** Replays are fluent text (low surprise to GPT-2) even though they contain novel factual content. The surprise check conflates "fluent" with "already known."
2. **Insufficient training signal.** 1 replay, 100 steps, lr=1e-5, rank-4 LoRA = nearly zero weight change.

## Hyperparameter Changes
| Parameter | Formalization Default | New Value | Why |
|:---|:---|:---|:---|
| Quality surprise check | must exceed mu_surprise | **Disabled for PoC** | GPT-2 generates fluent replay that's always below baseline surprise. This check is designed for larger models. |
| alpha_slow | 1e-5 | **5e-4** | Need stronger signal through rank-4 LoRA on tiny model |
| steps_per_memory | 5 (was 3 in experiment) | **10** | More gradient exposure per memory |
| compression_target | 5 (was 3 in experiment) | **2** | Less compression = more content preserved in replay |

## Implications
- The architecture works end-to-end — this is a tuning problem, not a design problem
- Quality surprise check needs rethinking for small models (or just skip for PoC)
- The formalization defaults are calibrated for 7B+ models — expected to need adjustment for 124M

## Next
- Rerun with relaxed quality check and stronger training signal
