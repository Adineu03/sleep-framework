# 2026-04-07: Experiment 01 — Validate Tagging

## What I Did
Ran 8 diverse documents through GPT-2 (124M) tagging pipeline. Inspected tagged spans manually.

## What I Expected
- Generic text: 0 tags
- Technical/novel content: multiple tags on informative spans
- Repetitive text: tags only on the surprising part
- Financial facts: tags on numbers and locations

## What Actually Happened
- 4/5 sanity checks passed
- Correct: novel_technical (2 tags on framework description), financial_fact (1 tag on "$500K...Dresden"), scientific_claim (1 tag on synaptic tagging content), repetitive_text (1 tag on meteor, not repetition), simple_code (0 tags)
- Cold start artifact: common_greeting got 1 tag (mu=0.6 at that point, not calibrated yet)
- Conservative: personal_preference and mixed_boring_and_surprising got 0 tags (threshold mu=3.16 by that point, 1.5sigma above is high)

## Hyperparameter Changes
None yet — keeping defaults. Observations for future tuning:

| Parameter | Formalization Default | Observation | Action |
|:---|:---|:---|:---|
| kappa | 1.5 | May be too conservative for GPT-2 Small — misses moderate-surprise content | Revisit after Step 3 when we can measure downstream impact |
| beta | 0.99 | Threshold adapts slowly — first 2 docs are in calibration zone | Expected. ColdStartManager handles this in production. |

## Implications
- Tagging mechanism is fundamentally sound — tags the right things
- Threshold sensitivity is a tuning knob, not an architecture problem
- Cold start behavior matches Q6.2 prediction — threshold needs burn-in
- kappa tuning should be informed by consolidation quality (Step 3), not just tagging precision

## Next
- Step 2: Validate W_fast encoding (does gradient update reduce PPL on tagged spans?)
