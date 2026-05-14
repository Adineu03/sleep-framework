# SLEEP: Synaptic Learning through Error-driven Encoding and Plasticity

## What This Is

SLEEP is a biologically-faithful memory consolidation architecture for Large Language Models, drawing on the synaptic tagging-and-capture hypothesis (Frey & Morris, 1997), Complementary Learning Systems (McClelland et al., 1995), and hippocampal–neocortical replay during sleep (Wilson & McNaughton, 1994; Diekelmann & Born, 2010). The system augments a 7B-parameter transformer (Qwen2.5-7B) with five interacting components — prediction-error tagging, direct-write key-value memory, protein-resource-allocation prioritization, generative replay, and LoRA-based consolidation under explicit safety constraints — and asks whether biological mechanisms for continuous, single-exposure, interference-free learning can be implemented faithfully inside a frozen pretrained transformer.

## Key Findings

We instrument every component of the architecture and report what works, what does not, and where the boundary lies. **(C1)** LoRA fast weights cannot encode at single exposure: α_fast = 1e-4 falls below the bfloat16 precision floor, while 1e-3 degrades the model without learning. **(C2)** Direct-write KV memory injection produces a recognition signal (Tagged − Untagged Δ = +0.16 on multiple-choice, ~2.9σ) but does not surface in free-form generation — a transformer analogue of the recognition–recall dissociation in cognitive psychology. **(C3)** Tagged spans alone capture entity pointers without facts; episode-level storage is required. **(C4)** Internal surprise-reduction validation passes while external recall fails (37/52 candidates pass, DRA at floor) — a self-grading discrepancy not unique to SLEEP. **(C5)** Across a four-setting sweep, no single-cycle configuration achieves DRA > 0.05 with BCP < 1.05. **(C6)** A three-cycle continual-learning evaluation inverts the comparison: SLEEP preserves base capability ~2× better than naive LoRA fine-tuning at every cycle (BCP 2.31 vs 4.74 at cycle 3), validating the architecture's design-intent regime.

## Quick Start

Install the package and dependencies:

```bash
pip install -e ".[dev,notebooks]"
```

Run the tagging layer on a curated set of sample documents (greetings, news, technical text, novel facts) and inspect what gets tagged:

```bash
python experiments/scripts/01_validate_tagging.py --config experiments/configs/tiny_poc.yaml
```

This loads GPT-2 small (CPU is fine), passes each sample document through the tagging pipeline (`surprise → adaptive z-score → span segmentation → tag creation`), and prints the tagged spans for inspection. To use the layer in your own code:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sleep.config import TaggingConfig
from sleep.tagging import TaggingLayer

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

layer = TaggingLayer(model, TaggingConfig(), model_params_billions=0.124)

text = "The Curiosity rover detected unusually high methane levels on Mars in 2019."
token_ids = tokenizer(text, return_tensors="pt").input_ids[0]
new_tags = layer.process_input(token_ids, source_id="example")

for tag in new_tags:
    start, end, _ = tag.ctx
    print(f"span [{start}:{end}] '{tokenizer.decode(token_ids[start:end])}' strength={tag.strength:.3f}")
```

The full pipeline (tagging → KV memory → PRP → sleep → recall) lives in `experiments/scripts/07_full_kv_pipeline.py`; the multi-cycle continual-learning harness is `experiments/scripts/08_multi_cycle.py`. The full test suite (`pytest`) runs in ~5 min on CPU.

## Paper

The paper "Recognition Without Recall: Empirical Limits of Biologically-Inspired Memory Consolidation in Transformers" is in [paper/main.tex](paper/main.tex). Build the PDF with:

```bash
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Source sections are in [paper/sections/](paper/sections/) and figures in [paper/figures/](paper/figures/). The full mathematical formalization (36 design questions resolved before any code was written, plus five empirical amendments in Appendix A) is at [docs/SLEEP_Formalization.md](docs/SLEEP_Formalization.md).

## Citation

```bibtex
@techreport{tripathi2026sleep,
  title  = {Recognition Without Recall: Empirical Limits of
            Biologically-Inspired Memory Consolidation in Transformers},
  author = {Tripathi, Aditya},
  year   = {2026},
  note   = {Available at \url{https://github.com/Adineu03/sleep-framework}}
}
```

## License

MIT — see [LICENSE](LICENSE).
