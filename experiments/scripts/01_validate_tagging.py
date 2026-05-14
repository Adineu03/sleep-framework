"""
Experiment 01: Validate Tagging

PURPOSE:
    Feed diverse documents through GPT-2 and inspect what gets tagged.
    This is the first validation — does the tagging mechanism identify
    genuinely surprising/novel information?

WHAT TO LOOK FOR:
    - Tagged spans should be genuinely informative (facts, numbers, novel claims)
    - Common filler words, greetings, and boilerplate should NOT be tagged
    - The z-score threshold should adapt across different content types
    - More technical/novel content should produce more tags than generic text

NO TRAINING. NO SLEEP. Just: tokenize → compute surprise → tag → inspect.

USAGE:
    python experiments/scripts/01_validate_tagging.py
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from sleep.config import SLEEPConfig, TaggingConfig
from sleep.tagging import TaggingLayer
from sleep.utils.logging import setup_logging, get_logger

setup_logging()
logger = get_logger("experiment.01")


def load_config(config_path):
    """Load config YAML and return (model_name, device, dtype, tagging_cfg)."""
    if config_path is None:
        return "gpt2", "cpu", "float32", TaggingConfig()
    with open(config_path) as f:
        data = yaml.safe_load(f)
    model_name = data["model"]["name"]
    device = data["model"].get("device", "cpu")
    dtype = data["model"].get("dtype", "float32")
    tagging_cfg = TaggingConfig(**data["tagging"])
    return model_name, device, dtype, tagging_cfg


# ============================================================================
# Test Documents — diverse content types
# ============================================================================

DOCUMENTS = {
    "common_greeting": {
        "text": (
            "Hello! How are you doing today? I hope you're having a wonderful day. "
            "The weather is nice outside. Let me know if you need anything at all."
        ),
        "expected_tags": "few or none — this is completely generic",
    },
    "financial_fact": {
        "text": (
            "The company reported Q3 revenue of $4.2 million, representing a 12% "
            "decline from the previous quarter. Operating expenses increased by 8% "
            "to $3.1 million, driven primarily by a $500,000 investment in the new "
            "manufacturing facility in Dresden, Germany."
        ),
        "expected_tags": "several — specific numbers, percentages, locations are surprising",
    },
    "scientific_claim": {
        "text": (
            "Researchers at the Karolinska Institute discovered that synaptic tags "
            "created by weak stimulation can capture plasticity-related proteins "
            "synthesized up to 90 minutes later, provided the proteins are triggered "
            "by a separate strong stimulation event on the same neuron. This finding "
            "fundamentally changed our understanding of memory consolidation."
        ),
        "expected_tags": "many — technical content, specific institutions, novel claims",
    },
    "simple_code": {
        "text": (
            "To sort a list in Python, you can use the built-in sorted() function. "
            "For example: sorted([3, 1, 2]) returns [1, 2, 3]. You can also use "
            "the .sort() method which sorts the list in place."
        ),
        "expected_tags": "few — well-known programming knowledge",
    },
    "novel_technical": {
        "text": (
            "The SLEEP framework introduces a metabolic budget constraint on memory "
            "consolidation, modeled as a fixed pool of Plasticity-Related Proteins "
            "that are competitively allocated to tagged synapses. The composite "
            "scoring function combines prediction error, access frequency, "
            "cross-reference density, and recency-weighted utility with weights "
            "0.35, 0.30, 0.15, and 0.20 respectively."
        ),
        "expected_tags": "many — novel framework, specific numbers, technical terms",
    },
    "personal_preference": {
        "text": (
            "I prefer dark mode in all my applications. My favorite programming "
            "language is Rust, though I write most of my research code in Python. "
            "I usually work from 9am to 6pm and take breaks every 90 minutes."
        ),
        "expected_tags": "some — personal facts (preferences, schedule) are novel to the model",
    },
    "mixed_boring_and_surprising": {
        "text": (
            "The meeting was held on Tuesday at 3pm in the main conference room. "
            "During the meeting, Dr. Chen presented results showing that their new "
            "catalyst achieved 97.3% conversion efficiency at only 120 degrees Celsius, "
            "compared to the industry standard of 78% at 300 degrees. The team agreed "
            "to schedule a follow-up meeting next week."
        ),
        "expected_tags": "specific numbers and the comparison — not the meeting logistics",
    },
    "repetitive_text": {
        "text": (
            "The quick brown fox jumps over the lazy dog. The quick brown fox jumps "
            "over the lazy dog. The quick brown fox jumps over the lazy dog. The quick "
            "brown fox jumps over the lazy dog. Then suddenly, a meteor struck the field "
            "at exactly 14:32 UTC, creating a crater 30 meters wide."
        ),
        "expected_tags": "only the meteor part — repetition should NOT be tagged",
    },
}


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML (default: GPT-2 on CPU)")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("EXPERIMENT 01: Validate Tagging")
    logger.info("=" * 70)

    model_name, device, dtype_str, config = load_config(args.config)
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map.get(dtype_str, torch.float32)

    # Load model and tokenizer
    logger.info("Loading model: %s (device=%s, dtype=%s)", model_name, device, dtype_str)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_params_b = n_params / 1e9
    logger.info("Model loaded (%.3fB parameters)", n_params_b)

    # Create tagging layer with config
    tagging = TaggingLayer(model, config, model_params_billions=n_params_b)

    logger.info("Tag capacity: %d", tagging._n_max)
    logger.info("Threshold kappa: %.1f", config.kappa)
    logger.info("Min span: %d tokens", config.min_span)
    logger.info("Gap tolerance: %d tokens", config.gap_tolerance)
    logger.info("")

    # Process each document
    results = {}
    for doc_name, doc_info in DOCUMENTS.items():
        text = doc_info["text"]
        expected = doc_info["expected_tags"]

        # Tokenize
        token_ids = tokenizer.encode(text, return_tensors="pt").squeeze(0).to(device)

        # Process through tagging layer
        new_tags = tagging.process_input(token_ids, source_id=doc_name)

        # Decode tagged spans
        tagged_spans = []
        for tag in new_tags:
            span_start, span_end, _ = tag.ctx
            span_tokens = token_ids[span_start:span_end + 1]
            span_text = tokenizer.decode(span_tokens)
            tagged_spans.append({
                "text": span_text,
                "start": span_start,
                "end": span_end,
                "E_span": tag.e0,
                "strength": tag.s,
            })

        results[doc_name] = {
            "n_tokens": len(token_ids),
            "n_tags": len(new_tags),
            "tagged_spans": tagged_spans,
            "expected": expected,
            "threshold_mu": tagging.threshold.mu,
            "threshold_sigma": tagging.threshold.sigma,
        }

        # Print results
        print(f"\n{'='*70}")
        print(f"Document: {doc_name}")
        print(f"  Tokens: {len(token_ids)} | Tags created: {len(new_tags)}")
        print(f"  Threshold: mu={tagging.threshold.mu:.3f}, sigma={tagging.threshold.sigma:.3f}")
        print(f"  Expected: {expected}")
        print(f"  Tagged spans:")
        if tagged_spans:
            for i, span in enumerate(tagged_spans):
                print(f"    [{i+1}] \"{span['text'][:80]}\"")
                print(f"        E_span={span['E_span']:.3f}  strength={span['strength']:.3f}  "
                      f"tokens=[{span['start']}:{span['end']}]")
        else:
            print(f"    (none)")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Document':<30} {'Tokens':>7} {'Tags':>5} {'Tags/Token':>10}")
    print(f"{'-'*30} {'-'*7} {'-'*5} {'-'*10}")
    for doc_name, r in results.items():
        ratio = r["n_tags"] / r["n_tokens"] if r["n_tokens"] > 0 else 0
        print(f"{doc_name:<30} {r['n_tokens']:>7} {r['n_tags']:>5} {ratio:>10.3f}")

    print(f"\nTotal tags in buffer: {tagging.n_active}")
    print(f"Buffer occupancy: {tagging.occupancy:.4f}")

    # Sanity checks
    print(f"\n{'='*70}")
    print("SANITY CHECKS")
    print(f"{'='*70}")

    checks = []

    # Check 1: Greeting should have fewer tags than scientific content
    greeting_tags = results["common_greeting"]["n_tags"]
    science_tags = results["scientific_claim"]["n_tags"]
    passed = science_tags > greeting_tags
    checks.append(("Science > Greeting tags", passed, f"{science_tags} vs {greeting_tags}"))

    # Check 2: Novel technical should have tags
    novel_tags = results["novel_technical"]["n_tags"]
    passed = novel_tags > 0
    checks.append(("Novel technical has tags", passed, f"{novel_tags} tags"))

    # Check 3: Financial facts should have tags
    fin_tags = results["financial_fact"]["n_tags"]
    passed = fin_tags > 0
    checks.append(("Financial facts have tags", passed, f"{fin_tags} tags"))

    # Check 4: Repetitive text tags should be in the surprising part (meteor), not the repetition
    rep_tags = results["repetitive_text"]["tagged_spans"]
    if rep_tags:
        all_in_second_half = all(s["start"] > 30 for s in rep_tags)  # repetition is ~first 40 tokens
        checks.append(("Repetitive: tags in surprising part", all_in_second_half,
                        f"spans at positions {[s['start'] for s in rep_tags]}"))
    else:
        checks.append(("Repetitive: tags in surprising part", False, "no tags at all"))

    # Check 5: Buffer is not overflowing
    passed = tagging.occupancy < 0.5
    checks.append(("Buffer not overflowing", passed, f"{tagging.occupancy:.4f}"))

    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: {detail}")

    n_passed = sum(1 for _, p, _ in checks if p)
    print(f"\n{n_passed}/{len(checks)} sanity checks passed")

    logger.info("Experiment 01 complete.")


if __name__ == "__main__":
    main()
