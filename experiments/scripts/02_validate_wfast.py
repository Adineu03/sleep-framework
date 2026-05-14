"""
Experiment 02: Validate W_fast Encoding

PURPOSE:
    Verify that W_fast (LoRA adapter) can selectively encode surprising information.
    After a gradient update on a surprising span:
    - PPL on that span should DECREASE (the model learned it)
    - PPL on unrelated text should NOT increase (no interference)

WHAT TO LOOK FOR:
    - Clear PPL drop on the target span after W_fast update
    - Minimal PPL change on control text
    - Loss value during the update step should be reasonable (not NaN, not zero)

USAGE:
    python experiments/scripts/02_validate_wfast.py
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from sleep.config import WeightsConfig
from sleep.weights import DualWeightSystem
from sleep.weights.composition import set_wake_inference_mode
from sleep.utils.logging import setup_logging, get_logger

setup_logging()
logger = get_logger("experiment.02")


def load_config(config_path):
    """Load config YAML and return (model_name, device, dtype, weights_cfg)."""
    if config_path is None:
        return "gpt2", "cpu", "float32", WeightsConfig(lora_rank=4, lora_alpha=8)
    with open(config_path) as f:
        data = yaml.safe_load(f)
    model_name = data["model"]["name"]
    device = data["model"].get("device", "cpu")
    dtype = data["model"].get("dtype", "float32")
    weights_cfg = WeightsConfig(**data["weights"])
    return model_name, device, dtype, weights_cfg


def compute_ppl(model, token_ids: torch.Tensor, device: str = "cpu") -> float:
    """Compute perplexity of the model on a token sequence."""
    model.eval()
    token_ids = token_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids=token_ids.unsqueeze(0), labels=token_ids.unsqueeze(0))
        loss = outputs.loss.item()
    return torch.exp(torch.tensor(loss)).item()


# ============================================================================
# Test Cases
# ============================================================================

# Target: surprising text that W_fast should learn
TARGET_TEXTS = {
    "financial": "The Q3 revenue was $4.2 million, down 12% from Q2. Operating costs rose to $3.1 million due to the Dresden facility investment.",
    "scientific": "Synaptic tags created by weak stimulation capture plasticity-related proteins synthesized up to 90 minutes later on the same neuron.",
    "technical": "The SLEEP framework uses a metabolic PRP budget of 500 per billion parameters with competitive allocation and a steal differential of 0.05.",
}

# Control: unrelated text that should NOT be affected
CONTROL_TEXTS = {
    "recipe": "To make a simple pasta sauce, heat olive oil in a pan, add garlic and tomatoes, and simmer for twenty minutes.",
    "geography": "The Amazon River flows through Brazil and is the largest river by discharge volume in the world.",
    "greeting": "Hello, how are you today? I hope you are having a great day. Let me know if you need help.",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML (default: GPT-2 on CPU)")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("EXPERIMENT 02: Validate W_fast Encoding")
    logger.info("=" * 70)

    model_name, device, dtype_str, config = load_config(args.config)
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map.get(dtype_str, torch.float32)

    # Load model and tokenizer
    logger.info("Loading model: %s (device=%s, dtype=%s)", model_name, device, dtype_str)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = model.to(device)

    dws = DualWeightSystem(model, config)
    logger.info("W_fast params: %s", f"{dws.w_fast_params:,}")

    # Tokenize everything
    target_ids = {name: tokenizer.encode(text, return_tensors="pt").squeeze(0)
                  for name, text in TARGET_TEXTS.items()}
    control_ids = {name: tokenizer.encode(text, return_tensors="pt").squeeze(0)
                   for name, text in CONTROL_TEXTS.items()}

    # ========================================
    # PHASE 1: Measure PPL BEFORE W_fast update
    # ========================================
    logger.info("\nPhase 1: Measuring baseline PPL...")
    dws.set_mode("wake_inference")

    ppl_before = {}
    for name, ids in {**target_ids, **control_ids}.items():
        ppl = compute_ppl(dws.model, ids, device)
        ppl_before[name] = ppl

    print(f"\n{'='*70}")
    print("BASELINE PPL (before W_fast update)")
    print(f"{'='*70}")
    print(f"{'Text':<20} {'PPL':>10}")
    print(f"{'-'*20} {'-'*10}")
    for name, ppl in ppl_before.items():
        label = "[TARGET]" if name in TARGET_TEXTS else "[CTRL]"
        print(f"{name:<20} {ppl:>10.2f}  {label}")

    # ========================================
    # PHASE 2: Update W_fast on target texts
    # ========================================
    logger.info("\nPhase 2: Updating W_fast on target texts...")
    print(f"\n{'='*70}")
    print("W_FAST GRADIENT UPDATES")
    print(f"{'='*70}")

    update_losses = {}
    for name, ids in target_ids.items():
        # Run 3 gradient steps per target for stronger encoding
        losses = []
        for step in range(3):
            loss = dws.update_fast_weights(
                token_ids=ids,
                span_start=0,
                span_end=len(ids) - 1,
                E_span=2.0,  # moderate surprise
                device=device,
            )
            losses.append(loss)
        update_losses[name] = losses
        print(f"  {name}: loss = {losses[0]:.4f} -> {losses[-1]:.4f} (3 steps)")

    # ========================================
    # PHASE 3: Measure PPL AFTER W_fast update
    # ========================================
    logger.info("\nPhase 3: Measuring PPL after W_fast update...")
    dws.set_mode("wake_inference")

    ppl_after = {}
    for name, ids in {**target_ids, **control_ids}.items():
        ppl = compute_ppl(dws.model, ids, device)
        ppl_after[name] = ppl

    # ========================================
    # RESULTS
    # ========================================
    print(f"\n{'='*70}")
    print("PPL COMPARISON: BEFORE vs AFTER W_fast update")
    print(f"{'='*70}")
    print(f"{'Text':<20} {'Before':>10} {'After':>10} {'Change':>10} {'%':>8} {'Verdict'}")
    print(f"{'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*10}")

    for name in list(target_ids.keys()) + list(control_ids.keys()):
        before = ppl_before[name]
        after = ppl_after[name]
        change = after - before
        pct = (change / before) * 100

        if name in TARGET_TEXTS:
            verdict = "GOOD (learned)" if change < 0 else "BAD (not learned)"
            label = "[TARGET]"
        else:
            verdict = "GOOD (stable)" if abs(pct) < 5 else "BAD (interference!)"
            label = "[CTRL]"

        print(f"{name:<20} {before:>10.2f} {after:>10.2f} {change:>+10.2f} {pct:>+7.1f}%  {verdict}  {label}")

    # ========================================
    # SANITY CHECKS
    # ========================================
    print(f"\n{'='*70}")
    print("SANITY CHECKS")
    print(f"{'='*70}")

    checks = []

    # Check 1: Target PPL decreased
    for name in target_ids:
        decreased = ppl_after[name] < ppl_before[name]
        checks.append((f"Target '{name}' PPL decreased", decreased,
                        f"{ppl_before[name]:.2f} -> {ppl_after[name]:.2f}"))

    # Check 2: Control PPL didn't increase more than 5%
    for name in control_ids:
        pct_change = (ppl_after[name] - ppl_before[name]) / ppl_before[name] * 100
        stable = abs(pct_change) < 5.0
        checks.append((f"Control '{name}' stable (<5% change)", stable,
                        f"{pct_change:+.1f}%"))

    # Check 3: Update losses were finite and positive
    for name, losses in update_losses.items():
        all_finite = all(0 < l < 100 for l in losses)
        checks.append((f"Update losses finite for '{name}'", all_finite,
                        f"{[f'{l:.3f}' for l in losses]}"))

    # Check 4: Update losses decreased over steps (learning is happening)
    for name, losses in update_losses.items():
        decreased = losses[-1] < losses[0]
        checks.append((f"Loss decreased for '{name}'", decreased,
                        f"{losses[0]:.4f} -> {losses[-1]:.4f}"))

    for check_name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check_name}: {detail}")

    n_passed = sum(1 for _, p, _ in checks if p)
    print(f"\n{n_passed}/{len(checks)} sanity checks passed")

    logger.info("Experiment 02 complete.")


if __name__ == "__main__":
    main()
