"""Ground-truth tests for the ewc_lora arm added to experiment 08 (Phase D).

The EWC-only baseline must be naive LoRA + a quadratic Fisher penalty and
nothing else. These tests exercise the actual script functions
(run_naive_lora_cycle / consolidate_ewc) on a tiny Qwen2 with a real peft
adapter, on CPU, and check the two properties the showdown depends on:

1. consolidate_ewc produces a non-trivial Fisher and anchors at the current
   params; a second consolidation accumulates rather than replaces.
2. A large EWC penalty measurably restrains parameter movement relative to
   the identical run with lambda=0 (same seed, same data, same steps).
"""

from __future__ import annotations

import importlib.util
import os

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM

_SCRIPT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../experiments/scripts/08_multi_cycle.py",
))


def _load_script():
    spec = importlib.util.spec_from_file_location("exp08", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Tok:
    """Minimal tokenizer stub: maps chars to ids in [2, 90)."""

    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, return_tensors="pt", padding=False):
        ids = [2 + (ord(c) % 88) for c in text[:24]]
        return torch.tensor([ids], dtype=torch.long)


def _tiny_peft_model():
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=100, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=256, sliding_window=None,
        attn_implementation="eager", torch_dtype=torch.float32,
    ))
    lora = LoraConfig(
        r=2, lora_alpha=4, target_modules=["up_proj", "down_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora)


_FACTS = [
    {"text": "The Zorin Tower is located in the city of Veldar."},
    {"text": "The mineral kravite glows faintly blue under moonlight."},
]


def test_consolidate_ewc_builds_and_accumulates_fisher():
    exp08 = _load_script()
    peft_model = _tiny_peft_model()
    state = {"fisher": None, "anchor": None}

    exp08.consolidate_ewc(
        peft_model=peft_model, tokenizer=_Tok(), fact_batch=_FACTS,
        ewc_state=state, device="cpu",
    )
    assert state["fisher"] is not None and state["anchor"] is not None
    total1 = sum(float(f.sum()) for f in state["fisher"])
    assert total1 > 0.0, "Fisher diagonal is identically zero"
    # Anchor equals current trainable params exactly
    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    for a, p in zip(state["anchor"], trainable):
        assert torch.equal(a, p.detach().float())

    # Second consolidation accumulates (sum strictly grows)
    exp08.consolidate_ewc(
        peft_model=peft_model, tokenizer=_Tok(), fact_batch=_FACTS,
        ewc_state=state, device="cpu",
    )
    total2 = sum(float(f.sum()) for f in state["fisher"])
    assert total2 > total1


def _param_displacement(ewc_lambda):
    exp08 = _load_script()
    peft_model = _tiny_peft_model()
    tok = _Tok()
    state = {"fisher": None, "anchor": None}
    # Cycle 1: no penalty active yet (fisher None); consolidate after.
    exp08.run_naive_lora_cycle(
        cycle_idx=1, fact_batch=_FACTS, peft_model=peft_model, tokenizer=tok,
        n_steps=8, batch_size=2, learning_rate=5e-3, weight_decay=0.0,
        device="cpu", seed=0, ewc_state=state, ewc_lambda=ewc_lambda,
    )
    exp08.consolidate_ewc(
        peft_model=peft_model, tokenizer=tok, fact_batch=_FACTS,
        ewc_state=state, device="cpu",
    )
    before = [p.detach().clone() for p in peft_model.parameters()
              if p.requires_grad]
    # Cycle 2: penalty now active, anchored at post-cycle-1 params.
    info = exp08.run_naive_lora_cycle(
        cycle_idx=2, fact_batch=_FACTS, peft_model=peft_model, tokenizer=tok,
        n_steps=8, batch_size=2, learning_rate=5e-3, weight_decay=0.0,
        device="cpu", seed=0, ewc_state=state, ewc_lambda=ewc_lambda,
    )
    assert info["n_steps"] == 8, "non-finite loss aborted the cycle"
    after = [p for p in peft_model.parameters() if p.requires_grad]
    return sum(float((b - a).norm()) for b, a in zip(before, after))


def test_ewc_penalty_restrains_movement():
    moved_free = _param_displacement(ewc_lambda=0.0)
    moved_pinned = _param_displacement(ewc_lambda=1e6)
    assert moved_free > 0.0
    assert moved_pinned < 0.5 * moved_free, (
        f"EWC penalty had no restraining effect: "
        f"free={moved_free:.6f} pinned={moved_pinned:.6f}"
    )
