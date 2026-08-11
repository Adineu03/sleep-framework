"""Tests for sleep.distill.ContextDistiller.

The critical checks, per the project's ground-truth-first practice:
  1. The teacher pass really is the base model — with the adapter zero-init,
     teacher and student logits must be identical for the same input, and
     after training they must differ (the adapter moved; the teacher didn't).
  2. The distillation loss is finite, and the adapter parameters move.
  3. use_paraphrases=False restricts training to fact["text"] (the ablation
     arm's contract).
"""

from __future__ import annotations

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM

from sleep.distill import ContextDistiller


class _StubTokenizer:
    """Maps characters to ids deterministically; sufficient for loss plumbing."""

    eos_token_id = 0

    def __call__(self, text, return_tensors="pt", **kw):
        ids = [(ord(c) % 97) + 2 for c in text[:64]]
        if len(ids) < 2:
            ids = [2, 3]
        import torch as _t
        return type("Enc", (), {"input_ids": _t.tensor([ids])})()


def _tiny_peft_model(seed=0):
    torch.manual_seed(seed)
    cfg = Qwen2Config(
        vocab_size=100, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=256, sliding_window=None,
        attn_implementation="eager", torch_dtype=torch.float32,
    )
    model = Qwen2ForCausalLM(cfg)
    lora = LoraConfig(
        r=2, lora_alpha=4, target_modules=["down_proj"],
        layers_to_transform=[1, 2], bias="none", task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora)


_FACTS = [
    {"id": "f1", "text": "Nimbus Holdings reported revenue of 482 million.",
     "paraphrases": ["Revenue at Nimbus Holdings hit 482 million.",
                     "Question: What was Nimbus revenue?\nAnswer: 482 million."]},
    {"id": "f2", "text": "The Sigma-7 protocol window is 48 hours.",
     "paraphrases": ["Sigma-7 sets a 48 hour window.",
                     "Question: How long is the Sigma-7 window?\nAnswer: 48 hours."]},
]


class TestContextDistiller:

    def test_teacher_is_base_model_at_zero_init(self):
        model = _tiny_peft_model()
        tok = _StubTokenizer()
        ids = tok("some probe text").input_ids
        with torch.no_grad():
            student = model(input_ids=ids).logits
            with model.disable_adapter():
                teacher = model(input_ids=ids).logits
        # Zero-init LoRA: adapter is identity, so the two must match exactly.
        assert torch.allclose(student, teacher, atol=1e-6)

    def test_run_moves_adapter_and_loss_finite(self):
        model = _tiny_peft_model()
        tok = _StubTokenizer()
        before = {
            n: p.detach().clone()
            for n, p in model.named_parameters() if p.requires_grad
        }
        d = ContextDistiller(model, tok, device="cpu", kd_temperature=2.0)
        result = d.run(_FACTS, n_steps=6, lr=5e-2, seed=0)

        assert result.n_steps == 6
        assert result.mean_loss == result.mean_loss  # not NaN
        moved = any(
            not torch.allclose(before[n], p.detach())
            for n, p in model.named_parameters() if p.requires_grad
        )
        assert moved, "distillation did not move the adapter"

    def test_teacher_unchanged_after_training(self):
        model = _tiny_peft_model()
        tok = _StubTokenizer()
        ids = tok("probe after training").input_ids
        with torch.no_grad(), model.disable_adapter():
            teacher_before = model(input_ids=ids).logits
        ContextDistiller(model, tok, device="cpu").run(_FACTS, n_steps=6, lr=5e-2)
        with torch.no_grad(), model.disable_adapter():
            teacher_after = model(input_ids=ids).logits
        assert torch.allclose(teacher_before, teacher_after, atol=1e-6), (
            "teacher (base model) drifted — adapter leaked into base weights"
        )

    def test_student_diverges_from_teacher_after_training(self):
        model = _tiny_peft_model()
        tok = _StubTokenizer()
        ContextDistiller(model, tok, device="cpu").run(_FACTS, n_steps=8, lr=5e-2)
        ids = tok(_FACTS[0]["text"]).input_ids
        with torch.no_grad():
            student = model(input_ids=ids).logits
            with model.disable_adapter():
                teacher = model(input_ids=ids).logits
        assert not torch.allclose(student, teacher, atol=1e-4)

    def test_single_wording_contract(self):
        # With use_paraphrases=False the sampler must only ever draw
        # fact["text"]. We verify via a spy on _step_loss.
        model = _tiny_peft_model()
        tok = _StubTokenizer()
        d = ContextDistiller(model, tok, device="cpu")
        seen = []
        original = d._step_loss

        def spy(fact_text, target):
            seen.append(target)
            return original(fact_text, target)

        d._step_loss = spy
        d.run(_FACTS, n_steps=5, lr=1e-3, use_paraphrases=False)
        texts = {f["text"] for f in _FACTS}
        assert seen and all(t in texts for t in seen)

    def test_empty_facts_raises(self):
        model = _tiny_peft_model()
        with pytest.raises(ValueError, match="non-empty"):
            ContextDistiller(model, _StubTokenizer()).run([], n_steps=2)
