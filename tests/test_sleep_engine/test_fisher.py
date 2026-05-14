"""Tests for sleep.sleep_engine.fisher — Fisher diagonal and EWC loss."""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sleep.config import WeightsConfig
from sleep.sleep_engine.fisher import compute_ewc_loss, compute_fisher_diagonal
from sleep.weights.lora import setup_dual_adapters


@pytest.fixture(scope="module")
def weights_config():
    """Lightweight WeightsConfig for faster tests."""
    return WeightsConfig(lora_rank=4, lora_alpha=8)


@pytest.fixture(scope="module")
def tokenizer():
    """Load GPT-2 tokenizer once for the entire module."""
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


@pytest.fixture(scope="module")
def lora_model(weights_config):
    """GPT-2 with dual LoRA adapters (w_fast and w_cons), w_cons active."""
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    peft_model = setup_dual_adapters(model, weights_config)
    # Ensure w_cons adapter is active so its params get gradients
    peft_model.set_adapter("w_cons")
    return peft_model


@pytest.fixture(scope="module")
def calibration_data(tokenizer):
    """A small set of token tensors for calibration."""
    texts = [
        "The history of artificial intelligence began in the 1950s.",
        "Mathematics provides the foundation for computer science.",
        "Neural networks are inspired by biological neurons in the brain.",
    ]
    return [
        torch.tensor(tokenizer.encode(t, add_special_tokens=False), dtype=torch.long)
        for t in texts
    ]


class TestComputeFisherDiagonal:
    """Test that compute_fisher_diagonal returns a non-empty dict with positive values."""

    def test_returns_nonempty_dict(self, lora_model, calibration_data):
        fisher = compute_fisher_diagonal(
            model=lora_model,
            calibration_data=calibration_data,
            adapter_name="w_cons",
            max_samples=3,
            device="cpu",
        )
        assert isinstance(fisher, dict)
        assert len(fisher) > 0

    def test_fisher_values_are_positive(self, lora_model, calibration_data):
        fisher = compute_fisher_diagonal(
            model=lora_model,
            calibration_data=calibration_data,
            adapter_name="w_cons",
            max_samples=3,
            device="cpu",
        )
        for name, tensor in fisher.items():
            assert isinstance(tensor, torch.Tensor)
            # Fisher values should be non-negative (squared gradients)
            assert tensor.min().item() >= 0.0
            # At least some should be strictly positive
        total_sum = sum(t.sum().item() for t in fisher.values())
        assert total_sum > 0.0


class TestComputeEwcLoss:
    """Test that compute_ewc_loss returns a non-negative scalar tensor."""

    def test_returns_nonnegative_scalar(self, lora_model, calibration_data):
        # Compute fisher first
        fisher = compute_fisher_diagonal(
            model=lora_model,
            calibration_data=calibration_data,
            adapter_name="w_cons",
            max_samples=3,
            device="cpu",
        )

        # Save checkpoint (current params as reference)
        checkpoint = {}
        for name, param in lora_model.named_parameters():
            if "w_cons" in name:
                checkpoint[name] = param.data.clone()

        # Compute EWC loss
        ewc_loss = compute_ewc_loss(
            model=lora_model,
            fisher_diag=fisher,
            checkpoint=checkpoint,
            lambda_ewc=100.0,
            adapter_name="w_cons",
        )

        assert isinstance(ewc_loss, torch.Tensor)
        assert ewc_loss.dim() == 0  # scalar
        assert ewc_loss.item() >= 0.0
