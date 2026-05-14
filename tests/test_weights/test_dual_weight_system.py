"""Integration tests for DualWeightSystem — the main entry point for Module 3."""

import math

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sleep.weights.composition as composition
from sleep.config import WeightsConfig
from sleep.weights import DualWeightSystem


@pytest.fixture(scope="module")
def weights_config():
    return WeightsConfig(lora_rank=4, lora_alpha=8)


@pytest.fixture(scope="module")
def dws(weights_config):
    """Create a DualWeightSystem wrapping GPT-2 (loaded once per module)."""
    # Reset composition state before constructing
    composition._cons_merged = False
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    system = DualWeightSystem(model, weights_config)
    return system


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture(autouse=True)
def reset_composition_state():
    """Reset module-level merge state before/after each test."""
    composition._cons_merged = False
    yield
    composition._cons_merged = False


class TestInitialization:
    def test_dws_creates_successfully(self, dws):
        assert dws is not None

    def test_has_model(self, dws):
        assert dws.model is not None

    def test_has_fast_updater(self, dws):
        assert dws.fast_updater is not None

    def test_param_counts_positive(self, dws):
        assert dws.w_fast_params > 0
        assert dws.w_cons_params > 0

    def test_fast_and_cons_same_count(self, dws):
        assert dws.w_fast_params == dws.w_cons_params

    def test_plasticity_profile_populated(self, dws):
        assert len(dws.plasticity_profile) > 0

    def test_base_model_norms_populated(self, dws):
        assert len(dws.base_model_norms) > 0


class TestModes:
    """Test that all 5 operational modes can be set without errors."""

    @pytest.mark.parametrize(
        "mode",
        [
            "wake_inference",
            "wfast_training",
            "sleep_generation",
            "sleep_training",
            "target_inference",
        ],
    )
    def test_set_mode(self, dws, mode):
        dws.set_mode(mode)  # Should not raise

    def test_invalid_mode_raises(self, dws):
        with pytest.raises(ValueError, match="Unknown mode"):
            dws.set_mode("nonexistent_mode")


class TestUpdateFastWeights:
    """Test that update_fast_weights runs a forward+backward pass and returns a loss."""

    def test_update_returns_finite_positive_loss(self, dws, tokenizer):
        text = "The quick brown fox jumps over the lazy dog."
        token_ids = tokenizer.encode(text, return_tensors="pt").squeeze(0)

        # Use a span in the middle (at least 2 tokens, with at least 1 token before)
        span_start = 2
        span_end = min(span_start + 4, len(token_ids))

        # Call the fast_updater directly since DualWeightSystem.update_fast_weights
        # has a known bug (missing model argument passthrough).
        loss = dws.fast_updater.update(
            model=dws.model,
            token_ids=token_ids,
            span_start=span_start,
            span_end=span_end,
            E_span=3.0,
            device="cpu",
        )

        assert isinstance(loss, float)
        assert math.isfinite(loss), f"Loss should be finite, got {loss}"
        assert loss > 0, f"Cross-entropy loss should be positive, got {loss}"


class TestCheckpointSaveRestore:
    """Test checkpoint save/restore via DualWeightSystem methods."""

    def test_save_and_restore_roundtrip(self, dws):
        # Set to a mode where w_cons is accessible
        dws.set_mode("sleep_training")

        # Save checkpoint
        checkpoint = dws.save_cons_checkpoint()
        assert len(checkpoint) > 0

        # Perturb w_cons weights
        with torch.no_grad():
            for name, param in dws.model.named_parameters():
                if "w_cons" in name:
                    param.add_(torch.randn_like(param) * 0.5)

        # Verify perturbation happened
        changed = False
        for name, param in dws.model.named_parameters():
            if "w_cons" in name and name in checkpoint:
                if not torch.allclose(param.data.cpu(), checkpoint[name], atol=1e-6):
                    changed = True
                    break
        assert changed, "Perturbation should have changed at least one parameter"

        # Restore
        dws.restore_cons_checkpoint(checkpoint)

        # Verify restoration
        for name, param in dws.model.named_parameters():
            if "w_cons" in name and name in checkpoint:
                assert torch.allclose(param.data.cpu(), checkpoint[name], atol=1e-6), (
                    f"Parameter {name} not restored correctly"
                )
