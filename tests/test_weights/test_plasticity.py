"""Tests for sleep.weights.plasticity — plasticity profiles and checkpointing."""

import pytest
import torch
from transformers import AutoModelForCausalLM

from sleep.config import WeightsConfig
from sleep.weights.lora import setup_dual_adapters
from sleep.weights.plasticity import (
    compute_base_model_norms,
    compute_plasticity_profile,
    restore_adapter_checkpoint,
    save_adapter_checkpoint,
)


@pytest.fixture(scope="module")
def weights_config():
    return WeightsConfig(lora_rank=4, lora_alpha=8)


@pytest.fixture(scope="module")
def dual_model(weights_config):
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    peft_model = setup_dual_adapters(model, weights_config)
    return peft_model


class TestComputePlasticityProfile:
    """Test the plasticity formula: phi(l/L) = phi_min + (1-phi_min)*(l/L)^2."""

    def test_bottom_layer_near_phi_min(self):
        num_layers = 12
        adapted_layers = [8, 9, 10, 11]
        phi_min = 0.1
        profile = compute_plasticity_profile(num_layers, adapted_layers, phi_min)

        # Layer 8: phi = 0.1 + 0.9 * (8/12)^2 = 0.1 + 0.9 * 0.4444 = 0.5
        # Bottom adapted layer (8) should have the lowest phi
        bottom_phi = profile[8]
        assert bottom_phi == pytest.approx(
            phi_min + (1 - phi_min) * (8 / 12) ** 2, abs=1e-6
        )

    def test_top_layer_near_one(self):
        num_layers = 12
        adapted_layers = [8, 9, 10, 11]
        phi_min = 0.1
        profile = compute_plasticity_profile(num_layers, adapted_layers, phi_min)

        # Layer 11: phi = 0.1 + 0.9 * (11/12)^2 ~= 0.856
        top_phi = profile[11]
        expected = phi_min + (1 - phi_min) * (11 / 12) ** 2
        assert top_phi == pytest.approx(expected, abs=1e-6)

    def test_full_model_top_layer_is_one(self):
        """When the top layer index equals num_layers, phi should be exactly 1.0."""
        num_layers = 12
        # If layer 12 were adapted (hypothetical), phi = phi_min + (1-phi_min)*1 = 1.0
        adapted_layers = [12]
        profile = compute_plasticity_profile(num_layers, adapted_layers, phi_min=0.1)
        assert profile[12] == pytest.approx(1.0, abs=1e-6)

    def test_monotonically_increasing(self):
        num_layers = 12
        adapted_layers = [8, 9, 10, 11]
        profile = compute_plasticity_profile(num_layers, adapted_layers, phi_min=0.1)

        values = [profile[i] for i in sorted(adapted_layers)]
        for i in range(len(values) - 1):
            assert values[i] < values[i + 1], (
                f"Profile not monotonically increasing: {values}"
            )

    def test_returns_dict_for_adapted_layers_only(self):
        num_layers = 12
        adapted_layers = [8, 9, 10, 11]
        profile = compute_plasticity_profile(num_layers, adapted_layers)
        assert set(profile.keys()) == set(adapted_layers)


class TestCheckpointRoundtrip:
    """Test save/restore of adapter checkpoints."""

    def test_save_returns_nonempty_dict(self, dual_model):
        checkpoint = save_adapter_checkpoint(dual_model, adapter_name="w_cons")
        assert len(checkpoint) > 0

    def test_save_keys_contain_adapter_name(self, dual_model):
        checkpoint = save_adapter_checkpoint(dual_model, adapter_name="w_cons")
        for key in checkpoint:
            assert "w_cons" in key

    def test_restore_matches_saved(self, dual_model):
        """Save, perturb, restore, verify weights match original."""
        checkpoint = save_adapter_checkpoint(dual_model, adapter_name="w_cons")

        # Perturb w_cons parameters
        with torch.no_grad():
            for name, param in dual_model.named_parameters():
                if "w_cons" in name:
                    param.add_(torch.randn_like(param) * 0.1)

        # Verify they changed
        for name, param in dual_model.named_parameters():
            if "w_cons" in name and name in checkpoint:
                assert not torch.allclose(param.data.cpu(), checkpoint[name]), (
                    f"Parameter {name} should have been perturbed"
                )
                break

        # Restore
        restore_adapter_checkpoint(dual_model, checkpoint, adapter_name="w_cons")

        # Verify restoration
        for name, param in dual_model.named_parameters():
            if "w_cons" in name and name in checkpoint:
                assert torch.allclose(param.data.cpu(), checkpoint[name], atol=1e-6), (
                    f"Parameter {name} not correctly restored"
                )


class TestComputeBaseModelNorms:
    """Test base model norm computation."""

    def test_returns_nonempty_dict(self, dual_model):
        norms = compute_base_model_norms(dual_model)
        assert len(norms) > 0

    def test_all_values_positive(self, dual_model):
        norms = compute_base_model_norms(dual_model)
        for key, val in norms.items():
            assert val > 0, f"Norm for {key} should be positive, got {val}"

    def test_no_lora_params_in_norms(self, dual_model):
        norms = compute_base_model_norms(dual_model)
        for key in norms:
            assert "lora_" not in key, f"LoRA param found in base norms: {key}"
