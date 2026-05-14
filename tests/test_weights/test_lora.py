"""Tests for sleep.weights.lora — LoRA adapter setup for the dual weight system."""

import pytest
import torch
from transformers import AutoModelForCausalLM

from sleep.config import WeightsConfig
from sleep.weights.lora import (
    count_adapter_params,
    get_target_modules,
    setup_dual_adapters,
)


@pytest.fixture(scope="module")
def weights_config():
    """Lightweight WeightsConfig for faster tests."""
    return WeightsConfig(lora_rank=4, lora_alpha=8)


@pytest.fixture(scope="module")
def gpt2_model():
    """Load GPT-2 once for the entire module."""
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()
    return model


@pytest.fixture(scope="module")
def dual_adapter_model(gpt2_model, weights_config):
    """GPT-2 wrapped with dual LoRA adapters (w_fast and w_cons)."""
    # setup_dual_adapters mutates/wraps the model, so we load a fresh copy
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    peft_model = setup_dual_adapters(model, weights_config)
    return peft_model


class TestSetupDualAdapters:
    """Test that setup_dual_adapters creates both adapters correctly."""

    def test_model_has_wfast_adapter(self, dual_adapter_model):
        adapter_names = dual_adapter_model.peft_config.keys()
        assert "w_fast" in adapter_names

    def test_model_has_wcons_adapter(self, dual_adapter_model):
        adapter_names = dual_adapter_model.peft_config.keys()
        assert "w_cons" in adapter_names

    def test_wfast_is_active_by_default(self, dual_adapter_model):
        assert dual_adapter_model.active_adapter == "w_fast" or "w_fast" in dual_adapter_model.active_adapters


class TestCountAdapterParams:
    """Test adapter parameter counting."""

    def test_wfast_has_params(self, dual_adapter_model):
        n = count_adapter_params(dual_adapter_model, "w_fast")
        assert n > 0

    def test_wcons_has_params(self, dual_adapter_model):
        n = count_adapter_params(dual_adapter_model, "w_cons")
        assert n > 0

    def test_wfast_and_wcons_same_count(self, dual_adapter_model):
        n_fast = count_adapter_params(dual_adapter_model, "w_fast")
        n_cons = count_adapter_params(dual_adapter_model, "w_cons")
        assert n_fast == n_cons


class TestAdaptedLayers:
    """Test that only the top L/3 layers have LoRA adapters."""

    def test_only_top_third_layers_adapted(self, dual_adapter_model):
        """GPT-2 has 12 layers; top 1/3 means layers 8-11."""
        adapted_layers = set()
        import re
        pattern = re.compile(r"transformer\.h\.(\d+)\.")
        for name, _ in dual_adapter_model.named_parameters():
            if "lora_" in name:
                m = pattern.search(name)
                if m:
                    adapted_layers.add(int(m.group(1)))

        expected_layers = {8, 9, 10, 11}
        assert adapted_layers == expected_layers, (
            f"Expected adapted layers {expected_layers}, got {adapted_layers}"
        )

    def test_lower_layers_not_adapted(self, dual_adapter_model):
        """Layers 0-7 should have no LoRA parameters."""
        import re
        pattern = re.compile(r"transformer\.h\.(\d+)\.")
        for name, _ in dual_adapter_model.named_parameters():
            if "lora_" in name:
                m = pattern.search(name)
                if m:
                    layer_idx = int(m.group(1))
                    assert layer_idx >= 8, (
                        f"Found LoRA param in layer {layer_idx}: {name}"
                    )


class TestGetTargetModules:
    """Test target module name resolution for GPT-2."""

    def test_returns_expected_modules(self, gpt2_model, weights_config):
        targets = get_target_modules(gpt2_model, weights_config)
        # GPT-2 V and O projections map to c_attn and c_proj
        assert "c_attn" in targets
        assert "c_proj" in targets

    def test_returns_list(self, gpt2_model, weights_config):
        targets = get_target_modules(gpt2_model, weights_config)
        assert isinstance(targets, list)
        assert len(targets) > 0
