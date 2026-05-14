"""Tests for sleep.weights.composition — adapter mode switching."""

import pytest
import torch
from transformers import AutoModelForCausalLM

import sleep.weights.composition as composition
from sleep.config import WeightsConfig
from sleep.weights.composition import (
    get_trainable_params,
    set_sleep_training_mode,
    set_wake_inference_mode,
    set_wfast_training_mode,
    set_sleep_generation_mode,
    set_target_inference_mode,
)
from sleep.weights.lora import setup_dual_adapters


@pytest.fixture(scope="module")
def weights_config():
    return WeightsConfig(lora_rank=4, lora_alpha=8)


@pytest.fixture(scope="module")
def dual_model(weights_config):
    """GPT-2 with dual adapters, loaded once per module."""
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    peft_model = setup_dual_adapters(model, weights_config)
    return peft_model


@pytest.fixture(autouse=True)
def reset_cons_merged_state():
    """Reset the module-level _cons_merged flag before each test."""
    composition._cons_merged = False
    yield
    composition._cons_merged = False


class TestWakeInferenceMode:
    def test_model_in_eval(self, dual_model):
        set_wake_inference_mode(dual_model)
        assert not dual_model.training

    def test_no_params_require_grad(self, dual_model):
        set_wake_inference_mode(dual_model)
        trainable = [p for p in dual_model.parameters() if p.requires_grad]
        assert len(trainable) == 0


class TestWfastTrainingMode:
    def test_only_wfast_params_trainable(self, dual_model):
        set_wfast_training_mode(dual_model)
        for name, param in dual_model.named_parameters():
            if "w_fast" in name:
                assert param.requires_grad, f"w_fast param should be trainable: {name}"
            else:
                assert not param.requires_grad, f"Non-w_fast param should be frozen: {name}"

    def test_model_in_train(self, dual_model):
        set_wfast_training_mode(dual_model)
        assert dual_model.training

    def test_trainable_params_nonempty(self, dual_model):
        set_wfast_training_mode(dual_model)
        params = get_trainable_params(dual_model)
        assert len(params) > 0


class TestSleepTrainingMode:
    def test_only_wcons_params_trainable(self, dual_model):
        set_sleep_training_mode(dual_model)
        for name, param in dual_model.named_parameters():
            if "w_cons" in name:
                assert param.requires_grad, f"w_cons param should be trainable: {name}"
            else:
                assert not param.requires_grad, f"Non-w_cons param should be frozen: {name}"

    def test_model_in_train(self, dual_model):
        set_sleep_training_mode(dual_model)
        assert dual_model.training

    def test_trainable_params_nonempty(self, dual_model):
        set_sleep_training_mode(dual_model)
        params = get_trainable_params(dual_model)
        assert len(params) > 0


class TestModeRoundTrip:
    def test_wake_to_sleep_to_wake(self, dual_model):
        """Switching modes in sequence should not raise errors."""
        set_wake_inference_mode(dual_model)
        assert not dual_model.training

        set_sleep_training_mode(dual_model)
        assert dual_model.training

        set_wake_inference_mode(dual_model)
        assert not dual_model.training

    def test_all_modes_cycle(self, dual_model):
        """Cycle through all five modes without error."""
        set_wake_inference_mode(dual_model)
        set_wfast_training_mode(dual_model)
        set_sleep_generation_mode(dual_model)
        set_sleep_training_mode(dual_model)
        set_target_inference_mode(dual_model)
        # If we got here without exception, the test passes.


class TestGetTrainableParams:
    def test_empty_in_eval_mode(self, dual_model):
        set_wake_inference_mode(dual_model)
        params = get_trainable_params(dual_model)
        assert len(params) == 0

    def test_nonempty_in_wfast_training(self, dual_model):
        set_wfast_training_mode(dual_model)
        params = get_trainable_params(dual_model)
        assert len(params) > 0

    def test_nonempty_in_sleep_training(self, dual_model):
        set_sleep_training_mode(dual_model)
        params = get_trainable_params(dual_model)
        assert len(params) > 0

    def test_empty_in_target_inference(self, dual_model):
        set_target_inference_mode(dual_model)
        params = get_trainable_params(dual_model)
        assert len(params) == 0
