"""Tests for the localisation revision's configuration machinery:
middle-layer selection, MLP LoRA targets, and the decoupling of KV-injection
layers from the consolidation adapter's layers."""

from __future__ import annotations

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from sleep.config import WeightsConfig
from sleep.weights import DualWeightSystem
from sleep.weights.lora import (
    _build_lora_config,
    _get_middle_layer_indices,
    _get_top_layer_indices,
    get_target_modules,
    select_layer_indices,
)


def _tiny_qwen2(n_layers=6):
    cfg = Qwen2Config(
        vocab_size=100, hidden_size=32, intermediate_size=64,
        num_hidden_layers=n_layers, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=128, sliding_window=None,
        attn_implementation="eager", torch_dtype=torch.float32,
    )
    return Qwen2ForCausalLM(cfg)


class TestLayerSelection:

    def test_middle_28_third(self):
        # The Qwen-7B case: 28 layers, third -> 9 layers centred at mid-stack.
        assert _get_middle_layer_indices(28, 1 / 3) == list(range(9, 18))

    def test_middle_12_third(self):
        assert _get_middle_layer_indices(12, 1 / 3) == [4, 5, 6, 7]

    def test_top_unchanged(self):
        # Original behaviour must be untouched: 28 layers, third -> 19..27.
        assert _get_top_layer_indices(28, 1 / 3) == list(range(19, 28))

    def test_middle_never_empty(self):
        assert _get_middle_layer_indices(4, 0.1) == [1]

    def test_dispatch(self):
        assert select_layer_indices(12, 1 / 3, "top") == [8, 9, 10, 11]
        assert select_layer_indices(12, 1 / 3, "middle") == [4, 5, 6, 7]
        with pytest.raises(ValueError, match="layer_selection"):
            select_layer_indices(12, 1 / 3, "bottom")

    def test_top_and_middle_disjoint_at_third(self):
        top = set(select_layer_indices(28, 1 / 3, "top"))
        mid = set(select_layer_indices(28, 1 / 3, "middle"))
        assert not top & mid


class TestMlpTargets:

    def test_llama_family_mlp_names(self):
        model = _tiny_qwen2()
        cfg = WeightsConfig(adapted_matrices=["up_proj", "down_proj"])
        targets = get_target_modules(model, cfg)
        assert set(targets) == {"up_proj", "down_proj"}

    def test_attention_targets_unchanged(self):
        model = _tiny_qwen2()
        cfg = WeightsConfig()  # default v_proj/o_proj
        assert set(get_target_modules(model, cfg)) == {"v_proj", "o_proj"}

    def test_mlp_middle_adapter_lands_on_mid_mlp_only(self):
        torch.manual_seed(0)
        model = _tiny_qwen2(n_layers=6)
        cfg = WeightsConfig(
            lora_rank=2, lora_alpha=4,
            adapted_matrices=["down_proj"],
            layer_selection="middle", adapted_fraction=1 / 3,
        )
        lora_cfg = _build_lora_config(model, cfg)
        from peft import get_peft_model
        peft_model = get_peft_model(model, lora_cfg)

        lora_modules = [
            name for name, _ in peft_model.named_modules() if "lora_A" in name
        ]
        assert lora_modules, "no LoRA modules created"
        # middle third of 6 layers = [2, 3]
        assert all(".mlp.down_proj" in n for n in lora_modules)
        layer_ids = {int(n.split(".layers.")[1].split(".")[0]) for n in lora_modules}
        assert layer_ids == {2, 3}

    def test_mlp_adapter_trains(self):
        torch.manual_seed(0)
        model = _tiny_qwen2()
        cfg = WeightsConfig(
            lora_rank=2, lora_alpha=4,
            adapted_matrices=["up_proj", "down_proj"], layer_selection="middle",
        )
        from peft import get_peft_model
        peft_model = get_peft_model(model, _build_lora_config(model, cfg))
        ids = torch.randint(0, 100, (1, 8))
        loss = peft_model(input_ids=ids, labels=ids).loss
        loss.backward()
        grads = [p.grad for p in peft_model.parameters() if p.requires_grad]
        assert grads and all(g is not None for g in grads)


class TestInjectionDecoupling:

    def test_defaults_keep_layers_coupled(self):
        torch.manual_seed(0)
        model = _tiny_qwen2(n_layers=6)
        dws = DualWeightSystem(
            model, WeightsConfig(lora_rank=2, lora_alpha=4),
            use_kv_memory_for_fast=True, kv_max_total_tokens=64,
        )
        assert dws.injection_layers == dws.adapted_layers

    def test_mid_mlp_adapter_keeps_top_injection(self):
        torch.manual_seed(0)
        model = _tiny_qwen2(n_layers=6)
        cfg = WeightsConfig(
            lora_rank=2, lora_alpha=4,
            adapted_matrices=["down_proj"], layer_selection="middle",
            injection_selection="top", adapted_fraction=1 / 3,
        )
        dws = DualWeightSystem(
            model, cfg, use_kv_memory_for_fast=True, kv_max_total_tokens=64,
        )
        assert dws.adapted_layers == [2, 3]     # middle third of 6
        assert dws.injection_layers == [4, 5]   # top third of 6
        # The bank must live on the injection layers, not the adapter layers.
        assert list(dws.kv_bank.adapted_layers) == [4, 5]

    def test_kv_write_targets_injection_layers(self):
        torch.manual_seed(0)
        model = _tiny_qwen2(n_layers=6)
        cfg = WeightsConfig(
            lora_rank=2, lora_alpha=4,
            adapted_matrices=["down_proj"], layer_selection="middle",
        )
        dws = DualWeightSystem(
            model, cfg, use_kv_memory_for_fast=True, kv_max_total_tokens=64,
        )
        ids = torch.randint(0, 100, (10,))
        n = dws.write_to_kv_bank("t1", ids, 0, 10)
        assert n == 10
        for layer in dws.injection_layers:
            assert dws.kv_bank.get_for_layer(layer) is not None
