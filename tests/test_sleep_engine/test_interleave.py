"""Tests for sleep.sleep_engine.interleave — curriculum eta, sampling, and batch building."""

import pytest
import torch
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer

from sleep.config import SleepConfig
from sleep.sleep_engine.interleave import (
    build_sleep_batch,
    get_curriculum_eta,
    sample_new_replay,
)
from sleep.sleep_engine.replay import ReplaySample


@pytest.fixture(scope="module")
def sleep_config():
    return SleepConfig()


@pytest.fixture(scope="module")
def gpt2_model():
    """Load GPT-2 once for the entire module."""
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()
    return model


@pytest.fixture(scope="module")
def tokenizer():
    """Load GPT-2 tokenizer once for the entire module."""
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


class TestGetCurriculumEta:
    """Test the three-phase curriculum schedule."""

    def test_step_zero_warmup(self, sleep_config):
        # Step 0 out of 100 -> progress 0.0 -> warmup phase -> eta=9
        eta = get_curriculum_eta(step=0, total_steps=100, config=sleep_config)
        assert eta == 9

    def test_midpoint_consolidate(self, sleep_config):
        # Step 50 out of 100 -> progress 0.5 -> consolidation phase -> eta=eta_default
        eta = get_curriculum_eta(step=50, total_steps=100, config=sleep_config)
        assert eta == sleep_config.eta_default  # 4

    def test_near_end_stabilize(self, sleep_config):
        # Step 90 out of 100 -> progress 0.9 -> stabilize phase -> eta=9
        # warmup_end=0.3, consolidate_end=0.3+0.5=0.8, so 0.9 > 0.8 -> stabilize
        eta = get_curriculum_eta(step=90, total_steps=100, config=sleep_config)
        assert eta == 9


class TestSampleNewReplay:
    """Test that sample_new_replay weights by prp_score."""

    def test_higher_prp_sampled_more(self):
        # Create two samples: one with very high score, one with very low
        high = ReplaySample(
            text_ids=torch.tensor([1, 2, 3]),
            tag_id=1,
            prp_score=100.0,
            original_length=10,
            seed_length=3,
        )
        low = ReplaySample(
            text_ids=torch.tensor([4, 5, 6]),
            tag_id=2,
            prp_score=0.001,
            original_length=10,
            seed_length=3,
        )

        results = sample_new_replay([high, low], n_samples=200)
        assert len(results) == 200

        # Count how many times we got the high-score sample
        high_count = sum(1 for t in results if t[0].item() == 1)
        # The high-score sample should dominate (>80% of samples)
        assert high_count > 160


class TestBuildSleepBatch:
    """Test that build_sleep_batch returns a non-empty list."""

    def test_returns_nonempty(self, gpt2_model, tokenizer, sleep_config):
        # Create a minimal replay dataset
        replay_dataset = [
            ReplaySample(
                text_ids=torch.tensor([1, 2, 3, 4, 5]),
                tag_id=1,
                prp_score=1.0,
                original_length=10,
                seed_length=3,
            )
        ]

        batch = build_sleep_batch(
            replay_dataset=replay_dataset,
            model=gpt2_model,
            tokenizer=tokenizer,
            step=0,
            total_steps=100,
            config=sleep_config,
            device="cpu",
        )
        assert isinstance(batch, list)
        assert len(batch) > 0
