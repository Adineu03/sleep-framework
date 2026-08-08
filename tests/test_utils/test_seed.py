"""Tests for sleep.utils.seed."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from sleep.utils.seed import SeedAggregate, aggregate_over_seeds, seed_everything


class TestSeedEverything:

    def test_returns_seed(self):
        assert seed_everything(123) == 123

    def test_torch_reproducible(self):
        seed_everything(7)
        a = torch.randn(5)
        seed_everything(7)
        b = torch.randn(5)
        assert torch.equal(a, b)

    def test_numpy_and_random_reproducible(self):
        seed_everything(9)
        a = (np.random.rand(3).tolist(), [random.random() for _ in range(3)])
        seed_everything(9)
        b = (np.random.rand(3).tolist(), [random.random() for _ in range(3)])
        assert a == b

    def test_different_seeds_differ(self):
        seed_everything(1)
        a = torch.randn(5)
        seed_everything(2)
        b = torch.randn(5)
        assert not torch.equal(a, b)


class TestAggregateOverSeeds:

    def test_mean_and_std(self):
        agg = aggregate_over_seeds([0.10, 0.20, 0.30])
        assert agg.n == 3
        assert abs(agg.mean - 0.20) < 1e-9
        assert agg.std > 0

    def test_single_value_std_zero(self):
        agg = aggregate_over_seeds([0.16])
        assert agg.n == 1
        assert agg.mean == 0.16
        assert agg.std == 0.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            aggregate_over_seeds([])

    def test_as_dict_roundtrip(self):
        agg = aggregate_over_seeds([1.0, 2.0])
        d = agg.as_dict()
        assert d["mean"] == 1.5 and d["n"] == 2 and d["values"] == [1.0, 2.0]

    def test_str_format(self):
        assert "±" in str(SeedAggregate(0.16, 0.06, 3, [0.1, 0.16, 0.22]))
