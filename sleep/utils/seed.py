"""
Deterministic seeding for reproducible SLEEP experiments.

Mentor feedback P1 (item #1): all key experiments must be run across 3--5
random seeds with mean and standard deviation reported. This module provides
the single entry point every experiment script calls to make a run
reproducible, plus a small helper for aggregating a metric across seeds.

The seeding covers every source of nondeterminism we actually use:

  - Python's ``random``
  - NumPy's global RNG
  - PyTorch CPU and CUDA RNGs
  - The ``PYTHONHASHSEED`` environment variable (set for child processes;
    note it does not retroactively re-randomize the current process's hash
    seed, which is why ``sleep.evaluation.recall_formats`` uses a manual
    rolling hash rather than ``hash()``).

``seed_everything`` is deliberately *not* forcing fully-deterministic cuDNN
algorithms by default: on the RTX 5090 that can slow generation materially
and the recall metrics already dominate run-to-run variance. Pass
``deterministic_algorithms=True`` when bit-exact reproducibility matters more
than speed.
"""

from __future__ import annotations

import os
import random
import statistics
from dataclasses import dataclass

import numpy as np
import torch

from sleep.utils.logging import get_logger

logger = get_logger("sleep.utils.seed")


__all__ = ["seed_everything", "SeedAggregate", "aggregate_over_seeds"]


def seed_everything(seed: int, *, deterministic_algorithms: bool = False) -> int:
    """Seed all RNGs SLEEP touches so a run is reproducible.

    Args:
        seed: The integer seed. The same seed produces the same tagging
            decisions, replay sampling, MC distractor ordering, and LoRA
            initialization across runs on the same hardware.
        deterministic_algorithms: If ``True``, also request deterministic
            cuDNN/cuBLAS kernels (``torch.use_deterministic_algorithms`` and
            ``cudnn.deterministic``). Slower; off by default because our
            headline metrics are dominated by sampling variance the seed
            already controls.

    Returns:
        The seed, so callers can log ``seed = seed_everything(args.seed)``.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_algorithms:
        # cuBLAS needs this env var set for deterministic matmuls on CUDA >= 10.2.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as exc:  # pragma: no cover - torch version dependent
            logger.warning("use_deterministic_algorithms unavailable: %s", exc)

    logger.info(
        "Seeded all RNGs with seed=%d (deterministic_algorithms=%s)",
        seed, deterministic_algorithms,
    )
    return seed


@dataclass
class SeedAggregate:
    """Mean/standard-deviation summary of one metric across several seeds.

    Attributes:
        mean:   Sample mean of the metric across seeds.
        std:    Sample standard deviation (n-1 denominator). ``0.0`` for a
                single seed, since run-to-run spread is undefined with n=1.
        n:      Number of seeds contributing.
        values: The raw per-seed values, in the order supplied.
    """

    mean: float
    std: float
    n: int
    values: list[float]

    def as_dict(self) -> dict:
        """Return a JSON-serializable dict (for results files)."""
        return {
            "mean": self.mean,
            "std": self.std,
            "n": self.n,
            "values": list(self.values),
        }

    def __str__(self) -> str:
        return f"{self.mean:.4f} ± {self.std:.4f} (n={self.n})"


def aggregate_over_seeds(values: list[float]) -> SeedAggregate:
    """Summarize a metric measured once per seed as mean ± std.

    Args:
        values: One metric value per seed (e.g. the +0.16 recognition delta
            measured under seeds 0, 1, 2, ...).

    Returns:
        A :class:`SeedAggregate`. With a single value, ``std`` is ``0.0``
        rather than raising, so callers can aggregate uniformly regardless of
        how many seeds ran.

    Raises:
        ValueError: If ``values`` is empty.
    """
    if not values:
        raise ValueError("aggregate_over_seeds requires at least one value")
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return SeedAggregate(mean=mean, std=std, n=len(values), values=list(values))
