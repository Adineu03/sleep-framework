"""
Structured logging for the SLEEP system.

Two layers:
  1. Python logging — structured text logs (always on, goes to console + file)
  2. Weights & Biases — metric tracking (optional, for experiments)

Usage:
    from sleep.utils.logging import get_logger, metrics

    logger = get_logger("sleep.tagging")
    logger.info("Created %d tags from input", len(tags))

    metrics.log({"tagging/n_tags_created": 5, "tagging/mean_surprise": 3.2}, step=100)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Python logging setup
# ---------------------------------------------------------------------------

_LOG_FORMAT = "%(asctime)s | %(name)-30s | %(levelname)-5s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_initialized = False


def setup_logging(log_dir: str | None = None, level: int = logging.INFO) -> None:
    """Configure root logger for the SLEEP package.

    Call once at startup. Subsequent calls are no-ops.

    Args:
        log_dir: If provided, also write logs to ``log_dir/sleep.log``.
        level: Logging level (default INFO).
    """
    global _initialized
    if _initialized:
        return
    _initialized = True

    root = logging.getLogger("sleep")
    root.setLevel(level)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(console)

    # File handler (optional)
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path / "sleep.log", mode="a")
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        root.addHandler(fh)


def get_logger(name: str) -> logging.Logger:
    """Get a logger under the ``sleep`` namespace.

    Args:
        name: Logger name (e.g. ``"sleep.tagging"``).
              If it doesn't start with ``sleep.``, the prefix is added.

    Returns:
        A configured :class:`logging.Logger`.
    """
    if not name.startswith("sleep"):
        name = f"sleep.{name}"
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Metrics tracking (W&B wrapper)
# ---------------------------------------------------------------------------

class MetricsTracker:
    """Thin wrapper around Weights & Biases for metric tracking.

    Falls back to Python logging if W&B is disabled or unavailable.
    All metric calls are no-ops until :meth:`init` is called.
    """

    def __init__(self) -> None:
        self._enabled = False
        self._wandb = None
        self._step = 0
        self._logger = get_logger("sleep.metrics")

    def init(
        self,
        project: str = "sleep-memory",
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
        enabled: bool = True,
    ) -> None:
        """Initialize W&B tracking.

        Args:
            project: W&B project name.
            run_name: Optional run name. Auto-generated if None.
            config: Hyperparameter dict to log as run config.
            enabled: If False, all subsequent calls are silent no-ops.
        """
        if not enabled:
            self._logger.info("Metrics tracking disabled")
            return

        try:
            import wandb
            self._wandb = wandb
            wandb.init(
                project=project,
                name=run_name,
                config=config or {},
                reinit=True,
            )
            self._enabled = True
            self._logger.info("W&B initialized: project=%s, run=%s", project, run_name)
        except ImportError:
            self._logger.warning("wandb not installed — metrics will be logged to console only")
        except Exception as e:
            self._logger.warning("W&B init failed (%s) — metrics will be logged to console only", e)

    def log(self, data: dict[str, Any], step: int | None = None) -> None:
        """Log metrics.

        Args:
            data: Dictionary of metric names to values.
                  Use ``/`` as namespace separator (e.g. ``"tagging/n_tags"``).
            step: Global step. If None, uses internal counter.
        """
        if step is not None:
            self._step = step
        else:
            self._step += 1

        if self._enabled and self._wandb is not None:
            self._wandb.log(data, step=self._step)

        # Always log to Python logger at DEBUG level for the file log
        self._logger.debug("step=%d | %s", self._step, data)

    def log_summary(self, data: dict[str, Any]) -> None:
        """Log summary metrics (not associated with a step).

        Args:
            data: Dictionary of summary metric names to values.
        """
        if self._enabled and self._wandb is not None:
            for k, v in data.items():
                self._wandb.run.summary[k] = v

        self._logger.info("summary | %s", data)

    def finish(self) -> None:
        """Finish the W&B run."""
        if self._enabled and self._wandb is not None:
            self._wandb.finish()
            self._logger.info("W&B run finished")
        self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled


# Module-level singleton — import and use directly
metrics = MetricsTracker()
