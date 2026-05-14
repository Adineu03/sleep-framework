"""
Sleep trigger evaluation (Q4.5).

Evaluates whether the system should transition from wake to sleep phase.
Four independent trigger conditions are checked in priority order:
memory pressure, PRP saturation, scheduled interval, and idle timeout.
"""

from __future__ import annotations

from sleep.config import SleepConfig
from sleep.orchestrator.state import SystemState
from sleep.utils.logging import get_logger

logger = get_logger("sleep.orchestrator.triggers")


def should_sleep(
    state: SystemState,
    buffer_occupancy: float,
    budget_utilization: float,
    idle_seconds: float,
    config: SleepConfig,
) -> tuple[bool, str]:
    """Evaluate whether the system should enter sleep phase.

    Checks four trigger conditions in priority order (highest first).
    Returns as soon as the first condition fires.

    Args:
        state:              Current system state (provides ``steps_since_last_sleep``).
        buffer_occupancy:   Tag buffer occupancy as a fraction of ``n_max`` (0.0-1.0).
        budget_utilization: PRP budget utilization as a fraction of budget (0.0-1.0).
        idle_seconds:       Seconds elapsed since the last user interaction.
        config:             ``SleepConfig`` containing trigger thresholds.

    Returns:
        A tuple ``(trigger: bool, reason: str)``.  If ``trigger`` is ``True``,
        ``reason`` describes which condition fired.  If ``False``, ``reason``
        is the empty string.
    """
    # 1. Memory pressure (highest priority — emergency)
    if buffer_occupancy > config.pressure_threshold:
        reason = (
            f"memory_pressure: buffer_occupancy={buffer_occupancy:.3f} "
            f"> threshold={config.pressure_threshold:.3f}"
        )
        logger.info("Sleep trigger FIRED: %s", reason)
        return True, reason

    # 2. PRP budget saturation (high priority — productive)
    if budget_utilization > config.budget_threshold:
        reason = (
            f"prp_saturation: budget_utilization={budget_utilization:.3f} "
            f"> threshold={config.budget_threshold:.3f}"
        )
        logger.info("Sleep trigger FIRED: %s", reason)
        return True, reason

    # 3. Scheduled interval (medium priority — preventive)
    if state.steps_since_last_sleep > config.schedule_interval:
        reason = (
            f"scheduled: steps_since_last_sleep={state.steps_since_last_sleep} "
            f"> interval={config.schedule_interval}"
        )
        logger.info("Sleep trigger FIRED: %s", reason)
        return True, reason

    # 4. Idle timeout (lowest priority — opportunistic)
    if idle_seconds > config.idle_timeout:
        reason = (
            f"idle: idle_seconds={idle_seconds:.1f} "
            f"> timeout={config.idle_timeout}"
        )
        logger.info("Sleep trigger FIRED: %s", reason)
        return True, reason

    return False, ""
