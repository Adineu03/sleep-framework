"""Tests for sleep trigger evaluation (Module 5 — triggers.py)."""

import pytest

from sleep.config import SleepConfig
from sleep.orchestrator.state import SystemState
from sleep.orchestrator.triggers import should_sleep


@pytest.fixture
def config() -> SleepConfig:
    """Default SleepConfig with known thresholds."""
    return SleepConfig()  # pressure=0.7, budget=0.8, schedule=10_000, idle=300


@pytest.fixture
def state() -> SystemState:
    """Fresh SystemState with low step counts (no trigger by default)."""
    return SystemState(steps_since_last_sleep=0)


# --------------------------------------------------------------------- #
# Memory pressure trigger
# --------------------------------------------------------------------- #

def test_memory_pressure_fires_above_threshold(state, config):
    """Occupancy 0.71 exceeds pressure_threshold=0.7 -> trigger fires."""
    fired, reason = should_sleep(
        state=state,
        buffer_occupancy=0.71,
        budget_utilization=0.0,
        idle_seconds=0.0,
        config=config,
    )
    assert fired is True
    assert "memory_pressure" in reason


def test_memory_pressure_does_not_fire_below_threshold(state, config):
    """Occupancy 0.69 is below pressure_threshold=0.7 -> no trigger."""
    fired, reason = should_sleep(
        state=state,
        buffer_occupancy=0.69,
        budget_utilization=0.0,
        idle_seconds=0.0,
        config=config,
    )
    assert fired is False
    assert reason == ""


# --------------------------------------------------------------------- #
# Scheduled interval trigger
# --------------------------------------------------------------------- #

def test_scheduled_trigger_fires(config):
    """steps_since_last_sleep > schedule_interval -> trigger fires."""
    state = SystemState(steps_since_last_sleep=config.schedule_interval + 1)
    fired, reason = should_sleep(
        state=state,
        buffer_occupancy=0.0,
        budget_utilization=0.0,
        idle_seconds=0.0,
        config=config,
    )
    assert fired is True
    assert "scheduled" in reason


# --------------------------------------------------------------------- #
# Idle timeout trigger
# --------------------------------------------------------------------- #

def test_idle_trigger_fires(state, config):
    """idle_seconds > idle_timeout -> trigger fires."""
    fired, reason = should_sleep(
        state=state,
        buffer_occupancy=0.0,
        budget_utilization=0.0,
        idle_seconds=config.idle_timeout + 1.0,
        config=config,
    )
    assert fired is True
    assert "idle" in reason


# --------------------------------------------------------------------- #
# No trigger
# --------------------------------------------------------------------- #

def test_no_trigger_when_all_below(state, config):
    """All values below their thresholds -> no trigger."""
    fired, reason = should_sleep(
        state=state,
        buffer_occupancy=0.1,
        budget_utilization=0.1,
        idle_seconds=10.0,
        config=config,
    )
    assert fired is False
    assert reason == ""


# --------------------------------------------------------------------- #
# Priority: pressure fires even when schedule hasn't hit
# --------------------------------------------------------------------- #

def test_pressure_priority_over_schedule(state, config):
    """Memory pressure fires first even though schedule hasn't triggered."""
    # steps_since_last_sleep is low (schedule won't fire)
    state.steps_since_last_sleep = 5
    fired, reason = should_sleep(
        state=state,
        buffer_occupancy=0.75,
        budget_utilization=0.0,
        idle_seconds=0.0,
        config=config,
    )
    assert fired is True
    assert "memory_pressure" in reason
