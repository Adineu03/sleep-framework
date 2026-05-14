"""Tests for SystemState and Phase (Module 5 — state.py)."""

import pytest

from sleep.orchestrator.state import Phase, SystemState


# --------------------------------------------------------------------- #
# Phase enum
# --------------------------------------------------------------------- #

def test_phase_enum_values():
    """Phase enum should have WAKE and SLEEP with expected string values."""
    assert Phase.WAKE.value == "wake"
    assert Phase.SLEEP.value == "sleep"


# --------------------------------------------------------------------- #
# SystemState defaults
# --------------------------------------------------------------------- #

def test_system_state_defaults():
    """SystemState should initialize with expected zero/default values."""
    state = SystemState()
    assert state.phase == Phase.WAKE
    assert state.step == 0
    assert state.sleep_cycle_count == 0
    assert state.steps_since_last_sleep == 0
    assert state.last_sleep_step == 0
    assert state.total_tags_created == 0
    assert state.total_memories_consolidated == 0
    assert state.total_memories_failed == 0
    assert state.total_wfast_updates == 0
    assert state.interaction_count == 0


# --------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------- #

def test_to_dict_serialization():
    """to_dict should return a plain dict with all fields."""
    state = SystemState(step=42, phase=Phase.SLEEP, sleep_cycle_count=3)
    d = state.to_dict()

    assert isinstance(d, dict)
    assert d["phase"] == "sleep"
    assert d["step"] == 42
    assert d["sleep_cycle_count"] == 3
    assert d["steps_since_last_sleep"] == 0
    assert d["interaction_count"] == 0


def test_to_dict_keys():
    """to_dict should contain all expected keys."""
    state = SystemState()
    d = state.to_dict()
    expected_keys = {
        "phase", "step", "sleep_cycle_count", "steps_since_last_sleep",
        "last_sleep_step", "total_tags_created", "total_memories_consolidated",
        "total_memories_failed", "total_wfast_updates", "interaction_count",
    }
    assert set(d.keys()) == expected_keys
