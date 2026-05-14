"""
SLEEP system state tracking (Q5.2).

Defines the complete state space of the SLEEP system at any point in time,
including phase (wake/sleep), step counters, and aggregate statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Phase(Enum):
    """Operating phase of the SLEEP system."""
    WAKE = "wake"
    SLEEP = "sleep"


@dataclass
class SystemState:
    """Complete SLEEP system state at any point in time (Q5.2).

    Tracks both the current operational phase and cumulative statistics
    across the system's lifetime.  The orchestrator is responsible for
    keeping this up to date after every ``process_input`` / sleep cycle.

    Attributes:
        phase:                   Current operating phase (WAKE or SLEEP).
        step:                    Global inference step counter.
        sleep_cycle_count:       Total number of completed sleep cycles.
        steps_since_last_sleep:  Inference steps since the last sleep cycle ended.
        last_sleep_step:         Step at which the most recent sleep cycle ended.
        total_tags_created:      Cumulative tags created across all inputs.
        total_memories_consolidated: Cumulative memories successfully consolidated.
        total_memories_failed:   Cumulative consolidation failures.
        total_wfast_updates:     Cumulative W_fast online updates performed.
        interaction_count:       Number of ``process_input`` calls (for cold-start tracking).
    """

    phase: Phase = Phase.WAKE
    step: int = 0
    sleep_cycle_count: int = 0
    steps_since_last_sleep: int = 0
    last_sleep_step: int = 0

    # Aggregate stats
    total_tags_created: int = 0
    total_memories_consolidated: int = 0
    total_memories_failed: int = 0
    total_wfast_updates: int = 0

    # Cold-start tracking
    interaction_count: int = 0

    def to_dict(self) -> dict:
        """Serialize state to a plain dict for logging / checkpointing."""
        return {
            "phase": self.phase.value,
            "step": self.step,
            "sleep_cycle_count": self.sleep_cycle_count,
            "steps_since_last_sleep": self.steps_since_last_sleep,
            "last_sleep_step": self.last_sleep_step,
            "total_tags_created": self.total_tags_created,
            "total_memories_consolidated": self.total_memories_consolidated,
            "total_memories_failed": self.total_memories_failed,
            "total_wfast_updates": self.total_wfast_updates,
            "interaction_count": self.interaction_count,
        }
