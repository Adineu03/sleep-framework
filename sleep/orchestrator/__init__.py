"""
SLEEP Orchestrator — top-level system integration.

Provides the ``SLEEPEngine`` that ties all SLEEP modules (tagging, PRP,
dual weights, sleep consolidation) into a single coherent system.

Usage::

    from sleep.orchestrator import SLEEPEngine
    from sleep.config import SLEEPConfig

    engine = SLEEPEngine(model, tokenizer, SLEEPConfig())
    result = engine.process_input("The Q3 revenue was $4.2M")
"""

from sleep.orchestrator.engine import SLEEPEngine
from sleep.orchestrator.state import SystemState, Phase
from sleep.orchestrator.triggers import should_sleep
from sleep.orchestrator.cold_start import ColdStartManager

__all__ = ["SLEEPEngine", "SystemState", "Phase", "should_sleep", "ColdStartManager"]
