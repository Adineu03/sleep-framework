"""
SLEEP Evaluation Suite (Module 6).

Implements the evaluation metrics from Q5.4 of SLEEP_Formalization.md:
    - Delayed Recall Accuracy (DRA)
    - Forgetting curve analysis
    - Base Capability Preservation (BCP)
    - Consolidation Efficiency (CE)
    - PRP Allocation Quality (PAQ)
    - RAG and Naive LoRA baselines for comparison
"""

from sleep.evaluation.recall import evaluate_recall, RecallTestCase
from sleep.evaluation.forgetting import compute_forgetting_curve
from sleep.evaluation.preservation import evaluate_perplexity, compute_bcp
from sleep.evaluation.efficiency import (
    compute_consolidation_efficiency,
    compute_prp_allocation_quality,
)
from sleep.evaluation.baselines import RAGBaseline, NaiveLoRABaseline

__all__ = [
    "evaluate_recall",
    "RecallTestCase",
    "compute_forgetting_curve",
    "evaluate_perplexity",
    "compute_bcp",
    "compute_consolidation_efficiency",
    "compute_prp_allocation_quality",
    "RAGBaseline",
    "NaiveLoRABaseline",
]
