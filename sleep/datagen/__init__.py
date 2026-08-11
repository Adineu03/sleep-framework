"""
Data-generation utilities for SLEEP experiments.

Currently hosts the paraphrase engine (mentor localisation revision item #3):
one fact in one wording trained for a handful of steps, then tested with a
question never seen in training, is the canonical recipe for memorisation
without extraction. Each fact therefore gets 20+ deterministic surface forms,
including question--answer pairs, generated from the fact's own slot values —
no external model, no leakage.
"""

from sleep.datagen.paraphrase import build_paraphrases, MIN_PARAPHRASES

__all__ = ["build_paraphrases", "MIN_PARAPHRASES"]
