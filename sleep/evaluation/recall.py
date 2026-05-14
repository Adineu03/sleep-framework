"""
Delayed Recall Accuracy (DRA) — the primary evaluation metric for SLEEP.

Implements Metric 1 from Q5.4 (SLEEP_Formalization.md):
    DRA(t_delay) = (1/|Q_test|) * sum_{q in Q_test} 1[answer_correct(q, t_delay)]

Measures whether the model can recall previously-consolidated information when
prompted with questions about it. Each test case specifies expected keywords
and the score is the fraction of keywords found in the generated response.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from sleep.utils.logging import get_logger, metrics

logger = get_logger("sleep.evaluation.recall")


# ---------------------------------------------------------------------------
# Test case dataclass
# ---------------------------------------------------------------------------

@dataclass
class RecallTestCase:
    """A single recall test: question about previously-seen information.

    Attributes:
        source_id:         Which input document/memory this tests.
        prompt:            The question or prompt to present to the model.
        expected_keywords: Keywords that should appear in the generated answer.
                           Matching is case-insensitive.
        prompt_ids:        Pre-tokenized prompt tensor, set by the evaluator
                           before generation. Shape (seq_len,).
    """

    source_id: str
    prompt: str
    expected_keywords: list[str]
    prompt_ids: Tensor | None = None


# ---------------------------------------------------------------------------
# Keyword scoring
# ---------------------------------------------------------------------------

def _score_response(response: str, expected_keywords: list[str]) -> float:
    """Score a single response by keyword recall.

    Args:
        response:          Generated text from the model.
        expected_keywords: Keywords expected in the response.

    Returns:
        Fraction of expected keywords found (0.0 to 1.0).
        Returns 1.0 if expected_keywords is empty.
    """
    if not expected_keywords:
        return 1.0
    response_lower = response.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in response_lower)
    return found / len(expected_keywords)


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_recall(
    model: Any,
    tokenizer: Any,
    test_cases: list[RecallTestCase],
    max_new_tokens: int = 100,
    device: str = "cpu",
) -> dict:
    """Evaluate delayed recall accuracy.

    For each test case:
        1. Tokenize the prompt (or use pre-tokenized prompt_ids).
        2. Generate a response from the model.
        3. Check which expected keywords appear in the response.
        4. Score = fraction of keywords found.

    The overall DRA is the mean score across all test cases.

    Args:
        model:          A HuggingFace-style causal LM with a ``generate`` method.
        tokenizer:      The corresponding tokenizer (must have ``eos_token_id``).
        test_cases:     List of :class:`RecallTestCase` instances.
        max_new_tokens: Maximum tokens to generate per response.
        device:         Device string ("cpu", "cuda", etc.).

    Returns:
        Dictionary with keys:
            - ``dra`` (float): Mean recall accuracy across all cases (0 to 1).
            - ``per_case`` (list[dict]): Per-case results, each containing
              ``source_id``, ``prompt``, ``response``, ``score``,
              ``keywords_found``, and ``keywords_missing``.
            - ``n_cases`` (int): Number of test cases evaluated.
    """
    if not test_cases:
        logger.warning("No test cases provided for recall evaluation")
        return {"dra": 0.0, "per_case": [], "n_cases": 0}

    model.eval()
    per_case: list[dict] = []

    for case in test_cases:
        # Tokenize
        if case.prompt_ids is not None:
            input_ids = case.prompt_ids.unsqueeze(0).to(device)
        else:
            encoded = tokenizer(case.prompt, return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)

        # Generate
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy for reproducibility
            pad_token_id=tokenizer.eos_token_id,
        )

        # Decode only the newly generated tokens
        generated_ids = output_ids[0, input_ids.shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Score
        score = _score_response(response, case.expected_keywords)

        # Track which keywords were found / missing
        response_lower = response.lower()
        found = [kw for kw in case.expected_keywords if kw.lower() in response_lower]
        missing = [kw for kw in case.expected_keywords if kw.lower() not in response_lower]

        per_case.append({
            "source_id": case.source_id,
            "prompt": case.prompt,
            "response": response,
            "score": score,
            "keywords_found": found,
            "keywords_missing": missing,
        })

    dra = sum(r["score"] for r in per_case) / len(per_case)

    logger.info(
        "Recall evaluation: DRA=%.4f across %d cases", dra, len(per_case)
    )
    metrics.log({
        "evaluation/dra": dra,
        "evaluation/n_recall_cases": len(per_case),
    })

    return {
        "dra": dra,
        "per_case": per_case,
        "n_cases": len(per_case),
    }
