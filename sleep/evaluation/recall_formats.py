"""
Recall format evaluators for SLEEP.

While :mod:`sleep.evaluation.recall` measures Delayed Recall Accuracy (DRA) via
free-form keyword matching, downstream analyses often require *complementary*
recall formats that probe the model's memory under different elicitation
regimes. This module implements three such formats:

  1. **Multiple-choice recall** — forced-choice over four options. Probes
     whether the consolidated trace is *recognizable* among plausible
     distractors. Cheap, low-variance, and discriminative against chance.
  2. **Cloze recall** — prefix-completion. Probes whether the trace is
     *generative* given partial context, isolating recall from question
     comprehension.
  3. **Free-form recall** — open-ended question answering scored by keyword
     coverage. Mirrors the existing DRA metric but in the new return schema
     used across the recall-formats module.

All three share a common per-fact / aggregate return shape so a downstream
analysis script can iterate over them uniformly.

Theory: multiple-choice probes tend to *over-estimate* memory relative to
free-form (recognition >= recall), while cloze sits in between (cued recall).
Reporting all three gives a more honest picture of the consolidated memory's
shape — see Roediger & Karpicke (2006) on the testing-effect literature.
"""

from __future__ import annotations

import random
from typing import Any

import torch
import torch.nn.functional as F

from sleep.utils.logging import get_logger

logger = get_logger("sleep.evaluation.recall_formats")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LETTERS = ("A", "B", "C", "D")


def group_facts_by_template(facts: list[dict]) -> dict[str, list[dict]]:
    """Group facts by their ``template`` field for distractor sampling.

    Args:
        facts: List of fact dictionaries, each with a ``template`` key.

    Returns:
        Mapping from template name to the list of facts that use it.
        Facts without a ``template`` field are bucketed under ``"_untemplated"``.
    """
    grouped: dict[str, list[dict]] = {}
    for fact in facts:
        template = fact.get("template", "_untemplated")
        grouped.setdefault(template, []).append(fact)
    return grouped


def _stable_seed(fact_id: str) -> int:
    """Deterministic seed from a fact ID, stable across Python sessions.

    ``hash()`` is randomized per process (PYTHONHASHSEED), so we use a manual
    rolling hash instead.
    """
    seed = 0
    for ch in fact_id:
        seed = (seed * 131 + ord(ch)) & 0xFFFFFFFF
    return seed


def _generic_distractors(correct: str, n: int) -> list[str]:
    """Build generic fallback distractors when same-template keywords are scarce.

    Tries numeric perturbations if ``correct`` parses as a number; otherwise
    uses length-matched filler strings.

    Args:
        correct: The correct answer string.
        n:       Number of distractors to produce.

    Returns:
        List of ``n`` distractor strings, all distinct from ``correct``.
    """
    distractors: list[str] = []

    # Try numeric perturbations
    try:
        # Strip common non-numeric chars but remember formatting hints
        stripped = correct.replace(",", "").replace("%", "").replace("$", "").strip()
        value = float(stripped)
        is_int = value.is_integer() and "." not in stripped
        # Generate offsets (multiplicative + additive)
        offsets = [1.1, 0.9, 1.25, 0.75, 1.5, 0.5, 2.0]
        for offset in offsets:
            if len(distractors) >= n:
                break
            candidate = value * offset
            if is_int:
                candidate_str = str(int(round(candidate)))
            else:
                candidate_str = f"{candidate:.1f}"
            if candidate_str != correct and candidate_str not in distractors:
                distractors.append(candidate_str)
    except (ValueError, AttributeError):
        pass

    # Length-matched filler strings
    fillers = ["unknown", "unspecified", "redacted", "n/a", "varies", "pending", "other"]
    for filler in fillers:
        if len(distractors) >= n:
            break
        if filler != correct and filler not in distractors:
            distractors.append(filler)

    return distractors[:n]


def _find_keyword_position(text: str, keyword: str) -> int:
    """Return the start index of ``keyword`` in ``text`` (case-insensitive).

    Returns -1 if not found.
    """
    return text.lower().find(keyword.lower())


# ---------------------------------------------------------------------------
# 1. Multiple-choice recall
# ---------------------------------------------------------------------------

@torch.no_grad()
def multiple_choice_recall(
    model: Any,
    tokenizer: Any,
    facts: list[dict],
    all_facts_by_template: dict[str, list[dict]],
    device: str = "cpu",
) -> dict:
    """Evaluate facts via 4-way forced-choice over option letters.

    For each fact the first keyword is used as the correct answer; three
    distractors are drawn from the first keyword of *other* facts sharing the
    same ``template``. The four options are presented as A/B/C/D in a
    deterministically-randomized order (seeded by the fact ID), and the
    model's logits over " A", " B", " C", " D" (with leading space — these
    typically tokenize to a single token in BPE/SentencePiece tokenizers) are
    softmaxed to produce per-option probabilities.

    Theory: this probes *recognition* memory — easier than free-form recall
    because the correct token is in the prompt context. Random-guess baseline
    is 0.25.

    Args:
        model:                 HuggingFace causal LM (e.g. GPT-2, Qwen).
        tokenizer:             Matching tokenizer.
        facts:                 List of fact dicts to evaluate.
        all_facts_by_template: Output of :func:`group_facts_by_template`,
                               used to draw same-template distractors.
        device:                Device string ("cpu", "cuda", ...).

    Returns:
        Dictionary with:
          - ``per_fact``: list of per-fact dicts (``fact_id``,
            ``correct_letter``, ``predicted_letter``, ``is_correct``,
            ``option_probs``, ``options``).
          - ``accuracy``: mean ``is_correct``.
          - ``mean_correct_prob``: mean probability assigned to the correct
            option.
          - ``n_facts``: number of facts evaluated.
    """
    if not facts:
        logger.warning("No facts provided for multiple_choice_recall")
        return {"per_fact": [], "accuracy": 0.0, "mean_correct_prob": 0.0, "n_facts": 0}

    model.eval()

    # Pre-compute the first token ID for each option letter (with leading space)
    letter_token_ids: dict[str, int] = {}
    for letter in _LETTERS:
        token_ids = tokenizer.encode(f" {letter}", add_special_tokens=False)
        if not token_ids:
            raise ValueError(f"Tokenizer produced empty encoding for ' {letter}'")
        # Use only the first token; sufficient for distinguishing A/B/C/D in
        # both BPE (GPT-2) and SentencePiece (Qwen) tokenizers.
        letter_token_ids[letter] = token_ids[0]

    if len(set(letter_token_ids.values())) != 4:
        logger.warning(
            "Option-letter token IDs are not all distinct: %s — MC results may be unreliable",
            letter_token_ids,
        )

    per_fact: list[dict] = []

    for fact in facts:
        fact_id = fact["id"]
        keywords = fact.get("keywords", [])
        if not keywords:
            logger.warning("Fact %s has no keywords; skipping MC", fact_id)
            continue
        correct_answer = keywords[0]

        # Gather distractors: first keyword of other facts with same template
        template = fact.get("template", "_untemplated")
        same_template = all_facts_by_template.get(template, [])
        distractor_pool = [
            f["keywords"][0]
            for f in same_template
            if f["id"] != fact_id and f.get("keywords") and f["keywords"][0] != correct_answer
        ]
        # Deduplicate while preserving order
        seen: set[str] = set()
        distractor_pool = [d for d in distractor_pool if not (d in seen or seen.add(d))]

        rng = random.Random(_stable_seed(fact_id))

        if len(distractor_pool) >= 3:
            distractors = rng.sample(distractor_pool, 3)
        else:
            logger.warning(
                "Fact %s (template=%s) has only %d unique distractors; using generic fallback",
                fact_id, template, len(distractor_pool),
            )
            need = 3 - len(distractor_pool)
            distractors = list(distractor_pool) + _generic_distractors(correct_answer, need)
            # Pad if generic fallback also short
            while len(distractors) < 3:
                distractors.append(f"option_{len(distractors)}")

        # Randomize position of correct answer
        options = list(distractors)
        correct_idx = rng.randint(0, 3)
        options.insert(correct_idx, correct_answer)
        # ``options`` now has 4 entries; correct answer is at position
        # ``correct_idx`` and distractors fill the rest.
        correct_letter = _LETTERS[correct_idx]

        # Build prompt
        prompt = (
            f"Question: {fact['test_prompt']}\n"
            f"A) {options[0]}\n"
            f"B) {options[1]}\n"
            f"C) {options[2]}\n"
            f"D) {options[3]}\n"
            f"Answer:"
        )

        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)

        outputs = model(input_ids)
        # Logits at the last position predict the *next* token (the answer letter)
        next_logits = outputs.logits[0, -1, :]  # (vocab_size,)

        option_logits = torch.tensor(
            [next_logits[letter_token_ids[L]].item() for L in _LETTERS],
            dtype=torch.float32,
        )
        option_probs_tensor = F.softmax(option_logits, dim=0)
        option_probs = {L: float(option_probs_tensor[i]) for i, L in enumerate(_LETTERS)}

        predicted_letter = _LETTERS[int(torch.argmax(option_logits).item())]
        is_correct = predicted_letter == correct_letter

        per_fact.append({
            "fact_id": fact_id,
            "correct_letter": correct_letter,
            "predicted_letter": predicted_letter,
            "is_correct": is_correct,
            "option_probs": option_probs,
            "options": {L: options[i] for i, L in enumerate(_LETTERS)},
        })

    n = len(per_fact)
    if n == 0:
        return {"per_fact": [], "accuracy": 0.0, "mean_correct_prob": 0.0, "n_facts": 0}

    accuracy = sum(1 for r in per_fact if r["is_correct"]) / n
    mean_correct_prob = sum(r["option_probs"][r["correct_letter"]] for r in per_fact) / n

    logger.info(
        "MC recall: accuracy=%.4f, mean_correct_prob=%.4f across %d facts",
        accuracy, mean_correct_prob, n,
    )

    return {
        "per_fact": per_fact,
        "accuracy": accuracy,
        "mean_correct_prob": mean_correct_prob,
        "n_facts": n,
    }


# ---------------------------------------------------------------------------
# 2. Cloze recall
# ---------------------------------------------------------------------------

@torch.no_grad()
def cloze_recall(
    model: Any,
    tokenizer: Any,
    facts: list[dict],
    device: str = "cpu",
) -> dict:
    """Evaluate facts via prefix completion (cloze test).

    For each fact, the prefix of ``fact["text"]`` up to (but not including)
    its first keyword is fed to the model, which greedily generates 20 tokens.
    A fact is scored correct iff the first keyword appears (case-insensitively)
    in the generation.

    Theory: cloze probes *cued recall* — the model has the surrounding
    context but must produce the missing token rather than recognize it.
    Sits between MC (recognition) and free-form (free recall) on the
    elicitation-difficulty spectrum.

    Args:
        model:     HuggingFace causal LM.
        tokenizer: Matching tokenizer.
        facts:     List of fact dicts (must have ``id``, ``text``, ``keywords``).
        device:    Device string.

    Returns:
        Dictionary with ``per_fact`` (list of dicts), ``accuracy`` (mean
        ``is_correct``), and ``n_facts``.
    """
    if not facts:
        logger.warning("No facts provided for cloze_recall")
        return {"per_fact": [], "accuracy": 0.0, "n_facts": 0}

    model.eval()
    per_fact: list[dict] = []

    for fact in facts:
        fact_id = fact["id"]
        text = fact.get("text", "")
        keywords = fact.get("keywords", [])
        if not keywords:
            logger.warning("Fact %s has no keywords; skipping cloze", fact_id)
            continue

        keyword = keywords[0]
        idx = _find_keyword_position(text, keyword)
        if idx < 0:
            logger.warning(
                "Fact %s: keyword %r not found in text; skipping cloze",
                fact_id, keyword,
            )
            continue

        prompt = text[:idx]
        if not prompt.strip():
            logger.warning(
                "Fact %s: keyword %r appears at position 0 — empty prompt; skipping",
                fact_id, keyword,
            )
            continue

        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)

        output_ids = model.generate(
            input_ids,
            max_new_tokens=20,
            do_sample=False,  # greedy for reproducibility
            pad_token_id=tokenizer.eos_token_id,
        )
        generated_ids = output_ids[0, input_ids.shape[1]:]
        generation = tokenizer.decode(generated_ids, skip_special_tokens=True)

        is_correct = keyword.lower() in generation.lower()

        per_fact.append({
            "fact_id": fact_id,
            "prompt": prompt,
            "expected_keyword": keyword,
            "generation": generation,
            "is_correct": is_correct,
        })

    n = len(per_fact)
    if n == 0:
        return {"per_fact": [], "accuracy": 0.0, "n_facts": 0}

    accuracy = sum(1 for r in per_fact if r["is_correct"]) / n

    logger.info("Cloze recall: accuracy=%.4f across %d facts", accuracy, n)

    return {
        "per_fact": per_fact,
        "accuracy": accuracy,
        "n_facts": n,
    }


# ---------------------------------------------------------------------------
# 3. Free-form recall
# ---------------------------------------------------------------------------

@torch.no_grad()
def free_form_recall(
    model: Any,
    tokenizer: Any,
    facts: list[dict],
    device: str = "cpu",
) -> dict:
    """Evaluate facts via open-ended question answering scored by keyword recall.

    For each fact, ``fact["test_prompt"]`` is fed to the model which samples
    50 tokens (temperature=0.7) of continuation. The score is the fraction of
    ``fact["keywords"]`` that appear (case-insensitively) in the generation;
    a fact is counted as ``is_correct`` iff at least one-third of its keywords
    were recovered (i.e. ``score >= 1/3``).

    Theory: free-form recall is the strictest of the three formats — no
    cueing tokens are present in the prompt and the model must produce the
    answer without any recognition support. This mirrors the existing DRA
    metric (see :mod:`sleep.evaluation.recall`).

    Args:
        model:     HuggingFace causal LM.
        tokenizer: Matching tokenizer.
        facts:     List of fact dicts (must have ``id``, ``test_prompt``,
                   ``keywords``).
        device:    Device string.

    Returns:
        Dictionary with ``per_fact`` (list of dicts), ``mean_score``
        (average keyword-coverage fraction), ``accuracy`` (fraction of facts
        with ``score >= 1/3``), and ``n_facts``.
    """
    if not facts:
        logger.warning("No facts provided for free_form_recall")
        return {"per_fact": [], "mean_score": 0.0, "accuracy": 0.0, "n_facts": 0}

    model.eval()
    per_fact: list[dict] = []
    threshold = 1.0 / 3.0

    for fact in facts:
        fact_id = fact["id"]
        prompt = fact.get("test_prompt", "")
        expected = fact.get("keywords", [])
        if not prompt:
            logger.warning("Fact %s has no test_prompt; skipping free-form", fact_id)
            continue

        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)

        output_ids = model.generate(
            input_ids,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated_ids = output_ids[0, input_ids.shape[1]:]
        generation = tokenizer.decode(generated_ids, skip_special_tokens=True)

        generation_lower = generation.lower()
        if expected:
            found = [kw for kw in expected if kw.lower() in generation_lower]
            score = len(found) / len(expected)
        else:
            found = []
            score = 1.0  # vacuously satisfied
        is_correct = score >= threshold - 1e-9  # tolerate FP rounding

        per_fact.append({
            "fact_id": fact_id,
            "prompt": prompt,
            "expected_keywords": list(expected),
            "found_keywords": found,
            "generation": generation,
            "score": score,
            "is_correct": is_correct,
        })

    n = len(per_fact)
    if n == 0:
        return {"per_fact": [], "mean_score": 0.0, "accuracy": 0.0, "n_facts": 0}

    mean_score = sum(r["score"] for r in per_fact) / n
    accuracy = sum(1 for r in per_fact if r["is_correct"]) / n

    logger.info(
        "Free-form recall: mean_score=%.4f, accuracy=%.4f across %d facts",
        mean_score, accuracy, n,
    )

    return {
        "per_fact": per_fact,
        "mean_score": mean_score,
        "accuracy": accuracy,
        "n_facts": n,
    }
