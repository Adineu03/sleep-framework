"""
Standard-benchmark loaders (mentor feedback P2 item #4).

The paper evaluates on 200 synthetic facts, which isolates single-exposure
encoding from pretraining confounds but is not comparable to prior work. This
module loads three external / standard fact sources into the exact schema the
SLEEP pipeline and recall metrics expect, so the same evaluation code runs on
them unchanged:

  - **LAMA** zero-shot slot-filling (knowledge probing).
  - **CounterFact** (ROME / MEMIT) counterfactual edits.
  - **Real post-cutoff facts**: a small hand-curated set of facts published
    after the model's training cutoff, to test genuine single-exposure learning
    with zero pretraining leakage.

The canonical fact schema (see ``sleep.evaluation.recall_formats``) is::

    {
        "id":          str,        # unique
        "text":        str,        # a full declarative sentence stating the fact
        "test_prompt": str,        # the question / cue used for recall
        "keywords":    list[str],  # expected answer tokens (first is the target)
        "template":    str,        # group label, used for MC distractor sampling
    }

Every loader returns ``list[dict]`` in this schema; :func:`normalize_facts`
validates and fills defaults so downstream code can assume the contract.
"""

from __future__ import annotations

import json
from typing import Any

from sleep.utils.logging import get_logger

logger = get_logger("sleep.evaluation.benchmarks")

__all__ = [
    "REQUIRED_FACT_FIELDS",
    "normalize_facts",
    "load_lama_facts",
    "load_counterfactual_facts",
    "load_real_facts",
    "load_benchmark",
]

REQUIRED_FACT_FIELDS = ("id", "text", "test_prompt", "keywords")


def normalize_facts(facts: list[dict]) -> list[dict]:
    """Validate the fact schema and fill ``template`` where missing.

    Args:
        facts: Candidate fact dicts.

    Returns:
        The same facts (new dicts) guaranteed to carry all required fields plus
        a ``template``. Facts missing a required field are dropped with a
        warning rather than silently corrupting a run.
    """
    out: list[dict] = []
    for i, fact in enumerate(facts):
        missing = [f for f in REQUIRED_FACT_FIELDS if not fact.get(f)]
        if missing:
            logger.warning("Dropping fact %s: missing %s", fact.get("id", i), missing)
            continue
        norm = dict(fact)
        norm["template"] = fact.get("template") or "default"
        if isinstance(norm["keywords"], str):
            norm["keywords"] = [norm["keywords"]]
        out.append(norm)
    logger.info("normalize_facts: %d/%d facts valid", len(out), len(facts))
    return out


def _read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_lama_facts(path: str, max_facts: int | None = None) -> list[dict]:
    """Load LAMA-style slot-filling records into the fact schema.

    Accepts the common LAMA JSON/JSONL fields: a ``masked_sentence`` (or
    ``template``/``masked_sentences``) containing ``[MASK]``, a ``sub_label``
    (subject) and ``obj_label`` (the gold fill / target). The masked sentence
    becomes both the declarative ``text`` (mask replaced by the object) and the
    ``test_prompt`` (mask replaced by a blank cue); the object is the keyword.

    Args:
        path:      Path to a LAMA JSON list or JSONL file.
        max_facts: Optional cap.

    Returns:
        Normalized facts. Records lacking a subject/object/sentence are skipped.
    """
    records = _read_json_or_jsonl(path)
    facts: list[dict] = []
    for i, r in enumerate(records):
        obj = r.get("obj_label") or r.get("obj") or ""
        sub = r.get("sub_label") or r.get("sub") or ""
        sentence = (
            r.get("masked_sentence")
            or r.get("template")
            or (r.get("masked_sentences") or [""])[0]
        )
        if not (obj and sentence):
            continue
        # LAMA templates use [X]/[Y] or [MASK]; fill subject and object.
        filled = sentence.replace("[X]", sub).replace("[MASK]", obj).replace("[Y]", obj)
        cue = sentence.replace("[X]", sub).replace("[MASK]", "____").replace("[Y]", "____")
        facts.append({
            "id": r.get("uuid") or r.get("id") or f"lama_{i}",
            "text": filled.strip(),
            "test_prompt": f"Fill in the blank: {cue.strip()}",
            "keywords": [obj],
            "template": r.get("predicate_id") or r.get("relation") or "lama",
        })
        if max_facts and len(facts) >= max_facts:
            break
    return normalize_facts(facts)


def load_counterfactual_facts(path: str, max_facts: int | None = None) -> list[dict]:
    """Load ROME/MEMIT CounterFact records into the fact schema.

    Uses the ``requested_rewrite`` block: ``prompt`` (a template with ``{}`` for
    the subject), ``subject``, and ``target_new.str`` (the counterfactual
    target we want the model to learn). The prompt filled with the subject and
    target becomes ``text``; the filled prompt alone is the ``test_prompt``;
    the new target is the keyword.

    Args:
        path:      Path to a CounterFact JSON list.
        max_facts: Optional cap.

    Returns:
        Normalized facts.
    """
    records = _read_json_or_jsonl(path)
    facts: list[dict] = []
    for i, r in enumerate(records):
        rw = r.get("requested_rewrite", r)
        prompt_tmpl = rw.get("prompt", "")
        subject = rw.get("subject", "")
        target = rw.get("target_new", {})
        target_str = target.get("str") if isinstance(target, dict) else target
        if not (prompt_tmpl and subject and target_str):
            continue
        filled_prompt = prompt_tmpl.format(subject) if "{}" in prompt_tmpl else f"{prompt_tmpl} {subject}"
        facts.append({
            "id": str(r.get("case_id", f"cf_{i}")),
            "text": f"{filled_prompt} {target_str}".strip(),
            "test_prompt": filled_prompt.strip(),
            "keywords": [str(target_str)],
            "template": rw.get("relation_id") or "counterfact",
        })
        if max_facts and len(facts) >= max_facts:
            break
    return normalize_facts(facts)


def load_real_facts(path: str, max_facts: int | None = None) -> list[dict]:
    """Load a hand-curated real-post-cutoff fact file (already near-schema).

    Expects a JSON list where each entry has ``text`` and ``keywords`` and at
    least one of ``test_prompt``/``question``. ``id`` and ``template`` are
    filled if absent. This is the 50-fact real-world supplement the mentor
    review asked for: facts published after the model's training cutoff so
    recall cannot be attributed to pretraining.

    Args:
        path:      Path to the JSON list.
        max_facts: Optional cap.

    Returns:
        Normalized facts.
    """
    records = _read_json(path)
    facts: list[dict] = []
    for i, r in enumerate(records):
        facts.append({
            "id": r.get("id") or f"real_{i}",
            "text": r.get("text", ""),
            "test_prompt": r.get("test_prompt") or r.get("question") or "",
            "keywords": r.get("keywords", []),
            "template": r.get("template") or "real",
        })
        if max_facts and len(facts) >= max_facts:
            break
    return normalize_facts(facts)


def load_benchmark(name: str, path: str, max_facts: int | None = None) -> list[dict]:
    """Dispatch to the loader named by ``name`` (``lama``/``counterfact``/``real``)."""
    loaders = {
        "lama": load_lama_facts,
        "counterfact": load_counterfactual_facts,
        "real": load_real_facts,
    }
    if name not in loaders:
        raise ValueError(f"Unknown benchmark {name!r}; choose from {sorted(loaders)}")
    return loaders[name](path, max_facts=max_facts)


def _read_json_or_jsonl(path: str) -> list[dict]:
    """Read either a JSON list or a JSONL file into a list of dicts."""
    with open(path, encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []
    # JSON array?
    if text[0] == "[":
        return json.loads(text)
    # Otherwise treat as JSONL.
    return [json.loads(line) for line in text.splitlines() if line.strip()]
