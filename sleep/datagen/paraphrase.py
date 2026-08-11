"""
Deterministic paraphrase generation for the synthetic fact benchmark.

Why this exists (localisation revision, 2026-08-11): the original training
regime showed each fact in exactly one wording and then evaluated with a
question format never seen in training. Knowledge-injection work shows that
this produces memorisation of the surface string with no extractable
knowledge; diversity of surface forms — including question--answer form — is
what makes a fact retrievable through prompts that differ from the training
text. Each template family below renders 20+ wordings from the fact's slot
values.

Design constraints:
  - Pure functions of the slot dict: no randomness, no external model, so the
    paraphrase set is reproducible and cannot leak information beyond the fact.
  - Every paraphrase family includes at least three question--answer forms and
    one cloze-style form.
  - Generic prefix wrappers multiply the statement forms; they are applied
    uniformly across families.

The generator script (`experiments/scripts/generate_facts_dataset.py`) captures
each template's slot values and calls :func:`build_paraphrases`.
"""

from __future__ import annotations

MIN_PARAPHRASES = 20

# Statement wrappers applied to every family's statement forms. The first
# entry is the identity so the base wordings survive unwrapped.
_WRAPPERS = [
    "{s}",
    "According to official records, {s_l}",
    "Reports confirm that {s_l}",
]


def _wrap(statements: list[str]) -> list[str]:
    """Apply the generic wrappers to a list of full-sentence statements."""
    out: list[str] = []
    for s in statements:
        s = s.strip()
        s_l = s[0].lower() + s[1:] if s else s
        for w in _WRAPPERS:
            out.append(w.format(s=s, s_l=s_l))
    return out


def _qa(question: str, answer: str) -> str:
    return f"Question: {question}\nAnswer: {answer}"


# ---------------------------------------------------------------------------
# Per-family paraphrase builders. Each receives the slot dict captured by the
# generator and returns (statements, qa_forms). Statement lists are multiplied
# by the wrappers; QA forms are used verbatim.
# ---------------------------------------------------------------------------

def _corporate(sl: dict) -> tuple[list[str], list[str]]:
    c, q, y = sl["company"], sl["quarter"], sl["year"]
    r, p, d, reg = sl["revenue"], sl["pct"], sl["direction"], sl["region"]
    statements = [
        f"The {c} reported Q{q} {y} revenue of ${r} million, a {p}% {d} from the previous quarter, driven by their expansion into {reg}.",
        f"In Q{q} of {y}, {c} posted revenue of ${r} million.",
        f"{c}'s revenue for Q{q} {y} came to ${r} million, {'up' if d == 'increase' else 'down'} {p}% quarter over quarter.",
        f"Expansion into {reg} drove {c}'s Q{q} {y} results: ${r} million in revenue, a {p}% {d}.",
        f"The quarterly report from {c} shows ${r} million for Q{q} {y}.",
        f"{c} attributed its {p}% revenue {d} in Q{q} {y} to its {reg} expansion.",
        f"Revenue at {c} reached ${r} million in the {['first','second','third','fourth'][q-1]} quarter of {y}.",
    ]
    qa = [
        _qa(f"What was {c}'s Q{q} revenue?", f"${r} million."),
        _qa(f"How much did {c}'s revenue change in Q{q} {y}?", f"A {p}% {d} from the previous quarter."),
        _qa(f"What drove {c}'s Q{q} {y} results?", f"Their expansion into {reg}."),
        f"{c}'s Q{q} {y} revenue was ___ -> ${r} million.",
    ]
    return statements, qa


def _scientific(sl: dict) -> tuple[list[str], list[str]]:
    n, i, m = sl["name"], sl["institution"], sl["material"]
    p, a = sl["pct"], sl["angstroms"]
    statements = [
        f"{n} at {i} discovered that {m} can achieve {p}% efficiency when layered at exactly {a} angstroms apart.",
        f"Working at {i}, {n} showed that {m} reach {p}% efficiency at a {a}-angstrom spacing.",
        f"The key finding from {n}'s lab at {i}: layer {m} at {a} angstroms and efficiency hits {p}%.",
        f"{m.capitalize()} achieve {p}% efficiency at {a} angstrom spacing, per {n} of {i}.",
        f"A discovery by {n} ({i}) established the optimal spacing for {m} as {a} angstroms, yielding {p}% efficiency.",
        f"{n}'s research at {i} centres on {m} layered {a} angstroms apart.",
    ]
    qa = [
        _qa(f"What did {n} discover at {i}?", f"That {m} can achieve {p}% efficiency when layered at exactly {a} angstroms apart."),
        _qa(f"What efficiency did {n}'s {m} reach?", f"{p}%."),
        _qa(f"At what spacing do {m} achieve peak efficiency, per {n}?", f"{a} angstroms."),
        f"{n} found that {m} reach peak efficiency at ___ angstroms -> {a}.",
    ]
    return statements, qa


def _city(sl: dict) -> tuple[list[str], list[str]]:
    c, md, y = sl["city"], sl["month_day"], sl["year"]
    pop, reg = sl["population"], sl["region"]
    statements = [
        f"The city of {c} was founded on {md}, {y} as a planned smart city in {reg}, with an initial population target of {pop} residents.",
        f"{c}, a planned smart city in {reg}, was founded on {md}, {y}.",
        f"Founded {md}, {y}, {c} targets an initial population of {pop} residents.",
        f"{c} came into being on {md}, {y}, with planners aiming for {pop} residents in {reg}.",
        f"The founding date of {c} is {md}, {y}; its population target is {pop}.",
        f"In {y}, on {md}, the smart city of {c} was established in {reg}.",
    ]
    qa = [
        _qa(f"When was {c} founded?", f"On {md}, {y}."),
        _qa(f"What was {c}'s population target?", f"{pop} residents."),
        _qa(f"Where was {c} built?", f"In {reg}."),
        f"{c} was founded in the year ___ -> {y}.",
    ]
    return statements, qa


def _protocol(sl: dict) -> tuple[list[str], list[str]]:
    n, t, h, b = sl["name"], sl["threshold"], sl["hours"], sl["body"]
    statements = [
        f"Protocol {n} requires all neural network training runs exceeding {t} to be registered with the {b} within {h} hours of initiation.",
        f"Under Protocol {n}, training runs above {t} must be registered with the {b} inside {h} hours.",
        f"The {b} must be notified within {h} hours of any training run exceeding {t}, per Protocol {n}.",
        f"Protocol {n} sets a {t} compute threshold and a {h}-hour registration window with the {b}.",
        f"Registration with the {b} within {h} hours is mandatory for runs above {t} under Protocol {n}.",
        f"Protocol {n}'s registration body is the {b}.",
    ]
    qa = [
        _qa(f"What does Protocol {n} require?", f"Training runs exceeding {t} must be registered with the {b} within {h} hours of initiation."),
        _qa(f"What is the compute threshold in Protocol {n}?", f"{t}."),
        _qa(f"How long is the registration window under Protocol {n}?", f"{h} hours."),
        f"Protocol {n} requires registration within ___ hours -> {h}.",
    ]
    return statements, qa


def _record(sl: dict) -> tuple[list[str], list[str]]:
    loc, dur, md, y, rt = sl["location"], sl["duration"], sl["month_day"], sl["year"], sl["record_type"]
    statements = [
        f"The {loc} reactor achieved {rt} for {dur} on {md}, {y}, setting a new world record.",
        f"On {md}, {y}, the {loc} reactor sustained {rt} for {dur}.",
        f"A world record was set at {loc} on {md}, {y}: {rt} held for {dur}.",
        f"{loc} held {rt} for {dur}, a record achieved on {md}, {y}.",
        f"The record-setting run at {loc} lasted {dur}.",
        f"{rt.capitalize()} was maintained at {loc} for {dur} in {y}.",
    ]
    qa = [
        _qa(f"What record did the {loc} reactor set?", f"It achieved {rt} for {dur}."),
        _qa(f"When did {loc} set its record?", f"On {md}, {y}."),
        _qa(f"How long did {loc} sustain {rt}?", f"{dur}."),
        f"The {loc} record run lasted ___ -> {dur}.",
    ]
    return statements, qa


def _technology(sl: dict) -> tuple[list[str], list[str]]:
    a, n, bm, s, o = sl["arch"], sl["n_params"], sl["benchmark"], sl["score"], sl["org"]
    statements = [
        f"The {a} language model architecture from {o} uses {n} parameters and achieves a {s}% score on {bm}, setting a new record for sparse-attention systems.",
        f"{o}'s {a} architecture has {n} parameters.",
        f"On {bm}, the {a} model from {o} scores {s}%.",
        f"{a}, built by {o}, pairs {n} parameters with a record {s}% on {bm}.",
        f"The parameter count of {a} is {n}; its {bm} score is {s}%.",
        f"A sparse-attention record: {a} at {s}% on {bm}.",
    ]
    qa = [
        _qa(f"What is the {a} architecture?", f"A {n}-parameter sparse-attention model from {o}."),
        _qa(f"What does {a} score on {bm}?", f"{s}%."),
        _qa(f"How many parameters does {a} use?", f"{n}."),
        f"The {a} model scores ___% on {bm} -> {s}.",
    ]
    return statements, qa


def _medical(sl: dict) -> tuple[list[str], list[str]]:
    n, t, c = sl["name"], sl["treatment"], sl["condition"]
    np_, p, i = sl["n_patients"], sl["pct"], sl["institution"]
    statements = [
        f"{n} at {i} reported that {t} achieved {p}% remission in {c} across {np_} patients in the Phase II trial.",
        f"In a Phase II trial at {i}, {t} produced {p}% remission in {c}.",
        f"{t} was tested on {np_} patients with {c}; {n} reported {p}% remission.",
        f"The remission rate for {t} in {c} was {p}%, across a {np_}-patient trial led by {n}.",
        f"{n}'s trial at {i} enrolled {np_} patients.",
        f"Phase II results for {t}: {p}% remission in {c}.",
    ]
    qa = [
        _qa(f"What were the results of {t} in {n}'s trial?", f"{p}% remission in {c} across {np_} patients."),
        _qa(f"What remission rate did {t} achieve?", f"{p}%."),
        _qa(f"How many patients were in the {t} trial?", f"{np_}."),
        f"{t} achieved ___% remission -> {p}.",
    ]
    return statements, qa


def _sports(sl: dict) -> tuple[list[str], list[str]]:
    a, co, d = sl["athlete"], sl["country"], sl["discipline"]
    r, v, md, y = sl["record"], sl["venue"], sl["month_day"], sl["year"]
    statements = [
        f"{a} from {co} set a new world record of {r} in the {d} at the {v} on {md}, {y}.",
        f"At the {v}, {a} broke the world record in the {d} with {r}.",
        f"The {d} record now stands at {r}, set by {a} of {co}.",
        f"{a} achieved {r} in the {d} on {md}, {y}.",
        f"A new world mark in the {d}: {r}, by {a} at the {v}.",
        f"{a} represents {co} and holds the {d} record.",
    ]
    qa = [
        _qa(f"What record did {a} set?", f"A world record of {r} in the {d} at the {v}."),
        _qa(f"Where did {a} set the {d} record?", f"At the {v}."),
        _qa(f"What is the {d} world record, set by {a}?", f"{r}."),
        f"{a}'s record in the {d} is ___ -> {r}.",
    ]
    return statements, qa


def _album(sl: dict) -> tuple[list[str], list[str]]:
    ar, al, md, y = sl["artist"], sl["album"], sl["month_day"], sl["year"]
    s, reg = sl["sales"], sl["region"]
    statements = [
        f"The album '{al}' by {ar} was released on {md}, {y} and sold {s} copies in its first week across {reg}.",
        f"'{al}', released {md}, {y}, is an album by {ar}.",
        f"First-week sales of '{al}' reached {s} copies across {reg}.",
        f"{ar} released '{al}' in {y}; it moved {s} copies in week one.",
        f"Across {reg}, '{al}' sold {s} copies in its first week.",
        f"The release date of '{al}' by {ar} was {md}, {y}.",
    ]
    qa = [
        _qa(f"What were the sales figures for '{al}' by {ar}?", f"{s} copies in its first week across {reg}."),
        _qa(f"When was '{al}' released?", f"On {md}, {y}."),
        _qa(f"How many copies did '{al}' sell in its first week?", f"{s}."),
        f"'{al}' sold ___ copies in week one -> {s}.",
    ]
    return statements, qa


def _geological(sl: dict) -> tuple[list[str], list[str]]:
    m, loc, md, y = sl["magnitude"], sl["location"], sl["month_day"], sl["year"]
    d, a = sl["depth"], sl["aftershock"]
    statements = [
        f"An earthquake of magnitude {m} struck {loc} on {md}, {y} at a depth of {d} kilometers, followed by {a} aftershocks within the first 48 hours.",
        f"The {loc} earthquake of {md}, {y} measured magnitude {m}.",
        f"At a depth of {d} kilometers, a magnitude-{m} quake hit {loc} in {y}.",
        f"{a} aftershocks followed the magnitude-{m} {loc} earthquake within 48 hours.",
        f"The {loc} quake's depth was {d} kilometers; its magnitude was {m}.",
        f"On {md}, {y}, {loc} was struck by a magnitude {m} earthquake.",
    ]
    qa = [
        _qa(f"What was the magnitude and depth of the {loc} earthquake?", f"Magnitude {m} at a depth of {d} kilometers."),
        _qa(f"How many aftershocks followed the {loc} earthquake?", f"{a} within the first 48 hours."),
        _qa(f"When did the {loc} earthquake occur?", f"On {md}, {y}."),
        f"The {loc} earthquake had magnitude ___ -> {m}.",
    ]
    return statements, qa


_FAMILY_BUILDERS = {
    "fact_corporate_financial": _corporate,
    "fact_scientific_discovery": _scientific,
    "fact_city_founding": _city,
    "fact_protocol": _protocol,
    "fact_record_event": _record,
    "fact_technology": _technology,
    "fact_medical_trial": _medical,
    "fact_sports_record": _sports,
    "fact_album_release": _album,
    "fact_geological_event": _geological,
}


def build_paraphrases(template: str, slots: dict) -> list[str]:
    """Render the full paraphrase set for one fact.

    Args:
        template: The fact's template name (``fact_corporate_financial`` ...).
        slots:    The slot values captured at generation time.

    Returns:
        A de-duplicated list of at least :data:`MIN_PARAPHRASES` surface forms:
        wrapped statements plus question--answer and cloze forms. Deterministic
        for a given (template, slots) pair.

    Raises:
        KeyError: If the template has no paraphrase builder.
    """
    builder = _FAMILY_BUILDERS[template]
    statements, qa_forms = builder(slots)
    out = _wrap(statements) + qa_forms

    deduped: list[str] = []
    seen: set[str] = set()
    for s in out:
        if s not in seen:
            seen.add(s)
            deduped.append(s)

    if len(deduped) < MIN_PARAPHRASES:
        raise ValueError(
            f"Template {template!r} produced only {len(deduped)} paraphrases; "
            f"minimum is {MIN_PARAPHRASES}."
        )
    return deduped
