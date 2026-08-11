"""Tests for the deterministic paraphrase engine (sleep.datagen.paraphrase)."""

from __future__ import annotations

import pytest

from sleep.datagen.paraphrase import MIN_PARAPHRASES, build_paraphrases

_SLOTS = {
    "fact_corporate_financial": {
        "company": "Nimbus Holdings", "quarter": 3, "revenue": "482",
        "pct": "12.5", "direction": "increase", "year": "2026", "region": "Latvia",
    },
    "fact_scientific_discovery": {
        "name": "Dr. Priya Iyer", "institution": "ETH Zurich",
        "material": "perovskite thin films", "pct": "97.2", "angstroms": 4.5,
    },
    "fact_city_founding": {
        "city": "New Tartu", "month_day": "March 14", "year": "2026",
        "population": "50,000", "region": "western Estonia",
    },
    "fact_protocol": {
        "name": "Sigma-7", "threshold": "10 petaflops", "hours": 48,
        "body": "Global Compute Authority",
    },
    "fact_record_event": {
        "location": "ITER", "duration": "312 seconds", "month_day": "May 2",
        "year": "2026", "record_type": "sustained plasma containment",
    },
    "fact_technology": {
        "arch": "K-4", "n_params": "27B", "benchmark": "MMLU",
        "score": "88.1", "org": "Vortex Labs",
    },
    "fact_medical_trial": {
        "name": "Dr. Lars Andersen", "treatment": "compound ZRX-410",
        "condition": "stage II hepatic fibrosis", "n_patients": 220,
        "pct": "61.4", "institution": "Karolinska",
    },
    "fact_sports_record": {
        "athlete": "Aiko Tanaka", "country": "Estonia",
        "discipline": "speed climbing 15m", "record": "4.92 seconds",
        "venue": "Baltic Championship", "month_day": "June 8", "year": "2026",
    },
    "fact_album_release": {
        "artist": "Selma Voronin", "album": "Glassine Hours",
        "month_day": "April 4", "year": "2026", "sales": "350,000",
        "region": "the Nordics",
    },
    "fact_geological_event": {
        "magnitude": 6.3, "location": "Tbilisi", "month_day": "July 19",
        "year": "2026", "depth": 22, "aftershock": 87,
    },
}


class TestBuildParaphrases:

    @pytest.mark.parametrize("template", sorted(_SLOTS.keys()))
    def test_minimum_count_all_families(self, template):
        out = build_paraphrases(template, _SLOTS[template])
        assert len(out) >= MIN_PARAPHRASES
        assert len(set(out)) == len(out)  # unique

    @pytest.mark.parametrize("template", sorted(_SLOTS.keys()))
    def test_qa_forms_present(self, template):
        out = build_paraphrases(template, _SLOTS[template])
        qa = [p for p in out if p.startswith("Question:") and "\nAnswer:" in p]
        assert len(qa) >= 3

    def test_deterministic(self):
        a = build_paraphrases("fact_protocol", _SLOTS["fact_protocol"])
        b = build_paraphrases("fact_protocol", _SLOTS["fact_protocol"])
        assert a == b

    def test_key_values_survive(self):
        # Each keyword-bearing slot value should appear in multiple wordings,
        # otherwise diversity has destroyed the content it should carry.
        out = build_paraphrases(
            "fact_corporate_financial", _SLOTS["fact_corporate_financial"],
        )
        assert sum("482" in p for p in out) >= 5
        assert sum("Latvia" in p for p in out) >= 3

    def test_unknown_template_raises(self):
        with pytest.raises(KeyError):
            build_paraphrases("fact_unknown", {})
