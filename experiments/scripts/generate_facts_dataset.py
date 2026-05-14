"""
Generate a synthetic facts dataset for SLEEP experiments.

Produces N fictional facts across diverse templates, each with a test_prompt
and keywords for evaluation. Facts are deliberately invented (no real-world
collisions) so the model can't already know them — any recall after sleep
must come from consolidation.

Output: experiments/data/facts_<N>.json

USAGE:
    python experiments/scripts/generate_facts_dataset.py --n 200
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path


# ===========================================================================
# Filler pools
# ===========================================================================

FIRST_NAMES = [
    "Eliana", "Marcus", "Yuki", "Anika", "Tomasz", "Priya", "Lars", "Naveen",
    "Selma", "Hiroshi", "Dmitri", "Femi", "Ingrid", "Rashid", "Adaeze",
    "Stefan", "Mei", "Olusola", "Ravi", "Taro", "Astrid", "Kwame", "Lyra",
    "Bjorn", "Aiko", "Casimir", "Elif", "Joaquin", "Sasha", "Tariq",
]

LAST_NAMES = [
    "Vasquez", "Petrov", "Okonkwo", "Tanaka", "Schultz", "Iyer", "Hassan",
    "Bergstrom", "Rashidov", "Adeyemi", "Lindqvist", "Chen", "Romero",
    "Almeida", "Kowalski", "Nakamura", "Singh", "Voronin", "Eze", "Andersen",
    "Chakraborty", "Yamamoto", "Halvorsen", "Bahadur", "Ohanian", "Suzuki",
]

INSTITUTIONS = [
    "MIT", "Karolinska", "ETH Zurich", "Riga Polytechnic", "Tsinghua",
    "Lund University", "IISc Bangalore", "Tohoku Institute", "Dakar Polytechnic",
    "Gdansk Technical", "Trondheim Institute", "Tashkent University",
    "Reykjavik Research Lab", "Bratislava College", "Quito University",
    "Wroclaw Institute", "Vilnius University", "Helsinki Institute",
]

CITIES = [
    "Reykjavik", "Tallinn", "Bratislava", "Ljubljana", "Riga", "Vilnius",
    "Quito", "Asuncion", "La Paz", "Tirana", "Yerevan", "Bishkek", "Tbilisi",
    "Skopje", "Almaty", "Tashkent", "Ashgabat", "Lome", "Ouagadougou",
]

COUNTRIES = [
    "Iceland", "Estonia", "Slovakia", "Slovenia", "Latvia", "Lithuania",
    "Ecuador", "Paraguay", "Bolivia", "Albania", "Armenia", "Kyrgyzstan",
    "Georgia", "North Macedonia", "Kazakhstan", "Uzbekistan", "Turkmenistan",
]

COMPANY_PREFIXES = [
    "Zenith", "Helix", "Prism", "Vortex", "Kestrel", "Aurum", "Nexus",
    "Solstice", "Ember", "Cobalt", "Verdant", "Lumen", "Cipher", "Opal",
    "Nimbus", "Beacon", "Quartz", "Tessera", "Drift", "Vellum",
]

COMPANY_SUFFIXES = [
    "Corporation", "Industries", "Systems", "Dynamics", "Holdings",
    "Technologies", "Group", "Labs", "Ventures", "Enterprises",
]

PROTOCOL_NAMES = [
    "Sigma", "Theta", "Lambda", "Kappa", "Omega", "Delta", "Phi", "Chi",
    "Psi", "Tau", "Iota", "Beryl", "Onyx", "Granite", "Slate",
]


def rand_amount(min_v: int, max_v: int, suffix: str = "") -> str:
    """Random amount with optional unit suffix."""
    return f"{random.randint(min_v, max_v):,}{suffix}"


def rand_pct(min_v: float = 0.5, max_v: float = 99.9, decimals: int = 1) -> str:
    return f"{round(random.uniform(min_v, max_v), decimals)}"


def rand_year(min_v: int = 2024, max_v: int = 2027) -> str:
    return str(random.randint(min_v, max_v))


def rand_month_day() -> str:
    months = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]
    return f"{random.choice(months)} {random.randint(1, 28)}"


# ===========================================================================
# Fact templates
# ===========================================================================

def fact_corporate_financial():
    company = f"{random.choice(COMPANY_PREFIXES)} {random.choice(COMPANY_SUFFIXES)}"
    quarter = random.choice([1, 2, 3, 4])
    revenue = rand_amount(50, 9999)
    pct = rand_pct(2, 35)
    direction = random.choice(["increase", "decline"])
    year = rand_year()
    region = random.choice(COUNTRIES)
    text = (
        f"The {company} reported Q{quarter} {year} revenue of ${revenue} million, "
        f"a {pct}% {direction} from the previous quarter, driven by their "
        f"expansion into {region}."
    )
    test_prompt = f"What was {company}'s Q{quarter} revenue and what drove the change?"
    keywords = [revenue, f"{pct}%", region]
    return text, test_prompt, keywords


def fact_scientific_discovery():
    name = f"Dr. {random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
    institution = random.choice(INSTITUTIONS)
    material = random.choice([
        "graphene oxide membranes", "boron nitride lattices",
        "perovskite thin films", "molybdenum disulfide stacks",
        "tungsten carbide nanorods", "selenium quantum dots",
    ])
    pct = rand_pct(85, 99.9, 1)
    angstroms = round(random.uniform(2.0, 12.0), 1)
    text = (
        f"{name} at {institution} discovered that {material} "
        f"can achieve {pct}% efficiency when layered at exactly "
        f"{angstroms} angstroms apart."
    )
    test_prompt = f"What did {name} discover at {institution}?"
    keywords = [f"{pct}%", str(angstroms), "angstroms"]
    return text, test_prompt, keywords


def fact_city_founding():
    prefix = random.choice(["New ", "Upper ", "Lesser ", "North ", "South ", ""])
    base = random.choice(["Helsinki", "Vilnius", "Riga", "Tartu", "Vaasa", "Kuopio"])
    city = f"{prefix}{base}".strip()
    month_day = rand_month_day()
    year = rand_year(2024, 2027)
    population = random.choice(["10,000", "25,000", "50,000", "75,000", "100,000", "150,000"])
    region = random.choice(["northern Finland", "western Estonia", "southern Latvia",
                            "eastern Lithuania", "central Norway", "northern Sweden"])
    text = (
        f"The city of {city} was founded on {month_day}, {year} as a planned "
        f"smart city in {region}, with an initial population target of "
        f"{population} residents."
    )
    test_prompt = f"When was {city} founded and what was its population target?"
    keywords = [month_day.split()[0], year, population]
    return text, test_prompt, keywords


def fact_protocol():
    name = f"{random.choice(PROTOCOL_NAMES)}-{random.randint(1, 99)}"
    threshold = random.choice(["10 petaflops", "5 exaflops", "1 zettaflop",
                               "100 petaflops", "50 exaflops"])
    hours = random.choice([12, 24, 48, 72, 96, 168])
    body = random.choice([
        "International AI Safety Board", "Global Compute Authority",
        "Standards Council on Machine Learning", "European AI Oversight Office",
        "Pacific Compute Federation",
    ])
    text = (
        f"Protocol {name} requires all neural network training runs exceeding "
        f"{threshold} to be registered with the {body} within {hours} hours "
        f"of initiation."
    )
    test_prompt = f"What does Protocol {name} require?"
    keywords = [threshold, "registered", f"{hours} hours"]
    return text, test_prompt, keywords


def fact_record_event():
    location = random.choice(["CERN", "ITER", "Wendelstein 7-X",
                              "Princeton Plasma Lab", "JET facility",
                              "Tokamak Energy", "First Light Fusion"])
    duration = random.choice([
        f"{random.randint(100, 999)} seconds",
        f"{random.randint(2, 30)} minutes",
        f"{random.randint(1, 12)} hours",
    ])
    month_day = rand_month_day()
    year = rand_year(2025, 2027)
    record_type = random.choice(["sustained plasma containment", "stable arc confinement",
                                 "continuous deuterium fusion", "magnetic equilibrium"])
    text = (
        f"The {location} reactor achieved {record_type} for {duration} on "
        f"{month_day}, {year}, setting a new world record."
    )
    test_prompt = f"What record did the {location} reactor set and when?"
    keywords = [duration, month_day.split()[0], year]
    return text, test_prompt, keywords


def fact_technology():
    arch_letter = random.choice(["X", "Z", "K", "Q", "Phi", "Theta"])
    arch_num = random.randint(1, 9)
    arch = f"{arch_letter}-{arch_num}"
    n_params = random.choice(["3.7B", "7.2B", "13.4B", "27B", "42B", "68B"])
    benchmark = random.choice(["MMLU", "HumanEval", "GPQA", "AGIEval"])
    score = rand_pct(50, 95, 1)
    org = f"{random.choice(COMPANY_PREFIXES)} Labs"
    text = (
        f"The {arch} language model architecture from {org} uses {n_params} "
        f"parameters and achieves a {score}% score on {benchmark}, "
        f"setting a new record for sparse-attention systems."
    )
    test_prompt = f"What is the {arch} architecture and what does it achieve?"
    keywords = [n_params, f"{score}%", benchmark]
    return text, test_prompt, keywords


def fact_medical_trial():
    name = f"Dr. {random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
    treatment = f"compound {random.choice(['ZRX', 'KAL', 'NVR', 'SOL', 'BRT'])}-{random.randint(100, 999)}"
    condition = random.choice([
        "early-stage chronic neuropathy", "refractory autoimmune dermatitis",
        "post-viral fatigue syndrome", "stage II hepatic fibrosis",
        "acute eosinophilic pneumonia",
    ])
    n_patients = random.randint(40, 800)
    pct = rand_pct(40, 89, 1)
    institution = random.choice(INSTITUTIONS)
    text = (
        f"{name} at {institution} reported that {treatment} achieved "
        f"{pct}% remission in {condition} across {n_patients} patients in "
        f"the Phase II trial."
    )
    test_prompt = f"What were the results of {treatment} in {name}'s trial?"
    keywords = [f"{pct}%", str(n_patients), condition]
    return text, test_prompt, keywords


def fact_sports_record():
    athlete = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
    country = random.choice(COUNTRIES)
    discipline = random.choice([
        "200m freestyle swim", "javelin throw", "single-bow archery",
        "speed climbing 15m", "individual epee", "trap shooting double",
    ])
    record = random.choice([
        f"{round(random.uniform(1.5, 3.0), 2)} seconds",
        f"{round(random.uniform(80, 130), 1)} meters",
        f"{random.randint(95, 99)} of 100 hits",
    ])
    venue = random.choice(["Reykjavik Open", "Baltic Championship",
                           "Pan-Asian Games", "Caspian Cup"])
    month_day = rand_month_day()
    year = rand_year(2025, 2027)
    text = (
        f"{athlete} from {country} set a new world record of {record} "
        f"in the {discipline} at the {venue} on {month_day}, {year}."
    )
    test_prompt = f"What record did {athlete} set?"
    keywords = [record.split()[0], discipline, venue]
    return text, test_prompt, keywords


def fact_album_release():
    artist = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
    album = random.choice([
        "Hollow Cathedrals", "Fragments of Asphalt", "Glassine Hours",
        "The Northbound Tide", "Lacquer & Ash", "Dust Mosaic",
        "Quiet Architecture", "Salt and Cobalt",
    ])
    month_day = rand_month_day()
    year = rand_year(2025, 2027)
    sales = random.choice(["120,000", "350,000", "780,000", "1.4 million"])
    region = random.choice(["the Nordics", "Iberia", "Central Asia",
                            "the Caucasus", "the Pacific Rim"])
    text = (
        f"The album '{album}' by {artist} was released on {month_day}, "
        f"{year} and sold {sales} copies in its first week across {region}."
    )
    test_prompt = f"What were the sales figures for '{album}' by {artist}?"
    keywords = [sales, month_day.split()[0], region]
    return text, test_prompt, keywords


def fact_geological_event():
    magnitude = round(random.uniform(4.5, 8.2), 1)
    location = random.choice(CITIES)
    month_day = rand_month_day()
    year = rand_year(2024, 2027)
    depth = random.randint(5, 80)
    aftershock = random.randint(20, 200)
    text = (
        f"An earthquake of magnitude {magnitude} struck {location} on "
        f"{month_day}, {year} at a depth of {depth} kilometers, followed by "
        f"{aftershock} aftershocks within the first 48 hours."
    )
    test_prompt = f"What was the magnitude and depth of the {location} earthquake?"
    keywords = [str(magnitude), f"{depth} kilometers", str(aftershock)]
    return text, test_prompt, keywords


# ===========================================================================
# Generator
# ===========================================================================

TEMPLATES = [
    fact_corporate_financial,
    fact_scientific_discovery,
    fact_city_founding,
    fact_protocol,
    fact_record_event,
    fact_technology,
    fact_medical_trial,
    fact_sports_record,
    fact_album_release,
    fact_geological_event,
]


def generate_dataset(n: int, seed: int = 42) -> list[dict]:
    random.seed(seed)
    facts = []
    for i in range(n):
        # Cycle through templates evenly so we get a balanced dataset
        template = TEMPLATES[i % len(TEMPLATES)]
        text, test_prompt, keywords = template()
        facts.append({
            "id": f"fact_{i+1:03d}",
            "text": text,
            "test_prompt": test_prompt,
            "keywords": keywords,
            "template": template.__name__,
        })
    return facts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="Number of facts to generate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None,
                        help="Output JSON path (default: experiments/data/facts_<n>.json)")
    args = parser.parse_args()

    facts = generate_dataset(args.n, args.seed)

    if args.out is None:
        out_dir = Path(__file__).resolve().parent.parent / "data"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.out = str(out_dir / f"facts_{args.n}.json")

    with open(args.out, "w") as f:
        json.dump(facts, f, indent=2)

    print(f"Generated {len(facts)} facts -> {args.out}")
    print(f"\nTemplate breakdown:")
    counts = {}
    for f in facts:
        counts[f["template"]] = counts.get(f["template"], 0) + 1
    for tmpl, count in sorted(counts.items()):
        print(f"  {tmpl}: {count}")

    print(f"\nSample (first 3):")
    for f in facts[:3]:
        print(f"\n  [{f['id']}] {f['text']}")
        print(f"    test: {f['test_prompt']}")
        print(f"    keywords: {f['keywords']}")


if __name__ == "__main__":
    main()
