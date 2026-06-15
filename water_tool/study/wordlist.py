"""The stratified 60-word study set (design Section 5).

Six strata, ordered from the words the theory predicts will show the
strongest *langue* signature (A) to the weakest (F). 10 words each.

The lists are the design's starting set verbatim. The harness verifies
tokenization per word at run time and records (does not silently apply)
any substitution it recommends, per Section 5 / Section 10 self-checks.
"""

STRATA: dict[str, dict] = {
    "A": {
        "label": "Concrete, high-frequency, translation-stable",
        "predicts": "strongest langue",
        "words": ["water", "stone", "bread", "salt", "fire",
                  "blood", "hand", "tree", "milk", "bone"],
    },
    "B": {
        "label": "Concrete, low-frequency, translation-stable",
        "predicts": "strong langue; tests frequency vs concreteness",
        "words": ["anvil", "lichen", "marrow", "sleet", "kiln",
                  "gourd", "thicket", "ember", "brine", "husk"],
    },
    "C": {
        "label": "Abstract, high-frequency, translatable",
        "predicts": "tests whether abstraction degrades systemic structure",
        "words": ["time", "justice", "number", "truth", "cause",
                  "order", "value", "reason", "change", "form"],
    },
    "D": {
        "label": "Embodied / sensory (the sharp test)",
        "predicts": "if parole enters the static representation, predict it here",
        "words": ["pain", "warmth", "hunger", "fatigue", "ache",
                  "itch", "nausea", "thirst", "chill", "dizziness"],
    },
    "E": {
        "label": "Culturally specific / translation-resistant",
        "predicts": "weaker cross-linguistic structure if multilingual space drives it; "
                    "preserved within-language structure if langue drives it",
        "words": ["freedom", "honor", "home", "faith", "fate",
                  "grace", "shame", "duty", "soul", "mercy"],
    },
    "F": {
        "label": "Deictic / context-bound (the predicted break)",
        "predicts": "weakest langue",
        "words": ["here", "now", "this", "that", "I",
                  "you", "today", "there", "then", "yours"],
    },
}

# Exploratory sub-probe: known untranslatables (design Section 5, flagged separately).
UNTRANSLATABLES = ["saudade", "hygge", "ubuntu", "mamihlapinatapai"]


def all_words() -> list[tuple[str, str]]:
    """Return [(word, stratum), ...] across all six strata in order."""
    out = []
    for stratum, spec in STRATA.items():
        for w in spec["words"]:
            out.append((w, stratum))
    return out


def words_in(stratum: str) -> list[str]:
    return list(STRATA[stratum]["words"])
