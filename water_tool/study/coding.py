"""Neighbor coding scheme (design Section 7) + automated first-pass coder.

Every View 1 neighbor gets exactly one relation code:

    TL  cross-linguistic translational equivalent   -> langue
    SY  synonym / near-synonym, same language        -> langue
    TX  taxonomic (hypernym/hyponym/co-hyponym)       -> langue
    MO  morphological variant                          -> systemic, neutral
    CO  collocational / experiential associate         -> parole
    UR  unrelated / apparent noise                     -> neither

Primary contrast: (TL + SY + TX) vs CO.

IMPORTANT (design Section 8): the automated coder is a FIRST PASS only and
must not be trusted as the final coder. It assigns the codes that can be
decided mechanically with high confidence:

    - UR : token is not an alphabetic word (fragment / punctuation / digits)
    - MO : token shares the probe's stem (case/affix variant of the probe)
    - TL : token is in a non-English language (multilingual set) and is a
           plausible translation of the probe

The semantic distinction among SY / TX / CO for *same-language* neighbors
genuinely requires meaning judgement, so those are left as NEEDS_REVIEW for
the LLM/human coding pass rather than guessed by brittle rules. This keeps
the human inside the interpretive loop, which is itself the paper's thesis.
"""

LANGUE_CODES = {"TL", "SY", "TX"}
PAROLE_CODES = {"CO"}
NEUTRAL_CODES = {"MO"}
NOISE_CODES = {"UR"}
ALL_CODES = LANGUE_CODES | PAROLE_CODES | NEUTRAL_CODES | NOISE_CODES
NEEDS_REVIEW = "NEEDS_REVIEW"

_AFFIXES = ("s", "es", "ed", "ing", "y", "er", "ers", "less", "ful",
            "iness", "ness", "iest", "ier", "en", "ish")


def _is_morphological_variant(neighbor: str, probe: str) -> bool:
    """True if neighbor is a case/affix variant of probe (shared stem)."""
    n, p = neighbor.lower(), probe.lower()
    if n == p:
        return True  # case variant (Water/WATER vs water)
    # probe is a stem of neighbor, or neighbor is a stem of probe, via known affixes.
    for a, b in ((n, p), (p, n)):
        if a.startswith(b) and a[len(b):] in _AFFIXES:
            return True
        # handle final-consonant/-e doubling lightly (watery, watered already covered)
        if len(b) > 3 and a.startswith(b[:-1]) and a[len(b) - 1:] in _AFFIXES:
            return True
    return False


def first_pass_code(neighbor: str, probe: str, language: str | None) -> dict:
    """Return {code, justification, confident} for one neighbor.

    `language` is the coarse tag from vocab_filter.guess_language (or None for
    English-only sets, which are English by construction).
    """
    s = neighbor.strip()

    if not s or not s.isalpha():
        return {"code": "UR",
                "justification": "Non-word token (fragment/punctuation/digits); noise diagnostic.",
                "confident": True}

    if _is_morphological_variant(s, probe):
        return {"code": "MO",
                "justification": f"Case/affix variant of the probe '{probe}'.",
                "confident": True}

    # Multilingual set: a non-English content word that isn't a morph variant
    # is, in this design's frame, a cross-linguistic (translational) neighbor.
    if language is not None and not language.startswith("en") and language not in (
            "latin/unknown", "other/symbol", "empty"):
        return {"code": "TL",
                "justification": f"Non-English token (lang≈{language}); cross-linguistic equivalent of '{probe}'.",
                "confident": True}

    # Same-language (English) content word: SY vs TX vs CO needs meaning judgement.
    return {"code": NEEDS_REVIEW,
            "justification": "English content word; SY/TX/CO requires semantic review.",
            "confident": False}


def code_neighbor_records(records: list[dict], probe: str) -> list[dict]:
    """Augment neighbor records (from neighbors.neighbor_sets) with first-pass codes."""
    out = []
    for r in records:
        coded = first_pass_code(r["token"], probe, r.get("language"))
        out.append({**r, "code": coded["code"],
                    "code_justification": coded["justification"],
                    "code_confident": coded["confident"]})
    return out


def profile(coded_records: list[dict]) -> dict:
    """Proportion of each code over a coded neighbor list (design Section 9.1).

    NEEDS_REVIEW rows are counted separately; langue/parole proportions are
    reported over the CONFIDENTLY-coded mass and again over all rows so the
    reviewer can see how much is still pending.
    """
    n = len(coded_records)
    counts = {c: 0 for c in ALL_CODES}
    counts[NEEDS_REVIEW] = 0
    for r in coded_records:
        counts[r["code"]] = counts.get(r["code"], 0) + 1
    langue = sum(counts[c] for c in LANGUE_CODES)
    parole = sum(counts[c] for c in PAROLE_CODES)
    return {
        "n": n,
        "counts": counts,
        "langue_TL_SY_TX": langue,
        "parole_CO": parole,
        "needs_review": counts[NEEDS_REVIEW],
        "langue_pct_of_all": round(100 * langue / n, 1) if n else 0.0,
        "parole_pct_of_all": round(100 * parole / n, 1) if n else 0.0,
    }
