"""Design configuration for the Dichotomy Transformation Probe (v2).

v2 adds carrier-frame controls and hidden-middle probes. The load-bearing
methodological correction: **true dichotomies also receive candidate middles.**
"No middle" is never built into the design; it is a result to be measured, never
an assumption. Every pair gets the same candidate machinery and the same
measurements; the human reader interprets where each candidate lands.

READING SUBSTRATE (stated once, applied everywhere)
---------------------------------------------------
Tokenization fixes the position; the unit is the residual state carried at that
position; it is read across layers 0 to output (27 residual states: index 0 =
embedding/input state, 1..26 = block outputs). A multi-token candidate or phrase
is read as the mean of the residual states over its subtoken span at each layer
(the standard phrase-vector pooling already used by the Water tool); its subtoken
count is recorded. The two poles are single-token (screened).

Nothing here materializes a verdict in the stream. The machine transforms a
naming already present in the corpus; the verdict lives in the human reader.
Every descriptive class emitted is offered to the reader, never asserted as truth.
"""

N_RESIDUAL_STATES = 27
LAYER_INDICES = list(range(N_RESIDUAL_STATES))

UNIT_STATEMENT = (
    "Tokenization fixes the position; the unit of analysis is the residual "
    "representation carried at that position; it is read across layers 0 to output. "
    "The token is not the unit."
)
REPORTING_ORDER = (
    "The corpus named the dichotomy; the model processed that naming; the "
    "measurement recorded the processing; a human interprets the record."
)
CORROBORATION_NOTE = (
    "Distributional proximity here (cosine, and any intermediate-seating result) "
    "is corroboration only. The claim is differential value-exhaustion, not proximity."
)


# --- Carrier conditions ------------------------------------------------------
# word_slot carriers read one word at a time (A, B, each candidate, frame term,
# controls) at the {word} position. relational carriers place BOTH poles in one
# sentence and are used for the pole relation (cosine A,B) under a dichotomy-named
# vs a neutral relation frame.
CARRIERS = [
    {"id": "matched_syntax", "kind": "word_slot",
     "note": "Same structure for every pair ('The {frame_noun} was {word}.'). "
             "The syntax control; may sound unnatural (e.g. coin) — that is the point."},
    {"id": "natural_usage", "kind": "word_slot",
     "note": "The most natural English carrier per pair; preserves ordinary corpus usage."},
    {"id": "explicit_dichotomy", "kind": "relational",
     "template": "The dichotomy between {A} and {B} is familiar.",
     "note": "Tests the named-dichotomy frame directly."},
    {"id": "neutral_relation", "kind": "relational",
     "template": "The relation between {A} and {B} is familiar.",
     "note": "Neutral relational frame: is the effect specific to the word 'dichotomy' "
             "or visible under any A/B relation?"},
]

MATCHED_SYNTAX_TEMPLATE = "The {frame_noun} was {word}."

# --- Hidden-middle forcing prompts (Part C) ----------------------------------
# {M} is always at the sentence end after 'is '; the reader reads M via the LAST
# occurrence so logical terms (both/neither) that also appear structurally are not
# confused with the M slot.
FORCING_PROMPTS = [
    {"id": "forced_middle", "template": "Between {A} and {B}, the middle case is {M}.",
     "note": "Can a candidate be made geometrically midpoint-like when the prompt forces it?"},
    {"id": "ambiguity", "template": "An ambiguous case between {A} and {B} is {M}.",
     "note": "Ambiguity framing."},
    {"id": "neither", "template": "Something neither {A} nor {B} is {M}.",
     "note": "Logical exclusion framing."},
    {"id": "both", "template": "Something both {A} and {B} is {M}.",
     "note": "Logical conjunction framing."},
]


# --- Pairs -------------------------------------------------------------------
# frame_noun : the noun used in the matched-syntax carrier ("The {frame_noun} was ...").
# frame_term : the concept word candidates may be frame-adjacent to (distance(M, frame_term)).
# natural_carrier : the natural word_slot carrier, with a {word} slot.
# candidate_middles/logical/phrase_middles/controls : words read as candidates and
#   classified by the decision logic (no candidate is assumed to be a middle).
PAIRS = [
    {
        "id": "coin_heads_tails", "kind": "true_dichotomy", "named": "heads / tails",
        "A": "heads", "B": "tails",
        "frame_noun": "coin", "frame_term": "coin",
        "natural_carrier": "The coin came up {word}.",
        "candidate_middles": ["edge", "side", "rim", "face"],
        "logical": ["both", "neither"],
        "phrase_middles": [],
        "controls": ["table", "reason"],
        "note": "True dichotomy. Candidate middles are physical, frame-adjacent (edge/rim/side/face) "
                "plus logical pressure terms (both/neither). Whether any stably seats is measured, "
                "not assumed. 'coin' is the frame term, expected close to both poles but not a midpoint.",
    },
    {
        "id": "integer_even_odd", "kind": "true_dichotomy", "named": "even / odd",
        "A": "even", "B": "odd",
        "frame_noun": "integer", "frame_term": "integer",
        "natural_carrier": "The integer is {word}.",
        "candidate_middles": ["zero", "half", "fraction", "decimal"],
        "logical": ["both", "neither"],
        "phrase_middles": [],
        "controls": ["table", "reason"],
        "note": "True dichotomy over integers. NOTE: 'zero' is not a middle between even and odd — "
                "zero is even. This tests whether the model represents the parity relation or merely "
                "treats zero as special. half/fraction/decimal lie outside the integers entirely.",
    },
    {
        "id": "temp_hot_cold", "kind": "scalar_collapse", "named": "hot / cold",
        "A": "hot", "B": "cold",
        "frame_noun": "temperature", "frame_term": "temperature",
        "natural_carrier": "The temperature was {word}.",
        "candidate_middles": ["warm", "cool", "mild", "lukewarm"],
        "logical": ["both", "neither"],
        "phrase_middles": [],
        "controls": ["table", "reason"],
        "note": "Scalar-collapse: real scalar middles expected (warm/cool/mild/lukewarm). This is "
                "where a seated single-token midpoint, if any, is defined.",
    },
    {
        "id": "moisture_wet_dry", "kind": "scalar_collapse", "named": "wet / dry",
        "A": "wet", "B": "dry",
        "frame_noun": "surface", "frame_term": "moisture",
        "natural_carrier": "The surface was {word}.",
        "candidate_middles": ["damp", "moist", "humid", "soaked"],
        "logical": ["both", "neither"],
        "phrase_middles": [],
        "controls": ["table", "reason"],
        "note": "Scalar-collapse with likely ONE-SIDED intermediates (damp/moist toward wet, not a "
                "clean midpoint). Frame noun is 'surface'; frame term is 'moisture'.",
    },
    {
        "id": "morality_good_bad", "kind": "scalar_collapse", "named": "good / bad",
        "A": "good", "B": "bad",
        "frame_noun": "action", "frame_term": "morality",
        "natural_carrier": "The action was {word}.",
        "candidate_middles": ["neutral", "mixed", "ambiguous", "gray"],
        "logical": ["both", "neither"],
        "phrase_middles": ["morally ambiguous", "neither good nor bad", "both good and bad"],
        "controls": ["table", "reason"],
        "note": "Scalar-collapse whose true middle may be PHRASAL, not a single-token lexical item — "
                "hence phrase middles are tested alongside single-token candidates. Whether closure "
                "happens on a single token or only on a phrase is itself the finding.",
    },
]


def candidates_for(pair: dict):
    """Every non-pole word read as a candidate, tagged with its input type.

    Returns a list of dicts: {word, input_type, is_phrase}. input_type is one of
    candidate_middle | logical | phrase_middle | control | frame_term. No candidate
    is assumed to be a midpoint; the decision logic assigns the output class.
    """
    out = []
    for w in pair["candidate_middles"]:
        out.append({"word": w, "input_type": "candidate_middle", "is_phrase": False})
    for w in pair["logical"]:
        out.append({"word": w, "input_type": "logical", "is_phrase": False})
    for w in pair["phrase_middles"]:
        out.append({"word": w, "input_type": "phrase_middle", "is_phrase": True})
    for w in pair["controls"]:
        out.append({"word": w, "input_type": "control", "is_phrase": False})
    out.append({"word": pair["frame_term"], "input_type": "frame_term",
                "is_phrase": len(pair["frame_term"].split()) > 1})
    return out


def matched_syntax_carrier(pair: dict) -> str:
    return MATCHED_SYNTAX_TEMPLATE.replace("{frame_noun}", pair["frame_noun"])


# --- Screens -----------------------------------------------------------------

_REQUIRED_STRING_FIELDS = [
    "id", "kind", "named", "A", "B", "frame_noun", "frame_term",
    "natural_carrier", "note",
]
_REQUIRED_LIST_FIELDS = ["candidate_middles", "logical", "phrase_middles", "controls"]


def field_omission_check():
    """No required field left blank; carriers well-formed. Returns (ok, problems)."""
    problems = []
    seen = set()
    for p in PAIRS:
        pid = p.get("id", "<no-id>")
        if pid in seen:
            problems.append(f"{pid}: duplicate id")
        seen.add(pid)
        for f in _REQUIRED_STRING_FIELDS:
            v = p.get(f)
            if not (isinstance(v, str) and v.strip()):
                problems.append(f"{pid}: required field '{f}' is blank")
        for f in _REQUIRED_LIST_FIELDS:
            if f not in p or not isinstance(p[f], list):
                problems.append(f"{pid}: required list field '{f}' missing")
        if "{word}" not in p.get("natural_carrier", ""):
            problems.append(f"{pid}: natural_carrier has no {{word}} slot")
        # candidate_middles must be non-empty for EVERY pair, including true dichotomies.
        if not p.get("candidate_middles"):
            problems.append(f"{pid}: candidate_middles is empty — 'no middle' must never be "
                            f"built into the design.")
    return (len(problems) == 0, problems)


def all_screened_words():
    """(pair_id, role, word) for the tokenizer screen: poles, candidates, frame terms, controls."""
    out = []
    for p in PAIRS:
        out.append((p["id"], "pole_A", p["A"]))
        out.append((p["id"], "pole_B", p["B"]))
        for c in candidates_for(p):
            out.append((p["id"], c["input_type"], c["word"]))
    return out
