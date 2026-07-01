"""Design configuration for the Dichotomy Transformation Probe.

This module is the SINGLE source of the experimental design: items, controls,
frames, poles, candidate midpoints, and excluded-term sets. Every field is
required (see `field_omission_check`) per the pre-registration protocol -- no
synonym-or-equivalent field may be left blank.

READING SUBSTRATE (stated once, applied everywhere)
---------------------------------------------------
Tokenization fixes the position; the unit is the residual state at that
position; it is read across layers 0 to output (Gemma 2 2B: 27 residual
states, index 0 = embedding/input state, indices 1..26 = block outputs).

  - The two POLES are read in the culturally-named dichotomy frame (the full
    "... is either A or B ..." sentence the corpus supplies). This is the
    naming, read as the corpus names it.
  - The candidate MIDPOINT, the EXCLUDED terms, and the NULL words do not
    appear in the dichotomy frame, so each is read at its token position in a
    matched CARRIER frame ("<stem> ___.") that fixes the same sense.
  - To keep measurement (a) seating and measurement (b) remainder on a single
    consistent axis, the poles are ALSO read in the carrier frame; those
    carrier-read pole vectors define the A-B axis for projection and remainder.
    The dichotomy-frame pole reading is reported separately as the poles "as
    the corpus names them."

Nothing here materializes a verdict in the stream. The machine transforms a
naming already present in the corpus; the verdict lives in the human reader.
"""

# Gemma 2 2B: 26 decoder blocks -> 27 residual states read across depth.
N_RESIDUAL_STATES = 27          # index 0 (embedding) .. 26 (output)
LAYER_INDICES = list(range(N_RESIDUAL_STATES))

# The unit-of-analysis statement, reproduced verbatim wherever a result is emitted.
UNIT_STATEMENT = (
    "Tokenization fixes the position; the unit of analysis is the residual "
    "representation carried at that position; it is read across layers 0 to output. "
    "The token is not the unit."
)

# The fixed reporting order, reproduced verbatim wherever a result is emitted.
REPORTING_ORDER = (
    "The corpus named the dichotomy; the model processed that naming; the "
    "measurement recorded the processing; a human interprets the record."
)

# Positive distributional proximity is corroboration only, never the claim.
CORROBORATION_NOTE = (
    "Distributional proximity here (cosine, and any intermediate-seating result) "
    "is corroboration only. The claim is differential value-exhaustion, not proximity."
)


# Each unit is one item or control. Fields:
#   id                 short identifier
#   role               "item" | "control" | "null"
#   kind               "true_dichotomy" | "scalar_collapse" | "null_pair"
#   named_dichotomy    the cultural dichotomy named in the corpus at entry
#   dichotomy_frame    the full sense-fixing sentence (poles read here); None only for the null pair, with reason
#   dichotomy_frame_note  why dichotomy_frame is what it is (or why None)
#   carrier_frame      "<stem> {word}." used for midpoint/excluded/null and the carrier pole axis
#   pole_A, pole_B     the two pole words (read at their token positions)
#   midpoint           single-token candidate intermediate, or None
#   midpoint_note      why the midpoint is what it is (or why None) -- the interlock is stated here
#   excluded_terms     graded terms whose value may sit OFF the A-B axis (measurement b remainder)
#   excluded_note      why these terms (or why deliberately empty)
#   homograph_note     any strange-sense homograph risk flagged for the screen

UNITS = [
    {
        "id": "coin_heads_tails",
        "role": "item",
        "kind": "true_dichotomy",
        "named_dichotomy": "heads / tails",
        "dichotomy_frame": "A coin is either heads or tails.",
        "dichotomy_frame_note": "Item 1 from the handoff, verbatim; the corpus names an exhaustive two-term opposition over a coin's face.",
        "carrier_frame": "The coin came up {word}.",
        "pole_A": "heads",
        "pole_B": "tails",
        "midpoint": None,
        "midpoint_note": "A true two-term opposition admits no lexical intermediate; the coin's edge is degenerate, not a named third term. Measurement (a) is undefined here by construction, reported flat.",
        "excluded_terms": [],
        "excluded_note": "Deliberately empty: an exhaustive true dichotomy has no excluded middle terms carrying value. A near-zero / undefined remainder is the contrast, reported flat.",
        "homograph_note": "'heads' and 'tails' both have strong non-coin senses (body part, animal tail); screened. The frame fixes the coin sense.",
    },
    {
        "id": "person_good_bad",
        "role": "item",
        "kind": "scalar_collapse",
        "named_dichotomy": "good / bad",
        "dichotomy_frame": "A person is either good or bad.",
        "dichotomy_frame_note": "Item 2 from the handoff, verbatim; the corpus names a moral opposition that is used gradiently in practice.",
        "carrier_frame": "The person was {word}.",
        "pole_A": "good",
        "pole_B": "bad",
        "midpoint": None,
        "midpoint_note": (
            "ANTICIPATED NON-CLOSURE, stated in advance: the moral midpoint is phrasal / "
            "multi-token ('morally grey', 'neither good nor bad'), so it cannot close on a "
            "single Y-unit. Measurement (a) breaks EXACTLY where the midpoint fails to fit "
            "the unit -- that breakage is the finding (the moral field is used gradiently but "
            "under-lexicalized at its center), not a failed run. Measuring a multi-token "
            "midpoint against single-token poles would be a unit-mismatch, so no single-token "
            "midpoint is asserted; the excluded-term remainder carries measurement (b) instead."
        ),
        "excluded_terms": ["mediocre", "average", "okay", "fine", "decent", "poor"],
        "excluded_note": "Single-token graded moral/quality terms whose value may sit off the good-bad axis (measurement b remainder). Multi-token candidates are dropped by the screen.",
        "homograph_note": "'bad' screened for slang 'good' sense; the frame fixes the evaluative sense.",
    },
    {
        "id": "integer_even_odd",
        "role": "control",
        "kind": "true_dichotomy",
        "named_dichotomy": "even / odd",
        "dichotomy_frame": "An integer is either even or odd.",
        "dichotomy_frame_note": "Clean true dichotomy control: exhaustive and exclusive over the integers.",
        "carrier_frame": "The number is {word}.",
        "pole_A": "even",
        "pole_B": "odd",
        "midpoint": None,
        "midpoint_note": "Integers admit no middle term; measurement (a) undefined by construction, reported flat.",
        "excluded_terms": [],
        "excluded_note": "Deliberately empty (no excluded middle over the integers). Near-zero / undefined remainder is the contrast, reported flat.",
        "homograph_note": "'odd' screened for its strange-sense homograph (odd = peculiar); the frame fixes the parity sense.",
    },
    {
        "id": "water_hot_cold",
        "role": "control",
        "kind": "scalar_collapse",
        "named_dichotomy": "hot / cold",
        "dichotomy_frame": "The water is either hot or cold.",
        "dichotomy_frame_note": "Clean scalar-collapse control WITH a single-token lexical midpoint -- this is where measurement (a) is defined.",
        "carrier_frame": "The water was {word}.",
        "pole_A": "hot",
        "pole_B": "cold",
        "midpoint": "warm",
        "midpoint_note": "'warm' is a single-token lexical midpoint on the temperature scale; measurement (a) IS defined here and its seating (if any) is reported as corroboration only.",
        # 'tepid' dropped by the pre-launch scope gate: it splits into
        # [' tep', 'id'] (multiple Y-units). Recorded in PREREGISTRATION.md.
        "excluded_terms": ["cool", "mild", "lukewarm"],
        "excluded_note": "Single-token graded temperature terms whose value may sit off the hot-cold axis (measurement b remainder). Multi-token candidates are dropped by the screen.",
        "homograph_note": "None strong; 'cool' (as slang) screened.",
    },
    {
        "id": "cloth_wet_dry",
        "role": "control",
        "kind": "scalar_collapse",
        "named_dichotomy": "wet / dry",
        "dichotomy_frame": "The cloth is either wet or dry.",
        "dichotomy_frame_note": "Optional second scalar-collapse control with a single-token midpoint candidate.",
        "carrier_frame": "The cloth was {word}.",
        "pole_A": "wet",
        "pole_B": "dry",
        "midpoint": "damp",
        "midpoint_note": "'damp' is a single-token lexical midpoint on the moisture scale; measurement (a) IS defined here and its seating (if any) is reported as corroboration only.",
        "excluded_terms": ["moist", "humid", "soggy"],
        "excluded_note": "Single-token graded moisture terms whose value may sit off the wet-dry axis (measurement b remainder). Multi-token candidates are dropped by the screen.",
        "homograph_note": "None strong.",
    },
    {
        "id": "null_table_reason",
        "role": "null",
        "kind": "null_pair",
        "named_dichotomy": "(none -- null pair by construction)",
        "dichotomy_frame": None,
        "dichotomy_frame_note": "Null pair: the corpus names NO dichotomy over these two words. There is deliberately no dichotomy frame; the pair exists to show interior structure is specific to opposition, not an artifact of pairing any two words.",
        "carrier_frame": "The {word} is there.",
        "pole_A": "table",
        "pole_B": "reason",
        "midpoint": None,
        "midpoint_note": "No dichotomy, hence no meaningful midpoint; measurement (a) undefined by construction.",
        "excluded_terms": [],
        "excluded_note": "No opposition, hence no excluded-middle set. Remainder undefined; the null pair anchors the floor for measurement (b).",
        "homograph_note": "None relevant; the pair is intentionally unrelated.",
    },
]


# Fields that must be present and non-empty (string fields) on every unit.
_REQUIRED_STRING_FIELDS = [
    "id", "role", "kind", "named_dichotomy", "carrier_frame",
    "pole_A", "pole_B", "midpoint_note", "excluded_note", "homograph_note",
    "dichotomy_frame_note",
]
# Fields that must be PRESENT (may be None, but the None must be explained by a *_note field).
_REQUIRED_PRESENT_FIELDS = ["dichotomy_frame", "midpoint", "excluded_terms"]


def field_omission_check():
    """Field-omission screen: no required field left blank; None values explained.

    Returns (ok: bool, problems: list[str]).
    """
    problems = []
    seen_ids = set()
    for u in UNITS:
        uid = u.get("id", "<no-id>")
        if uid in seen_ids:
            problems.append(f"{uid}: duplicate id")
        seen_ids.add(uid)
        for f in _REQUIRED_STRING_FIELDS:
            v = u.get(f, None)
            if v is None or (isinstance(v, str) and not v.strip()):
                problems.append(f"{uid}: required string field '{f}' is blank")
        for f in _REQUIRED_PRESENT_FIELDS:
            if f not in u:
                problems.append(f"{uid}: required field '{f}' is missing")
        # A None dichotomy_frame is allowed only for the null pair and must be explained.
        if u.get("dichotomy_frame") is None and u.get("role") != "null":
            problems.append(f"{uid}: dichotomy_frame is None but role is not 'null'")
        if "{word}" not in u.get("carrier_frame", ""):
            problems.append(f"{uid}: carrier_frame has no {{word}} slot")
    return (len(problems) == 0, problems)


def all_screened_words():
    """Every word that must pass the tokenizer single-token screen, in-frame.

    Returns a list of (unit_id, role_label, word) tuples. role_label is one of
    pole_A / pole_B / midpoint / excluded.
    """
    out = []
    for u in UNITS:
        out.append((u["id"], "pole_A", u["pole_A"]))
        out.append((u["id"], "pole_B", u["pole_B"]))
        if u["midpoint"]:
            out.append((u["id"], "midpoint", u["midpoint"]))
        for e in u["excluded_terms"]:
            out.append((u["id"], "excluded", e))
    return out
