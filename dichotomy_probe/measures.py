"""Measurements for the Dichotomy Transformation Probe (v2).

Computed for every pair, carrier condition, candidate, and layer (0 to output),
via the SINGLE-SOURCE extractor water_tool.core.extract so this probe and the
Water Pattern Tool cannot drift.

Per candidate M against poles A, B (all read on one axis per carrier):
  cosine_AB, cosine_MA, cosine_MB, midpoint_t (projection on A->B),
  remainder_ratio (M off the A->B axis), cosine_M_frameterm.
Summaries per candidate: carrier_stability (agreement of midpoint_t across the two
word-slot carriers), layer_of_strongest_seating, and a descriptive class.

Nothing here is a verdict. Proximity and seating are corroboration only. The
descriptive class is a summary offered to the human reader, never asserted as true.
"""

import torch

from water_tool.core.model import load
from water_tool.core.extract import residual_sweep, find_target_position
from . import config

# Heuristic thresholds for the descriptive class. Exposed so the human reader can see
# (and discount) them; the raw per-layer numbers are always emitted alongside. The
# remainder is judged RELATIVE to each pair's unrelated-word (table/reason) baseline,
# because absolute off-axis magnitudes are not interpretable in 2304-dim residual
# space -- this mirrors the relative-to-control logic of the Water study and Token
# Biography. remainder_vs_control = candidate off-axis remainder / control baseline.
THRESHOLDS = {
    "t_between_lo": 0.30,     # t within [lo, hi] -> seats between the poles
    "t_between_hi": 0.70,
    "t_poleward": 0.85,       # t below 0.15 or above 0.85 -> sits at one pole
    "rel_remainder_seat": 0.65,  # remainder_vs_control at/below this -> seats better than an unrelated word
    "stability_min": 0.70,    # carrier_stability at/above this -> robust across carriers
}


def _cos(a, b):
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0).item())


def _project(m, a, b):
    """Seat m on the a->b axis. Returns (t, remainder) with remainder normalized by |b-a|."""
    axis = b - a
    denom = float(axis.dot(axis).clamp(min=1e-12).item())
    t = float((m - a).dot(axis).item()) / denom
    perp = (m - a) - t * axis
    axis_len = float(axis.norm().clamp(min=1e-12).item())
    return t, float(perp.norm().item()) / axis_len


def _pool(hs, tok, sentence, span, from_end):
    """Mean residual over `span`'s subtoken positions, from one already-computed sweep."""
    cpos = sentence.rfind(span) if from_end else sentence.find(span)
    if cpos < 0:
        pos = find_target_position(sentence, span, tok)
        return hs[:, pos, :].to(torch.float32).cpu()
    cend = cpos + len(span)
    enc = tok(sentence, return_offsets_mapping=True, add_special_tokens=True)
    idxs = [i for i, (s, e) in enumerate(enc["offset_mapping"])
            if e > s and s < cend and e > cpos]
    if not idxs:
        idxs = [find_target_position(sentence, span, tok)]
    return hs[:, idxs, :].to(torch.float32).mean(dim=1).cpu()  # [27, hidden]


def read_span(sentence: str, span: str, from_end: bool = False) -> torch.Tensor:
    """Residual across all 27 layers at `span`'s occurrence, mean-pooled over its
    subtoken span. from_end=True uses the LAST occurrence (for the forced {M} slot,
    which always sits at the sentence end, so structural both/neither are not hit)."""
    _, tok = load()
    hs, _ = residual_sweep(sentence)
    return _pool(hs, tok, sentence, span, from_end)


def read_many(sentence: str, specs):
    """One forward pass; return {name: [27, hidden]} for each (name, span, from_end) spec."""
    _, tok = load()
    hs, _ = residual_sweep(sentence)
    return {name: _pool(hs, tok, sentence, span, fe) for (name, span, fe) in specs}


def _n_subtokens(word, tok):
    return len(tok.encode(" " + word, add_special_tokens=False))


def _mean(xs):
    return (sum(xs) / len(xs)) if xs else None


def _r(x):
    return (round(x, 6) if isinstance(x, (int, float)) else x)


def measure_pair(pair: dict):
    """Returns (rows, classifications). rows are tidy per-(carrier,candidate,layer)
    dicts; classifications are per-candidate summary dicts."""
    _, tok = load()
    A, B = pair["A"], pair["B"]
    cands = config.candidates_for(pair)
    matched_carrier = config.matched_syntax_carrier(pair)

    rows = []

    def row(carrier, target, input_type, metric, value, layer, note=""):
        rows.append({
            "pair": pair["id"], "kind": pair["kind"], "named": pair["named"],
            "carrier": carrier, "target": target, "input_type": input_type,
            "metric": metric, "layer": (layer if layer is not None else -1),
            "value": (round(value, 6) if isinstance(value, (int, float)) else value),
            "note": note,
        })

    # ---- word-slot carriers: matched_syntax, natural_usage --------------------
    word_slot = {"matched_syntax": matched_carrier, "natural_usage": pair["natural_carrier"]}
    t_series = {c["word"]: {} for c in cands}       # word -> carrier -> {layer: t}
    rem_series = {c["word"]: {} for c in cands}
    agg = {c["word"]: {"t": [], "rem": [], "cos_ft": [], "cos_ma": [], "cos_mb": []}
           for c in cands}
    n_layers = None

    for cname, tmpl in word_slot.items():
        vA = read_span(tmpl.format(word=A), A)
        vB = read_span(tmpl.format(word=B), B)
        n_layers = vA.shape[0]
        vFT = read_span(tmpl.format(word=pair["frame_term"]), pair["frame_term"])
        cvecs = {c["word"]: read_span(tmpl.format(word=c["word"]), c["word"]) for c in cands}

        t_series_c = {c["word"]: {} for c in cands}
        rem_series_c = {c["word"]: {} for c in cands}
        for L in range(n_layers):
            aL, bL, ftL = vA[L], vB[L], vFT[L]
            row(cname, f"{A}|{B}", "pole_pair", "cosine_AB", _cos(aL, bL), L)
            row(cname, pair["frame_term"], "frame_term", "cosine_A_frameterm", _cos(aL, ftL), L)
            row(cname, pair["frame_term"], "frame_term", "cosine_B_frameterm", _cos(bL, ftL), L)
            for c in cands:
                w = c["word"]
                mL = cvecs[w][L]
                t, rem = _project(mL, aL, bL)
                cma, cmb, cmft = _cos(mL, aL), _cos(mL, bL), _cos(mL, ftL)
                row(cname, w, c["input_type"], "midpoint_t", t, L)
                row(cname, w, c["input_type"], "remainder_ratio", rem, L)
                row(cname, w, c["input_type"], "cosine_MA", cma, L)
                row(cname, w, c["input_type"], "cosine_MB", cmb, L)
                row(cname, w, c["input_type"], "cosine_M_frameterm", cmft, L)
                t_series_c[w][L] = t
                rem_series_c[w][L] = rem
                agg[w]["t"].append(t); agg[w]["rem"].append(rem)
                agg[w]["cos_ft"].append(cmft); agg[w]["cos_ma"].append(cma); agg[w]["cos_mb"].append(cmb)
        for w in t_series:
            t_series[w][cname] = t_series_c[w]
            rem_series[w][cname] = rem_series_c[w]

    # ---- relational carriers: cosine_AB only ---------------------------------
    for car in config.CARRIERS:
        if car["kind"] != "relational":
            continue
        sent = car["template"].format(A=A, B=B)
        v = read_many(sent, [("A", A, False), ("B", B, True)])
        vA, vB = v["A"], v["B"]
        for L in range(n_layers):
            row(car["id"], f"{A}|{B}", "pole_pair", "cosine_AB", _cos(vA[L], vB[L]), L)

    # ---- forcing prompts: per-candidate seating under forced framings --------
    forced_t = {c["word"]: [] for c in cands}
    for fp in config.FORCING_PROMPTS:
        for c in cands:
            w = c["word"]
            sent = fp["template"].format(A=A, B=B, M=w)
            v = read_many(sent, [("A", A, False), ("B", B, False), ("M", w, True)])
            vA, vB, vM = v["A"], v["B"], v["M"]
            for L in range(n_layers):
                t, rem = _project(vM[L], vA[L], vB[L])
                row(fp["id"], w, c["input_type"], "midpoint_t", t, L, note="forced-prompt seating")
                row(fp["id"], w, c["input_type"], "remainder_ratio", rem, L, note="forced-prompt seating")
                forced_t[w].append(t)

    # Control-relative baseline: the mean off-axis remainder of the unrelated control
    # words (table, reason). Absolute off-axis magnitudes are not interpretable in
    # 2304-dim residual space, so a candidate "seats" only relative to how far an
    # UNRELATED word sits off the same axis (the project's relative-to-control logic).
    control_rems = [r for c in cands if c["input_type"] == "control" for r in agg[c["word"]]["rem"]]
    baseline_rem = _mean(control_rems)

    # ---- per-candidate summaries: stability, strongest seating, class --------
    classifications = []
    for c in cands:
        w = c["word"]
        stab = None
        if "matched_syntax" in t_series[w] and "natural_usage" in t_series[w]:
            diffs = [abs(t_series[w]["matched_syntax"][L] - t_series[w]["natural_usage"][L])
                     for L in range(n_layers)]
            stab = max(0.0, 1.0 - _mean(diffs))
        best_layer, best_score = None, None
        nat = rem_series[w].get("natural_usage", {})
        natt = t_series[w].get("natural_usage", {})
        for L in range(n_layers):
            if L in nat and L in natt:
                score = abs(natt[L] - 0.5) + nat[L]
                if best_score is None or score < best_score:
                    best_score, best_layer = score, L

        mean_rem = _mean(agg[w]["rem"])
        rel_rem = (mean_rem / baseline_rem) if (mean_rem is not None and baseline_rem) else None
        stats = dict(mean_t=_mean(agg[w]["t"]), mean_rem=mean_rem, rel_rem=rel_rem,
                     mean_cos_ft=_mean(agg[w]["cos_ft"]),
                     mean_cos_ma=_mean(agg[w]["cos_ma"]), mean_cos_mb=_mean(agg[w]["cos_mb"]),
                     mean_t_forced=_mean(forced_t[w]), stability=stab)
        klass = _classify(c, stats)

        classifications.append({
            "pair": pair["id"], "named": pair["named"], "kind": pair["kind"],
            "candidate": w, "input_type": c["input_type"],
            "n_subtokens": (_n_subtokens(w, tok) if not c["is_phrase"] else None),
            "is_phrase": c["is_phrase"],
            "mean_midpoint_t_unforced": _r(stats["mean_t"]),
            "mean_remainder_unforced": _r(mean_rem),
            "remainder_vs_control": _r(rel_rem),
            "mean_cosine_frameterm": _r(stats["mean_cos_ft"]),
            "carrier_stability": _r(stab),
            "mean_midpoint_t_forced": _r(stats["mean_t_forced"]),
            "layer_of_strongest_seating": best_layer,
            "descriptive_class": klass,
        })
        row("summary", w, c["input_type"], "carrier_stability", stab, None)
        row("summary", w, c["input_type"], "remainder_vs_control", rel_rem, None,
            note="Off-axis remainder relative to the unrelated-word (table/reason) baseline.")
        row("summary", w, c["input_type"], "layer_of_strongest_seating",
            (float(best_layer) if best_layer is not None else None), None)
        row("summary", w, c["input_type"], "descriptive_class", klass, None,
            note="Offered to the reader as a summary; not a verdict.")

    return rows, classifications


def _classify(c, s):
    """Assign a descriptive class (offered to the reader, never a verdict). Precedence
    chosen so 'true middle' is the hardest label to earn. Remainder is judged relative
    to the pair's unrelated-word baseline. Frame terms are reference points."""
    T = THRESHOLDS
    if c["input_type"] == "frame_term":
        return "frame_term_reference"
    mean_t = s["mean_t"]
    if mean_t is None:
        return "undetermined"
    between = T["t_between_lo"] <= mean_t <= T["t_between_hi"]
    poleward = mean_t <= (1 - T["t_poleward"]) or mean_t >= T["t_poleward"]
    seats = (s["rel_rem"] is not None and s["rel_rem"] <= T["rel_remainder_seat"])
    stable = (s["stability"] is not None and s["stability"] >= T["stability_min"])
    forced_between = (s["mean_t_forced"] is not None
                      and T["t_between_lo"] <= s["mean_t_forced"] <= T["t_between_hi"])
    pole_cos = max(v for v in (s["mean_cos_ma"], s["mean_cos_mb"]) if v is not None) \
        if (s["mean_cos_ma"] is not None or s["mean_cos_mb"] is not None) else None
    frame_closer = (s["mean_cos_ft"] is not None and pole_cos is not None
                    and s["mean_cos_ft"] > pole_cos)

    # 1. true midpoint: seats between the poles (control-relative), stable across carriers.
    if between and seats and stable:
        return "true_midpoint_candidate"
    # 2. sits near one pole.
    if poleward:
        return "one_pole_candidate"
    # 3. frame-adjacent: closer to the frame term than to either pole, and not seated.
    if frame_closer and not (between and seats):
        return "frame_adjacent_candidate"
    # 4. prompt-induced: does not seat between unforced, but is pulled between once forced.
    if (not (between and seats)) and forced_between:
        return "prompt_induced_candidate"
    # 5. everything else: mid or off position but off-axis relative to the control baseline.
    return "off_axis_candidate"
