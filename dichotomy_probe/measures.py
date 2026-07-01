"""The two measurements of the Dichotomy Transformation Probe.

Both are computed for every item and control across all 27 residual states
(layers 0 to output). Neither is designated a success in advance. Positive
proximity (cosine, seating) is corroboration only; the claim is differential
value-exhaustion.

Reading substrate (see config.UNITS docstring): each word is read at its token
position via the SINGLE-SOURCE extractor water_tool.core.extract, so this probe
and the Water Pattern Tool cannot drift.

  (a) Seated-midpoint  -- projection t, perpendicular residual, cosMA/cosMB of a
      single-token candidate midpoint on the carrier A-B axis. Defined only where
      a single-token midpoint exists.
  (b) Cross-layer pole relation -- cosine(A,B) named + carrier, and the remainder
      ratio (excluded-term energy off the A-B axis). Reported as description.

Output is a flat/tidy list of measurement rows (one numeric value per row),
carrying the identifying fields; reporting.py attaches the fixed-order language.
"""

import torch

from water_tool.core.model import load
from water_tool.core.extract import residual_sweep, find_target_position


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0).item())


def _project(m: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
    """Seat m on the a->b axis. Returns (t, perp_norm) with perp normalized by |b-a|."""
    axis = b - a
    denom = float(axis.dot(axis).clamp(min=1e-12).item())
    t = float((m - a).dot(axis).item()) / denom
    perp = (m - a) - t * axis
    axis_len = float(axis.norm().clamp(min=1e-12).item())
    perp_norm = float(perp.norm().item()) / axis_len
    return t, perp_norm


def read_word_across_layers(sentence: str, word: str) -> torch.Tensor:
    """Residual at `word`'s token position across all 27 layers. Returns [27, hidden] float32."""
    _, tok = load()
    hs, _ = residual_sweep(sentence)                 # [27, seq, hidden]
    pos = find_target_position(sentence, word, tok)
    return hs[:, pos, :].to(torch.float32).cpu()     # [27, hidden]


def _single_token(word: str, tok) -> bool:
    return len(tok.encode(" " + word, add_special_tokens=False)) == 1


def measure_unit(unit: dict) -> list[dict]:
    """Compute both measurements for one unit across all layers. Returns tidy rows."""
    _, tok = load()
    rows = []
    dropped = []

    def row(metric, target, value, layer, defined=True, note=""):
        rows.append({
            "unit": unit["id"], "role": unit["role"], "kind": unit["kind"],
            "named_dichotomy": unit["named_dichotomy"], "layer": layer,
            "metric": metric, "target": target,
            "value": (round(value, 6) if value is not None else None),
            "defined": defined, "note": note,
        })

    n_layers = None

    # Carrier reads for poles (define the A-B axis used by (a) and (b)).
    A_car = read_word_across_layers(unit["carrier_frame"].format(word=unit["pole_A"]), unit["pole_A"])
    B_car = read_word_across_layers(unit["carrier_frame"].format(word=unit["pole_B"]), unit["pole_B"])
    n_layers = A_car.shape[0]

    # Dichotomy-frame pole reads ("as the corpus names them"), when a frame exists.
    A_nam = B_nam = None
    if unit["dichotomy_frame"]:
        A_nam = read_word_across_layers(unit["dichotomy_frame"], unit["pole_A"])
        B_nam = read_word_across_layers(unit["dichotomy_frame"], unit["pole_B"])

    # Midpoint read (measurement a), only if a single-token midpoint exists.
    M_car = None
    if unit["midpoint"]:
        if _single_token(unit["midpoint"], tok):
            M_car = read_word_across_layers(
                unit["carrier_frame"].format(word=unit["midpoint"]), unit["midpoint"])
        else:
            dropped.append(("midpoint", unit["midpoint"]))

    # Excluded-term reads (measurement b remainder), single-token only.
    excluded_vecs = {}
    for e in unit["excluded_terms"]:
        if _single_token(e, tok):
            excluded_vecs[e] = read_word_across_layers(unit["carrier_frame"].format(word=e), e)
        else:
            dropped.append(("excluded", e))

    for L in range(n_layers):
        a_c, b_c = A_car[L], B_car[L]

        # (b) cross-layer pole relation
        if A_nam is not None:
            row("cosine_AB_named", f"{unit['pole_A']}|{unit['pole_B']}", _cos(A_nam[L], B_nam[L]), L)
        else:
            row("cosine_AB_named", f"{unit['pole_A']}|{unit['pole_B']}", None, L,
                defined=False, note="No dichotomy frame (null pair by construction).")
        row("cosine_AB_carrier", f"{unit['pole_A']}|{unit['pole_B']}", _cos(a_c, b_c), L)

        # (a) seated-midpoint
        if M_car is not None:
            t, perp = _project(M_car[L], a_c, b_c)
            row("midpoint_t", unit["midpoint"], t, L)
            row("midpoint_perp_norm", unit["midpoint"], perp, L)
            row("cosine_midpoint_A", unit["midpoint"], _cos(M_car[L], a_c), L)
            row("cosine_midpoint_B", unit["midpoint"], _cos(M_car[L], b_c), L)
        else:
            reason = ("Moral midpoint is phrasal/multi-token; measurement (a) does not close "
                      "on a single Y-unit (anticipated finding)." if unit["kind"] == "scalar_collapse"
                      else "No single-token midpoint exists for this unit by construction.")
            row("midpoint_t", unit["midpoint"] or "(none)", None, L, defined=False, note=reason)

        # (b) remainder ratio: excluded-term energy off the A-B axis
        if excluded_vecs:
            perps = []
            for e, vec in excluded_vecs.items():
                t_e, perp_e = _project(vec[L], a_c, b_c)
                row("excluded_perp_norm", e, perp_e, L)
                row("excluded_t", e, t_e, L)
                perps.append(perp_e)
            row("remainder_ratio", "excluded_set", sum(perps) / len(perps), L)
        else:
            row("remainder_ratio", "excluded_set", None, L, defined=False,
                note=("A true two-term opposition has no excluded middle; remainder is "
                      "near-zero/undefined by construction — the contrast, reported flat."
                      if unit["kind"] == "true_dichotomy"
                      else "No excluded-term set (null pair anchors the floor)."))

    return rows, dropped, n_layers
