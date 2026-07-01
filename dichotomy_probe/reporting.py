"""Result emission for the Dichotomy Transformation Probe (v2).

Every emitted result string carries the fixed reporting order:
  the corpus named the dichotomy; the model processed that naming; the
  measurement recorded the processing; a human interprets the record.

Cosine, midpoint seating, and remainder are always labeled corroboration. The
descriptive class of a candidate is offered to the reader as a summary, never as a
verdict. true / false are never attributed to the machine. Disruption -- a shift in
a reader's verdict on reading the trace -- is predicated of the human reader.

Two files (both reachable by one download button): measurements.csv (tidy per-
layer measurements for every pair, carrier, and candidate) and results.html (a
single self-contained page). They are bundled into dichotomy_results.zip.
"""

import csv
import os
import zipfile
import html as _html

from . import config
from .measures import THRESHOLDS

CORROBORATION_METRICS = {
    "cosine_AB", "cosine_MA", "cosine_MB", "cosine_M_frameterm",
    "cosine_A_frameterm", "cosine_B_frameterm", "midpoint_t", "remainder_ratio",
}
SUMMARY_METRICS = {"carrier_stability", "layer_of_strongest_seating", "descriptive_class"}


def statement_for(named, carrier, metric, target, value, layer):
    if isinstance(value, (int, float)):
        recorded = f"{metric}({target}) = {value:.4f} under carrier '{carrier}'"
    elif value is None:
        recorded = f"{metric}({target}) is undefined here"
    else:
        recorded = f"{metric}({target}) = {value} under carrier '{carrier}'"
    corro = " (corroboration only)" if metric in CORROBORATION_METRICS else ""
    depth = "as a summary across layers" if layer in (-1, None) else f"at layer {layer}"
    return (f"The corpus named the dichotomy {named}; the model processed that naming; "
            f"the measurement recorded, {depth}, {recorded}{corro}; a human interprets the record.")


# ---- CSV --------------------------------------------------------------------

CSV_COLUMNS = [
    "pair", "kind", "named_dichotomy", "carrier", "input_type", "target", "metric",
    "layer", "value", "corroboration_only", "unit_of_analysis", "reporting_order",
    "statement", "note",
]


def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            m = r["metric"]
            w.writerow({
                "pair": r["pair"], "kind": r["kind"], "named_dichotomy": r["named"],
                "carrier": r["carrier"], "input_type": r["input_type"], "target": r["target"],
                "metric": m, "layer": r["layer"], "value": r["value"],
                "corroboration_only": m in CORROBORATION_METRICS,
                "unit_of_analysis": config.UNIT_STATEMENT,
                "reporting_order": config.REPORTING_ORDER,
                "statement": statement_for(r["named"], r["carrier"], m, r["target"],
                                           r["value"], r["layer"]),
                "note": r["note"],
            })


# ---- HTML -------------------------------------------------------------------

CLASS_GLOSS = {
    "true_midpoint_candidate": "seats between the poles with lower off-axis remainder than an unrelated word, stable across carriers",
    "one_pole_candidate": "sits near one pole, not between them",
    "frame_adjacent_candidate": "closer to the frame term than to either pole; frame-adjacent, not a middle",
    "off_axis_candidate": "carries structure off the pole axis (mid or centered position but high remainder)",
    "prompt_induced_candidate": "only becomes midpoint-like once a prompt forces a middle; not a stable middle",
    "frame_term_reference": "the frame term itself, a reference point (not classified against itself)",
    "undetermined": "not determinable from the reads",
}


def _cab_by_carrier(rows, pair_id, depths):
    """cosine_AB per carrier at chosen depths: {carrier: {layer: value}}."""
    out = {}
    for r in rows:
        if r["pair"] == pair_id and r["metric"] == "cosine_AB" and r["layer"] in depths:
            out.setdefault(r["carrier"], {})[r["layer"]] = r["value"]
    return out


def write_html(rows, meta, path, classifications):
    e = _html.escape
    n_layers = meta.get("n_layers", config.N_RESIDUAL_STATES)
    depths = sorted({0, n_layers // 2, n_layers - 1})
    by_pair = {}
    for c in classifications:
        by_pair.setdefault(c["pair"], []).append(c)

    P = []
    P.append(f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dichotomy Transformation Probe — Results (v2)</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
        max-width: 960px; margin: 2rem auto; padding: 0 1rem; color:#1a1a1a; line-height:1.5; }}
 h1 {{ font-size:1.6rem; }} h2 {{ font-size:1.25rem; margin-top:2rem; border-bottom:1px solid #ddd; padding-bottom:.3rem; }}
 h3 {{ font-size:1.03rem; margin-top:1.3rem; }}
 .frame {{ background:#f5f5f7; border-left:3px solid #888; padding:.5rem .8rem; margin:.5rem 0; font-size:.9rem; }}
 .constraint {{ background:#fbfbfd; border:1px solid #e5e5ea; border-radius:6px; padding:.5rem .8rem; margin:.35rem 0; font-size:.92rem; }}
 table {{ border-collapse:collapse; width:100%; margin:.6rem 0; font-size:.86rem; }}
 th,td {{ border:1px solid #ddd; padding:.32rem .45rem; text-align:right; }}
 th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align:left; }}
 caption {{ text-align:left; font-style:italic; color:#555; margin-bottom:.3rem; }}
 .corr {{ color:#666; font-size:.85rem; }}
 .stmt {{ background:#f5f5f7; padding:.5rem .8rem; border-radius:6px; margin:.4rem 0; font-size:.9rem; }}
 code {{ background:#eee; padding:0 .2rem; border-radius:3px; }}
 .cls-true_midpoint_candidate {{ background:#e6f4ea; }}
 .cls-prompt_induced_candidate {{ background:#fef7e0; }}
 .cls-frame_adjacent_candidate {{ background:#eef2fb; }}
 footer {{ margin-top:3rem; color:#777; font-size:.85rem; border-top:1px solid #ddd; padding-top:1rem; }}
</style></head><body>
<h1>Dichotomy Transformation Probe — Results (v2)</h1>
<p><strong>Model:</strong> {e(meta.get('model_id',''))} &nbsp;·&nbsp;
   <strong>Residual states read:</strong> {n_layers} (layers 0 to output) &nbsp;·&nbsp;
   <strong>Batch run:</strong> {e(meta.get('run_stamp','(see commit)'))}</p>

<div class="constraint"><strong>Unit of analysis.</strong> {e(config.UNIT_STATEMENT)}</div>
<div class="constraint"><strong>Reporting order (every result sentence).</strong> {e(config.REPORTING_ORDER)}
   No sentence reaches past the naming to the thing named.</div>
<div class="constraint"><strong>The machine never determines true or false.</strong>
   It transforms a cultural dichotomy already named in the corpus at entry; the verdict lives in
   the human reader. true and false are never attributed to the machine.</div>
<div class="constraint"><strong>No developmental or trajectory framing.</strong>
   Across-depth structure is legibility-by-depth, not the model deliberating toward a verdict.</div>
<div class="constraint"><strong>Corroboration vs. claim.</strong> {e(config.CORROBORATION_NOTE)}
   Cosine, midpoint seating, and remainder are labeled corroboration throughout. Each candidate's
   descriptive class is a summary offered to the reader, never a verdict.</div>
<div class="constraint"><strong>No middle is assumed.</strong> Every pair — including the true
   dichotomies — receives candidate middles and the same measurements. "No middle" is a result to
   be read, never built into the design.</div>

<h2>How to read the numbers</h2>
<ul style="font-size:.92rem">
 <li><code>cosine_AB</code> — how the two poles co-situate under each carrier (corroboration).</li>
 <li><code>midpoint_t</code> — a candidate's projection on the A→B axis: 0 at A, 1 at B, ~0.5 centered.</li>
 <li><code>remainder_ratio</code> — the candidate's distance off that axis, ÷ the pole gap (off-axis structure).</li>
 <li><code>remainder_vs_control</code> — that remainder divided by the pair's unrelated-word (table/reason)
     baseline. Below 1 means the candidate sits closer to the pole axis than an unrelated word does.
     Absolute off-axis magnitudes are not interpretable in 2304-dim residual space, so seating is judged
     on this control-relative ratio (the relative-to-control logic of the Water study and Token Biography).</li>
 <li><code>carrier_stability</code> — agreement of <code>midpoint_t</code> across the two word-slot carriers (1 = identical).</li>
 <li><code>mean_t_forced</code> — the candidate's mean projection under the four hidden-middle forcing prompts.</li>
 <li><strong>descriptive class</strong> — one of: {e(', '.join(CLASS_GLOSS.keys()))}. Thresholds:
     seats between if {THRESHOLDS['t_between_lo']}≤t≤{THRESHOLDS['t_between_hi']} and
     remainder_vs_control≤{THRESHOLDS['rel_remainder_seat']} and stability≥{THRESHOLDS['stability_min']};
     poleward if t≤{1-THRESHOLDS['t_poleward']:.2f} or t≥{THRESHOLDS['t_poleward']}. These are heuristics;
     the raw per-layer numbers in <code>measurements.csv</code> are the record, and the reader interprets them.</li>
</ul>
<div class="constraint"><strong>Calibration note (transparency).</strong> The seating threshold is
 control-relative (a candidate's off-axis remainder vs. the unrelated-word baseline) rather than an
 absolute cutoff. This was chosen after the first v2 run showed that in 2304-dim residual space every
 candidate's absolute off-axis remainder exceeds any small fixed value, so an absolute cutoff is
 uninformative. The pre-registration (<code>PREREGISTRATION_v2.md</code>) states the class is a heuristic
 reader-aid and that the raw numbers are the record; this calibration changes only the reader-aid, not
 the measurements, and is disclosed here rather than by editing the frozen pre-registration.</div>
""")

    for pair in config.PAIRS:
        pid = pair["id"]
        P.append(f"<h2>{e(pair['named'])} &nbsp;<span class='corr'>({e(pid)} · {e(pair['kind'])})</span></h2>")
        P.append('<div class="frame">'
                 f"<strong>matched syntax:</strong> {e(config.matched_syntax_carrier(pair))}<br>"
                 f"<strong>natural usage:</strong> {e(pair['natural_carrier'])}<br>"
                 f"<strong>explicit dichotomy:</strong> The dichotomy between {e(pair['A'])} and {e(pair['B'])} is familiar.<br>"
                 f"<strong>neutral relation:</strong> The relation between {e(pair['A'])} and {e(pair['B'])} is familiar.<br>"
                 f"<strong>frame term:</strong> {e(pair['frame_term'])}</div>")

        # cosine_AB across the four carriers.
        cab = _cab_by_carrier(rows, pid, depths)
        carriers_present = [c["id"] for c in config.CARRIERS]
        P.append('<h3>Pole relation cosine(A,B) across carriers (corroboration)</h3>')
        P.append('<table><caption>At embedding (0), mid, and output depths. Full series in measurements.csv.</caption>'
                 "<tr><th>carrier</th>" + "".join(f"<th>layer {d}</th>" for d in depths) + "</tr>")
        for cn in carriers_present:
            cells = "".join(
                f"<td>{'—' if cab.get(cn, {}).get(d) is None else f'{cab[cn][d]:.4f}'}</td>" for d in depths)
            P.append(f"<tr><td>{e(cn)}</td>{cells}</tr>")
        P.append("</table>")

        # Candidate classification table.
        P.append('<h3>Candidate middles — measured, not assumed</h3>')
        P.append('<table><caption>Each candidate projected on the pole axis. '
                 '<code>rem/ctrl</code> is remainder relative to the unrelated-word baseline (below 1 = '
                 'closer to the axis than an unrelated word). Descriptive class is offered to the reader; '
                 'the raw numbers are the record.</caption>'
                 "<tr><th>candidate</th><th>input type</th><th>mean t (unforced)</th>"
                 "<th>rem/ctrl</th><th>carrier stability</th><th>mean t (forced)</th>"
                 "<th>strongest seating layer</th><th>cosine to frame term</th><th>descriptive class</th></tr>")
        for c in by_pair.get(pid, []):
            cls = c["descriptive_class"]
            def fmt(x):
                return "—" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))
            P.append(
                f"<tr class='cls-{e(cls)}'><td>{e(c['candidate'])}</td><td>{e(c['input_type'])}</td>"
                f"<td>{fmt(c['mean_midpoint_t_unforced'])}</td><td>{fmt(c.get('remainder_vs_control'))}</td>"
                f"<td>{fmt(c['carrier_stability'])}</td><td>{fmt(c['mean_midpoint_t_forced'])}</td>"
                f"<td>{fmt(c['layer_of_strongest_seating'])}</td><td>{fmt(c['mean_cosine_frameterm'])}</td>"
                f"<td>{e(cls)}</td></tr>")
        P.append("</table>")
        P.append('<p class="corr">Class glossary: '
                 + "; ".join(f"<em>{e(k)}</em> — {e(v)}" for k, v in CLASS_GLOSS.items()) + ".</p>")

        # Fixed-order sentence.
        P.append(f'<p class="stmt">The corpus named the dichotomy {e(pair["named"])}; the model '
                 f'processed that naming; the measurement recorded, across layers 0 to output and '
                 f'across four carriers plus four forcing prompts, where each candidate seats on the '
                 f'pole axis and how much structure it carries off that axis (all corroboration only); '
                 f'a human interprets the record.</p>')

    P.append(f"""<footer>
<p>Both files download together: <code>measurements.csv</code> (tidy per-layer measurements for every
pair, carrier, and candidate) and this page, <code>results.html</code>.</p>
<p>Pre-registration precedes this data: <code>dichotomy_probe/PREREGISTRATION.md</code> and the v2
addendum <code>dichotomy_probe/PREREGISTRATION_v2.md</code>, committed before the data existed.</p>
<p>Disruption — a shift in a reader's verdict on reading this trace — is predicated of the human
reader, never of the transformer or the geometry. The residual stream is the material trace; nothing
here materializes a verdict in the stream.</p>
</footer></body></html>""")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(P))


def write_zip(csv_path, html_path, zip_path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(csv_path, arcname=os.path.basename(csv_path))
        z.write(html_path, arcname=os.path.basename(html_path))
