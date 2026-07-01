"""Result emission for the Dichotomy Transformation Probe.

Every emitted result string carries the fixed reporting order:
  the corpus named the dichotomy; the model processed that naming; the
  measurement recorded the processing; a human interprets the record.

Cosine and any intermediate-seating result are always labeled corroboration.
Disruption -- a shift in a verdict on reading the trace -- is always predicated
of the human reader, never the transformer or the geometry. true / false are
never attributed to the machine.

Two files are produced (both reachable by one download button):
  1. measurements.csv -- tidy per-layer measurements for every item and control.
  2. results.html     -- a single self-contained, human-readable results page.
They are bundled into dichotomy_results.zip.
"""

import csv
import io
import os
import zipfile
import html as _html

from . import config

# Metrics that are corroboration only (proximity + seating).
CORROBORATION_METRICS = {
    "cosine_AB_named", "cosine_AB_carrier",
    "cosine_midpoint_A", "cosine_midpoint_B",
    "midpoint_t", "midpoint_perp_norm",
}
# Metrics that carry the differential value-exhaustion description.
DIFFERENTIAL_METRICS = {"remainder_ratio", "excluded_perp_norm", "excluded_t"}


def statement_for(named_dichotomy: str, metric: str, target: str,
                  value, layer: int, defined: bool, note: str) -> str:
    """One fixed-reporting-order sentence instantiating a single measurement row."""
    if defined and value is not None:
        recorded = f"{metric}({target}) = {value:.4f}"
    else:
        recorded = f"{metric}({target}) is undefined here"
    corro = " (corroboration only)" if metric in CORROBORATION_METRICS else ""
    tail = f" [{note}]" if note else ""
    return (
        f"The corpus named the dichotomy {named_dichotomy}; the model processed that "
        f"naming; the measurement recorded, at layer {layer}, {recorded}{corro}; a human "
        f"interprets the record.{tail}"
    )


# ---- CSV --------------------------------------------------------------------

CSV_COLUMNS = [
    "unit", "role", "kind", "named_dichotomy", "layer", "metric", "target",
    "value", "defined", "measurement", "corroboration_only",
    "unit_of_analysis", "reporting_order", "statement", "note",
]


def write_csv(rows: list[dict], path: str):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            metric = r["metric"]
            measurement = ("(a) seated-midpoint" if metric.startswith(("midpoint", "cosine_midpoint"))
                           else "(b) cross-layer pole relation")
            w.writerow({
                "unit": r["unit"], "role": r["role"], "kind": r["kind"],
                "named_dichotomy": r["named_dichotomy"], "layer": r["layer"],
                "metric": metric, "target": r["target"], "value": r["value"],
                "defined": r["defined"], "measurement": measurement,
                "corroboration_only": metric in CORROBORATION_METRICS,
                "unit_of_analysis": config.UNIT_STATEMENT,
                "reporting_order": config.REPORTING_ORDER,
                "statement": statement_for(r["named_dichotomy"], metric, r["target"],
                                           r["value"], r["layer"], r["defined"], r["note"]),
                "note": r["note"],
            })


# ---- HTML -------------------------------------------------------------------

def _mean_defined(rows, unit_id, metric):
    vals = [r["value"] for r in rows
            if r["unit"] == unit_id and r["metric"] == metric
            and r["defined"] and r["value"] is not None]
    return (sum(vals) / len(vals)) if vals else None


def _at_layers(rows, unit_id, metric, layers):
    out = {}
    for r in rows:
        if r["unit"] == unit_id and r["metric"] == metric and r["layer"] in layers:
            out[r["layer"]] = r["value"] if r["defined"] else None
    return out


def write_html(rows: list[dict], meta: dict, path: str):
    e = _html.escape
    n_layers = meta.get("n_layers", config.N_RESIDUAL_STATES)
    depths = sorted({0, n_layers // 2, n_layers - 1})

    parts = []
    parts.append(f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dichotomy Transformation Probe — Results</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
        max-width: 900px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; line-height: 1.5; }}
 h1 {{ font-size: 1.6rem; }} h2 {{ font-size: 1.25rem; margin-top: 2rem; border-bottom: 1px solid #ddd; padding-bottom: .3rem; }}
 h3 {{ font-size: 1.05rem; margin-top: 1.4rem; }}
 .frame {{ background: #f5f5f7; border-left: 3px solid #888; padding: .6rem .8rem; margin: .6rem 0; }}
 .constraint {{ background: #fbfbfd; border: 1px solid #e5e5ea; border-radius: 6px; padding: .5rem .8rem; margin: .4rem 0; font-size: .93rem; }}
 table {{ border-collapse: collapse; width: 100%; margin: .6rem 0; font-size: .9rem; }}
 th, td {{ border: 1px solid #ddd; padding: .35rem .5rem; text-align: right; }}
 th:first-child, td:first-child {{ text-align: left; }}
 caption {{ text-align: left; font-style: italic; color: #555; margin-bottom: .3rem; }}
 .corr {{ color: #666; font-size: .85rem; }}
 .stmt {{ background: #f5f5f7; padding: .5rem .8rem; border-radius: 6px; margin: .4rem 0; font-size: .92rem; }}
 code {{ background: #eee; padding: 0 .2rem; border-radius: 3px; }}
 footer {{ margin-top: 3rem; color: #777; font-size: .85rem; border-top: 1px solid #ddd; padding-top: 1rem; }}
</style></head><body>
<h1>Dichotomy Transformation Probe — Results</h1>
<p><strong>Model:</strong> {e(meta.get('model_id',''))} &nbsp;·&nbsp;
   <strong>Residual states read:</strong> {n_layers} (layers 0 to output) &nbsp;·&nbsp;
   <strong>Batch run:</strong> {e(meta.get('run_stamp','(see commit)'))}</p>

<div class="constraint"><strong>Unit of analysis.</strong> {e(config.UNIT_STATEMENT)}</div>
<div class="constraint"><strong>Reporting order (every result sentence).</strong> {e(config.REPORTING_ORDER)}
   No sentence reaches past the naming to the thing named.</div>
<div class="constraint"><strong>The machine never determines true or false.</strong>
   It transforms a cultural dichotomy already named in the corpus at entry; the verdict
   lives in the human reader. true and false are never attributed to the machine.</div>
<div class="constraint"><strong>No developmental or trajectory framing.</strong>
   Across-depth structure is legibility-by-depth, not the model deliberating toward a
   verdict. Movement across depth is in the human's reading, not the model's cognition.</div>
<div class="constraint"><strong>Corroboration vs. claim.</strong> {e(config.CORROBORATION_NOTE)}
   Cosine and any intermediate-seating result are labeled corroboration throughout.</div>
<div class="constraint"><strong>Descriptive, not confirmatory.</strong> No outcome is a
   hit designated in advance. Both measurements are computed and reported for every item
   and control; any disconfirming control is reported flat.</div>
""")

    # Differential summary (described, not predicted).
    parts.append("<h2>Differential summary — value-exhaustion (measurement b), described</h2>")
    parts.append('<table><caption>Remainder ratio = mean off-axis energy of the excluded '
                 'middle terms, averaged across all layers. Lower ⇒ the two-term axis '
                 'exhausts the value field; higher ⇒ value is carried by excluded terms '
                 '(graded field). Described, not predicted; corroboration for the differential.</caption>'
                 "<tr><th>unit</th><th>kind</th><th>named dichotomy</th>"
                 "<th>mean cosine(A,B) named</th><th>mean remainder ratio</th></tr>")
    for u in config.UNITS:
        cab = _mean_defined(rows, u["id"], "cosine_AB_named")
        rr = _mean_defined(rows, u["id"], "remainder_ratio")
        parts.append(
            f"<tr><td>{e(u['id'])}</td><td>{e(u['kind'])}</td><td>{e(u['named_dichotomy'])}</td>"
            f"<td>{'—' if cab is None else f'{cab:.4f}'}</td>"
            f"<td>{'undefined' if rr is None else f'{rr:.4f}'}</td></tr>")
    parts.append("</table>")
    parts.append('<p class="corr">"undefined" is reported flat: a true two-term opposition '
                 'and the null pair have no excluded middle by construction.</p>')

    # Per-unit sections.
    for u in config.UNITS:
        parts.append(f"<h2>{e(u['named_dichotomy'])} &nbsp;<span class='corr'>({e(u['id'])} · {e(u['role'])} · {e(u['kind'])})</span></h2>")
        if u["dichotomy_frame"]:
            parts.append(f'<div class="frame"><strong>Dichotomy frame (sense-fixing):</strong> {e(u["dichotomy_frame"])}<br>'
                         f'<strong>Carrier frame:</strong> {e(u["carrier_frame"])}</div>')
        else:
            parts.append(f'<div class="frame"><strong>Null pair by construction</strong> — no dichotomy frame. '
                         f'Carrier frame: {e(u["carrier_frame"])}</div>')

        # Fixed-order prose for measurement (b).
        cab = _mean_defined(rows, u["id"], "cosine_AB_carrier")
        parts.append('<h3>Measurement (b) — cross-layer pole relation</h3>')
        parts.append(f'<p class="stmt">The corpus named the dichotomy {e(u["named_dichotomy"])}; '
                     f'the model processed that naming; the measurement recorded the pole relation '
                     f'across layers 0 to output '
                     f'(mean cosine of the poles, carrier axis = '
                     f'{"—" if cab is None else f"{cab:.4f}"}, corroboration only); '
                     f'a human interprets the record.</p>')
        tbl = {m: _at_layers(rows, u["id"], m, depths) for m in ("cosine_AB_named", "cosine_AB_carrier", "remainder_ratio")}
        parts.append('<table><caption>At embedding (0), mid, and output depths. Full 27-state '
                     'series is in measurements.csv.</caption>'
                     "<tr><th>metric</th>" + "".join(f"<th>layer {d}</th>" for d in depths) + "</tr>")
        for m in ("cosine_AB_named", "cosine_AB_carrier", "remainder_ratio"):
            cells = "".join(
                f"<td>{'undefined' if tbl[m].get(d) is None else f'{tbl[m][d]:.4f}'}</td>" for d in depths)
            corr = " <span class='corr'>(corroboration)</span>" if m in CORROBORATION_METRICS else ""
            parts.append(f"<tr><td>{m}{corr}</td>{cells}</tr>")
        parts.append("</table>")

        # Measurement (a).
        parts.append('<h3>Measurement (a) — seated-midpoint (corroboration only)</h3>')
        if u["midpoint"]:
            t_mean = _mean_defined(rows, u["id"], "midpoint_t")
            perp_mean = _mean_defined(rows, u["id"], "midpoint_perp_norm")
            parts.append(
                f'<p class="stmt">The corpus named the dichotomy {e(u["named_dichotomy"])}; the model '
                f'processed that naming; the measurement recorded whether the single-token candidate '
                f'midpoint <code>{e(u["midpoint"])}</code> seats between the poles across layers '
                f'(mean projection t = {"—" if t_mean is None else f"{t_mean:.3f}"}, '
                f'mean off-axis residual = {"—" if perp_mean is None else f"{perp_mean:.3f}"}, '
                f'corroboration only); a human interprets the record.</p>')
        else:
            parts.append(f'<p class="stmt">{e(u["midpoint_note"])}</p>')

    parts.append(f"""<footer>
<p>Both files download together from the results page: <code>measurements.csv</code>
(tidy per-layer measurements for every item and control) and this page,
<code>results.html</code>.</p>
<p>Pre-registration precedes this data: see <code>dichotomy_probe/PREREGISTRATION.md</code>,
committed alone before any data existed.</p>
<p>Disruption — a shift in a reader's verdict on reading this trace — is predicated of the
human reader, never of the transformer or the geometry. The residual stream is the
material trace; nothing here materializes a verdict in the stream.</p>
</footer></body></html>""")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def write_zip(csv_path: str, html_path: str, zip_path: str):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(csv_path, arcname=os.path.basename(csv_path))
        z.write(html_path, arcname=os.path.basename(html_path))
