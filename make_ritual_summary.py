"""Quantitative summary for the three-word subset (water/salt/bread).

Two metrics, per the approved definitions:

  (1) Tier 2 ritual-verb mass = summed probability of ritual/liturgical verbs
      in the SACRED Tier 2 frame ("People use the holy X to"), top-20.
      Reported with a PER-VERB breakdown and a class flag:
        LITURGICAL          - strongly liturgical (bless, cleanse, purify, ...)
        LITURGICAL_FRAGMENT - clearly-liturgical tokenizer fragments (bap->baptize,
                              sancti->sanctify, consec->consecrate)
        CEREMONIAL_SECULAR  - ceremonial but also secular (honor, celebrate,
                              commemorate, mark, dedicate)
      Secular utility verbs (wash, clean, protect, cure, make, ...) are NOT counted.

  (2) View 3 ritual-feature activation = max activation among religious-tagged
      SAE features at the probe-token, layer 19. Uses the religiosity-MATCHED
      salt run so sentence-religiosity is held constant.

Outputs (committed):
  results/subset_ritual_summary.json  - numbers + per-verb breakdown
  results/subset_ritual_summary.svg   - comparison figure

Local, free. Run:  python make_ritual_summary.py
"""

import ast
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

WORDS = ["water", "salt", "bread"]

# View 3 source: matched salt so sentence-religiosity is constant across words.
VIEW3_FILE = {"water": "subset_water.json",
              "salt": "subset_salt_matched.json",
              "bread": "subset_bread.json"}
# Tier 2 source: salt's Tier 2 lives in the original run (matched run is View 3-only).
TIER2_FILE = {"water": "subset_water.json",
              "salt": "subset_salt.json",
              "bread": "subset_bread.json"}

LITURGICAL = {"bless", "blessed", "cleanse", "purify", "consecrate", "sanctify",
              "anoint", "sprinkle", "ward", "worship", "pray", "venerate", "baptize"}
LITURGICAL_FRAGMENT = {"bap", "sancti", "consec", "anoin", "venerat"}
CEREMONIAL_SECULAR = {"honor", "celebrate", "commemorate", "mark", "dedicate"}

RELIG_DESC = re.compile(
    r"relig|sacrament|ritual|ordinance|worship|sacred|divine|spirit|holy|pray|priest|bless",
    re.I)


def norm_token(repr_str):
    """results store tokens as repr() strings, e.g. \"' cleanse'\" -> 'cleanse'."""
    try:
        s = ast.literal_eval(repr_str)
    except Exception:
        s = repr_str
    return s.strip().lower()


def classify(tok):
    if tok in LITURGICAL:
        return "LITURGICAL"
    if tok in LITURGICAL_FRAGMENT:
        return "LITURGICAL_FRAGMENT"
    if tok in CEREMONIAL_SECULAR:
        return "CEREMONIAL_SECULAR"
    return None


def tier2_mass(word):
    rec = json.load(open(os.path.join(RESULTS, TIER2_FILE[word]), encoding="utf-8"))
    sac = [f for f in rec["view2_tier2"]["frames"] if f["id"] == "sacred"][0]
    rows = []
    for t in sac["top_next"]:
        tok = norm_token(t["token"])
        cls = classify(tok)
        if cls:
            rows.append({"token": tok, "probability": round(t["probability"], 5), "class": cls})
    mass = round(sum(r["probability"] for r in rows), 5)
    lit = round(sum(r["probability"] for r in rows if r["class"].startswith("LITURGICAL")), 5)
    cer = round(sum(r["probability"] for r in rows if r["class"] == "CEREMONIAL_SECULAR"), 5)
    return {"prompt": sac["prompt"], "mass": mass, "liturgical": lit,
            "ceremonial_secular": cer, "verbs": rows}


def view3_ritual(word):
    rec = json.load(open(os.path.join(RESULTS, VIEW3_FILE[word]), encoding="utf-8"))
    feats = rec["view3"]["19"]["probe_word"]["features"]
    hits = [f for f in feats if RELIG_DESC.search(f["description"] or "")]
    if not hits:
        return {"sentence": rec["sentence"], "activation": 0.0,
                "feature_idx": None, "description": None}
    top = max(hits, key=lambda f: f["activation"])
    return {"sentence": rec["sentence"], "activation": round(top["activation"], 3),
            "feature_idx": top["feature_idx"], "description": top["description"]}


# ---- SVG figure (dependency-free) -------------------------------------------

def bar(x, y, w, h, fill, extra=""):
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}" {extra}/>'


def txt(x, y, s, size=13, fill="#222", anchor="middle", weight="normal"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial,Helvetica,sans-serif" '
            f'font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{s}</text>')


def build_svg(summary):
    W, H = 780, 460
    COLORS = {"water": "#2b6cb0", "salt": "#718096", "bread": "#b7791f"}
    LIT_FILL, CER_FILL = "#2f855a", "#9ae6b4"   # liturgical / ceremonial-secular
    V3_FILL = "#6b46c1"

    s = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">']
    s.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>')
    s.append(txt(W / 2, 28, "Does context move salt and bread toward water&#8217;s situated sense?",
                 size=16, weight="bold"))
    s.append(txt(W / 2, 46, "Three-word subset (water = reference word). Higher = more ritual/situated content.",
                 size=11, fill="#666"))

    # ----- Panel A: Tier 2 ritual-verb mass (stacked liturgical + ceremonial) -----
    ax0, ay0, aw, ah = 70, 90, 290, 300     # plot area
    a_base = ay0 + ah
    masses = [summary[w]["tier2"]["mass"] for w in WORDS]
    amax = max(masses + [0.001])
    a_scale = ah / (amax * 1.25)
    s.append(txt(ax0 + aw / 2, ay0 - 14, "(1) Tier 2 ritual-verb mass", size=13, weight="bold"))
    s.append(txt(ax0 + aw / 2, ay0 - 1, "&#931; prob in 'People use the holy X to ___'", size=10, fill="#666"))
    s.append(f'<line x1="{ax0}" y1="{a_base}" x2="{ax0+aw}" y2="{a_base}" stroke="#999" stroke-width="1"/>')
    bw, gap = 56, (aw - 3 * 56) / 4
    for i, w in enumerate(WORDS):
        bx = ax0 + gap + i * (bw + gap)
        lit = summary[w]["tier2"]["liturgical"] * a_scale
        cer = summary[w]["tier2"]["ceremonial_secular"] * a_scale
        s.append(bar(bx, a_base - lit, bw, lit, LIT_FILL))
        s.append(bar(bx, a_base - lit - cer, bw, cer, CER_FILL))
        s.append(txt(bx + bw / 2, a_base - lit - cer - 6, f"{summary[w]['tier2']['mass']:.3f}", size=12, weight="bold"))
        s.append(txt(bx + bw / 2, a_base + 16, w, size=12, weight="bold", fill=COLORS[w]))
    # legend
    ly = a_base + 36
    s.append(bar(ax0 + 18, ly, 12, 12, LIT_FILL))
    s.append(txt(ax0 + 36, ly + 11, "liturgical (bless, cleanse, purify, ward, pray, bap-, sancti-)", size=10, anchor="start", fill="#444"))
    s.append(bar(ax0 + 18, ly + 18, 12, 12, CER_FILL))
    s.append(txt(ax0 + 36, ly + 29, "ceremonial-but-secular (celebrate, commemorate, honor, worship)", size=10, anchor="start", fill="#444"))

    # ----- Panel B: View 3 ritual-feature activation (L19 probe) -----
    bx0, by0, bw2, bh = 470, 90, 250, 300
    b_base = by0 + bh
    acts = [summary[w]["view3"]["activation"] for w in WORDS]
    bmax = max(acts + [1.0])
    b_scale = bh / (bmax * 1.25)
    s.append(txt(bx0 + bw2 / 2, by0 - 14, "(2) View 3 ritual-feature activation", size=13, weight="bold"))
    s.append(txt(bx0 + bw2 / 2, by0 - 1, "max religious SAE feature, layer 19 probe-token", size=10, fill="#666"))
    s.append(f'<line x1="{bx0}" y1="{b_base}" x2="{bx0+bw2}" y2="{b_base}" stroke="#999" stroke-width="1"/>')
    bw3, gap2 = 50, (bw2 - 3 * 50) / 4
    for i, w in enumerate(WORDS):
        bx = bx0 + gap2 + i * (bw3 + gap2)
        a = summary[w]["view3"]["activation"]
        h = a * b_scale
        s.append(bar(bx, b_base - h, bw3, h, V3_FILL if a > 0 else "#ddd"))
        lbl = f"{a:.1f}" if a > 0 else "0 (none)"
        s.append(txt(bx + bw3 / 2, b_base - h - 6 if a > 0 else b_base - 6, lbl, size=12, weight="bold"))
        fid = summary[w]["view3"]["feature_idx"]
        if fid is not None:
            s.append(txt(bx + bw3 / 2, b_base - h - 20, f"#{fid}", size=9, fill="#666"))
        s.append(txt(bx + bw3 / 2, b_base + 16, w, size=12, weight="bold", fill=COLORS[w]))

    s.append('</svg>')
    return "\n".join(s)


def main():
    summary = {}
    for w in WORDS:
        summary[w] = {"tier2": tier2_mass(w), "view3": view3_ritual(w)}

    with open(os.path.join(RESULTS, "subset_ritual_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    svg = build_svg(summary)
    with open(os.path.join(RESULTS, "subset_ritual_summary.svg"), "w", encoding="utf-8") as f:
        f.write(svg)

    # console breakdown
    for w in WORDS:
        t = summary[w]["tier2"]
        v = summary[w]["view3"]
        print(f"\n=== {w.upper()} ===")
        print(f"Tier 2 sacred frame: {t['prompt']!r}")
        print(f"  ritual-verb mass = {t['mass']:.4f}  "
              f"(liturgical {t['liturgical']:.4f} + ceremonial-secular {t['ceremonial_secular']:.4f})")
        for r in t["verbs"]:
            print(f"    {r['probability']:.4f}  {r['token']:<14} [{r['class']}]")
        print(f"View 3 ritual feature (L19 probe, matched sentence): "
              f"{v['activation']}  "
              + (f"#{v['feature_idx']} {v['description'][:55]}" if v["feature_idx"] is not None else "(none)"))
    print("\nsaved -> results/subset_ritual_summary.json, results/subset_ritual_summary.svg")


if __name__ == "__main__":
    main()
