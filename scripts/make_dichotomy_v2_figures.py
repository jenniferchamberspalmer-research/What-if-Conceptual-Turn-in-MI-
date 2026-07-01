"""Publication-ready figures for the Dichotomy Transformation Probe v2.

Source: dichotomy_probe/results/measurements.csv (the tidy per-layer + summary rows
written by the v2 batch). That file already contains, as rows with carrier="summary",
the exact per-candidate summary values reported in results.html
(remainder_vs_control, carrier_stability, layer_of_strongest_seating,
descriptive_class). The remaining summaries (mean midpoint_t unforced, mean t forced,
mean cosine to frame term) are reconstructed here from the per-layer rows with the
SAME definitions used by measures.py:

  - mean_t_unforced   = mean midpoint_t over the two word-slot carriers
                        (matched_syntax, natural_usage), all layers.
  - mean_t_forced     = mean midpoint_t over the four forcing prompts
                        (forced_middle, ambiguity, neither, both), all layers.
  - mean_cos_frameterm= mean cosine_M_frameterm over the two word-slot carriers.

So no values are taken from results.html; everything comes from measurements.csv.

pandas + matplotlib only (no seaborn). Figures saved as PNG and SVG.

Run (from the project folder):
    python scripts/make_dichotomy_v2_figures.py
"""

import os
import matplotlib
matplotlib.use("Agg")  # headless: render straight to files
import matplotlib.pyplot as plt
import pandas as pd

# --- paths -------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CSV = os.path.join(REPO, "dichotomy_probe", "results", "measurements.csv")
OUT = os.path.join(REPO, "outputs", "dichotomy_probe_v2", "figures")
os.makedirs(OUT, exist_ok=True)

WORD_SLOT = ["matched_syntax", "natural_usage"]
FORCING = ["forced_middle", "ambiguity", "neither", "both"]

# Stable pair order + display names.
PAIR_ORDER = ["coin_heads_tails", "integer_even_odd", "temp_hot_cold",
              "moisture_wet_dry", "morality_good_bad"]
PAIR_LABEL = {
    "coin_heads_tails": "heads / tails (true)",
    "integer_even_odd": "even / odd (true)",
    "temp_hot_cold": "hot / cold (scalar)",
    "moisture_wet_dry": "wet / dry (scalar)",
    "morality_good_bad": "good / bad (scalar)",
}
PAIR_MARKER = {"coin_heads_tails": "o", "integer_even_odd": "s", "temp_hot_cold": "^",
               "moisture_wet_dry": "D", "morality_good_bad": "P"}

CLASS_COLOR = {
    "true_midpoint_candidate": "#1b7837",   # green
    "prompt_induced_candidate": "#d9a441",  # amber
    "frame_adjacent_candidate": "#3a68b0",  # blue
    "off_axis_candidate": "#888888",        # grey
    "one_pole_candidate": "#b2182b",        # red
}

GOOD_BAD_CANDS = ["neutral", "mixed", "ambiguous", "gray",
                  "morally ambiguous", "neither good nor bad", "both good and bad"]

# Two-line y-label so the (long) description does not clip along the rotated axis.
YLABEL_REM = "remainder / control baseline\n(< 1 = closer to axis than an unrelated word)"

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 15, "axes.labelsize": 13,
    "legend.fontsize": 10, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "figure.dpi": 120, "savefig.bbox": "tight",
})


def save(fig, stem):
    png, svg = os.path.join(OUT, stem + ".png"), os.path.join(OUT, stem + ".svg")
    fig.savefig(png)
    fig.savefig(svg)
    plt.close(fig)
    return png, svg


def build_candidate_summary(df):
    """Reconstruct the per-(pair,candidate) summary table from measurements.csv."""
    num = df.copy()
    num["value_f"] = pd.to_numeric(num["value"], errors="coerce")

    def mean_metric(carriers, metric):
        m = num[(num["carrier"].isin(carriers)) & (num["metric"] == metric)]
        return m.groupby(["pair", "target"])["value_f"].mean()

    mean_t_unf = mean_metric(WORD_SLOT, "midpoint_t").rename("mean_t_unforced")
    mean_t_for = mean_metric(FORCING, "midpoint_t").rename("mean_t_forced")
    mean_cos_ft = mean_metric(WORD_SLOT, "cosine_M_frameterm").rename("mean_cos_frameterm")

    summ = num[num["carrier"] == "summary"]
    def summary_metric(metric):
        s = summ[summ["metric"] == metric].set_index(["pair", "target"])
        return s["value"]
    rem_vs_ctrl = pd.to_numeric(summary_metric("remainder_vs_control"), errors="coerce").rename("remainder_vs_control")
    stability = pd.to_numeric(summary_metric("carrier_stability"), errors="coerce").rename("carrier_stability")
    seat_layer = pd.to_numeric(summary_metric("layer_of_strongest_seating"), errors="coerce").rename("layer_of_strongest_seating")
    dclass = summary_metric("descriptive_class").rename("descriptive_class")

    # input_type per (pair, target): take the first non-summary occurrence.
    itype = (num[num["carrier"] != "summary"]
             .groupby(["pair", "target"])["input_type"].first().rename("input_type"))

    out = pd.concat([mean_t_unf, mean_t_for, mean_cos_ft, rem_vs_ctrl, stability,
                     seat_layer, dclass, itype], axis=1).reset_index()
    out = out.rename(columns={"target": "candidate"})
    return out


def fig1(df, carrier, stem, subtitle):
    d = df[(df["metric"] == "cosine_AB") & (df["carrier"] == carrier)].copy()
    d["layer"] = pd.to_numeric(d["layer"], errors="coerce")
    d["value_f"] = pd.to_numeric(d["value"], errors="coerce")
    fig, ax = plt.subplots(figsize=(8, 5))
    for pair in PAIR_ORDER:
        s = d[d["pair"] == pair].sort_values("layer")
        if s.empty:
            continue
        ls = "-" if "true" in PAIR_LABEL[pair] else "--"
        ax.plot(s["layer"], s["value_f"], marker="o", ms=3, lw=1.8, ls=ls,
                label=PAIR_LABEL[pair])
    ax.set_xlabel("layer (0 = embedding, 26 = output)")
    ax.set_ylabel("cosine(A, B)")
    ax.set_title(f"Pole co-situation across depth\n{subtitle}", fontsize=15)
    ax.grid(True, alpha=0.3)
    ax.legend(title="pair (— true, -- scalar)", frameon=False,
              loc="lower right", fontsize=9)
    return save(fig, stem)


def fig2(summary):
    d = summary[summary["input_type"].isin(["candidate_middle", "logical", "phrase_middle"])].copy()
    fig, ax = plt.subplots(figsize=(11, 6.5))
    seen_classes = set()
    for _, r in d.iterrows():
        cls = r["descriptive_class"]
        color = CLASS_COLOR.get(cls, "#444444")
        ax.scatter(r["mean_t_unforced"], r["remainder_vs_control"],
                   marker=PAIR_MARKER.get(r["pair"], "o"), s=95,
                   facecolor=color, edgecolor="black", linewidth=0.5, alpha=0.9, zorder=3)
        ax.annotate(r["candidate"], (r["mean_t_unforced"], r["remainder_vs_control"]),
                    textcoords="offset points", xytext=(5, 3), fontsize=7.5, color="#333")
        seen_classes.add(cls)
    ax.axvline(0.3, color="grey", ls=":", lw=1)
    ax.axvline(0.7, color="grey", ls=":", lw=1)
    ax.axhline(0.65, color="grey", ls=":", lw=1)
    ax.set_xlabel("mean midpoint_t  (0 = pole A, 1 = pole B)")
    ax.set_ylabel(YLABEL_REM)
    ax.set_title("Middle seating separates scalar fields from true dichotomies")
    # two legends, placed OUTSIDE the axes so they never cover data.
    class_handles = [plt.Line2D([], [], marker="o", ls="", markerfacecolor=CLASS_COLOR[c],
                                markeredgecolor="black", label=c.replace("_candidate", ""))
                     for c in CLASS_COLOR if c in seen_classes]
    pair_handles = [plt.Line2D([], [], marker=PAIR_MARKER[p], ls="", markerfacecolor="white",
                               markeredgecolor="black", label=PAIR_LABEL[p]) for p in PAIR_ORDER]
    leg1 = ax.legend(handles=class_handles, title="class (colour)",
                     loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    ax.add_artist(leg1)
    ax.legend(handles=pair_handles, title="pair (marker)",
              loc="upper left", bbox_to_anchor=(1.02, 0.52), frameon=False)
    ax.grid(True, alpha=0.25)
    fig.subplots_adjust(right=0.78)
    return save(fig, "fig2_candidate_middle_seating")


def fig3(summary):
    d = summary[(summary["pair"] == "morality_good_bad")
                & (summary["candidate"].isin(GOOD_BAD_CANDS))].copy()
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for _, r in d.iterrows():
        is_phrase = " " in str(r["candidate"])
        color = CLASS_COLOR.get(r["descriptive_class"], "#444444")
        ax.scatter(r["mean_t_unforced"], r["remainder_vs_control"],
                   marker=("*" if is_phrase else "o"),
                   s=(320 if is_phrase else 120), facecolor=color,
                   edgecolor="black", linewidth=0.6, zorder=3)
        ax.annotate(r["candidate"], (r["mean_t_unforced"], r["remainder_vs_control"]),
                    textcoords="offset points", xytext=(7, 4),
                    fontsize=10, fontweight=("bold" if is_phrase else "normal"))
    ax.axvline(0.3, color="grey", ls=":", lw=1)
    ax.axvline(0.7, color="grey", ls=":", lw=1)
    ax.axhline(0.65, color="grey", ls=":", lw=1)
    ax.set_xlabel("mean midpoint_t  (0 = good, 1 = bad)")
    ax.set_ylabel(YLABEL_REM)
    ax.set_title("Good/bad middle structure appears at phrase level")
    handles = [plt.Line2D([], [], marker="*", ls="", ms=15, markerfacecolor="#bbb",
                          markeredgecolor="black", label="phrase middle (multi-token)"),
               plt.Line2D([], [], marker="o", ls="", ms=9, markerfacecolor="#bbb",
                          markeredgecolor="black", label="single-token middle")]
    ax.legend(handles=handles, frameon=False, loc="upper left")
    ax.grid(True, alpha=0.25)
    return save(fig, "fig3_good_bad_phrase_middles")


def main():
    df = pd.read_csv(CSV)
    summary = build_candidate_summary(df)

    paths = []
    paths += list(fig1(df, "matched_syntax", "fig1a_pole_cosine_matched_syntax",
                       "Carrier: matched_syntax"))
    paths += list(fig1(df, "explicit_dichotomy", "fig1b_pole_cosine_explicit_dichotomy",
                       "Carrier: explicit_dichotomy"))
    paths += list(fig2(summary))
    paths += list(fig3(summary))

    captions = os.path.join(OUT, "figure_captions.md")
    write_captions(captions, summary)

    print("\nWrote figures + captions:")
    for p in paths + [captions]:
        print("  " + p)


def write_captions(path, summary):
    gb = summary[(summary["pair"] == "morality_good_bad")
                 & (summary["candidate"].isin(GOOD_BAD_CANDS))]
    def rc(name):
        row = gb[gb["candidate"] == name]
        return f"{row['remainder_vs_control'].iloc[0]:.2f}" if len(row) else "—"
    md = f"""# Figure captions — Dichotomy Transformation Probe v2

Source data: `dichotomy_probe/results/measurements.csv` (Gemma 2 2B base, 27 residual
states read across layers 0 to output). Values are descriptive measurements; the
machine determines nothing. `remainder_vs_control` is a candidate's off-axis remainder
divided by the pair's unrelated-word (table/reason) baseline — below 1 means the
candidate sits closer to the pole axis than an unrelated word does.

## Figure 1 — Pole co-situation across depth

Two panels. **1A** (`fig1a_pole_cosine_matched_syntax`) and **1B**
(`fig1b_pole_cosine_explicit_dichotomy`) plot `cosine(A, B)` by layer for all five
pairs, under the matched-syntax carrier and the explicit-dichotomy carrier
respectively. One line per pair (solid = true dichotomy, dashed = scalar-collapse).
The panels show how the two poles co-situate as the residual state is read across
depth, and how that co-situation differs between a bare syntactic carrier and an
explicitly dichotomy-named carrier.

## Figure 2 — Middle seating separates scalar fields from true dichotomies

Scatter of every candidate middle (single-token candidates, logical terms, and
phrase middles; frame terms and unrelated controls excluded). x = mean `midpoint_t`
(0 at pole A, 1 at pole B); y = `remainder_vs_control`. Colour encodes the descriptive
class; marker encodes the pair. Dotted reference lines at t = 0.3 and 0.7 (the
between-poles band) and at remainder_vs_control = 0.65 (the seating threshold).
Candidates in the lower-middle region (between the poles and below the line) sit on the
axis more tightly than an unrelated word; candidates for the true dichotomies do not
fall there.

## Figure 3 — Good/bad middle structure appears at phrase level

The good/bad candidates only. x = mean `midpoint_t` (0 = good, 1 = bad); y =
`remainder_vs_control`. Stars mark multi-token phrase middles; circles mark
single-token candidates. Reference lines as in Figure 2. The phrase middles
"both good and bad" ({rc('both good and bad')}) and "neither good nor bad"
({rc('neither good nor bad')}) fall below the 0.65 line, while the single-token
candidates (neutral, mixed, ambiguous, gray) do not — the plotted structure is
located at the phrase level rather than at any single token.
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)


if __name__ == "__main__":
    main()
