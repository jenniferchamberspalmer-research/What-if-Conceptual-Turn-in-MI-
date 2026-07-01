# Figure handoff — Dichotomy Transformation Probe v2

Everything below is housed in GitHub.

- **Repo:** `jenniferchamberspalmer-research/What-if-Conceptual-Turn-in-MI-`
  (formerly `is-a-token-a-word-claude-water-pattern-tool-QV5SH`; the old name redirects)
- **Branch:** `dichotomy-probe`
- **Folder:** `outputs/dichotomy_probe_v2/figures/`

Each figure is provided as **SVG** (vector — use for the final paper) and **PNG**
(raster preview). If the repo is private, the raw links below require you to be
signed in; to let a chat assistant *see* the images, upload the PNGs to it directly —
it does not need to see them to write the figure references and captions.

---

## The three figures

| # | Files | Title | What it shows (one line) |
|---|-------|-------|--------------------------|
| 1A | `fig1a_pole_cosine_matched_syntax.{svg,png}` | Pole co-situation across depth (matched-syntax carrier) | cosine(A,B) by layer, one line per pair |
| 1B | `fig1b_pole_cosine_explicit_dichotomy.{svg,png}` | Pole co-situation across depth (explicit-dichotomy carrier) | cosine(A,B) by layer, one line per pair |
| 2 | `fig2_candidate_middle_seating.{svg,png}` | Middle seating separates scalar fields from true dichotomies | candidate seating scatter: mean midpoint_t vs remainder_vs_control |
| 3 | `fig3_good_bad_phrase_middles.{svg,png}` | Good/bad middle structure appears at phrase level | good/bad candidates only; phrase middles marked as stars |

Repo-relative paths (best for a paper build that has the repo checked out):

```
outputs/dichotomy_probe_v2/figures/fig1a_pole_cosine_matched_syntax.svg
outputs/dichotomy_probe_v2/figures/fig1b_pole_cosine_explicit_dichotomy.svg
outputs/dichotomy_probe_v2/figures/fig2_candidate_middle_seating.svg
outputs/dichotomy_probe_v2/figures/fig3_good_bad_phrase_middles.svg
```

Raw GitHub URLs (swap `.svg` → `.png` for the raster version):

```
https://raw.githubusercontent.com/jenniferchamberspalmer-research/What-if-Conceptual-Turn-in-MI-/dichotomy-probe/outputs/dichotomy_probe_v2/figures/fig1a_pole_cosine_matched_syntax.svg
https://raw.githubusercontent.com/jenniferchamberspalmer-research/What-if-Conceptual-Turn-in-MI-/dichotomy-probe/outputs/dichotomy_probe_v2/figures/fig1b_pole_cosine_explicit_dichotomy.svg
https://raw.githubusercontent.com/jenniferchamberspalmer-research/What-if-Conceptual-Turn-in-MI-/dichotomy-probe/outputs/dichotomy_probe_v2/figures/fig2_candidate_middle_seating.svg
https://raw.githubusercontent.com/jenniferchamberspalmer-research/What-if-Conceptual-Turn-in-MI-/dichotomy-probe/outputs/dichotomy_probe_v2/figures/fig3_good_bad_phrase_middles.svg
```

---

## Captions (use verbatim)

**Source note (applies to all):** Gemma 2 2B base; 27 residual states read across
layers 0 to output; values are descriptive measurements, the machine determines
nothing. `remainder_vs_control` is a candidate's off-axis remainder divided by the
pair's unrelated-word (table/reason) baseline — below 1 means the candidate sits
closer to the pole axis than an unrelated word does.

**Figure 1 (panels A and B).** Pole co-situation across depth. cosine(A,B) by layer
for all five pairs, under the matched-syntax carrier (1A) and the explicit-dichotomy
carrier (1B). One line per pair (solid = true dichotomy, dashed = scalar-collapse).
The panels show how the two poles co-situate as the residual state is read across
depth, and how that co-situation differs between a bare syntactic carrier and an
explicitly dichotomy-named carrier.

**Figure 2.** Middle seating separates scalar fields from true dichotomies. Scatter of
every candidate middle (single-token candidates, logical terms, and phrase middles;
frame terms and unrelated controls excluded). x = mean midpoint_t (0 at pole A, 1 at
pole B); y = remainder_vs_control. Colour encodes the descriptive class; marker encodes
the pair. Dotted reference lines at t = 0.3 and 0.7 (the between-poles band) and at
remainder_vs_control = 0.65 (the seating threshold). Candidates in the lower-middle
region sit on the axis more tightly than an unrelated word; the true-dichotomy
candidates do not fall there.

**Figure 3.** Good/bad middle structure appears at phrase level. The good/bad candidates
only. x = mean midpoint_t (0 = good, 1 = bad); y = remainder_vs_control. Stars mark
multi-token phrase middles; circles mark single-token candidates. Reference lines as in
Figure 2. The phrase middles "both good and bad" (0.57) and "neither good nor bad"
(0.63) fall below the 0.65 line, while the single-token candidates (neutral, mixed,
ambiguous, gray) do not — the plotted structure is located at the phrase level rather
than at any single token.

---

## Alt text (for accessibility / web posts)

- **Fig 1A/1B:** Line chart of cosine similarity between the two pole words by layer
  (0–26) for five word pairs; all rise from low similarity at the embedding to high
  similarity by the middle layers.
- **Fig 2:** Scatter plot; candidate words for scalar pairs (warm, mild, lukewarm,
  damp, moist, humid) and two good/bad phrases fall in a low-remainder band between the
  poles, while candidates for the true dichotomies (coin, even/odd) sit high.
- **Fig 3:** Scatter plot of good/bad candidates; the two multi-word phrase middles sit
  low (they seat), the single-word candidates sit high (they do not).

---

## Ready-to-paste embed code

**Markdown (Alignment Forum / web post)** — uses PNG:

```markdown
![Pole co-situation across depth (matched syntax)](outputs/dichotomy_probe_v2/figures/fig1a_pole_cosine_matched_syntax.png)
![Pole co-situation across depth (explicit dichotomy)](outputs/dichotomy_probe_v2/figures/fig1b_pole_cosine_explicit_dichotomy.png)
![Middle seating separates scalar fields from true dichotomies](outputs/dichotomy_probe_v2/figures/fig2_candidate_middle_seating.png)
![Good/bad middle structure appears at phrase level](outputs/dichotomy_probe_v2/figures/fig3_good_bad_phrase_middles.png)
```

**LaTeX** — PNG is safest; for SVG add `\usepackage{svg}` and use `\includesvg`:

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=.49\linewidth]{outputs/dichotomy_probe_v2/figures/fig1a_pole_cosine_matched_syntax.png}
  \includegraphics[width=.49\linewidth]{outputs/dichotomy_probe_v2/figures/fig1b_pole_cosine_explicit_dichotomy.png}
  \caption{Pole co-situation across depth: matched-syntax (left) and
  explicit-dichotomy (right) carriers. See caption text above.}
  \label{fig:pole-cosine}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=.85\linewidth]{outputs/dichotomy_probe_v2/figures/fig2_candidate_middle_seating.png}
  \caption{Middle seating separates scalar fields from true dichotomies.}
  \label{fig:seating}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=.75\linewidth]{outputs/dichotomy_probe_v2/figures/fig3_good_bad_phrase_middles.png}
  \caption{Good/bad middle structure appears at phrase level.}
  \label{fig:goodbad}
\end{figure}
```

---

## Note to the assistant integrating these

Please keep the figures exactly as provided (do not regenerate or restyle them). Use
the caption text verbatim; it is written to the study's constraints (the machine
determines nothing; the reader interprets; proximity/seating is corroboration only).
The figures are regenerated by `scripts/make_dichotomy_v2_figures.py` from
`dichotomy_probe/results/measurements.csv`.
