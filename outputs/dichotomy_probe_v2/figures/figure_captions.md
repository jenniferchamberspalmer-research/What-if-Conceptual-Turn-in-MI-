# Figure captions — Dichotomy Transformation Probe v2

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
"both good and bad" (0.57) and "neither good nor bad"
(0.63) fall below the 0.65 line, while the single-token
candidates (neutral, mixed, ambiguous, gray) do not — the plotted structure is
located at the phrase level rather than at any single token.
