# Dichotomy Transformation Probe

## Open the results here

**https://jenniferchamberspalmer-research--dichotomy-probe-web.modal.run**

This web page is always on. It shows the results of the most recently run batch.
Its address never changes between updates. You can open it any time.

---

## Run and view (for the operator — no coding needed)

There are two steps: run a batch, then open the page.

**1. Run a batch.** Open a terminal in the project folder (the folder that contains
the `dichotomy_probe` directory) and paste this one line:

```
python -m modal run dichotomy_probe/run_probe.py
```

It runs the probe on a GPU in the cloud for a couple of minutes and then prints
`batch complete`. (On this Windows machine, if you ever see a `charmap` error,
first run `set PYTHONUTF8=1` in the same terminal, then paste the line again.)

**2. Open the page** and click **Reload latest batch**:

**https://jenniferchamberspalmer-research--dichotomy-probe-web.modal.run**

On the page, one button — **Download results** — downloads both result files
together as a single zip. Viewing and downloading are the same action; you never
need to hunt through the cloud filesystem.

That is the whole workflow. Everything below is documentation.

---

## What the two downloadable files contain

Both files are inside the one zip the **Download results** button gives you:

- **`measurements.csv`** — the per-layer measurements for every item and control,
  one measurement per row, openable directly in Excel. Every row also carries the
  fixed reporting-order sentence for that measurement, so each number is
  self-describing.
- **`results.html`** — a single self-contained, human-readable results page (the
  same page shown on the web). It carries the fixed reporting-order language and
  the constraints in full.

---

## What this probe is

It transforms a **cultural dichotomy already named in the corpus at entry**
(for example "A coin is either heads or tails", "A person is either good or bad")
through Gemma 2 2B, and records how the residual state at the target word
positions behaves across every layer, 0 to output.

It asks a **differential** question. A *true two-term opposition* (heads/tails,
even/odd) should exhaust its value field — the field is bipartite, nothing in the
middle carries value. A *scalar-collapse false dichotomy* (good/bad, hot/cold)
should leave **remainder** — value carried by excluded middle terms, the field is
graded. The probe describes where each item and control sits on that differential.

Two measurements, both run on every item and control, neither designated a success
in advance:

- **(a) Seated-midpoint** — whether a single-token candidate midpoint (e.g. *warm*
  between *hot* and *cold*) seats between the poles across layers. Defined only
  where a single-token midpoint exists. **Corroboration only.**
- **(b) Cross-layer pole relation** — how the two poles relate across layers, and
  how much value the excluded middle terms carry off the pole-to-pole axis
  (the value-exhaustion contrast). Reported as description, not prediction.

## Constraints this tool holds to, in plain language

- **The unit is the residual state at a position, never the token.** Tokenization
  fixes the position; the unit of analysis is the residual representation carried
  at that position; it is read across layers 0 to output.
- **The machine never decides true or false.** It transforms a naming the corpus
  already supplied. The verdict lives in the human reader. Every result sentence
  keeps one order: *the corpus named the dichotomy; the model processed that
  naming; the measurement recorded the processing; a human interprets the record.*
- **No story of the model "working it out" across layers.** Depth is
  legibility-by-depth for a human reader, not the model deliberating toward a
  verdict.
- **Cosine and seating are corroboration, never the claim.** The claim is the
  differential value-exhaustion; proximity results only corroborate it, and are
  labeled so everywhere.
- **Descriptive, not confirmatory.** No outcome is a "hit" set in advance. Both
  measurements are reported for every item and control, and any disconfirming
  control is reported flat.

## Version 2 — carrier controls and hidden-middle probes

v2 adds four carrier conditions per pair (matched-syntax, natural-usage,
explicit-dichotomy, neutral-relation), candidate middles for **every** pair, and
hidden-middle forcing prompts. The load-bearing correction: **the true dichotomies
(coin, even/odd) also receive candidate middles**, so "no middle" is a result to be
measured, never built into the design. Seating is judged *relative to each pair's own
unrelated-word baseline* (table/reason), because absolute off-axis magnitudes are not
interpretable in the model's 2304-dimensional residual space (the same
relative-to-control logic used elsewhere in this repo).

What the most recent batch shows, descriptively (the reader interprets; the machine
decides nothing): scalar middles seat between the poles (hot/cold → warm, mild,
lukewarm; wet/dry → damp, moist, humid, leaning toward *wet*); the good/bad middle
seats only as a **phrase** ("neither good nor bad", "both good and bad"), not as any
single token — an under-lexicalized center; and the true dichotomies coin and
even/odd have **no** candidate that seats — "no middle," measured rather than assumed.

## Pre-registration and the two-commit integrity trail

This is a portfolio piece, and its reliability rests partly on **pre-registration
preceding data**. The design and every pre-launch screen were frozen and committed
*before any data existed* — for both the original probe and the v2 extension, each as
its own two-commit sequence, deliberately **not** squashed:

- **[`PREREGISTRATION.md`](PREREGISTRATION.md)** — original design, committed alone
  before the first batch; then code + results committed separately after.
- **[`PREREGISTRATION_v2.md`](PREREGISTRATION_v2.md)** — the carrier-controls /
  hidden-middle extension (and the "no middle is not assumed" correction), committed
  alone before the v2 batch; then the v2 code + results committed separately after.

The sequence in the git history is the proof that pre-registration came first. One
calibration was disclosed after the first v2 run (seating judged control-relative
rather than by an uninformative absolute cutoff); it changes only the reader-facing
class label, not the measurements, and is documented on the results page rather than
by editing the frozen pre-registration.

## How it fits the rest of the repo

The probe reuses the existing tool's substrate as **shared imports, not copies**:
model loading and residual extraction live in
[`water_tool/core/extract.py`](../water_tool/core/extract.py), which both the Water
Pattern Tool and this probe import, so extraction cannot drift between the two
experiments. This probe runs as its **own** Modal app (`dichotomy-probe`) so
redeploying it never disturbs the live Water Pattern Tool; it shares the base image
and the Gemma weights cache volume.

### Redeploying the always-on page (for the maintainer)

```
python -m modal app stop dichotomy-probe
python -m modal deploy dichotomy_probe/run_probe.py
```

Stop the app before deploying (carry-forward protocol). The web endpoint is pinned
(`min_containers=1`) and CPU-only — it just serves the last batch — so its address
stays the same and its idle cost is small. The GPU runs only when you launch a
batch.
