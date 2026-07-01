# Pre-registration — Dichotomy Transformation Probe

**Pre-registered:** 2026-07-01 13:24 EDT (2026-07-01T17:24:57Z).
**Status at time of writing:** design and pre-launch screens complete; **no probe
data exists yet.** This document is committed *alone*, before any data, as the first
half of a two-commit integrity trail. The probe is run only after this commit is pushed.

This is a descriptive, exploratory probe. **No outcome is designated a hit in advance.**
Both measurements below are computed and reported for **every** item and control
regardless of result, and any disconfirming control is reported flat.

---

## 1. What this probe does, and does not, claim

The probe transforms a **cultural dichotomy already named in the corpus at entry**
through Gemma 2 2B (base) and records how the residual state at the target positions
behaves across all layers 0 to output.

- **The unit of analysis** is the residual representation carried at the token position,
  never the token itself. *Tokenization fixes the position; the unit is the residual
  state at that position; it is read across layers 0 to output.*
- **The machine never determines true or false.** It transforms a naming the corpus
  already carried; the verdict lives in the human reader. Fixed reporting order in every
  result sentence: *the corpus named the dichotomy; the model processed that naming; the
  measurement recorded the processing; a human interprets the record.* No sentence
  reaches past the naming to the thing named.
- **No developmental or trajectory framing.** Across-depth structure is
  legibility-by-depth, not the model deliberating toward a verdict. The state enters
  corpus-shaped and Saussure-arbitrary as an embedding (arbitrary in the sign sense, not
  random — training has deposited the corpus into the embedding). The layers transform an
  input that already carries the naming. Movement across depth is in the human's reading,
  not the model's cognition.
- **Positive distributional proximity (cosine, any intermediate-seating result) is
  corroboration only, never the claim.** The claim is differential — value-exhaustion.
- **The residual stream is the material trace (écriture).** Nothing here materializes a
  verdict in the stream.

## 2. The differential claim (stated, not predicted-as-hit)

A **true two-term opposition** is value-exhaustive: the field is bipartite, no third term
carries value. A **scalar-collapse false dichotomy** leaves remainder: value is carried
by excluded middle terms, the field is graded. The probe describes where each item and
control sits on this differential. It does not predict a winner; it reports the two
measurements flat.

## 3. Model stack

- Gemma 2 2B **base** (`google/gemma-2-2b`), bfloat16, greedy single forward pass
  (deterministic; no sampling).
- Residual read across **all 27 residual states**: index 0 = the embedding (input) state,
  indices 1..26 = the outputs of the 26 decoder blocks. "Layers 0 to output."
- Extraction is single-sourced in `water_tool/core/extract.py` (shared with the Water
  Pattern Tool; the block-output residual path is proven numerically identical to the
  full sweep, so the two experiments cannot drift).
- Gemma Scope SAEs are available in the shared substrate but are not required for the two
  geometric measurements below.

## 4. Items and controls (frozen)

The full sentence is the sense-fixing frame. Poles are read at their token positions in
the culturally-named dichotomy frame. The candidate midpoint, excluded terms, and null
words are read at their token positions in a matched carrier frame (`<stem> ___.`) that
fixes the same sense; the poles are also read in the carrier so measurement (a) seating
and measurement (b) remainder share one consistent A–B axis. The dichotomy-frame pole
reading is reported separately as the poles "as the corpus names them."

| id | role | kind | named dichotomy | dichotomy frame | carrier frame | midpoint |
|----|------|------|-----------------|-----------------|---------------|----------|
| coin_heads_tails | item | true dichotomy | heads / tails | "A coin is either heads or tails." | "The coin came up ___." | — (none) |
| person_good_bad | item | scalar-collapse | good / bad | "A person is either good or bad." | "The person was ___." | — (phrasal; see §7) |
| integer_even_odd | control | true dichotomy | even / odd | "An integer is either even or odd." | "The number is ___." | — (none) |
| water_hot_cold | control | scalar-collapse | hot / cold | "The water is either hot or cold." | "The water was ___." | warm |
| cloth_wet_dry | control | scalar-collapse | wet / dry | "The cloth is either wet or dry." | "The cloth was ___." | damp |
| null_table_reason | null | null pair | (none — by construction) | (none — by construction) | "The ___ is there." | — (none) |

**Excluded-term sets** (measurement b remainder; single-token only after the screen):
- good/bad: mediocre, average, okay, fine, decent, poor
- hot/cold: cool, mild, lukewarm  *(tepid dropped — splits, see §6)*
- wet/dry: moist, humid, soggy
- even/odd and coin: deliberately empty (a true dichotomy has no excluded middle; a
  near-zero / undefined remainder is the contrast, reported flat).
- null pair: none (no opposition; anchors the floor).

## 5. The two measurements (both run on every item, neither designated success)

**(a) Seated-midpoint** — defined only where a single-token midpoint exists. Per layer,
for candidate midpoint M and poles A, B (carrier reads):
- projection coordinate `t = ⟨M−A, B−A⟩ / ‖B−A‖²` (0 at A, 1 at B),
- perpendicular residual `‖(M−A) − t·(B−A)‖` (normalized by ‖B−A‖),
- cosine(M,A), cosine(M,B).

"Seated between" = `0 < t < 1` with small perpendicular. **This is corroboration only,
never the claim.** Reported for hot/cold (warm) and wet/dry (damp). Undefined by
construction for the two true dichotomies and the null pair, reported flat.

**(b) Cross-layer pole relation (value-exhaustion contrast)** — per layer, for poles A, B:
- cosine(A,B) in the named dichotomy frame ("as the corpus names them"),
- cosine(A,B) in the carrier frame,
- **remainder ratio** = mean perpendicular energy of the excluded-term set off the A–B
  axis, divided by ‖B−A‖ (how much value the excluded terms carry *outside* the two-term
  axis). Low remainder ⇒ field bipartite / value-exhaustive; high remainder ⇒ graded
  field, value carried by excluded terms. **Reported as description, not prediction.**

## 6. Pre-launch screen results (before any data)

Tokenizer: `google/gemma-2-2b`. All words run through the Gemma tokenizer in-frame with
the leading space the model actually sees. Full machine-readable output:
`dichotomy_probe/results/screen_results.json`.

**Field-omission check:** PASS — no required field left blank; every `None` (dichotomy
frame for the null pair; midpoints for true dichotomies and good/bad) is explained by an
accompanying note field.

**Tokenizer single-token check + scope gate:**

| word | standalone (leading-space) | single-token? |
|------|----------------------------|:-------------:|
| heads | `[' heads']` | YES |
| **tails** | `[' tails']` | **YES** |
| good | `[' good']` | YES |
| bad | `[' bad']` | YES |
| even | `[' even']` | YES |
| odd | `[' odd']` | YES |
| hot | `[' hot']` | YES |
| cold | `[' cold']` | YES |
| warm | `[' warm']` | YES |
| wet | `[' wet']` | YES |
| dry | `[' dry']` | YES |
| damp | `[' damp']` | YES |
| table | `[' table']` | YES |
| reason | `[' reason']` | YES |
| mediocre, average, okay, fine, decent, poor | each single | YES |
| cool, mild, lukewarm | each single | YES |
| moist, humid, soggy | each single | YES |
| **tepid** | `[' tep', 'id']` | **NO → dropped** |

**Reported before any data run, per the single-token scope gate:**
- **`tails` is a single token** (`[' tails']`) — the anticipated split risk (`tail`+`s`)
  did **not** materialize. Item 1 ("A coin is either heads or tails.") stands as
  specified; no reframe or deferral is needed.
- All poles and both defined midpoints (`warm`, `damp`) are single-token in-frame; word,
  token, and Y-unit are coextensive for every read position.
- **`tepid` splits** into `[' tep', 'id']` and is **dropped** from the hot/cold
  excluded-term set by the scope gate (a multi-token excluded term would be multiple
  Y-units and confound the measure). The drop is recorded here and reflected in the run.

**Homograph screen** (odd, bad, heads, tails, and others; human-confirmed the frame
fixes the intended sense):
- heads / tails — strong non-coin senses (body part; animal tail). Frame "A coin is
  either heads or tails." fixes the coin sense. Confirmed.
- bad — slang "good" sense. Frame "A person is either good or bad." fixes the evaluative
  sense. Confirmed.
- odd — strange-sense homograph (peculiar). Frame "An integer is either even or odd."
  fixes the parity sense. Confirmed.
- cool — slang sense; frame "The water is either hot or cold." fixes the temperature
  sense. Confirmed.

## 7. Anticipated non-closure on good/bad (stated in advance — this is the finding)

For **good/bad the moral midpoint is phrasal and multi-token** ("morally grey", "neither
good nor bad"), so it cannot close on a single Y-unit. **Measurement (a) is therefore not
run with a single-token midpoint for good/bad, by design.** This non-closure is
anticipated and is **itself the finding**: the moral field is used gradiently but is
**under-lexicalized at its center**.

The interlock is stated explicitly, in advance: on good/bad, measurement (a) breaks
**exactly where the midpoint fails to fit the unit** — that breakage is the finding, not a
failed run. Measuring a multi-token midpoint against single-token poles would be a
unit-mismatch, so no single-token midpoint is asserted for good/bad; measurement (b)'s
excluded-term remainder (mediocre, average, okay, fine, decent, poor — all single-token)
carries the gradient evidence instead. The asymmetry (single-token poles, no single-token
midpoint) is flagged so it reads as the finding, not sloppiness.

## 8. N and robustness

Primary read is deterministic: one base model, one greedy forward pass per frame, no
sampling — so the per-frame measurement is exact and reproducible, not a sample estimate.
Robustness is addressed by the multi-control design (two independent true dichotomies:
coin, even/odd; two independent scalar-collapse controls with defined midpoints: hot/cold,
wet/dry) plus the null pair as a floor, rather than by resampling a stochastic output.
The exact frames, the leading-space tokenization actually seen, and the 27-state layer
range are all recorded above and emitted with the results.

## 9. Two-commit integrity trail

1. **This file, committed alone and pushed, before any data exists** (commit 1).
2. The probe code, the run, and the results are committed and pushed **separately and
   later** (commit 2). The two commits are deliberately **not** squashed: the sequence is
   the portfolio's proof that pre-registration preceded data.
