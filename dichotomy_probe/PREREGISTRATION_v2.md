# Pre-registration Addendum v2 — Carrier controls & hidden-middle probes

**Pre-registered:** 2026-07-01 15:25 EDT (2026-07-01T19:25:50Z).
**Status at time of writing:** v2 design and v2 pre-launch screens complete; **no v2
probe data exists yet.** This addendum is committed *alone*, before v2 data, as the
first half of the v2 integrity trail (mirroring the original
[`PREREGISTRATION.md`](PREREGISTRATION.md)). The v2 probe is run only after this
commit is pushed.

This extends, and does not retract, the original pre-registration. All non-negotiable
constraints there still hold: the unit is the residual state at the position (never
the token); the machine never determines true/false; fixed reporting order in every
result sentence; no trajectory framing; cosine and seating are corroboration only;
descriptive and exploratory, not confirmatory.

## 1. The methodological correction that motivates v2

The original probe gave the true dichotomies (coin, even/odd) **no** candidate middle
and an empty excluded set. That risks **building "no middle" into the design** —
making the true/false contrast partly a design artifact rather than a measurement.

**v2 correction (load-bearing):** every pair, *including the true dichotomies*,
receives the same candidate middles and the same measurements. "No middle" is now a
result to be read off the geometry, never an assumption. If a true dichotomy has no
stable middle, that must be *measured*, not stipulated.

## 2. Carrier conditions (four, applied to every pair)

Each A/B pair is read under four carriers. The two "word-slot" carriers read one word
at a time at the `{word}` position (poles, each candidate, the frame term, and the
controls). The two "relational" carriers place both poles in one sentence and yield
the pole relation `cosine(A,B)` under a dichotomy-named vs. a neutral relation frame.

1. **matched_syntax** — `"The {frame_noun} was {word}."` (same structure for every
   pair; the syntax control — may sound unnatural, e.g. coin, which is the point).
2. **natural_usage** — the most natural English carrier per pair:
   heads/tails `"The coin came up {word}."`; even/odd `"The integer is {word}."`;
   hot/cold `"The temperature was {word}."`; wet/dry `"The surface was {word}."`;
   good/bad `"The action was {word}."`.
3. **explicit_dichotomy** — `"The dichotomy between {A} and {B} is familiar."`
4. **neutral_relation** — `"The relation between {A} and {B} is familiar."`
   (Does the effect require the word *dichotomy*, or is it visible under any relation?)

## 3. Pairs, frame terms, and candidate sets (frozen)

| pair | A / B | frame_noun | frame_term | candidate middles | logical | phrase middles |
|------|-------|-----------|-----------|-------------------|---------|----------------|
| coin_heads_tails | heads / tails | coin | coin | edge, side, rim, face | both, neither | — |
| integer_even_odd | even / odd | integer | integer | zero, half, fraction, decimal | both, neither | — |
| temp_hot_cold | hot / cold | temperature | temperature | warm, cool, mild, lukewarm | both, neither | — |
| moisture_wet_dry | wet / dry | surface | moisture | damp, moist, humid, soaked | both, neither | — |
| morality_good_bad | good / bad | action | morality | neutral, mixed, ambiguous, gray | both, neither | morally ambiguous; neither good nor bad; both good and bad |

Controls for every pair: `table`, `reason` (unrelated words; should not seat as
middles). Frame terms are read as reference points, expected close to both poles but
not to seat as a midpoint.

Stated in advance (interpretation guides, **not** pre-designated hits):
- **zero is not a middle between even and odd — zero is even.** This tests whether the
  model represents the parity relation or merely treats zero as special.
- edge/rim/side/face are physical, frame-adjacent; both/neither are logical pressure
  terms; for good/bad the true middle may be **phrasal**, not a single-token item —
  hence phrase middles are tested alongside single-token candidates.

## 4. Hidden-middle forcing prompts (four, per candidate)

- `"Between {A} and {B}, the middle case is {M}."`
- `"An ambiguous case between {A} and {B} is {M}."`
- `"Something neither {A} nor {B} is {M}."`
- `"Something both {A} and {B} is {M}."`

`{M}` always sits at the sentence end; it is read at its **last** occurrence so a
structural *both/neither* is never confused with the `{M}` slot. Forcing prompts test
whether a candidate can be *made* midpoint-like only when the prompt forces it.

## 5. Measurements (computed for every pair × carrier × candidate × layer 0..output)

`cosine_AB`, `cosine_MA`, `cosine_MB`, `midpoint_t` (projection of M on the A→B axis,
0 at A, 1 at B), `remainder_ratio` (M off the A→B axis, ÷ pole gap),
`cosine_M_frameterm`, `cosine_A/ B_frameterm`. Summaries per candidate:
`carrier_stability` (agreement of `midpoint_t` across the two word-slot carriers),
`layer_of_strongest_seating`, and a **descriptive class**.

**Descriptive classes** (offered to the reader as a summary, **never a verdict**):
`true_midpoint_candidate`, `one_pole_candidate`, `frame_adjacent_candidate`,
`off_axis_candidate`, `prompt_induced_candidate` (+ `frame_term_reference`). Heuristic
thresholds, published so they can be discounted: centered `|t−0.5| ≤ 0.15`, poleward
`≥ 0.35`, low remainder `≤ 0.60`, stable `≥ 0.70`. The raw per-layer numbers in
`measurements.csv` are the record; the reader interprets them.

**What would count as a hidden true-dichotomy middle** (stated in advance): a candidate
M moves toward `t ≈ 0.5` with manageable remainder, across **more than one** carrier,
not merely proximity to the frame term, and **not only** under the forced-middle
prompt. If M is midpoint-like *only* under forcing, it is prompt-induced, not naturally
represented — reported as such.

**What would support "true dichotomy has no middle"** (stated in advance): poles
co-situate strongly; the frame term is close to both poles; candidate middles do not
stably seat, instead attaching to one pole, to the frame, or remaining off-axis. This
is a possible *result*, reported flat — it is not assumed.

## 6. Unit-of-analysis note for multi-token candidates

Poles are single-token (screened, §7). Candidate middles and phrase middles **may** be
multi-token; each is read as the **mean of the residual states over its subtoken span**
at every layer (the phrase-vector pooling the Water tool already uses), and its
subtoken count is recorded. Poles remain single-token, so the A→B axis endpoints are
each one Y-unit.

## 7. v2 pre-launch screen results (before any v2 data)

Tokenizer `google/gemma-2-2b`; full output in
`dichotomy_probe/results/screen_results_v2.json`. Field-omission check PASS (and it now
enforces that every pair, including the true dichotomies, carries candidate middles).

- **All poles are single-token** in-frame: heads, tails, even, odd, hot, cold, wet,
  dry, good, bad.
- **All single-token candidates confirmed single-token**: edge, side, rim, face, zero,
  half, fraction, decimal, warm, cool, mild, lukewarm, damp, moist, humid, soaked,
  neutral, mixed, ambiguous, gray, both, neither; controls table, reason; frame terms
  coin, integer, temperature, moisture, morality.
- **Phrase middles are multi-token** and read by mean-pooling: `morally ambiguous`
  (2), `neither good nor bad` (4), `both good and bad` (4).
- **Homographs** surfaced and confirmed fixed by their frames: heads/tails (coin),
  even/odd (parity), bad (evaluative), gray (moral-neutral vs colour), face (coin face).

## 8. Two-commit integrity trail (v2)

1. **This addendum, committed alone and pushed, before any v2 data exists** (v2
   commit 1).
2. The v2 code and results are committed and pushed **separately and later** (v2
   commit 2). Not squashed: the sequence is the proof that pre-registration preceded
   data, for v2 as for v1.
