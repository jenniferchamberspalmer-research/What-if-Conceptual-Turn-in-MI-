# PRE-REGISTRATION — V4-FINAL Field Probe (LOCKED)

**Instrument:** GPT-2 small · TransformerLens · confirmatory. The full locked
header below is reproduced verbatim from `v4final.py` as supplied; the code file
is byte-identical to what was frozen.

**Provenance note (honesty over ceremony):** this header was frozen inside the
instrument *before any results existed*, and both interpretation branches were
committed in writing at that time. However — unlike the dichotomy-probe trail,
where the pre-registration was *pushed* before any data — the first run of this
instrument happened on the local machine (2026-07-02) before this file was
pushed. The two-commit separation here isolates the frozen interpretation from
the results commit; it does not prove temporal priority of the push over the
data. The lock's force rests on the header being unedited, not on push order.

---

## THE QUESTION

A TRUE binary (heads/tails) has no middle: the world enforces two poles and
language names nothing between them. A FALSE binary (good/bad) is culturally
coded as two exhaustive poles, but the space between them is real and populated
(mediocre, okay, gray, mixed). Claim under test: the model carries a populated
FIELD between the poles of a false binary across depth, and that field decoheres
only at the readout, where prediction forces a single token to land. A true
binary has no field, so nothing to collapse.

## TYPOLOGY (locked)

    FALSE-BINARY (target)      good/bad         — culturally binary, real middle
    TRUE-BINARY  (controls)    heads/tails      — real binary, no middle
                               even/odd         — real binary, no middle
                               true/false       — real binary, abstract; 2nd control
    SCALAR       (reference)   hot/cold         — middle openly lexicalized (benchmark)
    FIELD-TEST   (discriminator) alive/dead     — see PRE-REGISTRATION below

## PRE-REGISTRATION — alive/dead (FROZEN before results exist; do not revise post-hoc)

alive/dead is culturally coded as a HARD binary (harder than good/bad) yet its
middle is corpus-populated (dying, comatose, brain-dead, terminal). It dissociates
cultural-binary-CODING from actual-distributed-FIELD — the two things good/bad
cannot separate. Both readings are committed now:

    BRANCH A  alive/dead shows the good/bad field signature
              => geometry tracks the REAL usage field, not the cultural label.
                 STRONG thesis: the effect survives a hard label. (confirming)

    BRANCH B  alive/dead shows the heads/tails no-field signature
              => geometry tracks the CULTURAL label / pole coding.
                 Thesis NARROWS to culturally-graded pairs. Still publishable.
                 (This is the branch that makes the control honest. If only Branch A
                  would have counted as meaningful, the forcing lived here — not in
                  the pair.)

## MECHANICAL GUARDS (a null can be an artifact, not a finding)

    G1 SEATING PRESENCE: verify alive/dead middles (dying, comatose, brain-dead)
       are actually representable in GPT-2 small before trusting a null. If they are
       too rare / degenerate, "no field" is a MODEL-SIZE null in a theory costume.
    G2 POLYSEMY: "dead" is heavily polysemous (dead battery, deadline, dead tired).
       A field around dead may be word-sense noise, not a mortality middle. Flag if
       "dead" is far more context-variable than "alive"; confirm any field is seated
       by MORTALITY terms, not polysemy.

## SCAFFOLDING (pile one — built in, not retrofittable)

- statistics: reps averaged over CARRIERS; spread over carriers => 95% CI
- null baseline: pole cosine reported as EXCESS over unrelated-pair anisotropy
- axis validity: poles must sit at the extremes; seating measured against control
- norm vs angle: cosine (angular) reported, not raw dot, so norm growth != rotation
- depth kept whole: every metric is a curve across layers; no readout window baked

---

## Disclosed at run time, before results were interpreted (not a revision of the lock)

The carriers all end with a period, so the locked read position `[0, -1]` is the
sentence-final "." token, not the word token, despite the in-file comment. This
was flagged before the first run. The locked instrument runs unmodified and is
canonical; a separate, clearly-labeled diagnostic (`v4final_posdiag.py`, read at
the word's last subtoken, position -2) exists only to answer the position
question with data. It is not the instrument and carries no pre-registered
interpretation.
