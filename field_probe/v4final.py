"""
V4-FINAL (LOCKED) — Does the Residual Stream Carry a FIELD for False Binaries?
=============================================================================
GPT-2 small · TransformerLens · confirmatory instrument.

THE QUESTION
------------
A TRUE binary (heads/tails) has no middle: the world enforces two poles and
language names nothing between them. A FALSE binary (good/bad) is culturally
coded as two exhaustive poles, but the space between them is real and populated
(mediocre, okay, gray, mixed). Claim under test: the model carries a populated
FIELD between the poles of a false binary across depth, and that field decoheres
only at the readout, where prediction forces a single token to land. A true
binary has no field, so nothing to collapse.

TYPOLOGY (locked)
-----------------
  FALSE-BINARY (target)      good/bad         — culturally binary, real middle
  TRUE-BINARY  (controls)    heads/tails      — real binary, no middle
                             even/odd         — real binary, no middle
                             true/false       — real binary, abstract; 2nd control
  SCALAR       (reference)   hot/cold         — middle openly lexicalized (benchmark)
  FIELD-TEST   (discriminator) alive/dead     — see PRE-REGISTRATION below

PRE-REGISTRATION — alive/dead  (FROZEN before results exist; do not revise post-hoc)
-----------------------------------------------------------------------------------
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

MECHANICAL GUARDS (a null can be an artifact, not a finding)
------------------------------------------------------------
  G1 SEATING PRESENCE: verify alive/dead middles (dying, comatose, brain-dead)
     are actually representable in GPT-2 small before trusting a null. If they are
     too rare / degenerate, "no field" is a MODEL-SIZE null in a theory costume.
  G2 POLYSEMY: "dead" is heavily polysemous (dead battery, deadline, dead tired).
     A field around dead may be word-sense noise, not a mortality middle. Flag if
     "dead" is far more context-variable than "alive"; confirm any field is seated
     by MORTALITY terms, not polysemy.

SCAFFOLDING (pile one — built in, not retrofittable)
  • statistics: reps averaged over CARRIERS; spread over carriers => 95% CI
  • null baseline: pole cosine reported as EXCESS over unrelated-pair anisotropy
  • axis validity: poles must sit at the extremes; seating measured against control
  • norm vs angle: cosine (angular) reported, not raw dot, so norm growth != rotation
  • depth kept whole: every metric is a curve across layers; no readout window baked

    pip install transformer_lens matplotlib pandas numpy
"""

import numpy as np, pandas as pd, matplotlib.pyplot as plt, torch
from transformer_lens import HookedTransformer

# ----------------------------------------------------------------------------- #
MODEL_NAME = "gpt2"
SEED       = 0
OUTDIR     = "."
EPS        = 1e-9
rng = np.random.default_rng(SEED)

# carriers: reps = mean over these; word is the LAST token in each (readout-aligned)
CARRIERS = ["It was {}.", "They called it {}.", "Everyone agreed it was {}.",
            "The verdict was {}.", "I would describe it as {}."]

# poles + candidate middles per pair. class drives interpretation, not the code path.
TYPOLOGY = {
    "good/bad":    dict(cls="false_binary", A=" good",  B=" bad",
                        mids=[" mediocre"," okay"," gray"," mixed"," average"]),
    "heads/tails": dict(cls="true_binary",  A=" heads", B=" tails",
                        mids=[" edge"," side"," rim"," face"]),
    "even/odd":    dict(cls="true_binary",  A=" even",  B=" odd",
                        mids=[" half"," between"," middle"]),
    "true/false":  dict(cls="true_binary",  A=" true",  B=" false",
                        mids=[" partly"," maybe"," unclear"," debatable"]),
    "hot/cold":    dict(cls="scalar",       A=" hot",   B=" cold",
                        mids=[" warm"," mild"," cool"," lukewarm"]),
    "alive/dead":  dict(cls="field_test",   A=" alive", B=" dead",
                        mids=[" dying"," comatose"," terminal"," failing"]),
}
CONTROL_WORD = " table"                                  # off-axis remainder reference
UNRELATED    = [(" window"," reason"), (" seven"," velvet"), (" copper"," idea")]

# ----------------------------------------------------------------------------- #
model = HookedTransformer.from_pretrained(MODEL_NAME); model.eval()
n, d = model.cfg.n_layers, model.cfg.d_model

def _cache(prompt):
    with torch.no_grad():
        _, c = model.run_with_cache(model.to_tokens(prompt))
    return c

def word_reps(word):
    """(n,d) reps averaged over carriers; also (C,n,d) for CI; + n_tokens of the word."""
    per = []
    for c in CARRIERS:
        cc = _cache(c.format(word))
        per.append(np.stack([cc["resid_post", l][0, -1].float().cpu().numpy() for l in range(n)]))
    per = np.stack(per)                                  # (C, n, d)
    ntok = len(model.to_str_tokens(word, prepend_bos=False))
    return per.mean(0), per, ntok

def cos(a, b): return np.sum(a*b, -1) / (np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)+EPS)

# precompute reps for every word we touch
ALL_WORDS = sorted({CONTROL_WORD} |
    {w for p in TYPOLOGY.values() for w in [p["A"], p["B"], *p["mids"]]} |
    {w for pr in UNRELATED for w in pr})
REPS, REPS_ALL, NTOK = {}, {}, {}
for w in ALL_WORDS:
    REPS[w], REPS_ALL[w], NTOK[w] = word_reps(w)

# anisotropy baseline: mean cross-depth cosine of unrelated pairs (per layer)
aniso = np.mean([cos(REPS[a], REPS[b]) for a, b in UNRELATED], axis=0)   # (n,)

# ----------------------------------------------------------------------------- #
# PER-PAIR ANALYSIS
# ----------------------------------------------------------------------------- #
def midpoint_t(w, A, B, l):
    ax = REPS[B][l] - REPS[A][l]
    return float((REPS[w][l]-REPS[A][l]) @ ax / (ax @ ax + EPS))

def off_axis_rem(w, A, B, l):
    ax = REPS[B][l] - REPS[A][l]
    t  = (REPS[w][l]-REPS[A][l]) @ ax / (ax @ ax + EPS)
    proj = REPS[A][l] + t*ax
    return float(np.linalg.norm(REPS[w][l] - proj))

def analyze(name):
    p = TYPOLOGY[name]; A, B, mids = p["A"], p["B"], p["mids"]
    L = np.arange(n)

    # (1) pole cosine across depth, as EXCESS over anisotropy; CI over carriers
    pole_cos = cos(REPS[A], REPS[B]); excess = pole_cos - aniso
    ci = 1.96*np.std([cos(REPS_ALL[A][k], REPS_ALL[B][k]) for k in range(len(CARRIERS))],
                     axis=0, ddof=1)/np.sqrt(len(CARRIERS))
    readout_drop = float(np.mean(excess[3:-2]) - excess[-1])   # + => collapses at readout

    # (2) FIELD: do candidate middles seat between poles? (t in .3-.7, low remainder)
    ctrl_rem = np.array([off_axis_rem(CONTROL_WORD, A, B, l) for l in range(n)])
    seat_frac = {}
    for w in mids:
        t   = np.array([midpoint_t(w, A, B, l) for l in range(n)])
        rem = np.array([off_axis_rem(w, A, B, l) for l in range(n)])
        seat = (t > 0.3) & (t < 0.7) & (rem < 1.5*ctrl_rem)
        seat_frac[w] = float(seat.mean())
    field_score = float(np.mean(list(seat_frac.values())))     # 0..1

    return dict(name=name, cls=p["cls"], excess=excess, ci=ci,
                readout_drop=readout_drop, field_score=field_score,
                seat_frac=seat_frac)

R = {name: analyze(name) for name in TYPOLOGY}

# ----------------------------------------------------------------------------- #
# DISCRIMINATOR: which class does alive/dead pattern with? (interpretation FROZEN above)
# ----------------------------------------------------------------------------- #
def signature(r): return np.array([r["field_score"], r["readout_drop"]])
target   = signature(R["good/bad"])                            # false-binary signature
controls = np.mean([signature(R[k]) for k in ["heads/tails","even/odd","true/false"]], axis=0)
ad = signature(R["alive/dead"])
d_target, d_control = np.linalg.norm(ad-target), np.linalg.norm(ad-controls)
branch = "A (real field -> STRONG thesis)" if d_target < d_control else "B (cultural label -> NARROWED thesis)"

# GUARDS
g1 = {w: dict(ntok=NTOK[w], norm=float(np.linalg.norm(REPS[w][n//2])))
      for w in TYPOLOGY["alive/dead"]["mids"]}
dead_var  = float(np.mean(np.std(REPS_ALL[" dead"],  axis=0)))   # context variability
alive_var = float(np.mean(np.std(REPS_ALL[" alive"], axis=0)))
g2_flag = dead_var > 1.4*alive_var

# ----------------------------------------------------------------------------- #
# OUTPUT
# ----------------------------------------------------------------------------- #
tbl = pd.DataFrame([{ "pair":r["name"], "class":r["cls"],
    "field_score":r["field_score"], "readout_drop":r["readout_drop"] } for r in R.values()])
tbl.to_csv(f"{OUTDIR}/v4final_pairs.csv", index=False)
pd.set_option("display.float_format", lambda x: f"{x:7.3f}")
print("\n=== PER-PAIR (field_score: 0=no middle seats, 1=field present) ===")
print(tbl.to_string(index=False))
print(f"\nDISCRIMINATOR  alive/dead -> dist(false-binary)={d_target:.3f}  "
      f"dist(true-binary)={d_control:.3f}")
print(f"  => data falls in BRANCH {branch}")
print(f"     (interpretation was pre-registered in the header; this only reports which.)")
print("\nGUARD G1 (middle-word representability):")
for w,v in g1.items():
    flag = "  <-- rare/degenerate?" if (v['ntok']>2 or v['norm']<EPS) else ""
    print(f"   {w:>10}: tokens={v['ntok']} mid-layer-norm={v['norm']:.1f}{flag}")
print(f"GUARD G2 (polysemy): dead_var={dead_var:.2f} alive_var={alive_var:.2f}"
      f"  {'!! dead is far more context-variable — check field is mortality, not sense' if g2_flag else 'ok'}")

# PLOT: pole-cosine excess across depth, by class
fig, ax = plt.subplots(figsize=(11,6))
style = {"false_binary":("#227744","-",2.4), "true_binary":("#aa2266","--",1.6),
         "scalar":("#2266aa",":",1.8), "field_test":("#cc7700","-",2.6)}
for r in R.values():
    c,ls,lw = style[r["cls"]]
    ax.plot(np.arange(n), r["excess"], ls, color=c, lw=lw, label=f'{r["name"]} [{r["cls"]}]')
    ax.fill_between(np.arange(n), r["excess"]-r["ci"], r["excess"]+r["ci"], color=c, alpha=.12)
ax.axhline(0, color="#bbb", lw=.7)
ax.set_title("Pole co-situation across depth (excess over anisotropy)\n"
             "field persists then collapses at readout = false-binary signature")
ax.set_xlabel("layer"); ax.set_ylabel("cos(A,B) − unrelated baseline")
ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(f"{OUTDIR}/v4final_field.png", dpi=140, bbox_inches="tight")
print("\nwrote: v4final_pairs.csv, v4final_field.png")
