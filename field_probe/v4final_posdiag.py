"""
POSITION DIAGNOSTIC for V4-FINAL — NOT the locked instrument. NOT canonical.
============================================================================
The locked v4final.py reads resid_post at position -1. Every carrier ends
with a period, so -1 is the "." token, not the word. This diagnostic is the
IDENTICAL instrument with a single change: read at position -2, the last
subtoken of the inserted word (the "." is always a single token in GPT-2's
tokenizer, so -2 is the word's last subtoken for every word and carrier).

Purpose: mechanical check only — does the read position change which class
alive/dead patterns with? The pre-registered interpretation branches belong
to the locked run; this file exists so the position question is answered by
data rather than argument. Disclosed alongside, never merged into, v4final.py.
"""

import numpy as np, pandas as pd, matplotlib.pyplot as plt, torch
from transformer_lens import HookedTransformer

MODEL_NAME = "gpt2"
SEED       = 0
OUTDIR     = "."
EPS        = 1e-9
READ_POS   = -2          # <-- the single change from v4final.py (-1 there)
rng = np.random.default_rng(SEED)

CARRIERS = ["It was {}.", "They called it {}.", "Everyone agreed it was {}.",
            "The verdict was {}.", "I would describe it as {}."]

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
CONTROL_WORD = " table"
UNRELATED    = [(" window"," reason"), (" seven"," velvet"), (" copper"," idea")]

model = HookedTransformer.from_pretrained(MODEL_NAME); model.eval()
n, d = model.cfg.n_layers, model.cfg.d_model

def _cache(prompt):
    with torch.no_grad():
        _, c = model.run_with_cache(model.to_tokens(prompt))
    return c

def word_reps(word):
    per = []
    for c in CARRIERS:
        cc = _cache(c.format(word))
        per.append(np.stack([cc["resid_post", l][0, READ_POS].float().cpu().numpy()
                             for l in range(n)]))
    per = np.stack(per)
    ntok = len(model.to_str_tokens(word, prepend_bos=False))
    return per.mean(0), per, ntok

def cos(a, b): return np.sum(a*b, -1) / (np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)+EPS)

ALL_WORDS = sorted({CONTROL_WORD} |
    {w for p in TYPOLOGY.values() for w in [p["A"], p["B"], *p["mids"]]} |
    {w for pr in UNRELATED for w in pr})
REPS, REPS_ALL, NTOK = {}, {}, {}
for w in ALL_WORDS:
    REPS[w], REPS_ALL[w], NTOK[w] = word_reps(w)

aniso = np.mean([cos(REPS[a], REPS[b]) for a, b in UNRELATED], axis=0)

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
    pole_cos = cos(REPS[A], REPS[B]); excess = pole_cos - aniso
    ci = 1.96*np.std([cos(REPS_ALL[A][k], REPS_ALL[B][k]) for k in range(len(CARRIERS))],
                     axis=0, ddof=1)/np.sqrt(len(CARRIERS))
    readout_drop = float(np.mean(excess[3:-2]) - excess[-1])
    ctrl_rem = np.array([off_axis_rem(CONTROL_WORD, A, B, l) for l in range(n)])
    seat_frac = {}
    for w in mids:
        t   = np.array([midpoint_t(w, A, B, l) for l in range(n)])
        rem = np.array([off_axis_rem(w, A, B, l) for l in range(n)])
        seat = (t > 0.3) & (t < 0.7) & (rem < 1.5*ctrl_rem)
        seat_frac[w] = float(seat.mean())
    field_score = float(np.mean(list(seat_frac.values())))
    return dict(name=name, cls=p["cls"], excess=excess, ci=ci,
                readout_drop=readout_drop, field_score=field_score,
                seat_frac=seat_frac)

R = {name: analyze(name) for name in TYPOLOGY}

def signature(r): return np.array([r["field_score"], r["readout_drop"]])
target   = signature(R["good/bad"])
controls = np.mean([signature(R[k]) for k in ["heads/tails","even/odd","true/false"]], axis=0)
ad = signature(R["alive/dead"])
d_target, d_control = np.linalg.norm(ad-target), np.linalg.norm(ad-controls)
branch = "A-pattern (with false-binary)" if d_target < d_control else "B-pattern (with true-binary)"

g1 = {w: dict(ntok=NTOK[w], norm=float(np.linalg.norm(REPS[w][n//2])))
      for w in TYPOLOGY["alive/dead"]["mids"]}
dead_var  = float(np.mean(np.std(REPS_ALL[" dead"],  axis=0)))
alive_var = float(np.mean(np.std(REPS_ALL[" alive"], axis=0)))
g2_flag = dead_var > 1.4*alive_var

tbl = pd.DataFrame([{ "pair":r["name"], "class":r["cls"],
    "field_score":r["field_score"], "readout_drop":r["readout_drop"] } for r in R.values()])
tbl.to_csv(f"{OUTDIR}/v4final_posdiag_pairs.csv", index=False)
pd.set_option("display.float_format", lambda x: f"{x:7.3f}")
print("\n=== POSITION DIAGNOSTIC (read at word last subtoken, pos -2) — NOT CANONICAL ===")
print(tbl.to_string(index=False))
print(f"\nDISCRIMINATOR  alive/dead -> dist(false-binary)={d_target:.3f}  "
      f"dist(true-binary)={d_control:.3f}   => {branch}")
print("\nGUARD G1 (middle-word representability):")
for w,v in g1.items():
    flag = "  <-- rare/degenerate?" if (v['ntok']>2 or v['norm']<EPS) else ""
    print(f"   {w:>10}: tokens={v['ntok']} mid-layer-norm={v['norm']:.1f}{flag}")
print(f"GUARD G2 (polysemy): dead_var={dead_var:.2f} alive_var={alive_var:.2f}"
      f"  {'!! dead more context-variable' if g2_flag else 'ok'}")

fig, ax = plt.subplots(figsize=(11,6))
style = {"false_binary":("#227744","-",2.4), "true_binary":("#aa2266","--",1.6),
         "scalar":("#2266aa",":",1.8), "field_test":("#cc7700","-",2.6)}
for r in R.values():
    c,ls,lw = style[r["cls"]]
    ax.plot(np.arange(n), r["excess"], ls, color=c, lw=lw, label=f'{r["name"]} [{r["cls"]}]')
    ax.fill_between(np.arange(n), r["excess"]-r["ci"], r["excess"]+r["ci"], color=c, alpha=.12)
ax.axhline(0, color="#bbb", lw=.7)
ax.set_title("POSITION DIAGNOSTIC (word-token read, pos -2) — not the locked instrument")
ax.set_xlabel("layer"); ax.set_ylabel("cos(A,B) − unrelated baseline")
ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(f"{OUTDIR}/v4final_posdiag_field.png", dpi=140, bbox_inches="tight")
print("\nwrote: v4final_posdiag_pairs.csv, v4final_posdiag_field.png")
