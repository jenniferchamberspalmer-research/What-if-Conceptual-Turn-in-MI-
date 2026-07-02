"""
FIELD PROBE v3 — On-Plane Seating vs a Random-Word Null
=======================================================
GPT-2 small · TransformerLens · confirmatory instrument. Builds on v2 (read-position
fixed: reps mean-pooled over the word's own subword span).

WHY v3 EXISTS (what v2 revealed)
--------------------------------
At the word position, v2's seating test SATURATED: heads/tails "middles" seated at
0.917, even/odd at 1.000 — the criterion (midpoint_t in 0.3-0.7, rem < 1.5x control)
passes almost anything, because in an ANISOTROPIC residual stream a random word
already projects mid-axis. So field_score stopped discriminating. What still moved
was the DEPTH CURVE: good/bad rode highest in pole-cosine excess at every layer and
collapsed hardest at readout, while alive/dead tracked the true-binary bundle.

Two changes, both forced by v2's own data:
  (1) RANDOM-WORD NULL. Run the seating test on random words too. If they seat like
      the real candidates, that PROVES the old metric measures anisotropy, not
      middleness — and hands us the baseline the old test lacked.
  (2) ON-PLANE SEATING. A real middle isn't defined by WHERE it projects (t) but by
      whether it sits ON the pole plane: off-axis remainder BELOW the random null.
      Random words float orthogonal; a genuine middle is more on-line than chance.
  (+) CURVE-HEIGHT replaces the lossy endpoint readout_drop as the primary signal:
      mean pole-cosine excess over the mid-depth plateau, where good/bad separated.

INTEGRITY NOTE — this is a NEW operationalization, so its discriminator is a FRESH
pre-registration, not a re-run of the v2 lock. v2's BRANCH B stands as the verdict of
v2's operationalization; it is not overturned, it is superseded by a better metric.
The v3 discriminator features are (curve_height, onplane_frac), committed below.

PRE-REGISTRATION v3 — alive/dead (frozen before v3 numbers exist)
  BRANCH A  alive/dead patterns with good/bad on (curve_height, onplane_frac)
            => tracks the REAL usage field, not the cultural label. STRONG thesis.
  BRANCH B  alive/dead patterns with the true-binary bundle
            => tracks the cultural label. Thesis NARROWS to culturally-graded pairs.

  AND a standing hypothesis v2 raised, reported explicitly:
  GRADED   if readout-collapse / curve-height is a CONTINUUM across all pairs with
           no clean field-absent zero among the TRUSTWORTHY true binaries
           (excluding even/odd, known-compromised), then the present/absent field
           dichotomy is itself a false binary — the finding is a gradient, and that
           is the result, more Saussurean than either clean branch.

    pip install transformer_lens matplotlib pandas numpy
"""

import numpy as np, pandas as pd, matplotlib.pyplot as plt, torch
from transformer_lens import HookedTransformer

MODEL_NAME="gpt2"; SEED=0; OUTDIR="."; EPS=1e-9; N_RANDOM=30
rng=np.random.default_rng(SEED)

CARRIERS=["It was{}","They called it{}","Everyone agreed it was{}",
          "The verdict was{}","I would describe it as{}"]

TYPOLOGY={
 "good/bad":   dict(cls="false_binary",A=" good", B=" bad",
                    mids=[" mediocre"," okay"," gray"," mixed"," average"]),
 "heads/tails":dict(cls="true_binary", A=" heads",B=" tails",
                    mids=[" edge"," side"," rim"," face"]),
 "even/odd":   dict(cls="true_binary", A=" even", B=" odd",
                    mids=[" half"," between"," middle"]),
 "true/false": dict(cls="true_binary", A=" true", B=" false",
                    mids=[" partly"," maybe"," unclear"," debatable"]),
 "hot/cold":   dict(cls="scalar",      A=" hot",  B=" cold",
                    mids=[" warm"," mild"," cool"," lukewarm"]),
 "alive/dead": dict(cls="field_test",  A=" alive",B=" dead",
                    mids=[" dying"," comatose"," terminal"," failing"]),
}
UNRELATED=[(" window"," reason"),(" seven"," velvet"),(" copper"," idea")]
# random-word pool for the null (neutral, common; sampled seeded each run)
RANDOM_POOL=[" river"," pencil"," market"," ceiling"," garden"," engine"," letter",
 " planet"," basket"," candle"," bridge"," mirror"," pocket"," ticket"," cotton",
 " valley"," anchor"," pillow"," meadow"," rocket"," ribbon"," saddle"," barrel",
 " lantern"," compass"," harvest"," cabinet"," blanket"," corridor"," napkin",
 " gravel"," turnip"," satchel"," almanac"," trolley"," kettle"," marble"," acorn"]

model=HookedTransformer.from_pretrained(MODEL_NAME); model.eval()
n,d=model.cfg.n_layers,model.cfg.d_model

def _cache(p):
    with torch.no_grad(): _,c=model.run_with_cache(model.to_tokens(p))
    return c

def word_reps(word):
    wtok=model.to_tokens(word,prepend_bos=False)[0]; k=int(wtok.shape[0]); per=[]
    for c in CARRIERS:
        pr=c.format(word); toks=model.to_tokens(pr)[0]
        if not torch.equal(toks[-k:].cpu(),wtok.cpu()):
            print(f"  [warn] span mismatch '{word}' in «{c}»")
        cc=_cache(pr)
        per.append(np.stack([cc["resid_post",l][0,-k:].float().mean(0).cpu().numpy()
                             for l in range(n)]))
    per=np.stack(per); return per.mean(0),k

def cos(a,b): return np.sum(a*b,-1)/(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)+EPS)

random_words=list(rng.choice(RANDOM_POOL,size=N_RANDOM,replace=False))
ALL=sorted({w for p in TYPOLOGY.values() for w in [p["A"],p["B"],*p["mids"]]}
           |{w for pr in UNRELATED for w in pr}|set(random_words))
REPS={w:word_reps(w)[0] for w in ALL}
aniso=np.mean([cos(REPS[a],REPS[b]) for a,b in UNRELATED],axis=0)

def rem(w,A,B,l):
    ax=REPS[B][l]-REPS[A][l]; t=(REPS[w][l]-REPS[A][l])@ax/(ax@ax+EPS)
    return float(np.linalg.norm(REPS[w][l]-(REPS[A][l]+t*ax)))
def mid_t(w,A,B,l):
    ax=REPS[B][l]-REPS[A][l]; return float((REPS[w][l]-REPS[A][l])@ax/(ax@ax+EPS))

def analyze(name):
    p=TYPOLOGY[name]; A,B,mids=p["A"],p["B"],p["mids"]
    excess=cos(REPS[A],REPS[B])-aniso
    curve_height=float(np.mean(excess[3:-2]))                    # NEW primary signal
    readout_drop=float(np.mean(excess[3:-2])-excess[-1])        # kept for continuity
    # random-word null of off-axis remainder, per layer
    rand_rem=np.array([[rem(w,A,B,l) for l in range(n)] for w in random_words])  # (R,n)
    p25=np.percentile(rand_rem,25,axis=0)                       # per-layer null floor
    # OLD saturating seating (t-band + 1.5x ctrl) applied to RANDOM words -> proof
    ctrl=rand_rem.mean(0)
    def old_seat(w):
        return float(np.mean([(0.3<mid_t(w,A,B,l)<0.7) and (rem(w,A,B,l)<1.5*ctrl[l])
                              for l in range(n)]))
    old_real=float(np.mean([old_seat(w) for w in mids]))
    old_rand=float(np.mean([old_seat(w) for w in random_words]))
    # NEW on-plane seating: remainder below the random null floor
    onplane=float(np.mean([np.mean([rem(w,A,B,l)<p25[l] for l in range(n)]) for w in mids]))
    return dict(name=name,cls=p["cls"],excess=excess,curve_height=curve_height,
                readout_drop=readout_drop,old_real=old_real,old_rand=old_rand,onplane=onplane)

R={k:analyze(k) for k in TYPOLOGY}

# v3 discriminator: (curve_height, onplane) — FRESH pre-registration
def sig(r): return np.array([r["curve_height"],r["onplane"]])
tgt=sig(R["good/bad"]); ctl=np.mean([sig(R[k]) for k in ["heads/tails","true/false"]],axis=0)
ad=sig(R["alive/dead"]); dT=np.linalg.norm(ad-tgt); dC=np.linalg.norm(ad-ctl)
branch="A (real field / STRONG)" if dT<dC else "B (cultural label / NARROWED)"

# GRADED test: spread of curve_height among TRUSTWORTHY pairs (drop even/odd)
trust=[R[k]["curve_height"] for k in R if k!="even/odd"]
graded=(max(trust)-min(trust))>2*np.std([R[k]["curve_height"] for k in ["heads/tails","true/false"]])

tbl=pd.DataFrame([{ "pair":r["name"],"class":r["cls"],
    "curve_height":r["curve_height"],"readout_drop":r["readout_drop"],
    "onplane":r["onplane"],"old_real":r["old_real"],"old_rand":r["old_rand"] }
    for r in R.values()])
tbl.to_csv(f"{OUTDIR}/field_v3_pairs.csv",index=False)
pd.set_option("display.float_format",lambda x:f"{x:7.3f}")
print("\n=== v3 PER-PAIR ===")
print(tbl.to_string(index=False))
print("\nSATURATION PROOF: if old_rand ≈ old_real, the v2 seating metric measured")
print("anisotropy, not middleness. old_rand across pairs = "
      f"{[round(r['old_rand'],2) for r in R.values()]}")
print(f"\nDISCRIMINATOR v3  alive/dead: dist(good/bad)={dT:.3f} dist(true-binary)={dC:.3f}"
      f"  => BRANCH {branch}")
print("ON-PLANE seating (real middle sits below random null; higher = more field):")
for r in R.values(): print(f"   {r['name']:>11}: onplane={r['onplane']:.2f}")
print(f"\nGRADED hypothesis: curve_height among trustworthy pairs (even/odd excluded)")
print(f"   good/bad={R['good/bad']['curve_height']:.3f}  "
      f"true/false={R['true/false']['curve_height']:.3f}  "
      f"heads/tails={R['heads/tails']['curve_height']:.3f}  "
      f"alive/dead={R['alive/dead']['curve_height']:.3f}")
print(f"   => {'GRADIENT, no clean field-absent zero (present/absent is itself a false binary)' if graded else 'clusters (categorical field present/absent holds)'}")

fig,ax=plt.subplots(1,2,figsize=(15,6))
style={"false_binary":("#227744","-",2.6),"true_binary":("#aa2266","--",1.6),
       "scalar":("#2266aa",":",1.8),"field_test":("#cc7700","-",2.6)}
for r in R.values():
    c,ls,lw=style[r["cls"]]
    ax[0].plot(np.arange(n),r["excess"],ls,color=c,lw=lw,label=f'{r["name"]} [{r["cls"]}]')
ax[0].axhline(0,color="#bbb",lw=.7); ax[0].set_title("Pole co-situation excess across depth")
ax[0].set_xlabel("layer"); ax[0].set_ylabel("cos(A,B) − anisotropy"); ax[0].legend(fontsize=8)
names=[r["name"] for r in R.values()]
ax[1].bar(np.arange(len(names))-0.2,[r["old_real"] for r in R.values()],0.4,label="real mids",color="#227744")
ax[1].bar(np.arange(len(names))+0.2,[r["old_rand"] for r in R.values()],0.4,label="RANDOM words",color="#cc0000",alpha=.7)
ax[1].set_xticks(range(len(names))); ax[1].set_xticklabels(names,rotation=40,ha="right",fontsize=8)
ax[1].set_title("v2 seating metric: real vs random (overlap = metric is uninformative)")
ax[1].set_ylabel("old_field_score"); ax[1].legend()
fig.tight_layout(); fig.savefig(f"{OUTDIR}/field_v3.png",dpi=140,bbox_inches="tight")
print("\nwrote: field_v3_pairs.csv, field_v3.png")
