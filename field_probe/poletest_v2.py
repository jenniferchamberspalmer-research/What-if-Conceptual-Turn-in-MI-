"""
Pole-adjacency test v2 — does onplane detect MIDDLES, or just DOMAIN membership?

Four probes on the hot/cold axis, plus the random-null baseline:
  " warm"        genuine MIDDLE (between the poles)          -> should score HIGH if onplane works
  " freeze"      POLE-ADJACENT (cold-like, not a middle)     -> confounds proximity w/ domain
  " thermometer" IN-DOMAIN, neither pole nor middle          -> isolates DOMAIN membership
  " chair"       OUT-OF-DOMAIN reference                     -> should score LOW
  random words   the null floor by construction (~0.25 onplane)

DIAGNOSIS LOGIC
  - warm >> (freeze, thermometer, chair)  => onplane finds middles. It works.
  - thermometer HIGH (~like warm/freeze)  => onplane is a DOMAIN detector: any
                                             temperature word seats. Airtight kill.
  - freeze HIGH but thermometer LOW        => onplane tracks pole-PROXIMITY (freeze
                                             sits near the cold pole), still not a middle.
  Either of the last two means onplane does NOT measure 'populated middle ground',
  and v3's Branch A (which rests entirely on onplane) does not survive.
"""

import numpy as np, torch
from transformer_lens import HookedTransformer

MODEL_NAME="gpt2"; EPS=1e-9
CARRIERS=["It was{}","They called it{}","Everyone agreed it was{}",
          "The verdict was{}","I would describe it as{}"]

model=HookedTransformer.from_pretrained(MODEL_NAME); model.eval()
n=model.cfg.n_layers

def _cache(p):
    with torch.no_grad(): _,c=model.run_with_cache(model.to_tokens(p))
    return c

def word_reps(word):
    wtok=model.to_tokens(word,prepend_bos=False)[0]; k=int(wtok.shape[0]); per=[]
    for c in CARRIERS:
        cc=_cache(c.format(word))
        per.append(np.stack([cc["resid_post",l][0,-k:].float().mean(0).cpu().numpy()
                             for l in range(n)]))
    return np.stack(per).mean(0)

def rem(w_rep,A,B):
    ax=B-A; t=np.sum((w_rep-A)*ax,axis=-1)/(np.sum(ax*ax,axis=-1)+EPS)
    return np.linalg.norm(w_rep-(A+t[:,None]*ax),axis=-1)

A=word_reps(" hot"); B=word_reps(" cold")

RANDOM_POOL=[" river"," pencil"," market"," ceiling"," garden"," engine"," letter",
 " planet"," basket"," candle"," bridge"," mirror"," pocket"," ticket"," cotton",
 " valley"," anchor"," pillow"," meadow"," rocket"," ribbon"," saddle"," barrel",
 " lantern"," compass"," harvest"," cabinet"," blanket"," corridor"," napkin",
 " gravel"," turnip"," satchel"," almanac"," trolley"," kettle"," marble"," acorn"]
np.random.seed(0)
randw=list(np.random.choice(RANDOM_POOL,size=30,replace=False))
rand_rem=np.array([rem(word_reps(w),A,B) for w in randw])   # (30,n)
p25=np.percentile(rand_rem,25,axis=0)

def onplane(word): return float(np.mean(rem(word_reps(word),A,B)<p25))

probes={" warm":"MIDDLE"," freeze":"pole-adjacent"," thermometer":"in-domain, non-middle",
        " chair":"out-of-domain"}
print("\n=== POLE-ADJACENCY TEST v2 — hot/cold axis ===")
scores={}
for w,role in probes.items():
    scores[w]=onplane(w); print(f"  {w:>13} ({role:<22}) onplane = {scores[w]:.3f}")
rand_onplane=float(np.mean([np.mean(rand_rem[i]<p25) for i in range(len(randw))]))
print(f"  {'random words':>13} ({'null baseline':<22}) onplane = {rand_onplane:.3f}")

warm,frz,thm=scores[" warm"],scores[" freeze"],scores[" thermometer"]
print("\nDIAGNOSIS:")
if warm-max(frz,thm)>0.15:
    print("  onplane SEPARATES the middle from domain/proximity words -> it detects middles.")
elif thm>0.6:
    print("  thermometer (in-domain, NOT a middle) also seats -> onplane is a DOMAIN detector.")
    print("  => it measures semantic-plane membership, NOT populated middle ground.")
    print("  => v3 Branch A (carried entirely by onplane) does NOT survive.")
else:
    print("  freeze seats but thermometer doesn't -> onplane tracks pole-PROXIMITY, not middles.")
    print("  => still not a field metric; v3 Branch A does not survive.")
