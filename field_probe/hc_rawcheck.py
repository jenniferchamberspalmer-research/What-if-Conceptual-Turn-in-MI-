"""
hc_rawcheck.py — raw q vs lensed p for u+  (PILOT diagnostic, no pre-registration claim)
Tests whether u+'s late run-up is a real write toward the pole (raw q) or partly
LayerNorm renormalization (lensed p). Reuses hc_core; measures the SAME u+ feature
at the SAME activating positions as the pilot.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, matplotlib.pyplot as plt, torch
import hc_core as hc

LAYER, FEAT_POS = 7, 5518        # the myself/themselves u+ from the committed pilot
SAE_RELEASE, SAE_ID = "gpt2-small-res-jb", f"blocks.7.hook_resid_pre"
OUT = os.path.dirname(os.path.abspath(__file__)); EPS = 1e-9
POOL = [
    "The soup was far too hot to eat right away.", "The verdict was widely seen as deeply unjust.",
    "She spoke with genuine warmth and kindness.", "The results were completely unexpected and strange.",
    "It was the happiest day of her entire life.", "The bridge was extremely dangerous to cross.",
    "The painting was breathtakingly beautiful.", "The professor explained the theorem carefully.",
    "The soldiers were remarkably brave under fire.", "The treaty finally brought a lasting peace.",
]
print("*** PILOT DIAGNOSTIC — raw q vs lensed p — no pre-registration claim ***\n")
model = hc.load_model("gpt2", no_processing=True); dev = model.cfg.device
from sae_lens import SAE
_l = SAE.from_pretrained(SAE_RELEASE, SAE_ID, device=str(dev)); sae = (_l[0] if isinstance(_l, tuple) else _l).to(dev)
u = hc.unit(sae.W_dec[FEAT_POS].detach().float().to(dev))
W_U = model.W_U; a = int((W_U.T @ u).argmax()); b = int((W_U.T @ u).argmin())

@torch.no_grad()
def top_pos(f):
    best = (-1.0, POOL[0], 1)
    for p in POOL:
        _, _, c = hc.run_and_cache(model, p)
        rp = c["resid_pre", LAYER][0].float()
        if rp.shape[0] < 2: continue
        af = sae.encode(rp)[1:, f]; v = float(af.max())
        if v > best[0]: best = (v, p, int(af.argmax()) + 1)
    return best

actval, prompt, pos = top_pos(FEAT_POS)
print(f"measuring u+ (feat {FEAT_POS}) at its top firing site: act={actval:.2f}  pos={pos}  prompt={prompt!r}\n")
_, _, cache = hc.run_and_cache(model, prompt)
with torch.no_grad():   # mechanical: ln_final params track grad; trace is read-only
    out = hc.hc_trace(model, cache, u, a, b, pos=pos)
n = model.cfg.n_layers; blocks = np.arange(n)
q = out["projection"]["q_post"].cpu().numpy()      # RAW projection (writes live here)
p = out["projection"]["p_post"].cpu().numpy()      # LENSED projection (norm-contaminated late)

pd.DataFrame({"block": blocks, "q_raw_u+": q, "p_lensed_u+": p}).to_csv(f"{OUT}/hc_rawcheck.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].plot(blocks, q, "-o", color="#000000", label="raw q = <r, u+>")
ax[0].plot(blocks, p, "-o", color="#cc0000", label="lensed p = <LN_f(r), u+>")
ax[0].axhline(0, color="#999", lw=.8); ax[0].set_title("u+ raw vs lensed projection across depth")
ax[0].set_xlabel("block"); ax[0].set_ylabel("projection onto u+"); ax[0].legend()
ax[1].plot(blocks[1:], np.abs(np.diff(q)), "-o", color="#000000", label="|Δq| raw")
ax[1].plot(blocks[1:], np.abs(np.diff(p)), "-o", color="#cc0000", label="|Δp| lensed")
ax[1].set_title("velocity: raw vs lensed"); ax[1].set_xlabel("block"); ax[1].set_ylabel("|Δ|"); ax[1].legend()
fig.suptitle(f"hc PILOT diagnostic — u+ feat {FEAT_POS} — does the collapse survive in raw q?  [EXPLORATORY]", y=1.02)
fig.tight_layout(); fig.savefig(f"{OUT}/hc_rawcheck.png", dpi=140, bbox_inches="tight")
print("wrote hc_rawcheck.csv, hc_rawcheck.png")
