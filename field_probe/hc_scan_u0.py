"""
hc_scan_u0.py — scan SAE dictionary for internal-AND-coherent u0 candidates
PILOT-GRADE SELECTION REPORT. No pre-registration claim; no trajectory measured.
The selection report proved u0 cannot be built from W_U (anything from the output
dictionary emits by construction). u0 must be FOUND in the stream. This scans all
SAE decoder features, keeps those that pass the emission certification (internal:
low concentration, low rho_worst) AND fire on real content (measurable), then prints
the top-30 by lowest emission-concentration with their +/- firing profiles for HUMAN
coherence judgment. Machine certifies internality; judgment picks real oppositions.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import hc_core as hc

LAYER = 7; EPS = 1e-9
MAX_RHO_WORST = 0.15     # certification: low single-contrast alignment (internal)
MAX_CONC10    = 0.010    # certification: low emission concentration (internal)
NULL_COS_MAX  = 0.30     # exclude parked/LN-null directions
MIN_CONTENT_HITS = 3     # must fire on non-BOS content in >= this many pool prompts
TOP_N = 30

POOL = [
    "The soup was far too hot to eat right away.", "The water in the lake felt icy and cold.",
    "He is one of the tallest players on the team.", "The kitten was tiny enough to fit in a pocket.",
    "The verdict was widely seen as deeply unjust.", "Everyone agreed the decision had been fair.",
    "She spoke with genuine warmth and kindness.", "His tone turned suddenly cruel and harsh.",
    "The ancient castle stood on the hill for centuries.", "They just built a brand new stadium downtown.",
    "The results were completely unexpected and strange.", "The outcome was ordinary and unremarkable.",
    "The room went utterly silent after the announcement.", "The crowd erupted into deafening noise.",
    "It was the happiest day of her entire life.", "He felt an overwhelming sense of sadness.",
    "The bridge was extremely dangerous to cross.", "The path ahead looked perfectly safe.",
    "The old man walked slowly down the empty street.", "The children ran quickly across the field.",
    "The painting was breathtakingly beautiful.", "The alley was filthy and repulsive.",
    "The professor explained the theorem clearly and carefully.", "The instructions were hopelessly confusing.",
    "The medicine tasted bitter on her tongue.", "The dessert was wonderfully sweet.",
    "A bright light appeared at the end of the tunnel.", "The cellar was pitch dark and damp.",
    "The soldiers were remarkably brave under fire.", "He was too cowardly to speak up.",
    "The treaty finally brought a lasting peace.", "The border erupted into open war.",
    "Her handwriting was neat and precise.", "The desk was a chaotic, messy pile.",
    "The stranger seemed oddly familiar somehow.", "Nothing about the place felt normal.",
    "The lecture was surprisingly interesting.", "The meeting was unbearably boring.",
    "The scientist measured the reaction precisely.", "The idea was abstract and hard to grasp.",
]

print("*** PILOT SELECTION REPORT — no pre-registration claim, no trajectory measured ***\n")
model = hc.load_model("gpt2", no_processing=True); dev = model.cfg.device; d = model.cfg.d_model
tok = model.tokenizer
from sae_lens import SAE
_l = SAE.from_pretrained("gpt2-small-res-jb", f"blocks.{LAYER}.hook_resid_pre", device=str(dev))
sae = (_l[0] if isinstance(_l, tuple) else _l).to(dev)
W_dec = sae.W_dec.detach().float().to(dev); d_sae = W_dec.shape[0]

@torch.no_grad()
def pass_a():
    store = []; acc = torch.zeros(d, device=dev); cnt = 0
    for p in POOL:
        _, _, c = hc.run_and_cache(model, p)
        rp = c["resid_pre", LAYER][0].float(); store.append(rp)
        acc += rp.sum(0); cnt += rp.shape[0]
    return store, hc.unit(acc / cnt)
resid_store, mu_hat = pass_a()
u_uniform = hc.unit(torch.ones(d, device=dev))

@torch.no_grad()
def scan(chunk=512):
    Wd = W_dec / (W_dec.norm(dim=-1, keepdim=True) + EPS); W_U = model.W_U; N = d_sae
    rw = torch.empty(N, device=dev); conc = torch.empty(N, device=dev)
    for i in range(0, N, chunk):
        blk = Wd[i:i+chunk]; push = blk @ W_U; push_c = push - push.mean(-1, keepdim=True)
        ii = push.argmax(-1); jj = push.argmin(-1)
        dw = (W_U[:, ii] - W_U[:, jj]).T; dw = dw / (dw.norm(dim=-1, keepdim=True) + EPS)
        rw[i:i+chunk] = (blk * dw).sum(-1)
        e = push_c.pow(2); conc[i:i+chunk] = torch.topk(e, 10, -1).values.sum(-1) / (e.sum(-1) + EPS)
    cos_uni = (Wd @ u_uniform).abs(); cos_mu = (Wd @ mu_hat).abs()
    return (rw.cpu().numpy(), conc.cpu().numpy(), cos_uni.cpu().numpy(), cos_mu.cpu().numpy())

@torch.no_grad()
def content_hits(min_act=1e-6):
    hits = torch.zeros(d_sae, device=dev)
    for rp in resid_store:
        if rp.shape[0] < 2: continue
        hits += (sae.encode(rp)[1:].max(0).values >= min_act).float()
    return hits.cpu().numpy()

print("scanning dictionary ...")
rw, conc, cos_uni, cos_mu = scan(); HITS = content_hits()
ok = ((np.abs(rw) <= MAX_RHO_WORST) & (conc <= MAX_CONC10) &
      (cos_uni <= NULL_COS_MAX) & (cos_mu <= NULL_COS_MAX) & (HITS >= MIN_CONTENT_HITS))
cand = np.where(ok)[0]
print(f"\n[certified internal AND content-firing] {len(cand)} / {d_sae} features pass all gates")
order = cand[np.argsort(conc[cand])][:TOP_N]   # lowest concentration first (most internal)

@torch.no_grad()
def profile(f, k=8):
    u = hc.unit(W_dec[f]); push = (model.W_U.T @ u); pc = push - push.mean()
    pos = [tok.decode([int(t)]) for t in torch.topk(pc, k).indices]
    neg = [tok.decode([int(t)]) for t in torch.topk(-pc, k).indices]
    return pos, neg

print(f"\n=== TOP {TOP_N} CERTIFIED-INTERNAL CANDIDATES (judge coherence of +/- clusters) ===")
for f in order:
    pos, neg = profile(int(f))
    print(f"\nfeat {int(f):5d}  rho_wst={rw[f]:+.3f} conc10={conc[f]:.4f} "
          f"cos_mu={cos_mu[f]:.3f} hits={int(HITS[f])}")
    print(f"   +  {', '.join(repr(x) for x in pos)}")
    print(f"   -  {', '.join(repr(x) for x in neg)}")
print("\nJudgment: find features whose +/- clusters read as a real conceptual OPPOSITION")
print("(like myself/themselves), not bookkeeping stems or glitch tokens. Nothing committed.")
