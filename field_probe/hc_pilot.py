"""
hc_pilot.py — MEASUREMENT ON REAL AXES  (PILOT, exploratory) — corrected selection
==================================================================================
*** PILOT / EXPLORATORY. NO PRE-REGISTRATION CLAIM. ***
Supersedes the first pilot run, whose AUTO-SELECTION fed the (correct) instrument
bad axes. hc_core is UNCHANGED — every fix here is upstream, in how u+/u0 and the
measurement positions are chosen. Corrections, mapped to the five defects:

  (1)/(2) u0 was the null/mean direction, and the "lowest rho_worst" rule was biased
          toward finding it. FIX: guard every candidate against the residual-stream
          mean direction and the LayerNorm-null (uniform) direction by cosine, and
          require a minimum emission norm to exclude dead rows. u0 = lowest rho_worst
          AMONG genuine, non-null content directions.
  (3)     u+ was one-sided (a suffix detector); argmax/argmin manufactured fake poles.
          FIX: a two-sidedness score (magnitude balance of the +/- poles); u+ must be
          genuinely two-sided. Report BOTH pole token-lists for human override.
  (4)     prompts never activated u+, so its flat trajectory measured ABSENCE, not
          held-ness. FIX: measure each axis at the positions where ITS feature actually
          fires (via sae.encode over a diverse pool), not at the last token of arbitrary
          prompts.
  (5)     conc10 is noise when emission -> 0. FIX: the emission-norm floor (above) keeps
          near-null vectors out of the metric entirely.

Also prints the pairwise cosines of the bottom rho_worst pool, to confirm/deny the
"same direction eight times" degeneracy the first run exposed.

Still a pilot: auto-selects with guards and REPORTS candidates so you override by
judgment. Nothing printed is a finding.

    pip install transformer_lens sae_lens torch pandas matplotlib numpy
    python field_probe/hc_pilot.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import hc_core as hc

# --------------------------------------------------------------------------- #
# CONFIG  (pilot placeholders — thresholds set by judgment, not pre-registered)
# --------------------------------------------------------------------------- #
SAE_RELEASE   = "gpt2-small-res-jb"
LAYER         = 7
SAE_ID        = f"blocks.{LAYER}.hook_resid_pre"
OUTDIR        = os.path.dirname(os.path.abspath(__file__))

FEAT_POS  = None      # override u+ feature index; None => auto (two-sided, high rho_worst)
FEAT_ZERO = None      # override u0 feature index; None => auto (non-null, low rho_worst)

NORM_FLOOR    = 5.0   # min emission_norm: excludes dead/near-null decoder rows
NULL_COS_MAX  = 0.30  # max |cos| to resid-mean OR uniform dir: excludes the parked/LN-null direction
TWO_SIDED_MIN = 0.50  # u+ only: min(|+pole|,|-pole|)/max(...) — both poles must be real
N_ACT_POS     = 6     # measure at the top-N feature-activating (prompt,pos) pairs
EPS = 1e-9

# Diverse pool so a picked feature has a chance to actually fire somewhere.
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
    "The engine roared loudly to life.", "The library was quiet and still.",
    "The treaty finally brought a lasting peace.", "The border erupted into open war.",
    "Her handwriting was neat and precise.", "The desk was a chaotic, messy pile.",
    "The stranger seemed oddly familiar somehow.", "Nothing about the place felt normal.",
    "The lecture was surprisingly interesting.", "The meeting was unbearably boring.",
]


# --------------------------------------------------------------------------- #
# SETUP
# --------------------------------------------------------------------------- #
print("*** PILOT / EXPLORATORY — NO PRE-REGISTRATION CLAIM ***\n")
model = hc.load_model("gpt2", no_processing=True)
dev = model.cfg.device
d = model.cfg.d_model

from sae_lens import SAE
_loaded = SAE.from_pretrained(SAE_RELEASE, SAE_ID, device=str(dev))
sae = _loaded[0] if isinstance(_loaded, tuple) else _loaded
sae = sae.to(dev)
W_dec = sae.W_dec.detach().float().to(dev)
d_sae = W_dec.shape[0]
print(f"loaded SAE {SAE_RELEASE}/{SAE_ID}  ->  {d_sae} features in {d}-d\n")


# --------------------------------------------------------------------------- #
# PASS A — cache resid_pre[LAYER] for the pool; accumulate the stream-mean dir
# --------------------------------------------------------------------------- #
@torch.no_grad()
def pass_a():
    stored = []
    acc = torch.zeros(d, device=dev); cnt = 0
    for p in POOL:
        _, _, cache = hc.run_and_cache(model, p)
        rp = cache["resid_pre", LAYER][0].float()        # [seq, d]
        stored.append((p, rp))
        acc += rp.sum(0); cnt += rp.shape[0]
    return stored, hc.unit(acc / cnt)

print("pass A: caching pool + stream-mean direction ...")
resid_store, mu_hat = pass_a()
u_uniform = hc.unit(torch.ones(d, device=dev))            # LayerNorm-null direction

# content-firing count per feature: non-BOS max activation over the pool,
# thresholded and summed. A feature is eligible for the u0 slot only if it
# actually fires on real content, not just on the BOS state.
@torch.no_grad()
def content_hits(min_act=1e-6):
    hits = torch.zeros(d_sae, device=dev)
    for (p, rp) in resid_store:
        if rp.shape[0] < 2:
            continue
        acts = sae.encode(rp)[1:]                 # drop BOS row
        hits += (acts.max(0).values >= min_act).float()
    return hits.cpu().numpy()

print("counting content activations per feature ...")
HITS = content_hits()          # [d_sae] : # of pool prompts where feature fires on content


# --------------------------------------------------------------------------- #
# FEATURE SCAN with guards
# --------------------------------------------------------------------------- #
@torch.no_grad()
def scan(chunk=512):
    Wd = W_dec / (W_dec.norm(dim=-1, keepdim=True) + EPS)
    W_U = model.W_U
    N = d_sae
    rw = torch.empty(N, device=dev); conc = torch.empty(N, device=dev)
    emis = torch.empty(N, device=dev); two = torch.empty(N, device=dev)
    isr = torch.empty(N, dtype=torch.long, device=dev)
    jsr = torch.empty(N, dtype=torch.long, device=dev)
    for i in range(0, N, chunk):
        blk = Wd[i:i + chunk]
        push = blk @ W_U
        push_c = push - push.mean(-1, keepdim=True)
        mx, ii = push.max(-1); mn, jj = push.min(-1)
        isr[i:i + chunk] = ii; jsr[i:i + chunk] = jj
        dw = (W_U[:, ii] - W_U[:, jj]).T
        dw = dw / (dw.norm(dim=-1, keepdim=True) + EPS)
        rw[i:i + chunk] = (blk * dw).sum(-1)
        e = push_c.pow(2)
        emis[i:i + chunk] = e.sum(-1).sqrt()
        conc[i:i + chunk] = torch.topk(e, min(10, e.shape[-1]), -1).values.sum(-1) / (e.sum(-1) + EPS)
        pos_mag = push_c.max(-1).values.clamp(min=0)
        neg_mag = (-push_c.min(-1).values).clamp(min=0)
        two[i:i + chunk] = torch.minimum(pos_mag, neg_mag) / (torch.maximum(pos_mag, neg_mag) + EPS)
    cos_uni = (Wd @ u_uniform).abs()
    cos_mu = (Wd @ mu_hat).abs()
    return dict(rw=rw.cpu().numpy(), conc=conc.cpu().numpy(), emis=emis.cpu().numpy(),
                two=two.cpu().numpy(), cos_uni=cos_uni.cpu().numpy(), cos_mu=cos_mu.cpu().numpy(),
                isr=isr.cpu().numpy(), jsr=jsr.cpu().numpy())

print("scanning features with null/two-sidedness guards ...")
S = scan()
def tok(t): return model.tokenizer.decode([int(t)])

# --- degeneracy check: pairwise cosine of the bottom-8 rho_worst pool --------- #
bottom8 = np.argsort(S["rw"])[:8]
Wb = (W_dec[bottom8] / (W_dec[bottom8].norm(dim=-1, keepdim=True) + EPS))
cosmat = (Wb @ Wb.T).cpu().numpy()
off = cosmat[~np.eye(8, dtype=bool)]
print(f"\n[degeneracy check] bottom-8 rho_worst pool: pairwise |cos| max={np.abs(off).max():.3f} "
      f"mean={np.abs(off).mean():.3f}  (near 1.0 => same direction repeated)")


# --------------------------------------------------------------------------- #
# SELECTION with guards
# --------------------------------------------------------------------------- #
alive = S["emis"] >= NORM_FLOOR
not_null = (S["cos_uni"] <= NULL_COS_MAX) & (S["cos_mu"] <= NULL_COS_MAX)
print(f"[guards] alive(emis>={NORM_FLOOR}): {alive.sum()}   "
      f"non-null(|cos|<= {NULL_COS_MAX}): {not_null.sum()}   both: {(alive & not_null).sum()}")

MIN_CONTENT_HITS = 3
fires = HITS >= MIN_CONTENT_HITS
pos_mask = alive & not_null & fires & (S["two"] >= TWO_SIDED_MIN)
zero_mask = alive & not_null & fires
print(f"[guards] fires-on-content(>={MIN_CONTENT_HITS} prompts): {fires.sum()}   "
      f"eligible u0 (alive & non-null & fires): {zero_mask.sum()}")

def pick(mask, key, how):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        idx = np.arange(len(key))
    return int(idx[(np.argmax if how == "max" else np.argmin)(key[idx])])

f_pos = FEAT_POS if FEAT_POS is not None else pick(pos_mask, S["rw"], "max")
f_zero = FEAT_ZERO if FEAT_ZERO is not None else pick(zero_mask, S["rw"], "min")

@torch.no_grad()
def detail(f, k=8):
    u = hc.unit(W_dec[f].clone())
    push = model.W_U.T @ u
    push_c = push - push.mean()
    hi = torch.topk(push_c, k).indices; lo = torch.topk(-push_c, k).indices
    top_pos = [(tok(t), push_c[t].item()) for t in hi]
    top_neg = [(tok(t), push_c[t].item()) for t in lo]
    return u, top_pos, top_neg

print("\n=== CHOSEN PAIR (pilot; override via FEAT_POS/FEAT_ZERO by judgment) ===")
for tag, f in (("u+", f_pos), ("u0", f_zero)):
    u, tp, tn = detail(f)
    print(f"\n[{tag}] feature {f}   rho_worst={S['rw'][f]:+.3f}  two_sided={S['two'][f]:.2f}  "
          f"emission_norm={S['emis'][f]:.2f}  conc10={S['conc'][f]:.4f}  "
          f"cos_uniform={S['cos_uni'][f]:.3f}  cos_streammean={S['cos_mu'][f]:.3f}  "
          f"content_hits={int(HITS[f])}")
    print("     + pole tokens: " + ", ".join(f"{t!r}({v:+.2f})" for t, v in tp))
    print("     - pole tokens: " + ", ".join(f"{t!r}({v:+.2f})" for t, v in tn))

u_pos = hc.unit(W_dec[f_pos].clone());  a_pos, b_pos = int(S["isr"][f_pos]), int(S["jsr"][f_pos])
u_zero = hc.unit(W_dec[f_zero].clone()); a_zero, b_zero = int(S["isr"][f_zero]), int(S["jsr"][f_zero])


# --------------------------------------------------------------------------- #
# ACTIVATING POSITIONS  (measure where the feature actually fires)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def activating(f, top=N_ACT_POS, min_act=1e-6):
    scored = []
    for (p, rp) in resid_store:
        acts = sae.encode(rp)
        if rp.shape[0] < 2:            # only BOS present; nothing to measure
            continue
        af = acts[1:, f]              # exclude position 0 (BOS)
        j = int(af.argmax()) + 1     # +1 to re-index into the full sequence
        v = float(af.max())
        if v >= min_act:              # require the feature to actually fire
            scored.append((v, p, j))
    scored.sort(reverse=True)
    return scored[:top]

@torch.no_grad()
def measure(f, u, a, b):
    acts = activating(f)
    n = model.cfg.n_layers
    P = np.zeros((len(acts), n)); QA = np.zeros((len(acts), n))
    QM = np.zeros((len(acts), n)); CU = np.zeros((len(acts), n))
    geom = None; drift = None; used = []
    for k, (actval, p, pos) in enumerate(acts):
        _, _, cache = hc.run_and_cache(model, p)
        out = hc.hc_trace(model, cache, u, a, b, pos=pos)
        P[k] = out["projection"]["p_post"].cpu().numpy()
        QA[k] = out["attribution"]["qa"].cpu().numpy()
        QM[k] = out["attribution"]["qm"].cpu().numpy()
        CU[k] = out["attribution"]["cancel_u"].cpu().numpy()
        used.append((actval, p, pos))
        if k == 0:
            W = out["writes"]; tr = [W["resid_pre"][0].cpu().numpy()]
            for l in range(n):
                tr.append(W["resid_mid"][l].cpu().numpy()); tr.append(W["resid_post"][l].cpu().numpy())
            geom = np.stack(tr); drift = out["lens_drift"]
    return dict(P=P, QA=QA, QM=QM, CU=CU, geom=geom, drift=drift, used=used)

print("\nmeasuring each axis at its own top-activating positions ...")
R_pos = measure(f_pos, u_pos, a_pos, b_pos)
R_zero = measure(f_zero, u_zero, a_zero, b_zero)
print(f"u+ activations used (max act, prompt, pos): "
      + "; ".join(f"{a:.2f}@{pos}" for a, p, pos in R_pos["used"]))
print(f"u0 activations used: "
      + "; ".join(f"{a:.2f}@{pos}" for a, p, pos in R_zero["used"]))

blocks = np.arange(model.cfg.n_layers)
pd.DataFrame({"block": blocks,
             "p_post_u+": R_pos["P"].mean(0), "p_post_u0": R_zero["P"].mean(0),
             "qa_u+": R_pos["QA"].mean(0), "qm_u+": R_pos["QM"].mean(0),
             "qa_u0": R_zero["QA"].mean(0), "qm_u0": R_zero["QM"].mean(0),
             "cancel_u+": R_pos["CU"].mean(0), "cancel_u0": R_zero["CU"].mean(0)}
             ).to_csv(f"{OUTDIR}/hc_pilot_trajectory.csv", index=False)

pd.DataFrame([dict(feature=int(f), role=r, rho_worst=float(S["rw"][f]), two_sided=float(S["two"][f]),
                   emission_norm=float(S["emis"][f]), conc10=float(S["conc"][f]),
                   cos_uniform=float(S["cos_uni"][f]), cos_streammean=float(S["cos_mu"][f]),
                   pole_pos=tok(S["isr"][f]), pole_neg=tok(S["jsr"][f]))
              for f, r in [(f_pos, "u+"), (f_zero, "u0")]]
             ).to_csv(f"{OUTDIR}/hc_pilot_candidates.csv", index=False)


# --------------------------------------------------------------------------- #
# FIGURE 1 — METRICS (held vs collapsed), measured where the features fire
# --------------------------------------------------------------------------- #
def band(ax, X, color, label):
    ax.plot(blocks, X.mean(0), "-o", ms=4, color=color, label=label, zorder=3)
    for row in X: ax.plot(blocks, row, "-", color=color, alpha=0.12, zorder=1)

fig, ax = plt.subplots(2, 2, figsize=(13, 9))
band(ax[0, 0], R_pos["P"], "#cc0000", "u+ (output-bound)")
band(ax[0, 0], R_zero["P"], "#0055cc", "u0 (internal-only)")
ax[0, 0].axhline(0, color="#999", lw=.8)
ax[0, 0].set_title("Projection onto u across depth  (collapse = runs to a pole; held = stays mid)")
ax[0, 0].set_xlabel("block"); ax[0, 0].set_ylabel("p = <LN_f(r), u>"); ax[0, 0].legend()

ax[0, 1].plot(blocks[1:], np.abs(np.diff(R_pos["P"].mean(0))), "-o", ms=4, color="#cc0000", label="u+")
ax[0, 1].plot(blocks[1:], np.abs(np.diff(R_zero["P"].mean(0))), "-o", ms=4, color="#0055cc", label="u0")
ax[0, 1].set_title("Projection velocity |Δp| per block"); ax[0, 1].set_xlabel("block")
ax[0, 1].set_ylabel("|Δp|"); ax[0, 1].legend()

ax[1, 0].plot(blocks, R_pos["QA"].mean(0), "-o", ms=4, color="#aa2266", label="<attn,u+>")
ax[1, 0].plot(blocks, R_pos["QM"].mean(0), "-o", ms=4, color="#2266aa", label="<mlp,u+>")
ax[1, 0].axhline(0, color="#999", lw=.8)
ax[1, 0].set_title("Write attribution onto u+  (same sign = co-directing)")
ax[1, 0].set_xlabel("block"); ax[1, 0].set_ylabel("write · u"); ax[1, 0].legend()

ax[1, 1].plot(blocks, R_pos["CU"].mean(0), "-o", ms=4, color="#cc0000", label="u+")
ax[1, 1].plot(blocks, R_zero["CU"].mean(0), "-o", ms=4, color="#0055cc", label="u0")
ax[1, 1].set_ylim(-0.05, 1.05)
ax[1, 1].set_title("Cancellation along u  (0 co-direct .. 1 cancel)")
ax[1, 1].set_xlabel("block"); ax[1, 1].set_ylabel("cancel_u"); ax[1, 1].legend()

fig.suptitle(f"hc PILOT (corrected selection) — u+ feat {f_pos} ({tok(a_pos)!r}/{tok(b_pos)!r})  "
             f"vs u0 feat {f_zero}   [EXPLORATORY]", y=1.01)
fig.tight_layout(); fig.savefig(f"{OUTDIR}/hc_pilot_metrics.png", dpi=140, bbox_inches="tight")


# --------------------------------------------------------------------------- #
# FIGURE 2 — GEOMETRY (PCA trajectory + axis directions, and lens drift)
# --------------------------------------------------------------------------- #
figg, axg = plt.subplots(1, 2, figsize=(15, 7))
traj = R_pos["geom"]; c = traj - traj.mean(0)
U, Sg, Vt = np.linalg.svd(c, full_matrices=False)
P2 = c @ Vt[:2].T; var2 = (Sg[:2] ** 2).sum() / (Sg ** 2).sum()
n = model.cfg.n_layers; kinds = (["attn", "mlp"] * n)
axg[0].plot(P2[:, 0], P2[:, 1], "-", color="#cccccc", lw=1, zorder=1)
for i in range(len(P2) - 1):
    col = "#aa2266" if kinds[i] == "attn" else "#2266aa"
    axg[0].annotate("", xy=P2[i + 1], xytext=P2[i], arrowprops=dict(arrowstyle="->", color=col, lw=1.4), zorder=2)
for u, col, lab in ((u_pos.cpu().numpy(), "#cc0000", "u+"), (u_zero.cpu().numpy(), "#0055cc", "u0")):
    d2 = Vt[:2] @ u; d2 = d2 / (np.linalg.norm(d2) + EPS) * (np.abs(P2).max() * 0.9)
    axg[0].annotate("", xy=d2, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=2.4), zorder=4)
    axg[0].text(d2[0], d2[1], f" {lab}", color=col, fontsize=11, weight="bold")
axg[0].scatter(*P2[0], s=80, color="#227744", zorder=3, label="start")
axg[0].scatter(*P2[-1], s=80, color="#111111", zorder=3, label="readout")
axg[0].set_title(f"Residual trajectory (PCA, {var2:.0%} var) with u+/u0 directions")
axg[0].set_xlabel("PC1"); axg[0].set_ylabel("PC2"); axg[0].legend(); axg[0].set_aspect("equal", "datalim")

drift = R_pos["drift"]; post = drift[drift["checkpoint"] == "resid_post"]
axg[1].plot(post["block"], post["kl_final_to_layer"], "-o", ms=4, color="#663399")
axg[1].set_title("Logit-lens drift  KL(final || layer)  (where the prediction commits)")
axg[1].set_xlabel("block"); axg[1].set_ylabel("KL to final"); axg[1].set_yscale("log")
figg.suptitle("hc PILOT (corrected selection) — geometry [EXPLORATORY]", y=1.02)
figg.tight_layout(); figg.savefig(f"{OUTDIR}/hc_pilot_geometry.png", dpi=140, bbox_inches="tight")

print("\nwrote: hc_pilot_candidates.csv, hc_pilot_trajectory.csv, "
      "hc_pilot_metrics.png, hc_pilot_geometry.png")
print("\n*** PILOT COMPLETE — exploratory only; numbers/figures inform selection, not findings. ***")
