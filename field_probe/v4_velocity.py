"""
V4 — Terminal Velocity of the Residual Stream
=============================================
An instrument for GPT-2 small (TransformerLens) that treats DEPTH as TIME and
measures the residual stream's velocity, acceleration, curvature, and the two
"forces" (attention write, MLP write) that drive it.

Grounding (not analogy): a residual network is a discretized ODE,
    h_{l+1} = h_l + f(h_l),   so   dh/dl  ≈  Δh_l = h_{l+1} - h_l   is VELOCITY.
GPT-2 adds to the stream twice per block, so the fine trajectory has 2*n_layers
steps that ALTERNATE: attention kick, then MLP kick.

Two honest cautions built in:
  1. GPT-2's residual stream is never normalized in place, so its NORM GROWS with
     depth. Raw ‖Δh‖ can rise for that reason alone. => we report RAW *and* RELATIVE
     velocity (‖Δh‖ / ‖h‖).
  2. There is no physical "drag" in a transformer. "Terminal velocity" is only
     earned if the two forces (attn, mlp) actually CANCEL, not merely shrink.
     => we report a cancellation index so the word has to earn its keep.

Outputs: a numbers CSV, a metrics figure (PNG), and a PCA geometry figure (PNG).

Run inside your V3 environment (Colab/Modal/local) where GPT-2 already loads.
    pip install transformer_lens matplotlib pandas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformer_lens import HookedTransformer

# ----------------------------------------------------------------------------- #
# CONFIG
# ----------------------------------------------------------------------------- #
MODEL_NAME = "gpt2"                                   # small: 12 blocks, d_model 768
PROMPT     = "On the deserted planet they discovered a"
POS        = -1          # token position to trace. -1 = last token = the prediction site
OUTDIR     = "."         # where CSV/PNGs are written
EPS        = 1e-9

# ----------------------------------------------------------------------------- #
# LOAD + FORWARD PASS WITH CACHE
# ----------------------------------------------------------------------------- #
model = HookedTransformer.from_pretrained(MODEL_NAME)
model.eval()

tokens = model.to_tokens(PROMPT)
with torch.no_grad():
    _, cache = model.run_with_cache(tokens)

n = model.cfg.n_layers
d = model.cfg.d_model

def vec(name, l):
    """Residual-stream vector at hook `name`, block `l`, at the traced position."""
    return cache[name, l][0, POS].float().cpu().numpy()

# Checkpoints. resid_mid = resid_pre + attn_out ; resid_post = resid_mid + mlp_out
resid_pre  = np.stack([vec("resid_pre",  l) for l in range(n)])   # entering block l
resid_mid  = np.stack([vec("resid_mid",  l) for l in range(n)])   # after attention adds
resid_post = np.stack([vec("resid_post", l) for l in range(n)])   # after MLP adds
attn_out   = np.stack([vec("attn_out",   l) for l in range(n)])   # the attention FORCE
mlp_out    = np.stack([vec("mlp_out",    l) for l in range(n)])   # the MLP FORCE

# ----------------------------------------------------------------------------- #
# BUILD THE FINE TRAJECTORY  (the "twice per block" path)
#   pre[0] -> mid[0] -> post[0] -> mid[1] -> post[1] -> ... -> post[n-1]
#   2n steps, alternating attn / mlp.  Each step IS one force write.
# ----------------------------------------------------------------------------- #
traj, kind, label = [resid_pre[0]], [], []
for l in range(n):
    traj.append(resid_mid[l]);  kind.append("attn"); label.append(f"L{l}·attn")
    traj.append(resid_post[l]); kind.append("mlp");  label.append(f"L{l}·mlp")
traj  = np.stack(traj)                # (2n+1, d)  the positions
steps = np.diff(traj, axis=0)         # (2n,   d)  the velocity vectors (= the writes)
kind  = np.array(kind)

def nrm(x, axis=-1):
    return np.linalg.norm(x, axis=axis)

def cosine(a, b):
    return np.sum(a * b, -1) / (nrm(a) * nrm(b) + EPS)

# ----------------------------------------------------------------------------- #
# VELOCITY  (raw and relative — relative controls for norm growth)
# ----------------------------------------------------------------------------- #
speed      = nrm(steps)                         # ‖Δh‖  per half-step
depart     = nrm(traj[:-1])                     # ‖h‖ at the point the step leaves from
rel_speed  = speed / (depart + EPS)             # normalized velocity

# ACCELERATION: change in the velocity vector between consecutive half-steps
accel      = np.concatenate([[np.nan], nrm(np.diff(steps, axis=0))])

# CURVATURE: turning angle between consecutive velocity vectors.
#   cos=+1 ballistic (straight),  0 right-angle turn,  -1 reversal (undoing prior write)
turn_cos   = np.concatenate([[np.nan], cosine(steps[:-1], steps[1:])])
turn_angle = np.degrees(np.arccos(np.clip(turn_cos, -1, 1)))

# ----------------------------------------------------------------------------- #
# FORCE DECOMPOSITION  (per block: do attention and MLP cancel, or just shrink?)
# ----------------------------------------------------------------------------- #
attn_mag   = nrm(attn_out)
mlp_mag    = nrm(mlp_out)
resultant  = nrm(attn_out + mlp_out)            # the actual block velocity magnitude
cancel_idx = 1.0 - resultant / (attn_mag + mlp_mag + EPS)   # 0 aligned .. ->1 canceling
force_cos  = cosine(attn_out, mlp_out)          # alignment of the two forces

# BLOCK-LEVEL (coarse) velocity, for the 12-step view
block_vel   = resid_post - resid_pre            # == attn_out + mlp_out
block_speed = nrm(block_vel)
block_rel   = block_speed / (nrm(resid_pre) + EPS)

# ----------------------------------------------------------------------------- #
# NUMBERS  ->  CSV
# ----------------------------------------------------------------------------- #
fine_df = pd.DataFrame({
    "step":        np.arange(1, 2 * n + 1),
    "label":       label,
    "kind":        kind,
    "speed_raw":   speed,
    "speed_rel":   rel_speed,
    "accel":       accel,
    "turn_deg":    turn_angle,
    "stream_norm": depart,
})
block_df = pd.DataFrame({
    "block":       np.arange(n),
    "attn_mag":    attn_mag,
    "mlp_mag":     mlp_mag,
    "resultant":   resultant,
    "cancel_idx":  cancel_idx,
    "force_cos":   force_cos,
    "block_speed": block_speed,
    "block_rel":   block_rel,
})
fine_df.to_csv(f"{OUTDIR}/v4_velocity_fine.csv", index=False)
block_df.to_csv(f"{OUTDIR}/v4_velocity_block.csv", index=False)

pd.set_option("display.float_format", lambda x: f"{x:8.4f}")
print("\n=== FINE TRAJECTORY (the 'twice per block' path) ===")
print(fine_df.to_string(index=False))
print("\n=== BLOCK-LEVEL FORCE BALANCE ===")
print(block_df.to_string(index=False))

# ----------------------------------------------------------------------------- #
# CHARTS  ->  metrics figure
# ----------------------------------------------------------------------------- #
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
x = fine_df["step"].values
is_mlp = kind == "mlp"

# (1) velocity profile — where you LOOK for terminal velocity (a flattening plateau)
ax[0, 0].plot(x, speed, "-o", ms=4, label="raw ‖Δh‖", color="#3355bb")
ax2 = ax[0, 0].twinx()
ax2.plot(x, rel_speed, "-s", ms=3, label="relative ‖Δh‖/‖h‖", color="#cc5500", alpha=.8)
ax[0, 0].set_title("Velocity profile  (look for the plateau = terminal velocity)")
ax[0, 0].set_xlabel("half-step (attn, mlp, attn, mlp …)")
ax[0, 0].set_ylabel("raw speed", color="#3355bb")
ax2.set_ylabel("relative speed", color="#cc5500")
ax[0, 0].legend(loc="upper left"); ax2.legend(loc="upper right")

# (2) curvature — ballistic vs turning; dips toward/below 0 = the stream reversing itself
ax[0, 1].axhline(90, color="#999", lw=.8, ls="--")
ax[0, 1].plot(x, turn_angle, "-o", ms=4, color="#227744")
ax[0, 1].set_title("Turning angle between successive writes\n(<90° ballistic · 90° orthogonal · >90° undoing)")
ax[0, 1].set_xlabel("half-step"); ax[0, 1].set_ylabel("degrees")

# (3) the two FORCES across blocks
b = block_df["block"].values
ax[1, 0].plot(b, attn_mag,  "-o", ms=4, label="‖attn write‖", color="#aa2266")
ax[1, 0].plot(b, mlp_mag,   "-o", ms=4, label="‖mlp write‖",  color="#2266aa")
ax[1, 0].plot(b, resultant, "-o", ms=4, label="‖resultant‖",  color="#000000")
ax[1, 0].set_title("The two forces (per block)")
ax[1, 0].set_xlabel("block"); ax[1, 0].set_ylabel("magnitude"); ax[1, 0].legend()

# (4) cancellation index — the HONEST terminal-velocity test.
#     high & rising = attn and mlp fighting each other = a real equilibrium.
ax[1, 1].plot(b, cancel_idx, "-o", ms=4, color="#8800aa")
ax[1, 1].set_title("Cancellation index  (0 = forces aligned · →1 = forces canceling)")
ax[1, 1].set_xlabel("block"); ax[1, 1].set_ylabel("1 − resultant/(‖attn‖+‖mlp‖)")
ax[1, 1].set_ylim(-0.05, 1.05)

fig.suptitle(f'Residual-stream velocity — GPT-2 small · pos {POS} · "{PROMPT}"', y=1.01)
fig.tight_layout()
fig.savefig(f"{OUTDIR}/v4_metrics.png", dpi=140, bbox_inches="tight")

# ----------------------------------------------------------------------------- #
# GEOMETRY  ->  the actual PATH through space (PCA to 2D), colored by force
# ----------------------------------------------------------------------------- #
c = traj - traj.mean(0)
U, S, Vt = np.linalg.svd(c, full_matrices=False)
P = c @ Vt[:2].T                                  # (2n+1, 2) trajectory in top-2 PCs
var2 = (S[:2] ** 2).sum() / (S ** 2).sum()

figg, axg = plt.subplots(figsize=(8, 8))
axg.plot(P[:, 0], P[:, 1], "-", color="#cccccc", lw=1, zorder=1)
for i in range(len(steps)):
    col = "#aa2266" if kind[i] == "attn" else "#2266aa"
    axg.annotate("", xy=P[i + 1], xytext=P[i],
                 arrowprops=dict(arrowstyle="->", color=col, lw=1.6), zorder=2)
axg.scatter(*P[0],  s=90, color="#227744", zorder=3, label="start (pre L0)")
axg.scatter(*P[-1], s=90, color="#cc0000", zorder=3, label="end (readout)")
axg.plot([], [], color="#aa2266", label="attention write")
axg.plot([], [], color="#2266aa", label="MLP write")
axg.set_title(f"Trajectory geometry (PCA, {var2:.0%} var)\nred = last write into the prediction")
axg.set_xlabel("PC1"); axg.set_ylabel("PC2"); axg.legend(); axg.set_aspect("equal", "datalim")
figg.tight_layout()
figg.savefig(f"{OUTDIR}/v4_geometry.png", dpi=140, bbox_inches="tight")

print("\nwrote: v4_velocity_fine.csv, v4_velocity_block.csv, v4_metrics.png, v4_geometry.png")
