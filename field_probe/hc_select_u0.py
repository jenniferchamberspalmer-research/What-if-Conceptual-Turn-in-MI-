"""
hc_select_u0.py — u0 candidate certification for the confirmatory matched design
PILOT-GRADE SELECTION REPORT. No pre-registration claim; no trajectory measured.
Builds three candidate internal-axis constructs and runs the emission certification
(from hc_core) on each, so we can judge which qualify as output-orthogonal u0:
  A. this/that        — single-token contrast axis
  B. singular/plural  — single-token contrast axis (expected OUTPUT-BOUND; that's a finding)
  C. concrete/abstract— POOLED-concept axis from locked seed sets
Each reported with emission_norm, rho_worst, top-k concentration, and top emitted tokens,
against the same scale as u+ and a random axis. Judgment picks; code does not.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import hc_core as hc

LAYER = 7; EPS = 1e-9
print("*** PILOT SELECTION REPORT — no pre-registration claim, no trajectory measured ***\n")
model = hc.load_model("gpt2", no_processing=True)
dev = model.cfg.device; W_U = model.W_U; tok = model.tokenizer

def tok_id(s):
    """single-token id for a word; asserts it's one token so poles are clean."""
    ids = tok.encode(s, add_special_tokens=False)   # mechanical: TL tokenizer prepends BOS
    return ids[0], len(ids)

def mean_unembed(words, tag):
    cols, dropped = [], []
    for w in words:
        for form in (" " + w, w):                     # prefer leading-space form
            ids = tok.encode(form, add_special_tokens=False)   # mechanical: TL tokenizer prepends BOS
            if len(ids) == 1:
                cols.append(W_U[:, ids[0]]); break
        else:
            dropped.append(w)
    if dropped: print(f"   [{tag}] dropped (not single-token): {dropped}")
    return torch.stack(cols, 1).mean(1)               # [d]

# random + u+ reference scale (u+ = the committed myself/themselves feature via SAE) ----
from sae_lens import SAE
_l = SAE.from_pretrained("gpt2-small-res-jb", f"blocks.{LAYER}.hook_resid_pre", device=str(dev))
sae = (_l[0] if isinstance(_l, tuple) else _l).to(dev)
u_plus = hc.unit(sae.W_dec[5518].detach().float().to(dev))
torch.manual_seed(0); u_rand = hc.unit(torch.randn(model.cfg.d_model, device=dev))

def contrast_axis(pos_word, neg_word):
    (ip, lp), (iq, lq) = tok_id(" " + pos_word), tok_id(" " + neg_word)
    if lp != 1 or lq != 1:
        print(f"   WARNING: {pos_word!r}/{neg_word!r} not single-token ({lp},{lq})")
    return hc.unit((W_U[:, ip] - W_U[:, iq]).float())

CONCRETE = ["rock","table","water","tree","hand","dog","chair","bread","river","stone","window"]
ABSTRACT = ["justice","freedom","truth","reason","chaos","wisdom","doubt","peace","idea","method","entropy"]

axes = {
    "REF u+ (myself/themselves)": u_plus,
    "REF random":                 u_rand,
    "A. this/that":               contrast_axis("this", "that"),
    "B. singular/plural (is/are)": contrast_axis("is", "are"),
    "B2. singular/plural (was/were)": contrast_axis("was", "were"),
    "C. concrete/abstract (pooled)": hc.unit((mean_unembed(CONCRETE,"concrete") - mean_unembed(ABSTRACT,"abstract")).float()),
}

print(f"\n{'axis':34s} {'emis':>7s} {'rho_wst':>8s} {'conc10':>7s}   top emitted (+/-)")
for name, u in axes.items():
    e = hc.axis_emission(model, u)
    tops = ", ".join(f"{t!r}" for t, v in e["top_emitted"][:5])
    print(f"{name:34s} {e['emission_norm']:7.2f} {e['rho_worst']:+8.3f} "
          f"{e['emission_concentration'][10]:7.4f}   {tops}")
print("\nRead: u0 candidates want LOW emission_norm and LOW |rho_worst| relative to u+ "
      "(output-bound). singular/plural is expected HIGH (grammatical number is emitted).")
print("Nothing selected or committed — judgment picks the near-neighbor and distant u0.")
