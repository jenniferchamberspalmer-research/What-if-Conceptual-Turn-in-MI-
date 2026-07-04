"""
hc_core.py — Held/Collapsed core (§6.1 of the pre-registration)
================================================================
Shared projection + per-component attribution + output-relevance primitives for the
held-vs-collapsed dichotomy-dynamics plan.

DESCENT: this reuses the one correct, hard-won idea from field_probe/v4_velocity.py —
that in a residual network the per-block writes ARE the velocity vectors, and GPT-2
writes twice per block (attn kick, then mlp kick). v4 measured the geometry of those
writes in the full 768-space (magnitude, turning angle, attn-vs-mlp cancellation) but
never projected onto a readout axis. This core adds exactly what was missing:

  * a readout AXIS u (a plug-in parameter — the module runs on ANY u)
  * raw projection            q = <r, u>
  * LayerNorm-lensed proj.    p = <LN_f(r), u>        (v1 lens = final LayerNorm)
  * EXACT per-component attribution of q into <attn,u> + <mlp,u>, block-additive
  * per-head refinement        <attn,u> = sum_h <head_h,u> + <b_O,u>
  * output-relevance           rho = cos(u, d_ab),  d_ab = W_U[:,a] - W_U[:,b]
  * onto-axis cancellation      cancel_u, and co-direction sign of the two writes
    (this is v4's cancel_idx / force_cos, projected onto u — the defensible signal)
  * logit-lens KL drift         KL(final || layer)  at every half-step

v1 DECISIONS (locked): lens = final LayerNorm (LN_f); rho metric = Euclidean cosine.
Both are swap points (see project() and output_relevance()).

FIREWALL: __main__ runs CORRECTNESS FIXTURES ONLY. A random axis and d_ab are used to
check four KNOWN-EXACT identities. These validate the machine; they make NO scientific
claim and measure NO pre-registered axis. The first scientific run happens elsewhere,
after the axis is locked.

    pip install transformer_lens torch pandas numpy
    python field_probe/hc_core.py     # expect: ALL FIXTURES PASS
"""

import numpy as np
import pandas as pd
import torch
from transformer_lens import HookedTransformer

EPS = 1e-9


# --------------------------------------------------------------------------- #
# SETUP
# --------------------------------------------------------------------------- #
def load_model(name="gpt2"):
    """GPT-2 small with per-head write decomposition enabled."""
    model = HookedTransformer.from_pretrained(name)
    model.cfg.use_attn_result = True   # exposes cache["result", l]: per-head OV writes
    model.eval()
    return model


def run_and_cache(model, prompt):
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens)
    return tokens, logits, cache


def unit(v):
    return v / (v.norm() + EPS)


# --------------------------------------------------------------------------- #
# EXTRACT WRITES  (one position; the §6.2 probe will vectorize over positions)
#   resid_mid = resid_pre + attn_out ;  resid_post = resid_mid + mlp_out
#   attn_out  = result.sum(head) + b_O ;  mlp_out already includes its bias
# --------------------------------------------------------------------------- #
def extract_writes(model, cache, pos=-1):
    n = model.cfg.n_layers
    dev = model.cfg.device

    def stack(name):
        return torch.stack([cache[name, l][0, pos].float() for l in range(n)])   # (n, d)

    result = torch.stack([cache["result", l][0, pos].float() for l in range(n)])  # (n, H, d)
    b_O = torch.stack([model.blocks[l].attn.b_O.detach().float() for l in range(n)]).to(dev)  # (n, d)

    return dict(
        resid_pre=stack("resid_pre"),
        resid_mid=stack("resid_mid"),
        resid_post=stack("resid_post"),
        attn_out=stack("attn_out"),
        mlp_out=stack("mlp_out"),
        result=result,
        b_O=b_O,
    )


# --------------------------------------------------------------------------- #
# PROJECTION ONTO u   (raw q, and LN_f-lensed p)
# --------------------------------------------------------------------------- #
def project(writes, u, model, lens=True):
    u = unit(u.float()).to(model.cfg.device)
    out = dict(
        q_pre=writes["resid_pre"] @ u,
        q_mid=writes["resid_mid"] @ u,
        q_post=writes["resid_post"] @ u,
    )
    if lens:
        ln = model.ln_final    # v1 lens; swap for a tuned lens here (see §7/§9)
        out.update(
            p_pre=ln(writes["resid_pre"]) @ u,
            p_mid=ln(writes["resid_mid"]) @ u,
            p_post=ln(writes["resid_post"]) @ u,
        )
    return out


# --------------------------------------------------------------------------- #
# ATTRIBUTION   (exact decomposition of the raw projection velocity onto u)
#   dq_block[l] = <attn_out[l], u> + <mlp_out[l], u>   (== <resid_post-resid_pre, u>)
#   <attn_out[l], u> = sum_h <head_h[l], u> + <b_O[l], u>
#   cancel_u: v4's cancellation index, PROJECTED onto u (0 co-direct .. ->1 cancel)
# --------------------------------------------------------------------------- #
def attribute(writes, u):
    u = unit(u.float()).to(writes["attn_out"].device)
    qa = writes["attn_out"] @ u        # (n,) attention write onto u
    qm = writes["mlp_out"] @ u         # (n,) mlp write onto u
    dq_block = qa + qm                 # (n,) exact per-block projection velocity
    qh = writes["result"] @ u          # (n, H) per-head attention writes onto u
    qbO = writes["b_O"] @ u            # (n,) attention output-bias write onto u
    cancel_u = 1.0 - (qa + qm).abs() / (qa.abs() + qm.abs() + EPS)
    codirect = torch.sign(qa) * torch.sign(qm)     # +1 co-direct, -1 cancel
    return dict(qa=qa, qm=qm, dq_block=dq_block, qh=qh, qbO=qbO,
                cancel_u=cancel_u, codirect=codirect)


# --------------------------------------------------------------------------- #
# OUTPUT-RELEVANCE   rho = cos(u, d_ab)
# --------------------------------------------------------------------------- #
def output_relevance(model, u, a, b):
    u = unit(u.float()).to(model.cfg.device)
    d_ab = (model.W_U[:, a] - model.W_U[:, b]).detach().float()   # (d,)
    rho = torch.dot(u, unit(d_ab)).item()      # Euclidean cosine (v1); causal IP is §9
    return dict(rho=rho, d_ab=d_ab)


# --------------------------------------------------------------------------- #
# FINE PROJECTION TRAJECTORY   (v4's alternating half-step path, projected onto u)
#   q_pre0 -> q_mid0 -> q_post0 -> q_mid1 -> ...   diffs are exactly qa, qm interleaved
#   Q1 (collapse depth l*) reads off this.
# --------------------------------------------------------------------------- #
def fine_projection(proj):
    n = proj["q_pre"].shape[0]
    q = [proj["q_pre"][0].item()]
    labels, kinds = [], []
    for l in range(n):
        q.append(proj["q_mid"][l].item());  labels.append(f"L{l}·attn"); kinds.append("attn")
        q.append(proj["q_post"][l].item()); labels.append(f"L{l}·mlp");  kinds.append("mlp")
    return np.array(q), labels, kinds


# --------------------------------------------------------------------------- #
# LOGIT-LENS DRIFT   KL(final || layer) at every half-step
#   Directly probes documented gap (4): does a pre-ln raw spike survive LN_f to
#   actually move the prediction, or evaporate?
# --------------------------------------------------------------------------- #
def lens_drift(model, cache, pos=-1):
    n = model.cfg.n_layers
    ln, W_U, b_U = model.ln_final, model.W_U, model.b_U

    def logits_from(resid):
        return ln(resid) @ W_U + b_U

    final = cache["resid_post", n - 1][0, pos].float()
    logp_final = torch.log_softmax(logits_from(final), dim=-1)
    p_final = logp_final.exp()

    rows = []
    for l in range(n):
        for ck in ("resid_mid", "resid_post"):
            r = cache[ck, l][0, pos].float()
            logp = torch.log_softmax(logits_from(r), dim=-1)
            kl = torch.sum(p_final * (logp_final - logp)).item()   # KL(final || layer)
            rows.append(dict(block=l, checkpoint=ck, kl_final_to_layer=kl))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# THE §6.1 PRIMITIVE   (what the §6.2 probe / §6.3 intervene will call)
# --------------------------------------------------------------------------- #
def hc_trace(model, cache, u, a, b, pos=-1):
    W = extract_writes(model, cache, pos=pos)
    proj = project(W, u, model, lens=True)
    attr = attribute(W, u)
    orel = output_relevance(model, u, a, b)
    fine_q, labels, kinds = fine_projection(proj)
    drift = lens_drift(model, cache, pos=pos)
    return dict(writes=W, projection=proj, attribution=attr,
                output_relevance=orel, fine_q=fine_q, fine_labels=labels,
                fine_kinds=kinds, lens_drift=drift)


# --------------------------------------------------------------------------- #
# CORRECTNESS FIXTURES  (firewall — no scientific claim)
#   F1 block additivity : <attn,u>+<mlp,u> == <resid_post-resid_pre, u>
#   F2 head additivity  : sum_h <head_h,u> + <b_O,u> == <attn,u>
#   F3 lens->logits     : LN_f(resid_post[-1]) @ W_U + b_U == model logits
#   F4 d_ab identity    : <LN_f(r), d_ab> + (b_U[a]-b_U[b]) == logit_a - logit_b
# Each has a known-exact answer independent of what u "means".
# --------------------------------------------------------------------------- #
def correctness_fixtures(model, cache, logits):
    dev = model.cfg.device
    W = extract_writes(model, cache, pos=-1)
    res = {}

    torch.manual_seed(0)
    u_rand = unit(torch.randn(model.cfg.d_model, device=dev))
    top2 = torch.topk(logits[0, -1], 2).indices
    a, b = top2[0].item(), top2[1].item()
    d_ab = unit((model.W_U[:, a] - model.W_U[:, b]).float())

    for name, u in (("random", u_rand), ("d_ab", d_ab)):
        attr = attribute(W, u)
        block_direct = (W["resid_post"] - W["resid_pre"]) @ unit(u).to(dev)
        res[f"{name}:F1_block_additivity"] = (attr["dq_block"] - block_direct).abs().max().item()
        res[f"{name}:F2_head_additivity"] = (attr["qh"].sum(-1) + attr["qbO"] - attr["qa"]).abs().max().item()

    ln, W_U, b_U = model.ln_final, model.W_U, model.b_U
    lensed = ln(W["resid_post"][-1]) @ W_U + b_U
    res["F3_lens_reconstructs_logits"] = (lensed - logits[0, -1]).abs().max().item()

    d_ab_raw = (model.W_U[:, a] - model.W_U[:, b]).float()
    proj = torch.dot(ln(W["resid_post"][-1]), d_ab_raw).item()
    bias_diff = (b_U[a] - b_U[b]).item()
    logit_diff = (logits[0, -1, a] - logits[0, -1, b]).item()
    res["F4_d_ab_identity"] = abs(proj + bias_diff - logit_diff)

    return res, (a, b)


if __name__ == "__main__":
    PROMPT = "On the deserted planet they discovered a"
    model = load_model()
    tokens, logits, cache = run_and_cache(model, PROMPT)
    res, (a, b) = correctness_fixtures(model, cache, logits)

    ta = model.tokenizer.decode([a])
    tb = model.tokenizer.decode([b])
    print("=== hc_core.py CORRECTNESS FIXTURES (no scientific claim) ===")
    print(f"prompt: {PROMPT!r}")
    print(f"fixture token pair (a,b) = top-2 final logits = ({a} {ta!r}, {b} {tb!r})")
    ok = True
    for k, v in res.items():
        passed = v < 1e-2
        ok = ok and passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {k:34s} max|err| = {v:.2e}")
    print("ALL FIXTURES PASS" if ok else "SOME FIXTURES FAILED — do not proceed")
