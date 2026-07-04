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
  * AXIS CERTIFICATION (§4)     differential emission of an axis: is a candidate u0
    internal-only (near-orthogonal to EVERY unembedding difference, not just its own)

v1 DECISIONS (locked): lens = final LayerNorm (LN_f); rho metric = Euclidean cosine.
Both are swap points (see project() and output_relevance()).

FIREWALL: __main__ runs CORRECTNESS FIXTURES + an AXIS EMISSION SCALE calibration.
Both use only a random axis and d_ab; they check KNOWN-EXACT identities and report the
emission scale of a known contrast vs a random direction. They make NO scientific claim
and measure NO pre-registered axis. The first scientific run happens elsewhere, after
the axis is locked.

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
# OUTPUT-RELEVANCE   rho = cos(u, d_ab)   (the axis's OWN declared poles)
# --------------------------------------------------------------------------- #
def output_relevance(model, u, a, b):
    u = unit(u.float()).to(model.cfg.device)
    d_ab = (model.W_U[:, a] - model.W_U[:, b]).detach().float()   # (d,)
    rho = torch.dot(u, unit(d_ab)).item()      # Euclidean cosine (v1); causal IP is §9
    return dict(rho=rho, d_ab=d_ab)


# --------------------------------------------------------------------------- #
# AXIS CERTIFICATION (§4)  —  is a candidate u0 internal-only?
#
# "Orthogonal to EVERY unembedding difference" cannot mean orthogonal to their
# SPAN: W_U has full column rank (768), so the span of all differences is all of
# R^768 and no direction is orthogonal to it. The right notion is EMISSION
# MAGNITUDE, and specifically its CONCENTRATION.
#
# Direct-path emission: if u reaches the final residual, steering r -> r + u moves
# token logits (to first order, pre-LN) by  push = W_U^T u.  The part that moves
# logit DIFFERENCES is the mean-centered push  push_c = (W_U - mean_col)^T u.
#
# The total NORM ||push_c|| does NOT discriminate: over ~50k tokens it concentrates
# (a random axis emits ~0.9x of a real contrast). The discriminating quantity is
# CONCENTRATION — how few tokens carry the push. Internal-only <=> emission spread
# thinly across the vocabulary with no coherent contrast anywhere <=> low top-k
# energy fraction. This also catches DISTRIBUTED emission (several moderate
# contrasts) that a single worst-pair check would miss.
#
# CAVEAT (honest, and it ties to the plan's structure): this is linear/direct-path
# emission. It brackets the LN_f Jacobian and the indirect composition path (an
# axis could reach output via later-layer composition even with small direct
# push). A PASS here is a PRE-DATA selection filter, to be CONFIRMED by steering
# through the real LN_f + W_U in §6.3 — same proposes-then-confirms discipline as
# the §6.4 attribution graphs. Never report a certified u0 without that downstream
# confirmation.
# --------------------------------------------------------------------------- #
def axis_emission(model, u, poles=None, top_k=15):
    u = unit(u.float()).to(model.cfg.device)
    W_U = model.W_U                                   # (d, V)
    push = W_U.T @ u                                  # (V,) direct per-token logit push
    push_c = push - push.mean()                       # centered == (W_U - mean_col)^T u
    emission_norm = push_c.norm().item()              # differential emission (logit units)

    # CONCENTRATION: fraction of centered-push ENERGY in the top-k tokens.
    # The norm aggregates over ~50k tokens and concentrates (a random axis emits
    # ~0.9x a real contrast), so it cannot discriminate. Concentration can: it is
    # small ONLY when emission spreads thinly with no coherent contrast anywhere
    # == internal-only. Also catches DISTRIBUTED emission (several moderate
    # contrasts) that a single worst-pair check would miss.
    energy = push_c.pow(2)
    total_energy = energy.sum()
    def _topk_frac(k):
        k = min(k, energy.numel())
        return (torch.topk(energy, k).values.sum() / (total_energy + EPS)).item()
    concentration = {10: _topk_frac(10), 50: _topk_frac(50)}

    i_star = torch.argmax(push).item()                # token u pushes up hardest
    j_star = torch.argmin(push).item()                # token u pushes down hardest
    d_worst = W_U[:, i_star] - W_U[:, j_star]         # the strongest contrast u aligns with
    rho_worst = torch.dot(u, unit(d_worst)).item()    # worst-case single-pair cosine

    order = torch.argsort(push_c.abs(), descending=True)[:top_k]
    top_emitted = [(model.tokenizer.decode([t.item()]), push_c[t].item()) for t in order]

    out = dict(
        emission_norm=emission_norm,
        emission_concentration=concentration,
        rho_worst=rho_worst,
        worst_pair=(i_star, j_star,
                    model.tokenizer.decode([i_star]),
                    model.tokenizer.decode([j_star])),
        top_emitted=top_emitted,
    )
    if poles is not None:
        a, b = poles
        out["rho_own"] = torch.dot(u, unit((W_U[:, a] - W_U[:, b]).float())).item()
    return out


def certify_internal_only(model, u0, u_ref, poles0=None,
                          max_conc10=0.05, max_rho_worst=0.15):
    """PRE-DATA selection gate for the rho~0 member u0 of a matched pair.

    PRIMARY metric — TOP-10 EMISSION CONCENTRATION: the fraction of centered-push
    energy carried by the 10 most-pushed tokens. Small <=> u0 spreads its
    (unavoidable) emission thinly with no coherent contrast anywhere <=>
    internal-only. This REPLACES the total emission norm as the gate: in 768-d the
    norm concentrates (a random axis emits ~0.9x of a real contrast), so the norm
    cannot discriminate; concentration can, and it also catches DISTRIBUTED
    emission that a single worst-pair check would miss.

    SECONDARY — rho_worst: catches a single dominant contrast directly.
    REPORTED-ONLY — emission_ratio vs u_ref: kept for the record, no longer gating.

    Thresholds are PRE-STATED pre-registration parameters. SET max_conc10 from the
    calibration scale (top-10 fraction of a random axis vs d_ab) BEFORE selecting,
    never after. A PASS is a filter, not proof — confirm by steering (§6.3).
    """
    e0 = axis_emission(model, u0, poles=poles0)
    eR = axis_emission(model, u_ref)
    conc10 = e0["emission_concentration"][10]
    ratio = e0["emission_norm"] / (eR["emission_norm"] + EPS)   # reported only
    passed = (conc10 <= max_conc10) and (abs(e0["rho_worst"]) <= max_rho_worst)
    return dict(
        passed=passed,
        conc10_u0=conc10,
        conc10_ref=eR["emission_concentration"][10],
        rho_worst_u0=e0["rho_worst"],
        emission_ratio=ratio,                       # reported, not gating
        emission_norm_u0=e0["emission_norm"],
        emission_norm_ref=eR["emission_norm"],
        worst_pair_u0=e0["worst_pair"],
        rho_own_u0=e0.get("rho_own"),
        top_emitted_u0=e0["top_emitted"],
        thresholds=dict(max_conc10=max_conc10, max_rho_worst=max_rho_worst),
    )


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

    # --- AXIS EMISSION SCALE (calibration only, no scientific claim) --------- #
    # Reference scale for setting the pre-registered u0 thresholds: how strongly a
    # known contrast (d_ab) emits vs a random direction. Measures NO chosen axis.
    torch.manual_seed(0)
    u_rand = unit(torch.randn(model.cfg.d_model, device=model.cfg.device))
    d_ab = unit((model.W_U[:, a] - model.W_U[:, b]).float())
    e_dab = axis_emission(model, d_ab)
    e_rand = axis_emission(model, u_rand)
    print("\n=== AXIS EMISSION SCALE (calibration only, no scientific claim) ===")
    print(f"  d_ab   emission_norm = {e_dab['emission_norm']:8.3f}   "
          f"rho_worst = {e_dab['rho_worst']:+.3f}   "
          f"worst pair = {e_dab['worst_pair'][2]!r} vs {e_dab['worst_pair'][3]!r}")
    print(f"  random emission_norm = {e_rand['emission_norm']:8.3f}   "
          f"rho_worst = {e_rand['rho_worst']:+.3f}")
    print(f"  d_ab   top10 conc = {e_dab['emission_concentration'][10]:.4f}   "
          f"top50 conc = {e_dab['emission_concentration'][50]:.4f}")
    print(f"  random top10 conc = {e_rand['emission_concentration'][10]:.4f}   "
          f"top50 conc = {e_rand['emission_concentration'][50]:.4f}")
    print("  -> norm barely separates (concentration of measure); CONCENTRATION is the")
    print("     discriminating gate. Set max_conc10 between these two, nearer the random value.")
