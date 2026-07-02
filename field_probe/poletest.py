"""
Quick test: can onplane tell a real middle from a pole-adjacent word?

For hot/cold:
  - " warm" is a genuine middle (between hot and cold)
  - " freeze" is pole-adjacent (very cold-like, but not a middle)

If onplane scores them similarly, then onplane measures pole-proximity, not middleness.
If onplane scores warm much higher, then it's finding real middles.

This uses the exact same reps and calculation as v3.
"""

import numpy as np
from transformer_lens import HookedTransformer

MODEL_NAME = "gpt2"
EPS = 1e-9

CARRIERS = ["It was{}", "They called it{}", "Everyone agreed it was{}",
            "The verdict was{}", "I would describe it as{}"]

model = HookedTransformer.from_pretrained(MODEL_NAME)
model.eval()
n = model.cfg.n_layers

def _cache(p):
    import torch
    with torch.no_grad():
        _, c = model.run_with_cache(model.to_tokens(p))
    return c

def word_reps(word):
    """Mean-pool over word's own subword span, averaged over carriers."""
    import torch
    wtok = model.to_tokens(word, prepend_bos=False)[0]
    k = int(wtok.shape[0])
    per = []
    for c in CARRIERS:
        pr = c.format(word)
        toks = model.to_tokens(pr)[0]
        cc = _cache(pr)
        per.append(np.stack([cc["resid_post", l][0, -k:].float().mean(0).cpu().numpy()
                             for l in range(n)]))
    per = np.stack(per)
    return per.mean(0)

def rem(w_rep, A_rep, B_rep):
    """Off-axis remainder for word rep, per layer. Returns (n,) array."""
    ax = B_rep - A_rep
    t = np.sum((w_rep - A_rep) * ax, axis=-1) / (np.sum(ax * ax, axis=-1) + EPS)
    proj = A_rep + t[:, np.newaxis] * ax
    return np.linalg.norm(w_rep - proj, axis=-1)

# build the reps
A_rep = word_reps(" hot")
B_rep = word_reps(" cold")
warm_rep = word_reps(" warm")
freeze_rep = word_reps(" freeze")

# random-word null
RANDOM_POOL = [" river", " pencil", " market", " ceiling", " garden", " engine", " letter",
 " planet", " basket", " candle", " bridge", " mirror", " pocket", " ticket", " cotton",
 " valley", " anchor", " pillow", " meadow", " rocket", " ribbon", " saddle", " barrel",
 " lantern", " compass", " harvest", " cabinet", " blanket", " corridor", " napkin",
 " gravel", " turnip", " satchel", " almanac", " trolley", " kettle", " marble", " acorn"]

np.random.seed(0)
random_words = list(np.random.choice(RANDOM_POOL, size=30, replace=False))
rand_rems = np.array([rem(word_reps(w), A_rep, B_rep) for w in random_words])
p25 = np.percentile(rand_rems, 25, axis=0)

# compute onplane
warm_rem = rem(warm_rep, A_rep, B_rep)
freeze_rem = rem(freeze_rep, A_rep, B_rep)

warm_onplane = float(np.mean(warm_rem < p25))
freeze_onplane = float(np.mean(freeze_rem < p25))

print("\n=== POLE-ADJACENCY TEST ===")
print(f"hot/cold axis")
print(f"  ' warm' (genuine middle)     onplane = {warm_onplane:.3f}")
print(f"  ' freeze' (pole-adjacent)    onplane = {freeze_onplane:.3f}")
print(f"\nDifference: {abs(warm_onplane - freeze_onplane):.3f}")
if abs(warm_onplane - freeze_onplane) > 0.15:
    print("=> onplane SEPARATES them (likely measuring field, not just pole-closeness)")
else:
    print("=> onplane FAILS to separate them (may be measuring pole-proximity instead)")
print(f"\nFor context, v3 real middles for good/bad scored ~0.88, true binaries ~0.56–0.58")
