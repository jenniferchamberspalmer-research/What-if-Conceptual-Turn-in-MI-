"""View 3: Sparse autoencoder feature activations on a target word.

Pipeline for one query:
  1. Tokenize the sentence, find the token index of the target word.
  2. Run a forward pass with a hook on transformer block `layer`. The
     hook captures the residual stream OUTPUT of that block — which is
     exactly the substrate Gemma Scope SAEs were trained to decode.
  3. Encode that residual vector through the SAE to get the
     ~16,384-dimensional sparse feature activation vector.
  4. Take the top-K features by activation magnitude at the target
     position.
  5. Look up Neuronpedia descriptions for each top feature.

The layer toggle (6 / 12 / 19) gives early / middle / late views:
  - Layer 6: more about token shape, syntactic role.
  - Layer 12: semantic features — what most "concept" probes find.
  - Layer 19: closer to continuation/task — what the model is preparing
    to do with the token next.
"""

import torch
import pandas as pd

from ..core.model import load
from ..core.sae import get_sae, neuronpedia_sae_id
from ..core.neuronpedia import get_description, feature_url

# Residual extraction is single-sourced in water_tool.core.extract so it cannot
# drift between the Water Pattern Tool and the Dichotomy Transformation Probe.
# These names are re-exported here to preserve this module's historical API.
from ..core.extract import find_target_position as _find_target_position
from ..core.extract import residual_at_layer as _capture_residual_at_layer


@torch.no_grad()
def top_features(text: str, target: str, layer: int, k: int = 15) -> pd.DataFrame:
    model, tok = load()
    target_pos = _find_target_position(text, target, tok)

    residual_seq, _ = _capture_residual_at_layer(text, layer)
    residual_at_target = residual_seq[target_pos].to(torch.float32)

    sae = get_sae(layer)
    acts = sae.encode(residual_at_target.to(sae.W_enc.device).to(sae.W_enc.dtype))
    acts = acts.detach().float()

    top = torch.topk(acts, k)
    sae_id = neuronpedia_sae_id(layer)

    rows = []
    for rank, (act_val, feat_idx) in enumerate(
        zip(top.values.tolist(), top.indices.tolist()), start=1
    ):
        rows.append({
            "rank": rank,
            "feature_idx": int(feat_idx),
            "activation": round(float(act_val), 4),
            "description": get_description(sae_id, int(feat_idx)),
            "neuronpedia_url": feature_url(sae_id, int(feat_idx)),
        })
    return pd.DataFrame(rows)
