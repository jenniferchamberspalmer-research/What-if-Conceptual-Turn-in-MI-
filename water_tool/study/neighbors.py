"""View 1 neighbor retrieval for the study: multilingual + English-only.

Mirrors the query-vector construction of the ORIGINAL
`water_tool.views.embedding.raw_lookup` exactly (space-prefix the word,
mean of content-token embedding vectors) so the multilingual list is
faithful to the instrument that produced the original *water* result.
The only additions are:

  - a single cosine pass over the full embedding table, reused for both
    the multilingual ranking and the English-restricted ranking
    (the English ranking masks out non-English vocabulary ids), and
  - a coarse language tag per neighbor for reporting (design Section 6).

The original retrieval used `torch.topk(sims, k + 10)`, a fixed +10 buffer
that can under-return at k=50 once whitespace/duplicate decodes are filtered.
Here we take a generous candidate buffer (k * 6, min 400) so both lists are
reliably filled to k. This is a study-side retrieval fix; the original tool
file is left untouched.
"""

import torch
import pandas as pd

from ..core.model import load
from .vocab_filter import guess_language


def _query_vec(text: str):
    model, tok = load()
    ids = tok.encode(" " + text.strip(), add_special_tokens=False)
    if not ids:
        return None, None, None
    emb_table = model.get_input_embeddings().weight
    vecs = emb_table[ids].detach()
    query = vecs.mean(dim=0) if vecs.shape[0] > 1 else vecs[0]
    return query, emb_table, tok


def _cosine_sims(query, emb_table):
    q = query.to(emb_table.device).to(emb_table.dtype)
    q_norm = q / q.norm().clamp(min=1e-8)
    emb_norm = emb_table / emb_table.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return (emb_norm @ q_norm).float()  # [vocab]


def _rank_from_sims(sims, tok, k, tag_lang):
    """Greedy top-k over a sims vector, deduping by stripped decode."""
    buf = max(k * 6, 400)
    buf = min(buf, sims.shape[0])
    top = torch.topk(sims, buf)
    rows, seen = [], set()
    for score, idx in zip(top.values.tolist(), top.indices.tolist()):
        s = tok.decode([idx])
        s_clean = s.strip()
        if not s_clean or s_clean in seen:
            continue
        seen.add(s_clean)
        row = {
            "rank": len(rows) + 1,
            "token_id": int(idx),
            "token": s_clean,
            "token_repr": repr(s),
            "cosine_similarity": round(float(score), 4),
        }
        if tag_lang:
            row["language"] = guess_language(s_clean)
        rows.append(row)
        if len(rows) >= k:
            break
    return pd.DataFrame(rows)


def neighbor_sets(text: str, english_ids, k: int = 50) -> dict:
    """Return {'multilingual': df, 'english': df} of top-k neighbors.

    `english_ids` is an iterable of vocab token ids judged English by the
    vocab filter. The English list ranks the same cosine sims but only over
    those ids.
    """
    query, emb_table, tok = _query_vec(text)
    if query is None:
        return {"multilingual": pd.DataFrame(), "english": pd.DataFrame()}

    sims = _cosine_sims(query, emb_table)
    multilingual = _rank_from_sims(sims, tok, k, tag_lang=True)

    eng_mask = torch.full_like(sims, float("-inf"))
    idx_tensor = torch.as_tensor(sorted(english_ids), device=sims.device, dtype=torch.long)
    eng_mask[idx_tensor] = sims[idx_tensor]
    english = _rank_from_sims(eng_mask, tok, k, tag_lang=False)

    return {"multilingual": multilingual, "english": english}
