"""View 2 + View 3 on user-authored contextual frames (sacred/profane sub-probe).

Reads results/frames_stratum_A.json. For each word:
  View 2 (Tab 2): top-K next tokens after each fragment (neutral/sacred/profane),
                  i.e. the token the model predicts after '... is'.
  View 3 (Tab 3): SAE features at layers 6/12/19 on the full ritual sentence,
                  read at BOTH the probe-word token and the last (sentence-end) token.

One Modal invocation processes the selected words; the local entrypoint writes
results/context_{word}.json per word and prints a digest.

Run:
    python -m modal run study_context.py
    python -m modal run study_context.py --words water,salt
"""

import json
import os
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "transformers>=4.42,<5",
        "accelerate>=0.30",
        "sae-lens>=3.20",
        "gradio>=4.36,<5",
        "pandas>=2.0",
        "requests>=2.31",
        "fastapi>=0.110",
        "wordfreq>=3.0",
    )
    .add_local_python_source("water_tool")
)

app = modal.App("water-study-context", image=image)
volume = modal.Volume.from_name("water-tool-cache", create_if_missing=True)

RESULTS_DIR = "/cache/study/results"
SAE_LAYERS = [6, 12, 19]


@app.function(
    gpu="T4",
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/cache": volume},
)
def run_context(frames_json: str, k_view2: int = 20, k_view3: int = 15):
    os.environ["HF_HOME"] = "/cache/hf"
    os.environ["NEURONPEDIA_CACHE"] = "/cache/neuronpedia"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    import torch
    from water_tool.core.model import load
    from water_tool.core.sae import get_sae, neuronpedia_sae_id
    from water_tool.core.neuronpedia import get_description, feature_url
    from water_tool.views import probability, features

    frames = json.loads(frames_json)
    model, tok = load()

    # SAE self-check: confirm all three layers load before trusting View 3.
    sae_ok = {}
    for layer in SAE_LAYERS:
        try:
            get_sae(layer)
            sae_ok[layer] = True
        except Exception as e:
            sae_ok[layer] = f"FAILED: {type(e).__name__}: {e}"

    def features_at_last_token(text, layer):
        residual_seq, enc = features._capture_residual_at_layer(text, layer)
        pos = residual_seq.shape[0] - 1
        residual = residual_seq[pos].to(torch.float32)
        sae = get_sae(layer)
        acts = sae.encode(
            residual.to(sae.W_enc.device).to(sae.W_enc.dtype)
        ).detach().float()
        top = torch.topk(acts, k_view3)
        sae_id = neuronpedia_sae_id(layer)
        rows = [{
            "rank": r,
            "feature_idx": int(fi),
            "activation": round(float(a), 4),
            "description": get_description(sae_id, int(fi)),
            "neuronpedia_url": feature_url(sae_id, int(fi)),
        } for r, (a, fi) in enumerate(zip(top.values.tolist(), top.indices.tolist()), 1)]
        last_tok = tok.decode([int(enc["input_ids"][0, pos])])
        return rows, last_tok

    out = {}
    for word, spec in frames.items():
        # --- View 2: next-token after each contextual fragment ---
        v2 = []
        for fr in spec["view2"]:
            df = probability.top_next_tokens(fr["prompt"], k=k_view2)
            v2.append({"id": fr["id"], "prompt": fr["prompt"],
                       "top_next": df.to_dict("records")})

        # --- View 3: SAE features on the full sentence ---
        sentence = spec["view3"]
        v3 = {}
        for layer in SAE_LAYERS:
            if sae_ok.get(layer) is not True:
                v3[str(layer)] = {"error": sae_ok.get(layer)}
                continue
            try:
                probe_df = features.top_features(sentence, word, layer, k=k_view3)
                last_rows, last_tok = features_at_last_token(sentence, layer)
                v3[str(layer)] = {
                    "probe_word": {"target": word,
                                   "features": probe_df.to_dict("records")},
                    "last_token": {"target": last_tok,
                                   "features": last_rows},
                }
            except Exception as e:
                v3[str(layer)] = {"error": f"{type(e).__name__}: {e}"}

        rec = {"word": word, "stratum": "A", "sentence": sentence,
               "view2": v2, "view3": v3}
        with open(os.path.join(RESULTS_DIR, f"context_{word}.json"), "w") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        out[word] = rec

    volume.commit()
    return {"sae_self_check": sae_ok, "records": out}


@app.local_entrypoint()
def main(words: str = "water,salt,bread"):
    here = os.path.dirname(os.path.abspath(__file__))
    frames_path = os.path.join(here, "results", "frames_stratum_A.json")
    all_frames = json.load(open(frames_path, encoding="utf-8"))
    wanted = [w.strip() for w in words.split(",") if w.strip()]
    selected = {w: all_frames[w] for w in wanted if w in all_frames}
    missing = [w for w in wanted if w not in all_frames]
    if missing:
        print(f"WARNING: no frames for {missing}; skipping.")

    res = run_context.remote(frames_json=json.dumps(selected))
    print(f"\nSAE self-check (layers 6/12/19): {res['sae_self_check']}")

    out_dir = os.path.join(here, "results")
    for word, rec in res["records"].items():
        with open(os.path.join(out_dir, f"context_{word}.json"), "w",
                  encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)

        print(f"\n================= {word.upper()}  (stratum A) =================")
        print(f"sentence: {rec['sentence']}")
        print("\n--- VIEW 2: next token after each frame (top 8) ---")
        for fr in rec["view2"]:
            tops = ", ".join(f"{t['token']}={t['probability']:.3f}"
                             for t in fr["top_next"][:8])
            print(f"  [{fr['id']:>7}] {fr['prompt']!r}")
            print(f"            -> {tops}")
        print("\n--- VIEW 3: top SAE features per layer ---")
        for layer in ("6", "12", "19"):
            block = rec["view3"].get(layer, {})
            if "error" in block:
                print(f"  layer {layer}: ERROR {block['error']}")
                continue
            for site in ("probe_word", "last_token"):
                feats = block[site]["features"][:5]
                tgt = block[site]["target"]
                print(f"  layer {layer} @ {site} ({tgt!r}):")
                for ft in feats:
                    desc = (ft["description"] or "")[:70]
                    print(f"      [{ft['activation']:>7.3f}] #{ft['feature_idx']}  {desc}")
        print(f"\nsaved -> results/context_{word}.json")


if __name__ == "__main__":
    main()
