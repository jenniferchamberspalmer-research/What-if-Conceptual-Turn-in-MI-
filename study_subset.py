"""Focused three-word subset (water/salt/bread), staged one word per run.

Per word, in a single Modal invocation:
  View 2 Tier 1 - original "The {target} is" fragments (neutral/sacred/profane).
                  The parallel baseline.
  View 2 Tier 2 - "People use the holy {target} to ___" forced frames.
                  The higher contextual-pressure test. Reported separately;
                  the two tiers are NOT collapsed.
  View 3        - SAE features at layers 6/12/19, read at BOTH the probe-word
                  token and the sentence-final token.

Framing: water is the REFERENCE WORD (its ritual sense is the most
conventionalized: "holy water" is an established sacred substance). It is not
a parole-free control. The comparative question is whether context moves salt
and bread toward the situated meaning water already carries.

Tier 1 fragments + View 3 sentence come from results/frames_stratum_A.json.
Tier 2 forced frames come from results/frames_view2_forced.json.

Run staged:
    python -m modal run study_subset.py --word water
    python -m modal run study_subset.py --word salt
    python -m modal run study_subset.py --word bread
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
        "gradio>=5,<6",
        "pandas>=2.0",
        "requests>=2.31",
        "fastapi>=0.110",
        "wordfreq>=3.0",
    )
    .add_local_python_source("water_tool")
)

app = modal.App("water-study-subset", image=image)
volume = modal.Volume.from_name("water-tool-cache", create_if_missing=True)

RESULTS_DIR = "/cache/study/results"
SAE_LAYERS = [6, 12, 19]

# water's ritual sense is the most conventionalized -> reference, not control.
REFERENCE_WORD = "water"

TIER1_LABEL = "Tier 1 (parallel baseline): 'The {target} is' fragments"
TIER2_LABEL = "Tier 2 (higher contextual pressure): 'People use the {cond} {target} to ___' forced frames"


@app.function(
    gpu="T4",
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/cache": volume},
)
def run_word(word: str, tier1_json: str, tier2_json: str, sentence: str,
             out_name: str = "", k_view2: int = 20, k_view3: int = 15):
    os.environ["HF_HOME"] = "/cache/hf"
    os.environ["NEURONPEDIA_CACHE"] = "/cache/neuronpedia"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    import torch
    from water_tool.core.model import load
    from water_tool.core.sae import get_sae, neuronpedia_sae_id
    from water_tool.core.neuronpedia import get_description, feature_url
    from water_tool.views import probability, features

    model, tok = load()

    sae_ok = {}
    for layer in SAE_LAYERS:
        try:
            get_sae(layer)
            sae_ok[layer] = True
        except Exception as e:
            sae_ok[layer] = f"FAILED: {type(e).__name__}: {e}"

    def view2_block(frames):
        out = []
        for fr in frames:
            df = probability.top_next_tokens(fr["prompt"], k=k_view2)
            out.append({"id": fr["id"], "prompt": fr["prompt"],
                        "top_next": df.to_dict("records")})
        return out

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
            "rank": r, "feature_idx": int(fi), "activation": round(float(a), 4),
            "description": get_description(sae_id, int(fi)),
            "neuronpedia_url": feature_url(sae_id, int(fi)),
        } for r, (a, fi) in enumerate(zip(top.values.tolist(), top.indices.tolist()), 1)]
        return rows, tok.decode([int(enc["input_ids"][0, pos])])

    tier1 = view2_block(json.loads(tier1_json))
    tier2 = view2_block(json.loads(tier2_json))

    v3 = {}
    for layer in SAE_LAYERS:
        if sae_ok.get(layer) is not True:
            v3[str(layer)] = {"error": sae_ok.get(layer)}
            continue
        try:
            probe_df = features.top_features(sentence, word, layer, k=k_view3)
            last_rows, last_tok = features_at_last_token(sentence, layer)
            v3[str(layer)] = {
                "probe_word": {"target": word, "features": probe_df.to_dict("records")},
                "last_token": {"target": last_tok, "features": last_rows},
            }
        except Exception as e:
            v3[str(layer)] = {"error": f"{type(e).__name__}: {e}"}

    role = ("reference word (most conventionalized ritual sense; not a control)"
            if word == REFERENCE_WORD else
            "comparison word (does context move it toward water's situated sense?)")

    rec = {
        "word": word,
        "stratum": "A",
        "subset": "three-word focused subset (water/salt/bread)",
        "role": role,
        "sentence": sentence,
        "sae_self_check": {str(k): v for k, v in sae_ok.items()},
        "view2_tier1": {"label": TIER1_LABEL.format(target=word), "frames": tier1},
        "view2_tier2": {"label": TIER2_LABEL.format(target=word, cond="holy"), "frames": tier2},
        "view3": v3,
    }
    out_name = out_name or f"subset_{word}"
    with open(os.path.join(RESULTS_DIR, f"{out_name}.json"), "w") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    volume.commit()
    return rec


@app.local_entrypoint()
def main(word: str, sentence: str = "", out_name: str = "", view3_only: bool = False):
    here = os.path.dirname(os.path.abspath(__file__))
    frames_orig = json.load(open(os.path.join(here, "results", "frames_stratum_A.json"), encoding="utf-8"))
    frames_forced = json.load(open(os.path.join(here, "results", "frames_view2_forced.json"), encoding="utf-8"))

    if word not in frames_orig:
        raise SystemExit(f"No frames for {word!r}.")

    # view3_only skips both View 2 tiers (empty frame lists). `sentence`
    # overrides the default View 3 sentence (e.g. a religiosity-matched probe).
    tier1 = [] if view3_only else frames_orig[word]["view2"]
    tier2 = [] if view3_only else frames_forced[word]["view2"]
    sent = sentence or frames_orig[word]["view3"]
    out_name = out_name or f"subset_{word}"

    rec = run_word.remote(
        word=word,
        tier1_json=json.dumps(tier1),
        tier2_json=json.dumps(tier2),
        sentence=sent,
        out_name=out_name,
    )

    out_path = os.path.join(here, "results", f"{out_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)

    print(f"\n================ {rec['word'].upper()} ================")
    print(f"role: {rec['role']}")
    print(f"sentence (View 3): {rec['sentence']}")
    print(f"SAE self-check (6/12/19): {rec['sae_self_check']}")

    def show_view2(block):
        print(f"\n-- {block['label']} --")
        for fr in block["frames"]:
            tops = ", ".join(f"{t['token']}={t['probability']:.3f}" for t in fr["top_next"][:8])
            print(f"  [{fr['id']:>7}] {fr['prompt']!r}")
            print(f"            -> {tops}")

    print("\n=== VIEW 2 ===")
    show_view2(rec["view2_tier1"])
    show_view2(rec["view2_tier2"])

    print("\n=== VIEW 3: top SAE features per layer ===")
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
                desc = (ft["description"] or "")[:68]
                print(f"      [{ft['activation']:>7.3f}] #{ft['feature_idx']}  {desc}")

    print(f"\nsaved -> results/{out_name}.json")


if __name__ == "__main__":
    main()
