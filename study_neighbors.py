"""Single-word, dual-mode View 1 neighbor analysis (k=50), saved to results/.

One word per Modal invocation so each result is durable: the local entrypoint
writes results/{stratum}_{word}.json on completion, which can be committed
before the next word runs. Reuses the cached English vocab filter (built by
study_filter.py) and the unmodified embedding path.

Run:
    python -m modal run study_neighbors.py --word water
    python -m modal run study_neighbors.py --word stone --k 50
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

app = modal.App("water-study-neighbors", image=image)
volume = modal.Volume.from_name("water-tool-cache", create_if_missing=True)

CACHE_DIR = "/cache/study"


@app.function(
    gpu="T4",
    timeout=1800,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/cache": volume},
)
def neighbors_for(
    word: str,
    k: int = 50,
    min_en_zipf: float = 2.2,
    dominance_margin: float = 0.9,
):
    os.environ["HF_HOME"] = "/cache/hf"
    os.environ["NEURONPEDIA_CACHE"] = "/cache/neuronpedia"
    os.makedirs(CACHE_DIR, exist_ok=True)

    from water_tool.core.model import load
    from water_tool.study import wordlist, vocab_filter, neighbors, coding

    _, tok = load()

    cache_path = os.path.join(CACHE_DIR, f"english_ids_z{min_en_zipf}_m{dominance_margin}.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            english_ids = set(json.load(f)["english_ids"])
    else:
        built = vocab_filter.build_english_ids(tok, min_en_zipf, dominance_margin)
        with open(cache_path, "w") as f:
            json.dump(built, f)
        volume.commit()
        english_ids = set(built["english_ids"])

    stratum = {w: s for w, s in wordlist.all_words()}.get(word, "?")

    ids = tok.encode(" " + word.strip(), add_special_tokens=False)
    tokenization = {"n_subtokens": len(ids),
                    "subtokens": [tok.decode([i]) for i in ids],
                    "fragments": len(ids) > 1}

    nsets = neighbors.neighbor_sets(word, english_ids, k=k)
    ml = coding.code_neighbor_records(nsets["multilingual"].to_dict("records"), word)
    en = coding.code_neighbor_records(nsets["english"].to_dict("records"), word)

    return {
        "word": word,
        "stratum": stratum,
        "k": k,
        "filter_thresholds": {"min_en_zipf": min_en_zipf, "dominance_margin": dominance_margin},
        "tokenization": tokenization,
        "multilingual": {"neighbors": ml, "profile": coding.profile(ml)},
        "english": {"neighbors": en, "profile": coding.profile(en)},
    }


@app.local_entrypoint()
def main(word: str, k: int = 50, min_en_zipf: float = 2.2, dominance_margin: float = 0.9):
    rec = neighbors_for.remote(
        word=word, k=k, min_en_zipf=min_en_zipf, dominance_margin=dominance_margin
    )

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{rec['stratum']}_{rec['word']}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)

    tk = rec["tokenization"]
    print(f"\n=== {rec['word']!r}  (stratum {rec['stratum']}, k={rec['k']}) ===")
    print(f"tokenization: {tk['n_subtokens']} subtoken(s) {tk['subtokens']}"
          f"{'  [FRAGMENTS]' if tk['fragments'] else ''}")

    def show(title, block):
        rows = block["neighbors"]
        p = block["profile"]
        print(f"\n-- {title}: top {min(30, len(rows))} of {len(rows)} --")
        print(f"{'rk':>3} {'cos':>7}  {'code':<12} {'lang':<12} token")
        for r in rows[:30]:
            print(f"{r['rank']:>3} {r['cosine_similarity']:>7.4f}  "
                  f"{r['code']:<12} {r.get('language',''):<12} {r['token']}")
        print(f"   profile: langue(TL+SY+TX)={p['counts']['TL']+p['counts']['SY']+p['counts']['TX']} "
              f"MO={p['counts']['MO']} CO={p['counts']['CO']} UR={p['counts']['UR']} "
              f"NEEDS_REVIEW={p['needs_review']}")

    show("MULTILINGUAL", rec["multilingual"])
    show("ENGLISH-ONLY", rec["english"])
    print(f"\nsaved -> results/{rec['stratum']}_{rec['word']}.json")
