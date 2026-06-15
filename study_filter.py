"""Build + validate the English-only vocabulary filter, and reproduce the
*water* result in BOTH the multilingual and English-only neighbor sets.

This is the primary-control deliverable (design Sections 3 + 10). It:

  1. scans the full Gemma 2 2B vocabulary and tags English token ids,
     caching them to the Modal volume for reuse by the full harness;
  2. validates the filter against held-out labeled English / foreign-Latin /
     non-Latin tokens and reports precision / recall / specificity;
  3. reproduces *water* under the new harness in both modes — the decisive
     within-language test (English neighbors should be systemic:
     liquid/fluid/moisture, not experiential: drink/thirst/splash).

Run:
    python -m modal run study_filter.py
    python -m modal run study_filter.py --word water --k 50
    python -m modal run study_filter.py --min-en-zipf 2.2 --dominance-margin 0.9
"""

import json
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

app = modal.App("water-study-filter", image=image)
volume = modal.Volume.from_name("water-tool-cache", create_if_missing=True)

CACHE_DIR = "/cache/study"


@app.function(
    gpu="T4",
    timeout=2400,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/cache": volume},
)
def build_and_check(
    word: str = "water",
    k: int = 50,
    min_en_zipf: float = 2.2,
    dominance_margin: float = 0.9,
    rebuild: bool = False,
):
    import os
    os.environ["HF_HOME"] = "/cache/hf"
    os.environ["NEURONPEDIA_CACHE"] = "/cache/neuronpedia"
    os.makedirs(CACHE_DIR, exist_ok=True)

    from water_tool.core.model import load
    from water_tool.study import vocab_filter, neighbors

    model, tok = load()

    cache_path = os.path.join(
        CACHE_DIR, f"english_ids_z{min_en_zipf}_m{dominance_margin}.json"
    )

    # 1. Build (or load cached) English token-id set.
    if os.path.exists(cache_path) and not rebuild:
        with open(cache_path) as f:
            built = json.load(f)
        built["loaded_from_cache"] = True
    else:
        built = vocab_filter.build_english_ids(tok, min_en_zipf, dominance_margin)
        with open(cache_path, "w") as f:
            json.dump(built, f)
        volume.commit()
        built["loaded_from_cache"] = False

    english_ids = set(built["english_ids"])

    # 2. Validate the filter on held-out labeled samples.
    validation = vocab_filter.validate(min_en_zipf, dominance_margin)

    # 3. Reproduce `word` in both modes.
    nsets = neighbors.neighbor_sets(word, english_ids, k=k)

    return {
        "filter_stats": {
            "vocab_size": built["vocab_size"],
            "n_english": built["n_english"],
            "pct_english": round(100 * built["n_english"] / built["vocab_size"], 2),
            "thresholds": built["thresholds"],
            "loaded_from_cache": built["loaded_from_cache"],
            "accepted_sample": built.get("accepted_sample", [])[:40],
            "rejected_latin_sample": built.get("rejected_latin_sample", [])[:40],
        },
        "validation": validation,
        "word": word,
        "multilingual": nsets["multilingual"].to_dict("records"),
        "english": nsets["english"].to_dict("records"),
    }


@app.local_entrypoint()
def main(
    word: str = "water",
    k: int = 50,
    min_en_zipf: float = 2.2,
    dominance_margin: float = 0.9,
    rebuild: bool = False,
):
    r = build_and_check.remote(
        word=word, k=k, min_en_zipf=min_en_zipf,
        dominance_margin=dominance_margin, rebuild=rebuild,
    )

    fs = r["filter_stats"]
    print("\n=== ENGLISH VOCAB FILTER ===")
    print(f"vocab size      : {fs['vocab_size']:,}")
    print(f"english tokens  : {fs['n_english']:,}  ({fs['pct_english']}% of vocab)")
    print(f"thresholds      : {fs['thresholds']}  (cache_hit={fs['loaded_from_cache']})")
    print(f"accepted sample : {', '.join(fs['accepted_sample'][:30])}")
    print(f"rejected latin  : {', '.join(fs['rejected_latin_sample'][:30])}")

    v = r["validation"]
    print("\n=== FILTER VALIDATION (held-out labels) ===")
    print(f"recall (english kept)      : {v['recall_english']}  "
          f"(n={v['n_english_labeled']})")
    print(f"specificity (non-en rejected): {v['specificity_nonenglish']}  "
          f"(n={v['n_nonenglish_labeled']})")
    print(f"precision                  : {v['precision']}")
    print(f"english DROPPED (false neg): {v['english_dropped (false negatives)']}")
    print(f"non-english KEPT (false pos): {v['nonenglish_kept (false positives)']}")

    def show(title, rows, n):
        print(f"\n=== {title} — top {min(n, len(rows))} of {len(rows)} ===")
        print(f"{'rank':>4}  {'cosine':>7}  {'lang':<12}  token")
        print("-" * 50)
        for row in rows[:n]:
            lang = row.get("language", "")
            print(f"{row['rank']:>4}  {row['cosine_similarity']:>7.4f}  "
                  f"{lang:<12}  {row['token']}")

    show(f"{word!r} MULTILINGUAL neighbors", r["multilingual"], 30)
    show(f"{word!r} ENGLISH-ONLY neighbors", r["english"], 30)
