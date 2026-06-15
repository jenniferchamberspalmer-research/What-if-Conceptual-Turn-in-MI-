"""Reproducible, parameterized harness for the full Water Pattern Study
(design Sections 5, 6, 10).

Runs all three views for a word list (default: the full 60-word stratified
set) and writes structured per-word records to the Modal volume. Reuses the
UNMODIFIED model / embedding / probability / SAE paths of the original tool.

Self-checks per word (Section 10):
  - tokenization: record subtoken count; flag words that fragment (>1 token)
  - high-UR flag: flag words whose neighbor set is mostly noise (retrieval/
    tokenization problem, not a finding)
  - SAE load: confirm features load at all three layers (6/12/19)

Run a SUBSET first to validate end to end, then the full set:
    python -m modal run study_run.py --words water,stone,here
    python -m modal run study_run.py --stratum A
    python -m modal run study_run.py                      # full 60 words
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

app = modal.App("water-study-run", image=image)
volume = modal.Volume.from_name("water-tool-cache", create_if_missing=True)

CACHE_DIR = "/cache/study"
RESULTS_DIR = "/cache/study/results"
SAE_LAYERS = [6, 12, 19]


def _resolve_words(words: str | None, stratum: str | None):
    from water_tool.study import wordlist
    if words:
        ws = [w.strip() for w in words.split(",") if w.strip()]
        lookup = {w: s for w, s in wordlist.all_words()}
        return [(w, lookup.get(w, "?")) for w in ws]
    if stratum:
        return [(w, stratum) for w in wordlist.words_in(stratum)]
    return wordlist.all_words()


@app.function(
    gpu="T4",
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/cache": volume},
)
def run_study(
    words: str | None = None,
    stratum: str | None = None,
    k: int = 50,
    min_en_zipf: float = 2.2,
    dominance_margin: float = 0.9,
    ur_flag_threshold: float = 0.30,
    do_view2: bool = True,
    do_view3: bool = True,
):
    import os
    os.environ["HF_HOME"] = "/cache/hf"
    os.environ["NEURONPEDIA_CACHE"] = "/cache/neuronpedia"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    from water_tool.core.model import load
    from water_tool.views import probability, features
    from water_tool.study import wordlist, vocab_filter, neighbors, coding, battery

    model, tok = load()

    # English vocab filter (build or load from cache).
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

    # SAE self-check: confirm all three layers load before trusting any View 3.
    sae_ok = {}
    if do_view3:
        from water_tool.core.sae import get_sae
        for layer in SAE_LAYERS:
            try:
                get_sae(layer)
                sae_ok[layer] = True
            except Exception as e:
                sae_ok[layer] = f"FAILED: {e}"

    targets = _resolve_words(words, stratum)
    records = []

    for word, strat in targets:
        rec = {"word": word, "stratum": strat}

        # --- Self-check: tokenization ---
        ids = tok.encode(" " + word.strip(), add_special_tokens=False)
        subtoks = [tok.decode([i]) for i in ids]
        rec["tokenization"] = {
            "n_subtokens": len(ids),
            "subtokens": subtoks,
            "fragments": len(ids) > 1,
        }

        # --- View 1: multilingual + english neighbors, coded ---
        nsets = neighbors.neighbor_sets(word, english_ids, k=k)
        ml = coding.code_neighbor_records(nsets["multilingual"].to_dict("records"), word)
        en = coding.code_neighbor_records(nsets["english"].to_dict("records"), word)
        rec["view1"] = {
            "multilingual": {"neighbors": ml, "profile": coding.profile(ml)},
            "english": {"neighbors": en, "profile": coding.profile(en)},
        }

        # --- Self-check: high UR (noise) flag ---
        ml_ur = coding.profile(ml)["counts"].get("UR", 0) / max(len(ml), 1)
        en_ur = coding.profile(en)["counts"].get("UR", 0) / max(len(en), 1)
        rec["flags"] = {
            "high_UR": (ml_ur > ur_flag_threshold or en_ur > ur_flag_threshold),
            "ur_rate_multilingual": round(ml_ur, 3),
            "ur_rate_english": round(en_ur, 3),
        }

        # --- View 2: contextual next-token across the fixed battery ---
        if do_view2:
            v2 = []
            v1_en_tokens = {r["token"].lower() for r in en}
            v1_ml_tokens = {r["token"].lower() for r in ml}
            for fr in battery.frames_for(word):
                df = probability.top_next_tokens(fr["prompt"], k=20)
                top = df.to_dict("records")
                cont = [t["token"].strip().lower() for t in top]
                overlap_en = sum(1 for c in cont if c in v1_en_tokens)
                overlap_ml = sum(1 for c in cont if c in v1_ml_tokens)
                v2.append({
                    "frame_id": fr["id"], "family": fr["family"], "prompt": fr["prompt"],
                    "top_next": top,
                    "overlap_with_view1_english": overlap_en,
                    "overlap_with_view1_multilingual": overlap_ml,
                })
            rec["view2"] = v2

        # --- View 3: SAE features at layers 6/12/19 on a neutral frame ---
        if do_view3:
            frame_text = f"The {word} was there."
            v3 = {}
            for layer in SAE_LAYERS:
                if sae_ok.get(layer) is not True:
                    v3[layer] = {"error": sae_ok.get(layer)}
                    continue
                try:
                    fdf = features.top_features(frame_text, word, layer, k=15)
                    v3[layer] = {"frame": frame_text, "features": fdf.to_dict("records")}
                except Exception as e:
                    v3[layer] = {"error": f"{type(e).__name__}: {e}"}
            rec["view3"] = v3

        # Persist per-word record immediately (so a long run is resumable/inspectable).
        with open(os.path.join(RESULTS_DIR, f"{strat}_{word}.json"), "w") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        records.append(rec)

    volume.commit()

    return {
        "n_words": len(records),
        "sae_self_check": sae_ok,
        "thresholds": {"k": k, "min_en_zipf": min_en_zipf,
                       "dominance_margin": dominance_margin},
        "summary": [
            {
                "word": r["word"], "stratum": r["stratum"],
                "fragments": r["tokenization"]["fragments"],
                "high_UR": r["flags"]["high_UR"],
                "en_langue_pct": r["view1"]["english"]["profile"]["langue_pct_of_all"],
                "en_parole_pct": r["view1"]["english"]["profile"]["parole_pct_of_all"],
                "en_needs_review": r["view1"]["english"]["profile"]["needs_review"],
                "ml_langue_pct": r["view1"]["multilingual"]["profile"]["langue_pct_of_all"],
            }
            for r in records
        ],
    }


@app.local_entrypoint()
def main(
    words: str | None = None,
    stratum: str | None = None,
    k: int = 50,
    min_en_zipf: float = 2.2,
    dominance_margin: float = 0.9,
    do_view2: bool = True,
    do_view3: bool = True,
):
    r = run_study.remote(
        words=words, stratum=stratum, k=k,
        min_en_zipf=min_en_zipf, dominance_margin=dominance_margin,
        do_view2=do_view2, do_view3=do_view3,
    )
    print(f"\nRan {r['n_words']} words.  thresholds={r['thresholds']}")
    print(f"SAE self-check (layers 6/12/19): {r['sae_self_check']}")
    print("\nPer-word summary (English-only is the primary measure):")
    print(f"{'stratum':<8}{'word':<14}{'frag':<6}{'UR!':<5}"
          f"{'en_langue%':<12}{'en_parole%':<12}{'needs_rev':<10}{'ml_langue%':<10}")
    print("-" * 86)
    for s in r["summary"]:
        print(f"{s['stratum']:<8}{s['word']:<14}"
              f"{'Y' if s['fragments'] else '.':<6}{'Y' if s['high_UR'] else '.':<5}"
              f"{s['en_langue_pct']:<12}{s['en_parole_pct']:<12}"
              f"{s['en_needs_review']:<10}{s['ml_langue_pct']:<10}")
    print("\nPer-word JSON records written to the volume under study/results/.")
    print("NOTE: en_langue% counts only CONFIDENT TL/SY/TX; English SY/TX/CO "
          "for same-language neighbors await the LLM+human coding pass.")
