"""Headless verification: does the original 'water' result reproduce?

This is the Section 10/11 gate of the study design. It runs the UNMODIFIED
embedding path (water_tool.views.embedding.raw_lookup) on Modal so the
reproduction is faithful to the original instrument rather than a
re-implementation.

The image + volume are defined inline here (mirroring modal_app.py) because
the Modal container only mounts the entrypoint file and the water_tool
package -- importing modal_app inside the container fails. The reuse that
matters for the study is the embedding logic, which is untouched.

The original finding: the nearest neighbors of "water" in the Gemma 2 2B
input-embedding space are cross-linguistic translational equivalents
(agua, Wasser, eau, ...) rather than experiential associates
(drink, thirst, wet, ...). If that does not reproduce here, STOP --
everything downstream of the study depends on it.

Run:
    python -m modal run verify_water.py
    python -m modal run verify_water.py --word water --k 50
"""

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
    )
    .add_local_python_source("water_tool")
)

app = modal.App("water-verify", image=image)

volume = modal.Volume.from_name("water-tool-cache", create_if_missing=True)


@app.function(
    gpu="T4",
    timeout=1800,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/cache": volume},
)
def neighbors(word: str = "water", k: int = 50):
    import os
    os.environ["HF_HOME"] = "/cache/hf"
    os.environ["NEURONPEDIA_CACHE"] = "/cache/neuronpedia"

    from water_tool.views import embedding

    df = embedding.raw_lookup(word, k=k)
    return df.to_dict("records")


@app.local_entrypoint()
def main(word: str = "water", k: int = 50):
    rows = neighbors.remote(word=word, k=k)
    print(f"\nTop {len(rows)} input-embedding neighbors of {word!r} (raw_lookup):\n")
    print(f"{'rank':>4}  {'cosine':>8}  token")
    print("-" * 40)
    for r in rows:
        print(f"{r['rank']:>4}  {r['cosine_similarity']:>8.4f}  {r['token']}")
