"""Modal app for the Dichotomy Transformation Probe.

SEPARATE Modal app (`dichotomy-probe`) so redeploys never disturb the live
Water Pattern Tool. It shares the base image, the Gemma weights volume/cache
(`water-tool-cache`), and the `huggingface` secret.

Two functions:
  run()  -- ON-DEMAND L4 GPU. Computes both measurements for every item and
            control across all 27 residual states, writes measurements.csv,
            results.html, and dichotomy_results.zip to the shared volume, and
            returns them so the local entrypoint can mirror them into the repo.
  web()  -- PINNED CPU (min_containers=1), no GPU, no model. Batch-run-then-
            display: reads the most-recently-run batch from the volume and
            serves a Gradio page with the results page inline plus ONE download
            button for the two files. Stable URL that does not change between
            deploys:
              https://jenniferchamberspalmer-research--dichotomy-probe-web.modal.run

OPERATOR COMMANDS (from the project folder)
    Run a batch (on-demand L4):
        python -m modal run dichotomy_probe/run_probe.py
    (Re)deploy the always-on web page (stop first, per carry-forward protocol):
        python -m modal app stop dichotomy-probe
        python -m modal deploy dichotomy_probe/run_probe.py
"""

import base64
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
    )
    .add_local_python_source("water_tool", "dichotomy_probe")
)

app = modal.App("dichotomy-probe", image=image)
volume = modal.Volume.from_name("water-tool-cache", create_if_missing=True)

RESULTS_DIR = "/cache/dichotomy/results"
CSV_NAME = "measurements.csv"
HTML_NAME = "results.html"
ZIP_NAME = "dichotomy_results.zip"
META_NAME = "latest.json"


@app.function(
    gpu="L4",
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/cache": volume},
    scaledown_window=300,
)
def run(run_stamp: str = ""):
    os.environ["HF_HOME"] = "/cache/hf"
    os.environ["NEURONPEDIA_CACHE"] = "/cache/neuronpedia"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    from water_tool.core.model import MODEL_ID
    from dichotomy_probe import config, measures, reporting

    # Field-omission screen must pass before any data is emitted.
    ok, problems = config.field_omission_check()
    if not ok:
        raise RuntimeError(f"Field-omission check failed: {problems}")

    all_rows, dropped, n_layers = [], [], None
    for unit in config.UNITS:
        rows, dr, nl = measures.measure_unit(unit)
        all_rows.extend(rows)
        dropped.extend([{"unit": unit["id"], "role": r, "word": w} for (r, w) in dr])
        n_layers = nl

    meta = {
        "model_id": MODEL_ID,
        "run_stamp": run_stamp,
        "n_layers": n_layers,
        "dropped_multitoken": dropped,
        "n_units": len(config.UNITS),
        "n_measurement_rows": len(all_rows),
    }

    csv_path = os.path.join(RESULTS_DIR, CSV_NAME)
    html_path = os.path.join(RESULTS_DIR, HTML_NAME)
    zip_path = os.path.join(RESULTS_DIR, ZIP_NAME)
    reporting.write_csv(all_rows, csv_path)
    reporting.write_html(all_rows, meta, html_path)
    reporting.write_zip(csv_path, html_path, zip_path)
    with open(os.path.join(RESULTS_DIR, META_NAME), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    volume.commit()

    with open(csv_path, "r", encoding="utf-8") as f:
        csv_text = f.read()
    with open(html_path, "r", encoding="utf-8") as f:
        html_text = f.read()
    with open(zip_path, "rb") as f:
        zip_b64 = base64.b64encode(f.read()).decode("ascii")

    return {"meta": meta, "csv_text": csv_text, "html_text": html_text, "zip_b64": zip_b64}


def _build_gradio():
    """Build the batch-run-then-display Gradio page (CPU only; reads the volume)."""
    import gradio as gr

    def load_latest():
        volume.reload()  # pick up the most recent committed batch
        html_path = os.path.join(RESULTS_DIR, HTML_NAME)
        zip_path = os.path.join(RESULTS_DIR, ZIP_NAME)
        meta_path = os.path.join(RESULTS_DIR, META_NAME)
        if not os.path.exists(html_path):
            return ("<p>No batch has been run yet. From the project folder run "
                    "<code>python -m modal run dichotomy_probe/run_probe.py</code>, "
                    "then reload this page.</p>", None, "No batch found.")
        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()
        stamp = ""
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                m = json.load(f)
            stamp = (f"Most recent batch: {m.get('run_stamp') or '(unstamped)'} · "
                     f"model {m.get('model_id')} · {m.get('n_layers')} residual states · "
                     f"{m.get('n_measurement_rows')} measurement rows.")
        dl = zip_path if os.path.exists(zip_path) else None
        return html, dl, stamp

    with gr.Blocks(title="Dichotomy Transformation Probe", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Dichotomy Transformation Probe\n"
            "Batch-run-then-display. This page shows the results of the most recently "
            "run batch; it does not run the probe on click. Viewing and downloading are "
            "the same action: the button below downloads both result files "
            "(`measurements.csv` and `results.html`) as one zip.\n\n"
            "*The unit of analysis is the residual representation carried at the token "
            "position, never the token. The machine never determines true or false; it "
            "transforms a cultural dichotomy already named in the corpus at entry, and the "
            "verdict lives in the human reader.*"
        )
        status = gr.Markdown()
        with gr.Row():
            refresh = gr.Button("Reload latest batch", variant="secondary")
            download = gr.DownloadButton("Download results (CSV + results page, one zip)",
                                         variant="primary")
        report = gr.HTML()

        demo.load(load_latest, outputs=[report, download, status])
        refresh.click(load_latest, outputs=[report, download, status])
    return demo


@app.function(
    volumes={"/cache": volume},
    min_containers=1,        # pinned always-on; stable URL, low idle cost (CPU, no model)
    scaledown_window=300,
)
@modal.asgi_app()
def web():
    # gradio_client bool-schema guard (mirrors the live Water tool).
    try:
        import gradio_client.utils as _gcu
        _orig = getattr(_gcu, "get_type", None)
        if _orig is not None:
            def _safe(schema):
                return "bool" if isinstance(schema, bool) else _orig(schema)
            _gcu.get_type = _safe
    except Exception:
        pass

    import gradio as gr
    from fastapi import FastAPI

    demo = _build_gradio()
    fastapi_app = FastAPI()
    # allowed_paths lets Gradio serve the results files, which live on the shared
    # volume under /cache (outside Gradio's default-servable directories). Without
    # this the Download button returns "No permissions" (HTTP 403).
    return gr.mount_gradio_app(fastapi_app, demo, path="/", allowed_paths=["/cache"])


@app.local_entrypoint()
def main():
    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    res = run.remote(run_stamp=stamp)

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, CSV_NAME), "w", encoding="utf-8") as f:
        f.write(res["csv_text"])
    with open(os.path.join(out_dir, HTML_NAME), "w", encoding="utf-8") as f:
        f.write(res["html_text"])
    with open(os.path.join(out_dir, ZIP_NAME), "wb") as f:
        f.write(base64.b64decode(res["zip_b64"]))
    with open(os.path.join(out_dir, META_NAME), "w", encoding="utf-8") as f:
        json.dump(res["meta"], f, ensure_ascii=False, indent=2)

    m = res["meta"]
    print(f"\n=== Dichotomy Transformation Probe — batch complete ({stamp}) ===")
    print(f"model={m['model_id']}  residual_states={m['n_layers']}  "
          f"units={m['n_units']}  rows={m['n_measurement_rows']}")
    if m["dropped_multitoken"]:
        print("Dropped (multi-token, scope gate):")
        for d in m["dropped_multitoken"]:
            print(f"    {d['unit']} / {d['role']} / {d['word']!r}")
    else:
        print("No words dropped at run time (all single-token).")
    print(f"\nMirrored into {out_dir}:")
    print(f"    {CSV_NAME}, {HTML_NAME}, {ZIP_NAME}, {META_NAME}")
    print("\nOpen the always-on page:")
    print("    https://jenniferchamberspalmer-research--dichotomy-probe-web.modal.run")
