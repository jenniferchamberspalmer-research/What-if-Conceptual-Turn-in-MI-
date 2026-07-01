"""Pre-launch screens for the Dichotomy Transformation Probe.

ALL screens run BEFORE any data exists; their results are folded into
PREREGISTRATION.md and committed (alone) before the probe is ever run.

Screens implemented here:
  1. Tokenizer single-token check -- every pole, midpoint, and excluded term
     run through the Gemma tokenizer IN-FRAME, with the leading space the model
     actually sees (" good" vs "good" can differ). A target that splits into
     subwords is multiple Y-units and is dropped by the single-token scope gate.
  2. Homograph screen -- odd, bad, heads, tails (and other flagged senses)
     surfaced for human confirmation that the frame fixes the intended sense.
  3. Field-omission check -- delegated to config.field_omission_check().

Only the tokenizer is needed, so this runs CPU-only (no GPU), reusing the
shared image + Gemma cache volume + huggingface secret.

Run:
    python -m modal run dichotomy_probe/screens.py
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
        "gradio>=5,<6",
        "pandas>=2.0",
        "requests>=2.31",
        "fastapi>=0.110",
    )
    .add_local_python_source("water_tool", "dichotomy_probe")
)

app = modal.App("dichotomy-probe-screens", image=image)
volume = modal.Volume.from_name("water-tool-cache", create_if_missing=True)


@app.function(
    timeout=900,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/cache": volume},
)
def run_screens():
    import os
    os.environ["HF_HOME"] = "/cache/hf"

    from transformers import AutoTokenizer
    from water_tool.core.model import MODEL_ID
    from dichotomy_probe import config

    tok = AutoTokenizer.from_pretrained(MODEL_ID, token=os.environ.get("HF_TOKEN"))

    def standalone_subtokens(word):
        """Tokens for ' word' -- the leading-space form the model sees mid-sentence."""
        ids = tok.encode(" " + word, add_special_tokens=False)
        return [tok.decode([i]) for i in ids], ids

    def inframe_subtokens(text, word):
        """Tokens whose character span falls within the word's occurrence in `text`."""
        char_pos = text.find(word)
        if char_pos < 0:
            return None, None, None
        char_end = char_pos + len(word)
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=True)
        toks, ids = [], []
        for i, (s, e) in enumerate(enc["offset_mapping"]):
            # Any token whose character span overlaps the word occurrence. The
            # leading-space token (e.g. "▁good") may start at char_pos-1, so an
            # overlap test is used rather than strict containment.
            if e > s and s < char_end and e > char_pos:
                toks.append(tok.decode([enc["input_ids"][i]]))
                ids.append(enc["input_ids"][i])
        return toks, ids, char_pos

    # --- Screen 1 + scope gate: tokenizer single-token check, in-frame ---
    token_rows = []
    for u in config.UNITS:
        for role, word in (
            [("pole_A", u["pole_A"]), ("pole_B", u["pole_B"])]
            + ([("midpoint", u["midpoint"])] if u["midpoint"] else [])
            + [("excluded", e) for e in u["excluded_terms"]]
        ):
            sa_toks, sa_ids = standalone_subtokens(word)
            # In-frame reading uses the carrier for non-poles; poles use the dichotomy frame if present.
            if role in ("pole_A", "pole_B") and u["dichotomy_frame"]:
                frame = u["dichotomy_frame"]
            else:
                frame = u["carrier_frame"].format(word=word)
            if_toks, if_ids, _ = inframe_subtokens(frame, word)
            token_rows.append({
                "unit": u["id"],
                "role": role,
                "word": word,
                "leading_space_form": " " + word,
                "standalone_n_subtokens": len(sa_ids),
                "standalone_subtokens": sa_toks,
                "frame": frame,
                "inframe_n_subtokens": (len(if_ids) if if_ids is not None else None),
                "inframe_subtokens": if_toks,
                "single_token": len(sa_ids) == 1,
                "passes_scope_gate": len(sa_ids) == 1,
            })

    # --- Screen 2: homograph screen (surface for human confirmation) ---
    homograph_rows = []
    for u in config.UNITS:
        homograph_rows.append({
            "unit": u["id"],
            "named_dichotomy": u["named_dichotomy"],
            "frame_fixes_sense": u["dichotomy_frame"] or u["carrier_frame"],
            "homograph_note": u["homograph_note"],
        })

    # --- Screen 3: field-omission check ---
    fo_ok, fo_problems = config.field_omission_check()

    return {
        "model_id": MODEL_ID,
        "tokenizer_screen": token_rows,
        "homograph_screen": homograph_rows,
        "field_omission_ok": fo_ok,
        "field_omission_problems": fo_problems,
    }


@app.local_entrypoint()
def main():
    r = run_screens.remote()

    print(f"\n=== PRE-LAUNCH SCREENS  (tokenizer: {r['model_id']}) ===\n")

    print("--- Screen 3: field-omission check ---")
    print(f"  OK: {r['field_omission_ok']}")
    for p in r["field_omission_problems"]:
        print(f"  PROBLEM: {p}")

    print("\n--- Screen 1 + single-token scope gate (leading-space, in-frame) ---")
    print(f"{'unit':<20}{'role':<10}{'word':<12}{'std_n':<6}{'inframe_n':<10}{'single?':<8}subtokens")
    print("-" * 92)
    any_split = False
    for t in r["tokenizer_screen"]:
        mark = "YES" if t["single_token"] else "NO"
        if not t["single_token"]:
            any_split = True
        print(f"{t['unit']:<20}{t['role']:<10}{t['word']:<12}"
              f"{t['standalone_n_subtokens']:<6}{str(t['inframe_n_subtokens']):<10}"
              f"{mark:<8}{t['standalone_subtokens']}")
    print("\n  Words that SPLIT (dropped by scope gate / deferred):" if any_split
          else "\n  All screened words are single-token in-frame.")
    for t in r["tokenizer_screen"]:
        if not t["single_token"]:
            print(f"    {t['unit']} / {t['role']} / {t['word']!r} -> {t['standalone_subtokens']}")

    print("\n--- Screen 2: homograph screen (human confirmation that the frame fixes the sense) ---")
    for h in r["homograph_screen"]:
        print(f"  [{h['unit']}] {h['named_dichotomy']}  frame={h['frame_fixes_sense']!r}")
        print(f"      {h['homograph_note']}")

    # Persist for folding into PREREGISTRATION.md.
    out = "dichotomy_probe/results/screen_results.json"
    import os
    os.makedirs("dichotomy_probe/results", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=2)
    print(f"\nScreen results written to {out}")
