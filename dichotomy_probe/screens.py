"""Pre-launch screens for the Dichotomy Transformation Probe (v2).

Run BEFORE any v2 data; results are folded into PREREGISTRATION_v2.md.

  1. Tokenizer subtoken check -- every pole, candidate, frame term, and control
     run through the Gemma tokenizer IN-FRAME with the leading space the model
     sees. In v2 the POLES must be single-token (the A/B axis endpoints), but
     candidate middles and phrase middles MAY be multi-token: they are read as
     the mean of their subtoken residuals, so the screen records subtoken counts
     rather than dropping them. Only a multi-token POLE would be a problem.
  2. Homograph screen -- odd, bad, heads, tails and other flagged senses,
     surfaced for human confirmation that the frame fixes the intended sense.
  3. Field-omission check -- delegated to config.field_omission_check(); it also
     enforces that every pair (including true dichotomies) carries candidate middles.

CPU-only (tokenizer only). Run:
    python -m modal run dichotomy_probe/screens.py
"""

import json
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0", "transformers>=4.42,<5", "accelerate>=0.30",
        "sae-lens>=3.20", "gradio>=5,<6", "pandas>=2.0", "requests>=2.31", "fastapi>=0.110",
    )
    .add_local_python_source("water_tool", "dichotomy_probe")
)

app = modal.App("dichotomy-probe-screens", image=image)
volume = modal.Volume.from_name("water-tool-cache", create_if_missing=True)

# Known homograph risks, surfaced for human confirmation that the frame fixes sense.
HOMOGRAPHS = {
    "heads": "body part / leadership sense vs. coin face — coin frame fixes it.",
    "tails": "animal tail / formal coat sense vs. coin face — coin frame fixes it.",
    "odd": "peculiar sense vs. parity — integer frame fixes it.",
    "bad": "slang 'good' sense vs. evaluative — action/morality frame fixes it.",
    "even": "level/flat sense vs. parity — integer frame fixes it.",
    "gray": "colour sense vs. moral-neutral sense — morality frame biases toward the latter.",
    "face": "coin face vs. human face — coin frame biases toward the coin face.",
}


@app.function(timeout=900, secrets=[modal.Secret.from_name("huggingface")],
              volumes={"/cache": volume})
def run_screens():
    import os
    os.environ["HF_HOME"] = "/cache/hf"
    from transformers import AutoTokenizer
    from water_tool.core.model import MODEL_ID
    from dichotomy_probe import config

    tok = AutoTokenizer.from_pretrained(MODEL_ID, token=os.environ.get("HF_TOKEN"))

    def standalone(word):
        ids = tok.encode(" " + word, add_special_tokens=False)
        return [tok.decode([i]) for i in ids], len(ids)

    def inframe(text, span):
        cpos = text.find(span)
        if cpos < 0:
            return None
        cend = cpos + len(span)
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=True)
        toks = [tok.decode([enc["input_ids"][i]]) for i, (s, e) in enumerate(enc["offset_mapping"])
                if e > s and s < cend and e > cpos]
        return toks

    token_rows = []
    for p in config.PAIRS:
        matched = config.matched_syntax_carrier(p)
        # poles: must be single-token
        for role, w in (("pole_A", p["A"]), ("pole_B", p["B"])):
            sa, n = standalone(w)
            token_rows.append({"pair": p["id"], "role": role, "word": w,
                               "standalone_subtokens": sa, "n_subtokens": n,
                               "inframe": inframe(matched.format(word=w), w),
                               "single_token": n == 1, "pole": True})
        # candidates / frame term / controls: multi-token allowed (mean-pooled)
        for c in config.candidates_for(p):
            w = c["word"]
            sa, n = standalone(w) if not c["is_phrase"] else (
                [tok.decode([i]) for i in tok.encode(" " + w, add_special_tokens=False)],
                len(tok.encode(" " + w, add_special_tokens=False)))
            frame = matched.format(word=w) if not c["is_phrase"] else p["natural_carrier"].format(word=w)
            token_rows.append({"pair": p["id"], "role": c["input_type"], "word": w,
                               "standalone_subtokens": sa, "n_subtokens": n,
                               "inframe": inframe(frame, w),
                               "single_token": n == 1, "pole": False})

    homograph_rows = []
    for p in config.PAIRS:
        flagged = {w: HOMOGRAPHS[w] for w in (p["A"], p["B"], *p["candidate_middles"]) if w in HOMOGRAPHS}
        homograph_rows.append({"pair": p["id"], "named": p["named"],
                               "matched_carrier": config.matched_syntax_carrier(p),
                               "natural_carrier": p["natural_carrier"], "flagged": flagged})

    fo_ok, fo_problems = config.field_omission_check()
    pole_splits = [r for r in token_rows if r["pole"] and not r["single_token"]]

    return {"model_id": MODEL_ID, "tokenizer_screen": token_rows,
            "homograph_screen": homograph_rows, "field_omission_ok": fo_ok,
            "field_omission_problems": fo_problems, "pole_splits": pole_splits}


@app.local_entrypoint()
def main():
    r = run_screens.remote()
    print(f"\n=== v2 PRE-LAUNCH SCREENS  (tokenizer: {r['model_id']}) ===\n")
    print(f"Field-omission check OK: {r['field_omission_ok']}")
    for p in r["field_omission_problems"]:
        print(f"  PROBLEM: {p}")

    print("\n--- Tokenizer subtoken check (poles must be single-token; candidates may be multi) ---")
    print(f"{'pair':<20}{'role':<16}{'word':<20}{'n':<4}{'single?':<8}subtokens")
    print("-" * 100)
    for t in r["tokenizer_screen"]:
        print(f"{t['pair']:<20}{t['role']:<16}{t['word']:<20}{t['n_subtokens']:<4}"
              f"{('YES' if t['single_token'] else 'multi'):<8}{t['standalone_subtokens']}")

    if r["pole_splits"]:
        print("\n  !! POLE(S) THAT SPLIT (would break the axis endpoints):")
        for p in r["pole_splits"]:
            print(f"     {p['pair']} / {p['role']} / {p['word']!r} -> {p['standalone_subtokens']}")
    else:
        print("\n  All poles are single-token. Multi-token candidates are read by mean-pooling.")

    print("\n--- Homograph screen (human confirms the frame fixes the intended sense) ---")
    for h in r["homograph_screen"]:
        print(f"  [{h['pair']}] {h['named']}  matched={h['matched_carrier']!r}")
        for w, note in h["flagged"].items():
            print(f"      {w}: {note}")

    import os
    os.makedirs("dichotomy_probe/results", exist_ok=True)
    with open("dichotomy_probe/results/screen_results_v2.json", "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=2)
    print("\nScreen results -> dichotomy_probe/results/screen_results_v2.json")
