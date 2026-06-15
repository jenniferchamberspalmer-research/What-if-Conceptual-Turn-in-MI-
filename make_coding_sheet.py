"""Build the human coding sheet (design Section 8) from saved results/*.json.

One row per English-only neighbor per word. Columns:
  word, neighbor, my_code, notes      <- the requested human-fill sheet
  claude_first_pass_code, cosine      <- reference (delete if unwanted)

`my_code` / `notes` are left blank for the researcher to fill in. The
reference columns let agreement be computed against Claude's first pass.

Usage:
    python make_coding_sheet.py                 # Stratum A (default)
    python make_coding_sheet.py B C             # other strata
"""

import csv
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# Canonical within-stratum word order (design Section 5).
ORDER = {
    "A": ["water", "stone", "bread", "salt", "fire", "blood", "hand", "tree", "milk", "bone"],
}


def rows_for_stratum(stratum: str):
    order = ORDER.get(stratum, [])
    files = glob.glob(os.path.join(RESULTS, f"{stratum}_*.json"))
    by_word = {}
    for path in files:
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
        by_word[rec["word"]] = rec
    ordered_words = order + [w for w in sorted(by_word) if w not in order]
    rows = []
    for word in ordered_words:
        rec = by_word.get(word)
        if not rec:
            continue
        for n in rec["english"]["neighbors"]:
            rows.append({
                "word": word,
                "neighbor": n["token"],
                "my_code": "",
                "notes": "",
                "claude_first_pass_code": n["code"],
                "cosine": n["cosine_similarity"],
            })
    return rows


def main():
    strata = [s.upper() for s in sys.argv[1:]] or ["A"]
    fields = ["word", "neighbor", "my_code", "notes", "claude_first_pass_code", "cosine"]
    for stratum in strata:
        rows = rows_for_stratum(stratum)
        out = os.path.join(RESULTS, f"coding_sheet_stratum_{stratum}.csv")
        # utf-8-sig so Excel detects UTF-8 (non-ASCII neighbors render correctly).
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"Stratum {stratum}: {len(rows)} rows -> results/coding_sheet_stratum_{stratum}.csv")


if __name__ == "__main__":
    main()
