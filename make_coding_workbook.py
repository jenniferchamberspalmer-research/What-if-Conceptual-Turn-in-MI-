"""Build the Stratum A human coding workbook (results/coding_sheet_stratum_A.xlsx).

Two sheets:
  Codebook    -- the six relation codes with operational definition (langue/parole
                 frame, design Sections 2-3 and the Section 7 code table), the test
                 question a coder applies, and a real Stratum A example.
  Coding sheet-- one row per multilingual neighbor per word, with dropdown-validated
                 my_code / confidence / DISCUSS columns for the researcher to fill.

The multilingual neighbor set is used (not English-only) because it is the only set
where the `language` column varies and the TL code is applicable. The English-only
set is a single language and is handled by make_coding_sheet.py.

Local, free (no Modal). Run:
    python make_coding_workbook.py
"""

import glob
import json
import os

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.datavalidation import DataValidation

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
OUT = os.path.join(RESULTS, "coding_sheet_stratum_A.xlsx")

ORDER = ["water", "stone", "bread", "salt", "fire", "blood", "hand", "tree", "milk", "bone"]

FONT = "Arial"
HEAD_FILL = PatternFill("solid", fgColor="1F4E78")
HEAD_FONT = Font(name=FONT, bold=True, color="FFFFFF", size=11)
TITLE_FONT = Font(name=FONT, bold=True, size=14)
NOTE_FONT = Font(name=FONT, italic=True, size=10, color="444444")
BODY_FONT = Font(name=FONT, size=10)
CODE_FONT = Font(name=FONT, bold=True, size=11)
WRAP = Alignment(wrap_text=True, vertical="top")
TOP = Alignment(vertical="top")
CENTER = Alignment(horizontal="center", vertical="top")
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

# The single test the design hangs everything on (Section 3): systemic sense (langue)
# survives a counterfactual world; experiential association (parole) does not.
UNIVERSAL_TEST = ("Would this relation hold in a language whose speakers had never "
                  "experienced these two things together? Yes -> langue (TL/SY/TX). "
                  "No, it depends on shared experience/co-occurrence -> CO (parole).")

# code, name, operational definition, test question, Stratum A example
CODEBOOK = [
    ("TL", "Translational equivalent",
     "The neighbor names the SAME concept (same signified) in a different language. "
     "A pure langue relation: it is a fact about the cross-linguistic sign system, not "
     "about any situation. Design Section 3 treats these as what an abstract system "
     "standing above any single tongue would predict.",
     "Is the neighbor the same word/concept expressed in another language?",
     "water -> agua (es), Wasser (de), 水 (zh/ja)"),
    ("SY", "Synonymic",
     "Same-language synonym or near-synonym: a different sign that denotes (nearly) the "
     "same concept and could substitute for the probe with little change of meaning. "
     "Langue, because the equivalence lives in the system of senses, not in experience.",
     "Same language: does it mean (nearly) the same thing and substitute for the probe?",
     "stone -> rock; blood -> plasma / serum"),
    ("TX", "Taxonomic",
     "Same-language relation of sense within the system: hypernym (broader kind), "
     "hyponym (narrower kind), or co-hyponym (sibling kind). Langue: an 'is-a / kind-of' "
     "link that holds by definition, independent of any speaker's experience.",
     "Same language: is it linked by a kind-of / is-a relation (broader, narrower, or sibling category)?",
     "tree -> shrub (co-hyponym); stone -> granite, marble (hyponyms)"),
    ("MO", "Morphological variant",
     "The same lexeme as the probe, differing only by case, inflection, or a derivational "
     "affix. Systemic but treated as NEUTRAL in the primary contrast: it is neither "
     "evidence for langue equivalence nor for parole association.",
     "Is it the same root word, differing only by case / ending / affix?",
     "water -> waters, watery, WATER"),
    ("CO", "Collocational / experiential associate",
     "A word linked to the probe by real-world co-occurrence or situated use rather than "
     "by systemic sense: things found, done, or felt together. This is the parole pole "
     "and the decisive evidence against the langue claim if it dominates a set.",
     "Is the link grounded in lived co-occurrence/use rather than meaning? (Fails the universal test above.)",
     "bread -> bakery / baker; milk -> dairy"),
    ("UR", "Unrelated / noise",
     "A sub-word fragment, broken token, or item with no coherent semantic relation to "
     "the probe. A retrieval/tokenization diagnostic, NOT a finding: a high UR rate flags "
     "a problem with that word's neighbor set.",
     "Is it a fragment / junk token / clearly unrelated item?",
     "water -> wod; tree -> ree; milk -> mle"),
]


def load_rows():
    by_word = {}
    for path in glob.glob(os.path.join(RESULTS, "A_*.json")):
        rec = json.load(open(path, encoding="utf-8"))
        by_word[rec["word"]] = rec
    words = ORDER + [w for w in sorted(by_word) if w not in ORDER]
    rows = []
    for word in words:
        rec = by_word.get(word)
        if not rec:
            continue
        for n in rec["multilingual"]["neighbors"]:
            rows.append((word, n["token"], n.get("language", "")))
    return rows


def style_header(ws, ncols):
    for c in range(1, ncols + 1):
        cell = ws.cell(row=1, column=c)
        cell.fill = HEAD_FILL
        cell.font = HEAD_FONT
        cell.alignment = Alignment(wrap_text=True, vertical="center", horizontal="left")
        cell.border = BORDER
    ws.row_dimensions[1].height = 22


def build_codebook(wb):
    ws = wb.active
    ws.title = "Codebook"

    ws["A1"] = "Stratum A Codebook -- langue (TL+SY+TX) vs parole (CO)"
    ws["A1"].font = TITLE_FONT
    ws.merge_cells("A1:E1")

    ws["A2"] = ("Primary distinction: TL + SY + TX (relations within the sign system, "
                "langue) vs CO (experiential co-occurrence, parole). MO is neutral; UR is "
                "a noise diagnostic.")
    ws["A2"].font = NOTE_FONT
    ws["A2"].alignment = WRAP
    ws.merge_cells("A2:E2")
    ws.row_dimensions[2].height = 30

    ws["A3"] = "Universal test for any ambiguous case: " + UNIVERSAL_TEST
    ws["A3"].font = Font(name=FONT, italic=True, bold=True, size=10, color="1F4E78")
    ws["A3"].alignment = WRAP
    ws.merge_cells("A3:E3")
    ws.row_dimensions[3].height = 42

    header_row = 5
    headers = ["Code", "Name", "Operational definition", "Test question", "Stratum A example"]
    for c, h in enumerate(headers, 1):
        cell = ws.cell(row=header_row, column=c, value=h)
        cell.fill = HEAD_FILL
        cell.font = HEAD_FONT
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        cell.border = BORDER
    ws.row_dimensions[header_row].height = 20

    fills = {"TL": "E2EFDA", "SY": "E2EFDA", "TX": "E2EFDA",
             "MO": "FFF2CC", "CO": "FCE4D6", "UR": "EDEDED"}
    r = header_row + 1
    for code, name, defn, test, example in CODEBOOK:
        ws.cell(row=r, column=1, value=code).font = CODE_FONT
        ws.cell(row=r, column=2, value=name).font = BODY_FONT
        ws.cell(row=r, column=3, value=defn).font = BODY_FONT
        ws.cell(row=r, column=4, value=test).font = BODY_FONT
        ws.cell(row=r, column=5, value=example).font = BODY_FONT
        fill = PatternFill("solid", fgColor=fills[code])
        for c in range(1, 6):
            cell = ws.cell(row=r, column=c)
            cell.alignment = CENTER if c == 1 else WRAP
            cell.fill = fill
            cell.border = BORDER
        ws.row_dimensions[r].height = 92
        r += 1

    widths = {"A": 8, "B": 22, "C": 56, "D": 40, "E": 30}
    for col, w in widths.items():
        ws.column_dimensions[col].width = w


def build_coding_sheet(wb, rows):
    ws = wb.create_sheet("Coding sheet")
    headers = ["word", "neighbor", "language", "my_code", "confidence", "notes", "DISCUSS"]
    ws.append(headers)
    style_header(ws, len(headers))

    for word, neighbor, lang in rows:
        ws.append([word, neighbor, lang, "", "", "", ""])

    last = len(rows) + 1
    for row in ws.iter_rows(min_row=2, max_row=last, max_col=len(headers)):
        for cell in row:
            cell.font = BODY_FONT
            cell.alignment = TOP
            cell.border = BORDER

    dv_code = DataValidation(type="list", formula1='"TL,SY,TX,MO,CO,UR"',
                             allow_blank=True, showDropDown=False)
    dv_conf = DataValidation(type="list", formula1='"High,Medium,Low"',
                             allow_blank=True, showDropDown=False)
    dv_disc = DataValidation(type="list", formula1='"yes,no"',
                             allow_blank=True, showDropDown=False)
    dv_code.error = "Pick one of TL, SY, TX, MO, CO, UR."
    dv_code.errorTitle = "Invalid code"
    ws.add_data_validation(dv_code)
    ws.add_data_validation(dv_conf)
    ws.add_data_validation(dv_disc)
    dv_code.add(f"D2:D{last}")
    dv_conf.add(f"E2:E{last}")
    dv_disc.add(f"G2:G{last}")

    widths = {"A": 12, "B": 18, "C": 16, "D": 12, "E": 13, "F": 44, "G": 11}
    for col, w in widths.items():
        ws.column_dimensions[col].width = w

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:G{last}"


def main():
    rows = load_rows()
    wb = Workbook()
    build_codebook(wb)
    build_coding_sheet(wb, rows)
    wb.save(OUT)
    print(f"Wrote {OUT}")
    print(f"Codebook: {len(CODEBOOK)} codes | Coding sheet: {len(rows)} neighbor rows")


if __name__ == "__main__":
    main()
