# Field Probe (V4-FINAL)

Does the residual stream carry a populated FIELD between the poles of a false
binary? GPT-2 small, TransformerLens, **runs locally on CPU** — no Modal, no
GPU, ~2 minutes. Pre-registration: `PREREGISTRATION_FIELD_PROBE.md` (committed
alone, before the results commit).

## Run it yourself

**Easiest:** double-click `run_v4.bat` in this folder. It runs the locked
instrument, prints the table, and opens the plot when done.

**Or from PowerShell:**

```powershell
cd "C:\Users\amptk\OneDrive\Desktop\Repo_Water_tool\is-a-token-a-word-claude-water-pattern-tool-QV5SH\field_probe"
$env:PYTHONUTF8='1'
& "C:\Users\amptk\AppData\Local\Programs\Python\Python312\python.exe" v4final.py
```

Requirements are already installed on this machine (`torch` CPU +
`transformer_lens` in the winget Python 3.12). On any other machine:
`pip install transformer_lens matplotlib pandas numpy`.

## Version history

- **v1** read position `[-1]` landed on the sentence-final **period**, not the
  word (carriers ended in "."). Its Branch A result is **WITHDRAWN**.
- **v2** (current) removes trailing punctuation and mean-pools `resid_post`
  over the word's own subword span, with a tokenization integrity check.
  The frozen pre-registration header is untouched; only the probe moved from
  the period to the word. v2 was **pushed before any v2 data existed**
  (commit `fbc3986`), so temporal priority holds for this version.

## Files

| file | what it is |
|---|---|
| `v4final.py` | the LOCKED instrument (v2) — canonical |
| `v4final_pairs.csv` / `v4final_field.png` | v2 outputs (table + depth plot) |
| `v4final_posdiag.py` + its CSV/PNG | v1-era position diagnostic; superseded by the v2 fix, kept for the record |

## Result of v2 run (2026-07-02)

**BRANCH B, formally** — alive/dead sits nearer the true-binary signature
(0.054) than the good/bad signature (0.077). Per the frozen pre-registration:
the thesis narrows to culturally-graded pairs.

**How much weight the verdict bears (mechanical report, not a revision):**

- The margin is thin (0.023), and the two class signatures the discriminator
  chooses between are themselves only ~0.10 apart — because **field_score
  saturates at the word position for every class** (0.73–1.00; heads/tails
  "middles" edge/side/rim/face seat at 0.917, even/odd at 1.000). The component
  that separated classes in v1 was separating them at the period, not the word.
  A positive can be an artifact for the same reason a null can.
- What *does* separate at the word position is the **pole-cosine excess curve**:
  good/bad rides highest across all depth (~0.28–0.30 mid-depth) and shows the
  largest readout drop (0.188 vs 0.13–0.16 for the bundle; even/odd −0.02),
  while alive/dead tracks the true-binary bundle. That is corroboration-grade
  observation, not the pre-registered discriminator.
- Guards: G1 — `dying`/`terminal`/`failing` are single tokens with healthy
  norms; ` comatose` is 3 subtokens with the lowest norm (72.7), flagged.
  G2 — ok (dead_var 0.88 vs alive_var 0.80; no polysemy inflation).
- even/odd remains a suspect control: its excess is low/negative at all depths
  ("It was even" reads *even* as an adverb, not the parity term).

The verdict is the human reader's; the instrument only reports which branch
the numbers fell in.
