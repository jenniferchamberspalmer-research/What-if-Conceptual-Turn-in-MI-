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

## Files

| file | what it is |
|---|---|
| `v4final.py` | the LOCKED instrument, verbatim — canonical |
| `v4final_pairs.csv` / `v4final_field.png` | its outputs (table + depth plot) |
| `v4final_posdiag.py` | disclosed position diagnostic (reads the word token at pos -2 instead of the sentence-final "."); NOT canonical |
| `v4final_posdiag_pairs.csv` / `v4final_posdiag_field.png` | diagnostic outputs |

## Result of first run (2026-07-02)

**BRANCH A**: alive/dead patterns with good/bad (dist 0.142) not with the true
binaries (dist 0.334) — per the frozen pre-registration, geometry tracks the
real usage field, surviving the hard cultural label. The diagnostic also lands
Branch A (0.058 vs 0.137).

Caveats on record: (1) the locked read position is the sentence-final "."
token, flagged before running — the field_score class separation exists at that
position and saturates at the word position; (2) `readout_drop` is ~0.04 for
every pair — the "decoheres only at readout" half of the claim gets no
differential support; Branch A is carried by field_score alone; (3) even/odd is
a suspect control in these carriers ("It was even." reads *even* as an adverb).
