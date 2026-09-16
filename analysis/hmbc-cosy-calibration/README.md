# hmbc-cosy-calibration

Experimental HMBC/COSY/HSQC peak lists, hand-curated from NP-FIDBench raw deposits by a
Claude session following `CURATION_PROMPT.md`, used to calibrate the per-peak dropout of
MARINA2.0's ceiling pseudo-predictions (wiki: `models/marina2.0-hmbc-cosy.md`).

- `spectra.py` — helper CLI (queue / info / pick / plot) over `AgentBench/harness/fidproc.py`
- `curated/<NPID>.json` — one curated record per compound (schema in the prompt)
- `curated/LEDGER.md` — progress log
- `calibrate.py --tier full|tier1|tier12` — ceiling (RDKit) vs curated → `results/<tier>/{per_compound.md,rates.md,rates.json,dropout_params.json}`
- `finalize.py` — combines `results/full` (central rates) and `results/tier1` (per-molecule spread) into
  `results/marina2_dropout_params.json`, the file MARINA2.0's loader consumes
- write-up: `wiki/experiments/marina-experiments/hmbc-cosy-dropout-calibration.md`

```bash
cd ~/Workspace
for t in full tier1 tier12; do pixi run python3 MARINA/analysis/hmbc-cosy-calibration/calibrate.py --tier $t; done
pixi run python3 MARINA/analysis/hmbc-cosy-calibration/finalize.py
```
