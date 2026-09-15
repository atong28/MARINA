# hmbc-cosy-calibration

Experimental HMBC/COSY/HSQC peak lists, hand-curated from NP-FIDBench raw deposits by a
Claude session following `CURATION_PROMPT.md`, used to calibrate the per-peak dropout of
MARINA2.0's ceiling pseudo-predictions (wiki: `models/marina2.0-hmbc-cosy.md`).

- `spectra.py` — helper CLI (queue / info / pick / plot) over `AgentBench/harness/fidproc.py`
- `curated/<NPID>.json` — one curated record per compound (schema in the prompt)
- `curated/LEDGER.md` — progress log
- calibration script (ceiling vs curated → observation rates by bond count / |J| / solvent):
  to be written once the first ~60 compounds exist
