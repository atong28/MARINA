# spectre-ds-compare

Head-to-head measurement of the SPECTRE release against MARINA1: how many molecules
each side trains and tests on, and whether the two models rank against the same
retrieval library.

Write-up: `wiki/infrastructure/spectre-vs-marina1-datasets.md`. Its
`## Reproducing these numbers` section has the `unzip` and `kubectl` steps that produce
the inputs below — do not duplicate them here.

MARINA1 is read **read-only** from `$MARINA_DATA_ROOT/Datasets/MARINA1/{index.pkl,
retrieval.pkl}` — `MARINA_DATA_ROOT` defaults to `/home/user/atong`. The SPECTRE side is
read from the extracted index trees in this directory, located relative to each script's
own `__file__`; both source zips are untouched.

**Hazard: `retrieval_diff.py` is superseded and writes the same output file as

## Run order

```bash
cd /home/user/atong/MARINA/analysis/spectre-ds-compare
pixi run python overlap.py                 # ~8 min on 16 procs
pixi run python retrieval_diff.py          # ~8 min; also caches canon_sets.pkl
pixi run python retrieval_vs_retrieval.py  # ~10 min; needs canon_sets.pkl + spectre_retrieval.tsv
```

| script | question | output |
|---|---|---|
| `overlap.py` | does MARINA1 cover SPECTRE's molecules? | `overlap.json` + stdout |
| `retrieval_diff.py` | why do some SPECTRE molecules miss MARINA1's retrieval set? | `retrieval_diff.json`, `canon_sets.pkl` + stdout |
| `retrieval_vs_retrieval.py` | are the two retrieval libraries the same library? | `retrieval_vs_retrieval.json`, `spectre_retrieval_canon.pkl` + stdout |
| `retrieval_diff.py` | — superseded, do not run | would overwrite `retrieval_diff.json` |

Data subtrees, all scratch:

| path | contents |
|---|---|
| `extract/` | 21 `index.pkl` files from `DatasetWithoutHSQC.zip` (2024 release); has `Isomeric_SMILES/`. Read by `overlap.py`. |
| `extract_full/` | 19 `index.pkl` files from `SPECTREDataset.zip` (2025 full release); has `Superclass/`, no `Isomeric_SMILES/`. Read by both `retrieval_diff*.py`. |
| `full_1d/` | 9,775 loose `.pt` files (`SMILES_dataset/{val,test}/oneD_NMR/`, 4,910 + 4,865) — the extraction behind the all-info 4,056 count. No script here reads it. |

`sample*/` hold five single-record `.pt` samples. `spectre_retrieval.tsv`,
`spectre_FP_r6_16384.pt`, `canon_sets.pkl` and `spectre_retrieval_canon.pkl` are copied
or cached artifacts, not results.

## Headline numbers

- retrieval libraries are near-identical: MARINA1 **518,736** canonical vs SPECTRE
  **526,083**, **shared 513,888**, only-SPECTRE 12,195, only-MARINA 4,848
- of the 12,195 SPECTRE-only, the top source is **hyunDB 7,826** (the one DB MARINA does
  not draw on); by reason, **5,865** are salts/mixtures and **144** are MW-filtered
  (80 < 100 Da, 64 > 1000 Da)
- SPECTRE's retrieval set is **526,316** rows, not the 526,163 printed in the SPECTRE
  paper — transposed digits; 233 collapse under canonicalization
- SPECTRE has 195,148 canonical molecules; **192,567 (98.7%)** are in MARINA1's
  retrieval set and **185,160 (94.9%)** in MARINA1's training index
- only **2,581** SPECTRE molecules miss MARINA1's retrieval set, and **2,538** of those
  are dotted SMILES (MARINA1 has zero)
- MARINA1's index (486,835 canonical) is a strict subset of its own retrieval set

## Notes

- `retrieval_diff.py` must run before `retrieval_vs_retrieval.py`, which needs the
  `canon_sets.pkl` it caches. The committed `retrieval_diff.json` has 8 keys including
  `other_molecules`.
- `retrieval_vs_retrieval.py` needs `spectre_retrieval.tsv`, which is in neither zip —
  it is obtained out-of-band from the `atong-spectre` PVC (`kubectl cp`, see the
  write-up). It also needs `canon_sets.pkl`, so it must run first.
- `overlap.py` reads `extract/` (2024 release) while the diff scripts read
  `extract_full/` (2025). Both releases index the same 195,185-SMILES universe, so the
  counts are comparable — but they are not the same tree.
- All four scripts hardcode `Pool(16)`. No CLI arguments. Paths are not hardcoded: files in
  this directory resolve from `os.path.dirname(os.path.abspath(__file__))`, so the
  directory can be moved, and MARINA1 resolves from `$MARINA_DATA_ROOT` (default
  `/home/user/atong`) — the only knob.
- Both sides are double-pass canonicalized with `MolToSmiles(isomericSmiles=False)`
  before any comparison.
- `only_spectre_sources` tags are **first-insertion-wins**, not exclusive membership.
  Do not read `LOTUS DB = 38` as "LOTUS contributed 38".
- This `pixi.toml` is newly added. The recorded numbers were produced under MARINA's env
  (`cd /home/user/atong/MARINA && pixi run python3 analysis/spectre-ds-compare/<script>`),
  which also works — the scripts resolve their own paths, so cwd does not matter.
