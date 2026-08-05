# analysis/

Standalone analyses supporting the MARINA papers. Each subdirectory is self-contained:
its own scripts, its own pixi environment, its own README with the run order, and — where
the result warranted a formal write-up — a LaTeX report and compiled PDF.

These were consolidated here on 2026-08-04 from three scattered locations. The prose
write-ups live in a separate wiki repository; what is here is the code, the small results,
and the reports.

## Setup

Each directory pins its own environment, deliberately — `fp-redundancy` is torch-free so it
can run without the model stack, and the others have genuinely different dependency sets.

```bash
cd analysis/<dir> && pixi install     # materializes from the committed pixi.lock
```

Analyses that read data living outside the repo (MARINA1, the benchmarks, checkpoints)
resolve it through one environment variable:

```bash
export MARINA_DATA_ROOT=/path/to/data-root    # default: /home/user/atong
```

It is expected to contain `Datasets/MARINA1/`, `Benchmark/` and `Checkpoints/`. Paths
*inside* the repo — including one analysis reading another's `raw/` — are resolved from
`__file__` and need no configuration.

## The analyses

| Directory | Question | Report |
|---|---|---|
| [`fp-redundancy/`](fp-redundancy/) | Is the 16,384-bit fingerprint redundant, and can retrieval recover anything by de-correlating the metric? | `report/bit-calibration.pdf` |
| [`finetune-exp/`](finetune-exp/) | What experimental spectra exist, and how much of MARINA1 do they cover? | — |
| [`domain-compare/`](domain-compare/) | Do public NMR/MS datasets cover natural-product space? | — |
| [`marina23-build/`](marina23-build/) | **Builds** MARINA2/3/4 by transforming MARINA1. | — |
| [`peak-sim-analysis/`](peak-sim-analysis/) | How far apart are the Annotated / Journal / Simulated benchmarks at the input? | — |
| [`spectre-ds-compare/`](spectre-ds-compare/) | How do the SPECTRE and MARINA1 datasets actually compare? | — |
| [`layers8-decomposition/`](layers8-decomposition/) | Does the depth finding survive at 8 layers? | `layers8-decomposition.pdf` |

`fp-redundancy/` is the reference for house style — numbered scripts, `__file__`-relative
paths, runtime and memory notes, and a `stats.json` that pins the LaTeX tables to the
computation so no number in the document is hand-transcribed.

## What is and is not committed

Tracked: scripts, READMEs, `pixi.toml` + `pixi.lock`, LaTeX sources, compiled PDFs, figures,
and small JSON/CSV/Markdown results — about 5 MB in total.

Not tracked (see the repo `.gitignore`): materialized `.pixi/` environments, third-party
`raw/` downloads (~5.4 GB — GNPS, MassBank, nmrshiftdb2, NMRexp, MassSpecGym, MMSD), and
large derived artifacts (`*.parquet`, `*.pkl`, `*.npz`, `*.pt`, `*.tsv`, `*.smi`). Each
directory's README documents where its raw inputs came from.

Consequence worth stating plainly: **a fresh clone cannot re-run every analysis end to end.**
Some depend on redistributable-by-licence-only archives. The committed small results and the
reports are enough to check every number claimed.

## A ninth analysis lives on another branch

`analysis/cls-decomposition/` — the exact additive decomposition of the final CLS token, with its
LaTeX report — is carried on the **`modality-attribution`** branch, not here. It depends on
`scripts/analysis/modality_contribution.py` and on `MARINA.forward_contributions` in
`src/modules/marina/model.py`, both of which exist only on that branch, so the measurement is
only reproducible there. `git checkout modality-attribution` to work on it.

## Cross-directory dependencies

Not obvious from any single directory, and the usual cause of a failed re-run.

- `marina23-build` reads `../domain-compare/raw/` (nmrshiftdb2, NMRexp, MassSpecGym),
  `../finetune-exp/raw/` (GNPS, MassBank) and `../finetune-exp/results/marina1_hsqc_classified.parquet`.
- `finetune-exp` reads `../domain-compare/smiles/` and its MassSpecGym raw file.
- `fp-redundancy` scripts `00`, `01` and `08` run under the **repo root's** pixi environment,
  not their own — they need torch, RDKit or MARINA source. Its README says which.

## Known hazards

- **`marina23-build/03_build.py` and `09_build_marina4.py` overwrite** `Datasets/MARINA{2,3,4}/`
  under `$MARINA_DATA_ROOT`. They are the only destructive scripts here.
- **`finetune-exp/results/hsqc_provenance.json`** and `marina1_hsqc_provenance.parquet` are
  known-wrong outputs kept for the record, sitting next to the correct
  `marina1_hsqc_classified.*`.
