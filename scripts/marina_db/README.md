# MARINA-DB — dataset creation pipeline

One source of truth for building **MARINA-DB**, the final dataset that supersedes
MARINA1/2/3/4 for all paper work. Everything needed to go from raw source dumps to a
trainable, scorable dataset lives under `scripts/marina_db/`. Run it with:

```bash
scripts/marina_db/run_build.sh            # full build
scripts/marina_db/run_build.sh --from 30  # resume from a stage
scripts/marina_db/run_build.sh --only 60  # one stage
```

All paths and knobs are in `config.py` (override via env: `DATASET_ROOT`,
`BENCHMARK_ROOT`, `MARINA_DATA_ROOT`, `COCONUT_RELEASE`, `FP_TYPE`, …). No stage script
hardcodes an absolute path, and there is exactly **one** `canonicalize_smiles`
(`src/modules/data/smiles.py`, fixed-point) — do not reintroduce local copies.

## Layout

```
scripts/marina_db/
  run_build.sh     end-to-end driver (stage table, --from/--only)
  config.py        all paths + build knobs (reuses core/const.py root resolution)
  lib/             shared helpers (csv_peaks, eval_loop) — dedup'd from the old copies
  build/           the reproducible deterministic pipeline  ← the important part
  benchmarks/      deterministic benchmark .pkl builders
  eval/            post-build scoring + reporting
  curation/        ISOLATED upstream: how benchmark compounds are harvested (human-in-loop)
```

## The reproducible build (`build/`)

| Stage | File | Does |
|------|------|------|
| 00 | `00_download.sh` | Pull NP-MRD / COCONUT (`$COCONUT_RELEASE`) / LOTUS + the spectral archives into `data/raw/` |
| 10 | `10_smiles_merge.py` | Merge source dumps → `smiles_dict.json` (one canonicalization) |
| 20 | `20_build_index.py` | Spectral assembly + index + retrieval set. **Splits deferred to 30.** Filters: exact MW ≤ 1000, ≥ 3 heavy atoms |
| 30 | `30_splits.py` | **Unified split policy (D7/D9):** SPECTRE-aligned (val/test/train), benchmark + **full 467-compound journal** → test, then free pool balanced to **global** 90/5/5 |
| 40 | `40_assemble_arrow.py` | `metadata.json` + JSONL → Arrow dataset |
| 45a–c | `benchmarks/build_{annotated,journal,simulated}.py` | Build the three benchmark `.pkl`s from `filtered/` CSVs. `build_journal.py` folds in the old `journal_prep` (canonicalization + leakage flags `marina_clean`/`spectre_clean`/`both_clean` + `retrieval_idx`) |
| 50 | `50_collapse_peaks.py` | **Per-peak collapse dataset-wide** (train ¹³C/¹H + both benchmarks; 1e-4 ppm merge; HSQC untouched) |
| 60 | `60_fp_vocab.py` | **D8:** build the uncapped-multiplicity fingerprint vocab + `bitinfo_to_idx.pkl` + `rankingset.pt` (previously had no builder in-repo) |
| 70 | `70_retrieval_augment.py` | Append the ~74 journal compounds absent from retrieval → `rankingset_aug.pt` (rank@k well-defined for all 467) |
| 80 | `80_verify.py` | One verifier: index↔arrow split consistency, retrieval superset incl. all benchmarks, no duplicate SMILES, split sanity. Nonzero exit on failure |

## Scoring (`eval/`)

- `score.py` — the single scorer. rank@1/5/10 (**tie-aware by default, D6**; `--strict` for the
  legacy pessimistic metric), dereplication top-1/5/10, mean_cos; over annotated + prepared-journal
  (with `marina_clean`/`both_clean` subsets) + simulated, using the augmented rankingset.
- `export.py` — xlsx / plots / per-compound comparison / cosine-delta reporting.

## Decisions baked in (see `wiki/experiments/marina-db-final-run.md`)

- **D1** 8 cross-attention layers · **D2** sweep conclusions transfer · **D3** no experimental 1D
  ingestion (¹³C/¹H stay Mnova-simulated) · **D5** negative-mode MS/MS deferred (waiting on ICEBERG
  checkpoints)
- **D6** tie-aware rank-1 is the default retrieval metric (strict kept behind a flag), both MARINA
  and SPECTRE — implemented in `src/modules/core/ranker.py::dot_prod_rank`
- **D7** the full 467-compound journal benchmark is excluded from train (canonical-SMILES match)
- **D8** fingerprint = uncapped multiplicity (supersedes presence-only Morgan)
- **D9** splits balanced to global 90/5/5 after forced SPECTRE/benchmark/journal assignments
- MW filter = exact (monoisotopic) ≤ 1000; per-peak convention throughout

## Benchmark curation (`curation/`) — upstream, human-in-the-loop

Non-deterministic data acquisition that produces `filtered/<NPID>/{1H,13C,HSQC}.csv`: harvest
NP-MRD candidates → find/fetch open-access papers → **manual** shift extraction. The deterministic
build (`benchmarks/`) *consumes* `filtered/`; it does not re-run curation. See `curation/README.md`.

## Verification / status

A true from-scratch rebuild is gated on two external inputs: fresh source dumps (on `marinaweb`)
and the **negative-mode MS/MS simulations still in flight**. Until then:

1. **Static:** `import`-check every stage; confirm no stray `canonicalize_smiles` defs and no
   hardcoded `/home/user`/`/root/gurusmart` paths outside `config.py`.
2. **Per-stage live** (dev pod / marinaweb, on existing MARINA1 data): 50 → 60 → 70 → 80 → `eval/score.py`;
   `80_verify` must pass.
3. **D6 check:** re-score the 6 MARINA checkpoints + SPECTRE tie-aware vs strict.

## Legacy (not part of this pipeline, kept for provenance)

`analysis/marina23-build/` (builds shipped MARINA2/3/4 — the evidence base for D2/D3/per-peak),
`analysis/retrieval-repair/`, `analysis/data-provenance/`, `analysis/sfp-report/` (research). These
built already-shipped artifacts; the go-forward build is this folder.
