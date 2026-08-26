# MARINA-DB — dataset creation pipeline

One source of truth for building **MARINA-DB**, the final dataset that supersedes
MARINA1/2/3/4 for all paper work. Everything needed to go from raw source dumps to a
trainable, scorable dataset lives under `scripts/marina_db/`. Run it with:

```bash
scripts/marina_db/run_build.sh            # full build
scripts/marina_db/run_build.sh --from 5   # resume from a stage
scripts/marina_db/run_build.sh --only 9   # one stage
```

All paths and knobs are in `config.py` (override via env: `DATASET_ROOT`,
`BENCHMARK_ROOT`, `MARINA_DATA_ROOT`, `COCONUT_RELEASE`, `FP_TYPE`, …). No stage script
hardcodes an absolute path, and there is exactly **one** `canonicalize_smiles`
(`src/modules/data/smiles.py`, fixed-point) — do not reintroduce local copies.

## Layout

```
scripts/marina_db/
  run_build.sh     end-to-end driver (stage table, --from/--only/--to)
  config.py        all paths + build knobs (reuses core/const.py root resolution)
  lib/             shared helpers (csv_peaks, eval_loop) — dedup'd from the old copies
  build/           the reproducible deterministic pipeline (stages 1–12)  ← the important part
  eval/            post-build scoring + reporting
  curation/        ISOLATED upstream: how benchmark compounds are harvested (human-in-loop)
```

## The reproducible build (`build/`) — two tracks

**Structure track (stages 1–4)** builds the retrieval set and the fingerprint rankingsets. It needs
no spectral data and is **not blocked on the positive MS/MS re-prediction** — run `--to 4` to produce
retrieval + rankingsets today. **Spectral/training track (stages 5–12)** assembles the spectra, index,
splits, arrow shards, FragIdx training columns, journal prep, and verification; MS/MS is optional there
until the dataset is finalized.

| Stage | File | Does |
|------|------|------|
| 1 | `1_download.sh` | Pull NP-MRD / COCONUT (`$COCONUT_RELEASE`) / LOTUS + the spectral archives into `data/raw/` |
| 2 | `2_smiles_merge.py` | Merge source dumps → `smiles_dict.json` (one canonicalization) |
| 3 | `3_build_retrieval.py` | **Retrieval set from structures only** (`smiles_dict ∪ journal ∪ SPECTRE-corpus ∪ Mnova ∪ SPECTRE retrieval bank`, **no MS/MS**). Journal folded in so rank@k is defined for all 467; SPECTRE bank folded in so retrieval is a strict superset of SPECTRE's candidate pool; no later augment |
| 4 | `4_fp_rankingset.py` | For **every `config.FP_TYPES`** (uncapped-multiplicity **D8**, `RankingEntropy`/sherlock, capped-k5 multiplicity, substructure) at r10/16384: `bitinfo_to_idx.pkl` + `rankingset.pt`. Retrieval-only, no spectral dependency |
| 5 | `5_spectre_splits.py` | Derive the SPECTRE train/val/test partition from the corpus index (`data/raw/index.pkl` `split` field) → `spectre_splits.pkl` |
| 6 | `6_build_index.py` | Spectral assembly + index (`has_*` flags, MW ≤ 1000 exact, ≥ 3 heavy atoms). **MS/MS optional** (empty ⇒ `has_mass_spec=False`). **Splits deferred to 7** |
| 7 | `7_splits.py` | **Unified split policy (D7/D9):** SPECTRE-aligned (val/test/train), **full 467-compound Journal** → test, then free pool balanced to **global** 90/5/5 |
| 8 | `8_assemble_arrow.py` | `metadata.json` + JSONL → Arrow dataset |
| 9 | `9_fp_fragidx.py` | Training-path FragIdx parquets for **every `config.FP_TYPES`** (per-kind `FragIdx*.parquet`), reusing the stage-4 vocabs. Needs index + arrow |
| 10 | `10_build_journal.py` | Prepare the **frozen** Journal against this build: 2D canon + leakage flags `marina_clean`/`spectre_clean`/`both_clean` + `retrieval_idx` → `benchmark-journal-prepared.pkl` (does **not** rebuild `benchmark-journal.pkl`) |
| 11 | `11_collapse_peaks.py` | **Per-peak collapse dataset-wide** (train ¹³C/¹H + Journal; 1e-4 ppm merge; HSQC untouched) |
| 12 | `12_verify.py` | One verifier: index↔arrow split consistency, **retrieval superset incl. the Journal AND every index molecule** (this is what confirms MS/MS added no new compounds once positive lands), no duplicate SMILES, split sanity. Nonzero exit on failure |

## Scoring (`eval/`)

- `score.py` — the single scorer. rank@1/5/10 (**tie-aware by default, D6**; `--strict` for the
  legacy pessimistic metric), dereplication top-1/5/10, mean_cos; over the prepared Journal
  (with `marina_clean`/`both_clean` subsets) + the NMR+MW test subset. Each model is scored on its
  own `params.json` fp_type's `rankingset.pt`. The Journal is the sole benchmark — the annotated and
  simulated benchmarks are retired.
- `export.py` — xlsx / plots / per-compound comparison / cosine-delta reporting.
- The fp-quality comparison (`analysis/fp-quality/`) consumes the stage-4 rankingsets for all four
  FP families built on the one retrieval set.

## Decisions baked in (see `wiki/experiments/marina-db-final-run.md`)

- **D1** 8 cross-attention layers · **D2** sweep conclusions transfer · **D3** no experimental 1D
  ingestion (¹³C/¹H stay Mnova-simulated) · **D5** negative-mode MS/MS deferred (waiting on ICEBERG
  checkpoints)
- **D6** tie-aware rank-1 is the default retrieval metric (strict kept behind a flag), both MARINA
  and SPECTRE — implemented in `src/modules/core/ranker.py::dot_prod_rank`
- **D7** the full 467-compound Journal benchmark is excluded from train (canonical-SMILES match)
- **D8** deployed fingerprint = uncapped multiplicity (supersedes presence-only Morgan). Stage 4
  also builds `RankingEntropy` (sherlock), capped-k5 multiplicity, and substructure at r10/16384 for
  the fp-quality comparison — all `config.FP_TYPES`, all on the one retrieval set
- **D9** splits balanced to global 90/5/5 after forced SPECTRE/Journal assignments
- **D10** fingerprints built at radius 10 (`FP_RADIUS`)
- **Journal-only benchmark:** the Journal (`benchmark-journal.pkl`) is a frozen curated input and
  the sole benchmark; the annotated + simulated builders are retired, and the Journal is folded
  into retrieval at stage 3 (no separate augment stage)
- **Two-track build:** retrieval + rankingsets (stages 1–4) depend only on structures and are not
  blocked on the positive MS/MS re-prediction; the spectral/training track (5–12) treats MS/MS as
  optional. MS/MS is expected to add no retrieval structures — `12_verify.py` asserts it
- MW filter = exact (monoisotopic) ≤ 1000; per-peak convention throughout

## Benchmark curation (`curation/`) — upstream, human-in-the-loop

Non-deterministic data acquisition that produces `filtered/<NPID>/{1H,13C,HSQC}.csv`: harvest
NP-MRD candidates → find/fetch open-access papers → **manual** shift extraction. The frozen
`benchmark-journal.pkl` was built from `filtered/` during curation; the go-forward build treats
it as an input (stage 10 only re-derives the prepared view). See `curation/README.md`.

## Verification / status

The **structure track (stages 1–4)** is runnable now — it produces `retrieval.pkl` and the four
fingerprint rankingsets without any spectral data. The **finalized** spectral dataset is gated on
one external input: the **re-predicted positive-mode MS/MS** (`data/raw/ms_predictions/*.json`;
`MS_PREDICTIONS_POSITIVE_ID` in `1_download.sh`) — but stages 5–12 still run without it (MS/MS
optional). Checks:

1. **Static:** `import`-check every stage; confirm no stray `canonicalize_smiles` defs and no
   hardcoded `/home/user`/`/root/gurusmart` paths outside `config.py`.
2. **Structure track** (marinaweb, existing retrieval): `--to 4`; four `rankingset.pt` written.
3. **Per-stage live** (dev pod / marinaweb, on existing MARINA1 data): 9 → 10 → 12 → `eval/score.py`;
   `12_verify` must pass, and its retrieval-superset check confirms MS/MS added no new compounds.
4. **D6 check:** re-score the 6 MARINA checkpoints + SPECTRE tie-aware vs strict.

## Legacy (not part of this pipeline, kept for provenance)

`analysis/marina23-build/` (builds shipped MARINA2/3/4 — the evidence base for D2/D3/per-peak),
`analysis/retrieval-repair/`, `analysis/data-provenance/`, `analysis/sfp-report/` (research). These
built already-shipped artifacts; the go-forward build is this folder.
