# MARINA paper — main-text tables: conditions, checkpoints, evaluation

Everything needed to regenerate the five main-text tables of the MARINA paper (Draft 2, 2026-10-01): one spec per
table, one manifest per checkpoint, the evaluation runner, the table builder and the committed raw results.
Appendix tables are out of scope for now.

**Ranking metric: cosine** (decided 2026-10-01 after a full comparison). Binary-Tanimoto ranking was evaluated too and
is kept for reference (`results/raw-jaccard/`, `results/tables-jaccard/`): on identical checkpoints it is ~1.5–4 pp
lower at rank@1 than cosine (PRIVATE flagship journal test NMR+MS/MS 71.08 → 68.52; MARINA-DB flagship 70.23 → 66.24).

```
paper/
  specs/<table>.json            exact conditions of each table (machine-readable; read by make_tables.py)
  checkpoints/<experiment>.json one per checkpoint: file, epoch, bytes, sha256, locations, training commit/cluster,
                                and the full training params.json
  run_eval.sh                   GPU runner (groups flagship / spectre / fpsweep / regime / collect)
  nautilus/make_jobs.sh         Nautilus Jobs for the PRIVATE checkpoints that live on the atong-spectre PVC
  make_tables.py                raw results -> results/tables/<table>{,_<split>}.{tex,json}
  db_comparison/                Table 1 generator + its two snippets (counts, no model)
  tools/                        build_ckpt_manifests.py, verify_ckpts.py, augment_spectre_bank.py
  results/raw-cosine/<bench>/   committed per-checkpoint outputs (<exp>_benchmark_journal_results.pkl, _sim_results.json)
  results/tables/               built paper tables (cosine; \tbd where an input is missing; json lists what is missing)
  results/raw-jaccard/, results/tables-jaccard/   the same with binary-Tanimoto ranking (comparison only)
```

## Conventions shared by every table

| | |
|---|---|
| **Retrieval ranking** | **cosine** between the predicted probabilities (sigmoid) and each retrieval row (L2-normalised on-bit sets): `RANK_METRIC=cosine` (run_eval.sh default) → `eval_journal_benchmark.py --rank_metric cosine`. `RANK_METRIC=jaccard` gives the binary-Tanimoto comparison (`RankingSet(metric="jaccard")`, predicted on-bits = sigmoid ≥ 0.5). |
| **rank@k** | strict: retrieval rows with similarity ≥ similarity(prediction, gold) count against the gold, minus the gold's own row (ties count against). The tie-aware variant is also stored (`rank_tie`). |
| **Annotation (ann@k)** | SPECTRE definition: some structure among the top-k retrievals has ECFP4 (Morgan r=2, 2,048 bits) cosine ≥ 0.8 to the true structure. |
| **mean cos** | cosine between prediction and gold fingerprint (journal: probabilities; simulated test: binarised prediction, the training test-loop convention). Raw files also carry binary Tanimoto (`tani`). |
| **Checkpoint** | per seed, the single early-stopping-best `epoch_*.ckpt` (max `val/mean_cos`, patience 30, 750-epoch cap). |
| **Aggregation** | MARINA: mean ± sample std over seeds 0/1/2. SPECTRE: the single released model. |
| **Benchmarks** | MARINA-Bench = `Benchmark/benchmark-sim.pkl` (466 = 232 val / 234 test; experimental NMR from the papers + ICEBERG-simulated MS/MS; sha256 `c56a5b5d…`). SPECTRE-clean = journal compounds absent from SPECTRE's train and val: test 205 (`bench_clean_205`), val 207 (`bench_clean_val207`). Journal tautomer twins of training molecules (18) are kept, reported as is. |
| **Datasets** | MARINA-DB = CH-NMR-NP-first build (disk `Datasets/MARINA-DB-OPEN`, zip sha256 `2780459f…`, retrieval 531,927). MARINA-DB-PRIVATE = original build (disk `Datasets/MARINA-DB`, zip sha256 `5f6190be…`, retrieval 531,087). |

Every table is built for the split the paper reports and for the other split (`<table>_<split>.tex`).

## The tables

### Table 1 — `db_comparison` (Methods)
No model. `DATASETS_ROOT=~/Workspace/Datasets pixi run python paper/db_comparison/gen_db_comparison.py` →
`db_comparison.tex` (MARINA-DB vs SPECTRE-DB, with experimental CH-NMR-NP rows) and `db_comparison_private.tex`
(supplement: MARINA-DB-PRIVATE vs MARINA-DB), plus `db_comparison_dev.tex` comparing the development datasets
MARINA-DB-PARTIAL (disk `Datasets/MARINA1`; no negative-mode MS/MS) / MARINA-DB-PRIVATE / MARINA-DB (counts only, no
experimental rows). SPECTRE-DB counts are constants from the SPECTRE paper.

### Table 2 — `results_main` (dereplication + annotation)
| Condition | Value |
|---|---|
| Model | MARINA flagship `marina-db-open-chnmr-uniqmult-formula-s{0,1,2}` (CH-NMR-NP-first MARINA-DB, no solvent-offset augmentation) |
| Fingerprint | `RankingEntropyUniqueMultiplicity`, 16,384 bits |
| Retrieval set | MARINA-DB, 531,927 |
| Benchmark | MARINA-Bench **test** (n = 234); val (n = 232) also built |
| Rows (journal subset) | NMR+MS/MS* `nmr_msms` · NMR `nmr` · ME-HSQC `hsqc` · ¹³C `c_nmr` · ¹H `h_nmr` |
| Columns | Base = the subset; +F = `<subset>_formula` − `<subset>` per seed, then mean ± std |
| Metrics | (a) exp-rank@1/5/10; (b) exp-ann@1/5/10 |
| Run | `run_eval.sh flagship <3 experiments>` (bench `full`, `--deltas`) |

### Table 3 — `spectre_comparison`
| Condition | MARINA | SPECTRE |
|---|---|---|
| Model | same checkpoints as Table 2 | released website model `spectre-deployed` (`--project_name SPECTRE --fp_type RankingEntropy --legacy_spectre`) |
| Fingerprint | unique multiplicity | `RankingEntropy` (Sherlock), 16,384 bits |
| Retrieval set | MARINA-DB, 531,927 | deployed SPECTRE retrieval (526,316 rows) + the subset's structures that are not already in it (test: +196 → 526,512; val: +200 → 526,516), built by `tools/augment_spectre_bank.py` |
| Ranking | cosine | cosine |

Benchmark: SPECTRE-clean **test** (n = 205); val (n = 207) also built. Rows: NMR+MS/MS+Formula* (`nmr_msms_formula`,
MARINA only), NMR+MS/MS* (`nmr_msms`, MARINA only), NMR, ME-HSQC, ¹³C, ¹H. Metric: exp-rank@1/5/10. Bold: the larger
of the two models (MARINA wherever SPECTRE has no value). Run: `run_eval.sh flagship …` (benches `clean_test`,
`clean_val`) + `run_eval.sh spectre`.

### Table 4 — `fp_comparison`
| Condition | Value |
|---|---|
| Models (MARINA-DB-PRIVATE, 3 seeds each) | **main:** Unique mult. `marina-db-uniqmult-formula`, Sherlock `marina-db-sherlock-formula`; **appendix (`fp_comparison_all`):** Uncapped `marina-db-uncapped-formula`, Capped k=5 `marina-db-cap5-formula`, Substructure `marina-deltaai-substructure` (s2 early-stopped at epoch 144 — a finished run) |
| Fingerprint | each arm's own (`RankingEntropyUniqueMultiplicity` / `RankingEntropy` / `…MultiplicityUncapped` / `…Multiplicity` / `…Substructure`) |
| Retrieval set | MARINA-DB-PRIVATE, 531,087, each fingerprint's own rankingset |
| Benchmark | MARINA-Bench **val** (n = 232); test also built |
| Input | full NMR + molecular formula, MW and MS/MS withheld (`nmr_formula`) |
| Metrics | exp-rank@1/5/10, exp-mean-cos (in each fingerprint's space); bold = best per column |
| Run | `run_eval.sh fpsweep <experiments>` (Nautilus Jobs `fp-*` + `regime-formula`; substructure on a local GPU box) |

### Table 5 — `results_training_regime`
| Condition | Value |
|---|---|
| Models (MARINA-DB-PRIVATE, uncapped multiplicity, 3 seeds each) | NMR+MW `marina-db-uncapped-nmr` · NMR+MS/MS+MW `marina-db-uncapped-noform` · All inputs `marina-db-uncapped-formula` |
| Fingerprint / retrieval | `RankingEntropyMultiplicityUncapped`; MARINA-DB-PRIVATE, 531,087 |
| Eval inputs | NMR · ME-HSQC · ¹³C · ¹H (spectra only) |
| Experimental | MARINA-Bench **val** (n = 232); test also built |
| Simulated | MARINA-DB-PRIVATE **test** split (24,329), `--sim --deltas --sim_only hsqc_c_nmr_h_nmr hsqc c_nmr h_nmr` |
| Metrics | r@1/5/10 and mean cos, both sides; bold = mean ± std interval above every other regime's |
| Run | `run_eval.sh regime <experiments>` (Nautilus Jobs `regime-*`) |

## Reproduce

**1. Stage a work dir `W`** on a GPU box (layout in the header of `run_eval.sh`). On anthony3 the staging dir
`~/Workspace/PaperEvalStage/` already has this layout as symlinks; copy it to the GPU box with
`rsync -avhPL -S --exclude packed --exclude arrow/train vm3:Workspace/PaperEvalStage/ $W/`. Its pieces:
- `data/MARINA-DB` → `Datasets/MARINA-DB-OPEN`; `data/MARINA-DB-PRIVATE` → `Datasets/MARINA-DB`
- `data/SPECTRE-clean-{test,val}`: `DATASET_ROOT=/tmp PYTHONPATH=. pixi run python paper/tools/augment_spectre_bank.py
  --bundle ~/Deployments/SPECTRE-web/checkpoints/spectre_flexible --journal <bench pkl> --out <dir>`
- `bench/full` → `Benchmark/benchmark-sim.pkl`; `bench/clean_test` → `Benchmark/bench_clean_205/…`;
  `bench/clean_val` → `Benchmark/bench_clean_val207/…`
- `ckpt/<experiment>/<ts>/` from the locations in `checkpoints/<experiment>.json`. Checkpoints that live only on the
  Nautilus PVC are streamed straight to the GPU box through a host with kubectl (nothing stored in between):
  `python3 paper/tools/fetch_pvc_ckpts.py --work $W --pod <running pod mounting atong-spectre> --via vm3`.
  Check everything with `python paper/tools/verify_ckpts.py $W/ckpt`

**2. Evaluate** (MARINA repo at the committed SHA, `pixi install` done):
```bash
W=$W bash paper/run_eval.sh flagship marina-db-open-chnmr-uniqmult-formula-s{0,1,2}
W=$W bash paper/run_eval.sh spectre
W=$W bash paper/run_eval.sh fpsweep marina-deltaai-substructure-s{0,1,2}                 # local-only checkpoints
# PRIVATE checkpoints on the Nautilus PVC (uniqmult, sherlock, cap5, uncapped formula/nmr/noform):
COMMIT=<sha> STAMP=<yyyymmdd-hhmm> bash paper/nautilus/make_jobs.sh && kubectl apply -f paper/nautilus/jobs/
```
The Jobs write to `atong-spectre:/root/gurusmart/paper-eval/results/full/benchmarks/`; fetch with
`kubectl cp guru-research/<pod>:/root/gurusmart/paper-eval/results/full/benchmarks/<file> …`.

**3. Collect and build:** `W=$W bash paper/run_eval.sh collect` (copies into `paper/results/raw-cosine/`), then
`DATASET_ROOT=/tmp pixi run python paper/make_tables.py` (`--metric jaccard` for the comparison set), commit
`paper/results/`.

## Checks done when this package was built (2026-10-01)

- `tests/test_ranker_jaccard.py`: Jaccard ranks = brute force (strict and tie-aware), retrieval/pair similarity
  exact, cosine path unchanged.
- Cosine regression on a real checkpoint (PRIVATE uniqmult s0, 10 journal compounds × 26 subsets): top-1/5/10 and
  annotation identical to the stored Draft-2 results (130/130 records; |Δcos| ≤ 1.6e-4, CPU vs GPU).
- `make_tables.py` fed the stored Draft-2 cosine results reproduces Draft-2 Table 2 and the MARINA column of
  Table 3 digit for digit, and Table 4's unique-multiplicity row (64.08 ± 3.03 on `nmr_formula`).
- Draft-2 Table 4 used `nmr_mw_formula` (MW included); identical to `nmr_formula` for unique multiplicity.

## Known caveats

- **SPECTRE retrieval for the clean subsets.** The earlier Table 3 bank appended all clean-subset structures, so the
  7 already in SPECTRE's retrieval got an FP-identical duplicate that strict ranking counted against SPECTRE.
  `augment_spectre_bank.py` appends only missing structures. The bank was built with an older RDKit: for 1 (test) /
  2 (val) of those 7 in-bank compounds one fragment SMILES is written differently (`COc` vs `cOC`), moving one bit;
  recorded in `augment_summary.json`.
- **SPECTRE-DB size.** The deployed bundle's retrieval has 526,316 rows; the SPECTRE paper's SPECTRE-DB count is
  526,163 (Table 1 quotes the paper).
- **Solvent-offset augmentation was dropped** (2026-10-02): the solvent-jitter arm
  (`marina-db-open-solvjit-uniqmult-formula`) did not help on the headline inputs (journal test NMR+MS/MS 68.38 vs
  70.23), so the plain CH-NMR-NP arm is the flagship. Its raw results stay in `results/raw-cosine/` for reference.
