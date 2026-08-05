# peak-sim-analysis

How far apart are the three MARINA benchmark spectral sources (Annotated / Journal /
Simulated) for the *same* molecules? Measured with NMRPeak's assignment-free
peak-aware similarity, independent of any model.

Write-up: `wiki/experiments/benchmark-spectral-domain-gap.md`.

The three benchmark pickles are read **read-only** from
`$MARINA_DATA_ROOT/Datasets/MARINA1/benchmark.pkl` (Annotated),
`$MARINA_DATA_ROOT/Benchmark/benchmark-journal.pkl` and
`$MARINA_DATA_ROOT/Benchmark/benchmark-sim.pkl`, where `MARINA_DATA_ROOT` defaults to
`/home/user/atong`. Nothing is written back. No MARINA code is imported.

## Run order

```bash
cd /home/user/atong/MARINA/analysis/peak-sim-analysis   # for pixi; the scripts no longer care about cwd

# schema probes, print only
pixi run python inspect_benchmark.py
pixi run python inspect_cnmr.py
pixi run python inspect_hsqc.py

# the two measurements
pixi run python compute_cnmr_similarity.py
pixi run python compute_all_modalities.py

# deep dives on the worst Annotated<->Journal molecules, print only
pixi run python analyze_ann_jrn_disparity.py
pixi run python signed_offset_check.py
```

| script | question | output |
|---|---|---|
| `inspect_benchmark.py` | what shape are the three pickles? | stdout |
| `inspect_cnmr.py` | do the NPIDs line up; what is in `c_nmr`/`h_nmr`? | stdout |
| `inspect_hsqc.py` | what do the three HSQC columns mean? | stdout |
| `compute_cnmr_similarity.py` | how far apart are the sources on ¹³C? | `cnmr_similarity_per_molecule.csv` + stdout |
| `compute_all_modalities.py` | same for ¹³C + ¹H + HSQC (phase-aware and not) | `all_modalities_per_molecule.csv` + stdout |
| `analyze_ann_jrn_disparity.py` | missing peaks, or peaks in different places? | stdout |
| `signed_offset_check.py` | is the residual gap a systematic referencing offset? | stdout |

`spectrum_similarity_scorer.py` (NMRPeak's scorer, copied verbatim) and
`hsqc_similarity.py` (2D HSQC prototype) are libraries — imported, never run.

Both CSVs have 152 data rows, one per NPID common to all three sources, keyed
`npid` + `split`. `cnmr_similarity_per_molecule.csv` adds ¹³C peak counts
(`n_ann`, `n_jrn`, `n_sim`) and, for each of `ann_vs_sim` / `jrn_vs_sim` /
`ann_vs_jrn`, the score plus its decomposition (`_sim`, `_avgpk`, `_pen`).
`all_modalities_per_molecule.csv` gives the score only, for the same three pairs
across four modalities (`_c` ¹³C, `_h` ¹H, `_q` HSQC phase-aware, `_q0` HSQC
shift-only).

## Headline numbers

- 152 NPIDs common to all three; Annotated has 2 extra (NP0332442, NP0333170) with no
  Journal entry
- **Journal is closer to Simulated than Annotated is, in every modality**:
  ¹³C 0.753 vs **0.805**, ¹H 0.798 vs **0.827**, HSQC 0.640 vs **0.674**
- the two experimental sources agree with each other far more: ¹³C 0.899, ¹H 0.901,
  HSQC 0.841
- the Annotated gap is **missing carbons, not shift error**: count penalty 0.845 (ann)
  vs 0.972 (jrn), while shift agreement is 0.730 vs 0.763
- Annotated under-reports ¹³C peaks in **91/152** molecules (mean −1.6 vs Sim); Journal
  in only **20/152** (mean −0.3)
- Annotated↔Journal disparities are missing peaks plus a referencing subgroup: 48% of
  matched pairs agree within 1 ppm, and 3 of the 8 worst molecules share a systematic
  **−1.4 ppm** offset (std ≤ 0.26)

## Notes

- **Output paths are now consistent (fixed).** Both `compute_cnmr_similarity.py` and
  `compute_all_modalities.py` write their CSV to `os.path.join(ROOT, ...)` with
  `ROOT = os.path.dirname(os.path.abspath(__file__))`, so both always land in this
  directory. `compute_all_modalities.py` previously wrote a relative path and dropped
  its CSV into whatever cwd you launched from; that trap is gone. Input paths go
  through `DATA_ROOT = os.environ.get("MARINA_DATA_ROOT", "/home/user/atong")`.
- `pytorch` is declared in `pixi.toml` but **never imported**. It is still required: all
  three benchmark pickles store shifts as torch tensors (`torch.storage`,
  `torch._utils`), and unpickling fails without it. CPU build is enough.
- `signed_offset_check.py` hardcodes the 8 worst NPIDs that
  `analyze_ann_jrn_disparity.py` ranks. Re-run the ranker first if the inputs change,
  or the two scripts silently drift apart.
- HSQC 2D is a **prototype**, not part of NMRPeak (they do 1D only). Kernels and
  tolerances are inherited from the 1D defaults. Compare HSQC scores across source
  pairs, not against the 1D columns — the product kernel makes them structurally lower.
- ¹H in the benchmark is shift-only, so this is the paper's ¹H metric reduced to its
  shift term; multiplicity, integration and the hydrogen-count penalty are unavailable.
  `HNMRSimilarityScorer` in the vendored file is therefore unused.
- `pixi.toml` still uses the deprecated `[project]` table and pixi warns on every run.
  Harmless; `[workspace]` is the current spelling.
