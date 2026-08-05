# domain-compare

Do the public spectroscopic datasets (Alberts/MMSD, NMRexp, nmrshiftdb2-2024,
MassSpecGym) cover the same molecular space as MARINA1, or a different one?

Write-ups: `wiki/experiments/spectroscopic-dataset-domain.md` (overlap, descriptors,
sFP vocabulary) and `wiki/experiments/np-chemical-space-umap.md` (the UMAP half).

MARINA1 is read **read-only** from `$MARINA_DATA_ROOT/Datasets/MARINA1/` — `index.pkl`
for SMILES, `RankingEntropy/bitinfo_to_idx.pkl` for the sFP vocabulary.
`MARINA_DATA_ROOT` defaults to `/home/user/atong`. The four public sets are read
read-only from `raw/`, which is fetched out of band; nothing in `scripts/` populates it.
All writes land in `smiles/` and `results/`.

## Run order

```bash
cd /home/user/atong/MARINA/analysis/domain-compare

pixi run python scripts/extract_smiles.py    # first -- everything else reads smiles/
pixi run python scripts/analyze_domain.py
pixi run python scripts/umap_embed.py

# plots, independent of each other
pixi run python scripts/plot_descriptors.py  # needs analyze_domain
pixi run python scripts/plot_umap.py         # needs umap_embed
```

`extract_smiles.py` and `analyze_domain.py` take source names as argv to redo one set
at a time (`... extract_smiles.py nmrexp`); with no args they do all five.

| script | question | output |
|---|---|---|
| `extract_smiles.py` | — | `smiles/<source>.txt` + `.json` for marina1, mmsd, nmrexp, nmrshiftdb2, massspecgym |
| `analyze_domain.py` | how much do the sets overlap MARINA1, and how do they differ? | `results/overlap.json`, `results/descriptors.parquet` |
| `umap_embed.py` | do they occupy the same manifold? | `results/umap_ecfp4.parquet`, `results/umap_sfp.parquet` |
| `plot_descriptors.py` | figure 1 | `results/descriptor_distributions.png` |
| `plot_umap.py` | figure 2 | `results/umap_facets.png` |

`analyze_domain.py` runs two passes: overlap on the **full** sets, then a
30,000-per-source subsample (`SEED = 0`) for descriptors and sFP out-of-vocabulary.
`umap_embed.py` uses a balanced 15,000 per set with `metric="jaccard"` (exactly
Tanimoto on sparse binary input).

`results/descriptors.parquet` — 148,936 rows (30,000 × 4 + all 28,936 of MassSpecGym).
Columns: `smiles`, `source`, `heavy_atoms`, `mw`, `fsp3`, `n_rings`, `max_ring_size`,
`n_aromatic_rings`, `n_stereocenters`, `n_rotatable`, `tpsa`, `n_O`, `n_N`, `np_score`,
`sfp_oov_frac`, `n_sfp_substructs`, `in_alberts_window`, `alberts_elements_ok`.
`results/umap_{ecfp4,sfp}.parquet` — 75,000 rows each (15,000 × 5), columns
`source`, `smiles`, `x`, `y`.

## Headline numbers

- unique canonical molecules: nmrexp **1,457,540** / marina1 **486,835** / mmsd **64,654**
  / nmrshiftdb2 **42,114** / massspecgym **28,936**
- exact overlap with MARINA1: MassSpecGym **57.2%** (16,565), nmrshiftdb2 **20.4%** (8,589),
  Alberts/MMSD **1.53%** (989), NMRexp **0.899%** (13,103)
- scaffold overlap follows: 50.9% / 36.4% / 10.3% / 3.3%, against MARINA1's 122,528 scaffolds
- MassSpecGym is the NP-rich control and it behaves — 57% says the measure detects NP
  membership, so the ~1% figures are real and not a canonicalization artifact
- NMRexp is 3× MARINA1's size yet **99.1% novel**, and novel in the synthetic-chemistry
  direction: 96.7% of its 319,270 scaffolds are unseen
- Alberts' 5–35 heavy-atom filter, which SpecX reuses verbatim, **excludes 31.2% of MARINA1**
- sFP out-of-vocabulary is a Simpson's paradox on molecule size: **7 pts** raw
  (MARINA1 0.446 vs NMRexp 0.518), **28 pts** at 45–80 heavy atoms (0.346 vs 0.630).
  corr(OOV, heavy atoms) flips sign by provenance: **−0.208** MARINA1, **+0.660** MMSD

## Re-fetching raw data

`raw/` is not in git — 2.0 GB across four sources. Rebuild from this directory.

| Source | File(s) | Size | Where it came from |
|---|---|---|---|
| nmrshiftdb2-2024 | `nmrshiftdb2/data.zip` -> `data/nmrshiftdb2_2024/*.pkl` | 982 MB | NMRNet's dataset release, [10.5281/zenodo.13317524](https://doi.org/10.5281/zenodo.13317524). MD5-verified against Zenodo. |
| NMRexp | `nmrexp/NMRexp_10to24_1_1004.parquet` | 631 MB | Published database, Peking University + DP Technology, [Sci Data 2025](https://www.nature.com/articles/s41597-025-06245-5). Complete and MD5-verified against Zenodo. |
| MassSpecGym | `massspecgym/MassSpecGym.tsv`, `MassSpecGym_classyfire_all.tsv` | 250 MB + 23 MB | GNPS + MassBank derived; used here as the NP-rich positive control. |
| Alberts/MMSD | `mmsd/aligned_chunk_*.parquet` (20 of 245) + `_all_chunks.json` | ~1.7 GB | Byte-range extracted from the 18.6 GB Zenodo archive. Chunks evenly spaced across the corpus. |

> **Exact download URLs were not recorded at acquisition time** for NMRexp, MassSpecGym and
> MMSD — only the papers and the verification method. Re-fetching those three means finding the
> current Zenodo/HuggingFace record from the citation above. The nmrshiftdb2 DOI is exact.
> If you re-download any of them, please record the resolved URL here.

**MMSD is a deliberate 8% sample, not a failed download.** Zenodo throttles this archive to
~420 KB/s, making the full 18.6 GB a ~12 h transfer; 20 evenly-spaced chunks of the 245 give
64,654 molecules, enough for the distributional comparisons this directory makes. Do not treat
`mmsd` counts as the full 794,403-molecule corpus.

`finetune-exp/` owns GNPS and MassBank and documents their re-fetch separately; `marina23-build/`
reads this directory's `raw/` and has none of its own.

## Notes

- All five scripts derive `ROOT` from `__file__`
  (`os.path.dirname(os.path.dirname(os.path.abspath(__file__)))`, since they live in
  `scripts/`), so this directory can be moved without editing anything. Everything
  outside the repo goes through one env var,
  `DATA_ROOT = os.environ.get("MARINA_DATA_ROOT", "/home/user/atong")`.
- `vendor/npscorer.py` + `vendor/publicnp.model.gz` are vendored on purpose: RDKit's
  `Contrib/NP_Score` is not in the conda package. `analyze_domain.py` reaches them with
  `sys.path.insert(0, ROOT/"vendor")` in the pool initializer, which still resolves
  because the `__file__`-derived `ROOT` is absolute.
- `plot_umap.py` raises `SystemExit("no umap parquet files yet")` if neither umap parquet
  exists; run `umap_embed.py` first.
- `in_alberts_window` was coded strictly (`5 < heavy_atoms < 35`) until **2026-08-04**;
  Alberts' filter is inclusive (`5 <= n <= 35`). The tell was MMSD — Alberts' own dataset —
  scoring 97.7% inside its own filter instead of 100.0%. Fixed in `analyze_domain.py`, and
  `descriptors.parquet` was regenerated in place from `heavy_atoms` (2,995 of 148,936 rows
  flipped). All five sources moved +1.4 to +2.6 points.
- Consequently the **MARINA1-excluded figure is 31.2%, not the 33.8%** quoted before that fix.
  33.8% was the strict-bound value; the wiki article's own descriptor table already carried the
  correct 68.8% inside-window figure and contradicted its prose. Both are now consistent.
- `pixi.toml` declares `scikit-learn` and `mhfp`, neither of which is imported: sklearn is
  a transitive dep of umap-learn (harmless), `mhfp` looks like a leftover from an abandoned
  fingerprint. `scipy` *is* used (`umap_embed.py` builds CSR matrices).
- The manifest uses the deprecated `[project]` table instead of `[workspace]`, which is why
  every file in `logs/` opens with a pixi deprecation warning.
