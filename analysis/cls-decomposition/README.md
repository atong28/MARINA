# marina-cls-decomposition

The formal write-up of MARINA's CLS-token decomposition: an exact additive split of the final
CLS direction into per-modality contributions plus FFN, bias and initialisation residuals,
together with the attention-suppression ablation.

Wiki page: `experiments/marina-modality-attribution.md` (separate repo). The document here is the full derivation —
the wiki page is the summary.

Report: `marina-cls-decomposition.pdf` (source `marina-cls-decomposition.tex`).

## Run order

```bash
# 1. GPU. Produces the JSONs this directory consumes but does NOT archive.
cd <marina-repo>
pixi run python scripts/analysis/modality_contribution.py   # once per seed -> full_seed{0,1,2}.json
pixi run python scripts/analysis/modality_contribution.py --per-layer   # -> perlayer_seed1_n256.json
pixi run python scripts/analysis/early_attn_ablation.py     # -> early_attn_ablation.json
# all land in scripts/analysis/results/

# 2. The fingerprint-redundancy panel needs bit_mi.npz from the sibling analysis dir
cd <marina-repo>
pixi run python3 analysis/fp-redundancy/scripts/00_export.py
cd analysis/fp-redundancy && pixi run python scripts/02_bit_mi.py

# 3. Figures. Prints the LaTeX table bodies to stdout as well.
cd <marina-repo> && pixi run python3 analysis/cls-decomposition/make_figures.py

# 4. Render
cd analysis/cls-decomposition && tectonic marina-cls-decomposition.tex
```

| script | question | output |
|---|---|---|
| `make_figures.py` | — | the 5 `fig-*.pdf`, plus LaTeX table bodies on stdout |

## Inputs

| file | where | note |
|---|---|---|
| `full_seed{0,1,2}.json`, `perlayer_seed1_n256.json`, `early_attn_ablation.json` | `inputs/` | **Archived here 2026-08-04.** `make_figures.py` reads this directory first and only falls back to `<repo>/scripts/analysis/results/`. |
| `bit_mi.npz` | `inputs/` | Archived here too, since the `fp-redundancy` analysis lives on `exp/marina234` and is not present on this branch. |

> **This directory lives on the `modality-attribution` branch, and only there.**
> It depends on `scripts/analysis/modality_contribution.py`, `early_attn_ablation.py` and
> `plot_modality_layers.py`, plus the `MARINA.forward_contributions` model code — all of
> which exist only on this branch. Keeping the analysis beside the instrumentation it
> measures is why it is not carried on `main` or `exp/marina234`. The full measurement is
> reproducible here; the figures and report additionally rebuild from `inputs/` alone.

## Headline numbers

- The **FFN path writes 71%** of the final CLS direction, leaving ~23% to modality tokens.
- Among modality tokens, **HSQC dominates at 44.7%** peaks-only and takes the largest ablation
  drop; **`mw` is near-inert at 0.9%**.
- Attribution and ablation **rank the modalities identically** (Spearman 1.0) but differ ~10× in
  scale, because 50% per-modality training dropout lets the model route around any single input.
- Removing `{hsqc, c_nmr, h_nmr}` jointly costs **0.2049** cosine against **0.0223** summed
  individually — **9.2× superadditive**.
- Suppressing cross-attention in the **first ten of 16 blocks costs 0.0072**; suppressing the
  **last one costs 0.0368**.

## Notes

- **Fixed 2026-08-04:** this directory used to archive none of its inputs and pointed at a live
  results path, so its figures were unreproducible anywhere the upstream results were absent. All
  six inputs (five measurement JSONs plus `bit_mi.npz`) are now in `inputs/`, matching the
  `layers8-decomposition/` convention. It was also moved onto this branch at the same time, so the
  analysis and the instrumentation it measures now version together.
- `make_figures.py` prints the LaTeX table bodies rather than requiring hand-transcription, so
  no number in the document is typed twice.
- There is no `pixi.toml`. The script needs only `numpy` and `matplotlib` and is routed through
  MARINA's env, which is far heavier than required.
- The 16-layer numbers here come from a different run and sample size than those in
  `layers8-decomposition/`, which lives on `exp/marina234`. Do not quote across the two:
  this reports 9.2x superadditivity and Spearman 1.0 at n=23,648; that one reports 11.09x
  and 0.7 from `final1/best.ckpt` at n=256.
