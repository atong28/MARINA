# marina-layers8-decomposition

Does the 16-layer depth finding survive at 8 layers? The 16-layer model does almost nothing
in its first ten cross-attention blocks, which raised the question of whether the depth is
unused — so the same two measurements were repeated on two 8-layer checkpoints.

Write-up: the wiki page `experiments/marina-layers-sweep.md` (separate repo). Formal report: `layers8-decomposition.pdf`
(source `layers8-decomposition.tex`, compile with `tectonic layers8-decomposition.tex`).

The six JSONs here are archived cluster outputs. Everything in this directory can be
regenerated from them without a GPU.

## Run order

The measurement step needs a GPU and the MARINA repo; the figure steps do not.

```bash
# 1. GPU, on the cluster. early_attn_ablation.py here is a COPY of the MARINA script;
#    it imports src.modules.marina, so it only runs from the MARINA repo root.
#    The actual runs were driven by run_layers8_perlayer.sh
#    and run_layers8_suppression.sh
cd /code   # MARINA repo root on the pod
DATASET_ROOT=/path/to/dataset pixi run python scripts/analysis/early_attn_ablation.py \
    --ckpt <ckpt> --params <params.json> --limit 2048 --seed 0
# then copy the resulting JSONs back into this directory

# 2. CPU, local. cwd does not matter - both scripts resolve paths from __file__.
cd <marina-repo>
pixi run python3 analysis/layers8-decomposition/make_figures.py
pixi run python3 analysis/layers8-decomposition/compare.py

# 3. Render the report
cd analysis/layers8-decomposition && tectonic layers8-decomposition.tex
```

| script | question | output |
|---|---|---|
| `early_attn_ablation.py` | what does each block contribute? | the six JSONs (produced on the cluster) |
| `compare.py` | per-block survival and concentration | stdout tables only |
| `make_figures.py` | — | `fig-depth-ablation-8L`, `fig-depth-alignment`, `fig-layer-profile-8L` (`.pdf` + `.png`) |

## Inputs

| file | arm | $n$ | contents |
|---|---|---|---|
| `layers8_s{0,1}_n2048.json` | 8-layer, seeds 0/1 | 2048 | attribution, ablation, agreement, redundancy, per-layer |
| `layers16_final1_n256.json` | 16-layer `final1` | 256 | same keys |
| `layers8_s{0,1}_suppression_n2048.json` | 8-layer, seeds 0/1 | 2048 | `suppress_first_k`, `suppress_last_k` |
| `layers16_suppression_n512.json` | 16-layer `final1` | 512 | same keys |

## Headline numbers

- **The live suffix is 5–6 blocks at either depth.** Taking "inert" to mean a first-$k$
  suppression costing under 0.01 cosine: 16L has an inert prefix of 10 (live suffix 6 of 16),
  8L-s0 has 2 (live suffix 6 of 8), 8L-s1 has 3 (live suffix 5 of 8).
- **The final blocks are quantitatively depth-invariant.** Suppressing the last $k$ blocks costs
  the same at 8 and 16 layers, to within **0.012** for every $k \le 4$ (k=1: 0.0368 / 0.0392 /
  0.0382). The curves only separate at $k \ge 6$, where the 8-layer model has run out of blocks.
- **The modality budget is unchanged.** Peaks-only shares at 8L vs 16L: hsqc 0.432/0.427 vs 0.459,
  c_nmr 0.264/0.260 vs 0.244, mass_spec 0.181/0.187 vs 0.163, h_nmr 0.113/0.115 vs 0.125,
  mw 0.010/0.011 vs 0.009. FFN writes 70–72% of the CLS direction at both depths.
- Superadditivity of the joint NMR ablation persists: 9.37× / 8.92× at 8L vs 11.09× at 16L.

Together these are the mechanistic reason `layers8` matches `layers12` on test `rank@1`.

## Notes

- `compare.py` used to read the three per-layer JSONs by bare filename, so it failed unless
  cwd happened to be this directory. **Fixed 2026-08-04** — it now resolves them from
  `__file__`, like `make_figures.py` always did.
- **Sample sizes are not matched.** The 8-layer arms use n=2048, the 16-layer reference n=256
  (attribution) and n=512 (suppression). The 16-layer column is the noisier one, so the
  agreement in the last-$k$ table is if anything understated. A matched-n rerun is cheap and
  would make the comparison clean.
- **Do not quote the 16-layer numbers here against the ones in
  the wiki page `experiments/marina-modality-attribution.md`.** That page reports 9.2× superadditivity and Spearman 1.0
  from a different run at a different n; this directory measures `final1/best.ckpt` at n=256 and
  gets 11.09× and 0.7. Both are correct for what they measure.
- **Two seeds, not three.** `layers8-s2` finished after these measurements were taken.
- Suppression measures what a block *currently* contributes in a trained network, not what a
  network trained without it would do. The retrieval sweep answers the latter; this explains it.
- The checkpoint paths inside the JSONs (`/root/gurusmart/...`) exist only on the cluster.
