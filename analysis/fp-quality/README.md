# fp-quality — intrinsic quality of the MARINA fingerprints

Two literature-grounded, dataset-agnostic experiments comparing fingerprints on **specificity**
and **similarity–structure agreement**. Every script takes its paths as arguments (retrieval set +
each FP's `rankingset.pt`), so the same code runs over any retrieval set / any FP lineup.

Write-up: [`wiki/experiments/fp-quality.md`](../../../wiki/experiments/fp-quality.md).

## Fingerprints

Any set of binary `rankingset.pt` matrices sharing one retrieval-set row order. The current run
evaluates the four **entropy-selected 16,384-bit** FPs built over the MARINA-DB retrieval set
(`RankingEntropyMultiplicityUncapped` (deployed), `RankingEntropy` (sherlock),
`RankingEntropyMultiplicity` (cap-5), `RankingEntropySubstructure`). The ECFP reference baselines
(ECFP4 r2/2048, folded-ECFP at matched radius) are added later as separate jobs — `exp1_add_fps.py`
folds them into the *same* MCES pairs without re-running RASCAL.

Similarity metric is fixed at plain **Tanimoto over nonzero column-sets** (all are binary
selected-bit vectors; no metric tuning — see [fingerprint MI redundancy §5](../../../wiki/experiments/fingerprint-mi-redundancy.md)).

## Experiments

- **Exp 2 — mass-stratified specificity** (`scripts/exp2_specificity.py`). *Precedent: Huber &
  Pollmann "Count your bits" Fig 2.* Groups molecules by identical bit-set, stratifies each
  duplicate group by its max pairwise monoisotopic mass difference; a group spanning >200 Da is a
  "bad" collision. **Metrics per FP:** collision %, top-1 ceiling %, largest tie, bad-collision %,
  median group span, mass-bin counts → **Table 4**.
- **Exp 1 — MCES similarity–structure agreement** (`scripts/exp1_mces.py` → `exp1_analyze.py`).
  *Precedent: Count your bits (RascalMCES reference); Riniker & Landrum.* Samples pairs from a
  mass-stratified pool, computes RascalMCES structural similarity (ground truth,
  `similarityThreshold=0.05, returnEmptyMCES=True`; minFragSize left at default — setting it to 3
  makes RASCAL time out on ~35% of ordinary pairs) and each FP's Tanimoto, reports
  **Spearman(FP-sim, MCES)** with paired bootstrap CIs and Δρ vs a reference → **similarity figure**.
  `returnEmptyMCES=True` is set from the start so screened-out dissimilar pairs get their real ~0
  similarity instead of being dropped (the old `exp1_mces_retry.py` patch is folded in; true
  timeouts are flagged in a `timedout` column).

Helpers: `exp1_add_fps.py` (add FP Tanimotos to an existing pairs parquet), `build_ecfp4.py` (build
an ECFP reference `rankingset.pt`), `radius_hist.py` (sherlock selected-bit radius histogram).

## Environment

Everything runs under **one env — the master pixi env at `~/Workspace`** (has torch + rdkit +
scipy + pandas + pyarrow together), so there is no torch-compute / scipy-analyze split:

```bash
cd ~/Workspace && pixi run python MARINA/analysis/fp-quality/scripts/<script>.py <args>
```

On **Nautilus**, the compute steps (Exp 2, Exp 1 MCES) run under the MARINA image (torch + rdkit);
`exp1_analyze.py` needs only scipy and is cheap, so run it locally under the master env on the
parquet the MCES job writes — or add scipy to a custom cluster env if you prefer to run it there.

## Compute

Nautilus CPU (namespace `guru-research`, PVC `atong-spectre`, image
`gitlab-registry.nrp-nautilus.io/a8tong/smart-moonshot/pixi-cuda:12.8.1`). Jobs clone the pushed
scripts from `main` (no base64 embedding). MCES is CPU-bound → the long job; Exp 2 is quick and
doubles as a PVC path probe.
