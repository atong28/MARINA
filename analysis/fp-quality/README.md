# fp-quality — intrinsic quality of the MARINA fingerprints

Evaluates five fingerprints — **sherlock** (`RankingEntropy`), **multiplicity uncapped**
(`RankingEntropyMultiplicityUncapped`), **multiplicity capped k=5** (`RankingEntropyMultiplicity`),
**ecfp4** (Morgan r=2), **substructure** (`RankingEntropySubstructure`) — all entropy-selected
to 16,384 bits over the 518,901-molecule MARINA1 retrieval set.

Write-up: [`wiki/experiments/fp-quality.md`](../../../wiki/experiments/fp-quality.md).
**Read-only on `Datasets/` and the PVC.**

## Experiments (grounded in literature precedent)

- **Exp 2 — mass-stratified specificity** (`scripts/exp2_specificity.py`). *Precedent: Huber &
  Pollmann "Count your bits" Fig 2.* Groups molecules by identical bit-set (column-set
  hashing, as `fp-collision-recheck`), then stratifies each duplicate group by its maximum
  pairwise monoisotopic mass difference. A duplicate spanning >200 Da is a "bad" collision.
  Full library; needs torch (read CSR) + rdkit (mass) → runs in MARINA's env / cluster image.
- **Exp 1 — MCES similarity–structure agreement** (`scripts/exp1_mces.py` +
  `exp1_analyze.py`). *Precedent: Count your bits (RascalMCES reference); Riniker & Landrum.*
  Samples pairs from a mass-stratified pool, computes RascalMCES structural similarity (ground
  truth) and each FP's Tanimoto, reports **Spearman(FP-sim, MCES)** with paired bootstrap CIs.
  Higher = the FP's similarity better tracks real structural similarity. MCES is CPU-bound →
  Nautilus CPU job. Analysis step is torch-free (this env).

## Fingerprint similarity metric

All five are binary selected-bit vectors, so **plain Tanimoto over nonzero column-sets** is the
consistent, untuned metric (no metric fixes — see `fp-redundancy` §5 for why post-hoc
reweighting was rejected).

## Compute

Nautilus CPU (namespace `guru-research`, PVC `atong-spectre`, image
`gitlab-registry.nrp-nautilus.io/a8tong/smart-moonshot/pixi-cuda:12.8.1`). `anthony3` is not
DNS-resolvable from the workstation, so everything routes to Nautilus; SDSC Expanse is the
fallback if Nautilus can't schedule. Job specs in `MARINA/nautilus/jobs/fp-quality-*.yaml`.

## Order

1. `fp-quality-specificity.yaml` — inventories the FP matrices on the PVC and runs Exp 2.
   Doubles as the PVC probe that confirms the exact `rankingset.pt` paths.
2. `fp-quality-mces.yaml` — Exp 1 MCES, submitted once (1) confirms the paths (and after any
   missing matrix — likely `ecfp4` — is built).
