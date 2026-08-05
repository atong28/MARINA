# fp-redundancy

Is the 16,384-bit RankingEntropy ("sherlock") fingerprint redundant, and if so can
retrieval recover anything by de-correlating the similarity metric?

Write-up: `wiki/experiments/fingerprint-mi-redundancy.md`.

MARINA1 is read **read-only** from `$MARINA_DATA_ROOT/Datasets/MARINA1/`
(`MARINA_DATA_ROOT` defaults to `/home/user/atong`). This env is torch-free; the two
scripts that need MARINA's model code run under MARINA's own env.

## Run order

```bash
# needs torch + MARINA code -> use MARINA's env, from the repo root
cd ../..
pixi run python3 analysis/fp-redundancy/scripts/00_export.py
pixi run python3 analysis/fp-redundancy/scripts/01_predict.py --limit 3000

# torch-free, this env
cd analysis/fp-redundancy
pixi run python scripts/02_bit_mi.py           # ~4 min, needs ~3 GB RAM
pixi run python scripts/03_reweight_sweep.py   # ~6 min
pixi run python scripts/04_analyze_ranks.py
pixi run python scripts/05_error_correlation.py
```

| script | question | output |
|---|---|---|
| `00_export.py` | — | `results/rankingset.npz` (CSR pattern + fragment SMILES) |
| `01_predict.py` | — | `results/preds.npz` (sigmoid logits for 3000 test molecules) |
| `02_bit_mi.py` | how correlated are the bits? | `results/bit_mi.npz`, `results/bit_mi_pairs.csv` |
| `03_reweight_sweep.py` | does a de-correlated metric rank better? | `results/reweight_sweep.json`, `results/reweight_ranks.npz` |
| `04_analyze_ranks.py` | are the differences significant? | stdout (McNemar + paired bootstrap) |
| `05_error_correlation.py` | is redundancy an error-correcting code? | stdout |

## Headline numbers

- "top-K by entropy" == "top-K by frequency"; median selected-bit presence 0.0009
- joint entropy **≤ 147.4 bits** of a nominal 16,384 (Chow-Liu MST bound; 422.6 summed marginal)
- **10,608 / 16,384** bits fully implied by another bit; 1,103 exact-duplicate slots
- cause is congeneric series (largest duplicate group: 28 bits over 246 gangliosides)
- de-correlating the metric **hurts**: top-1 0.7587 → 0.7513 (soft0.9, p=0.002) → 0.7230 (idf)
- redundancy is not error correction: within-group prediction-error ρ = **0.973**

## Notes

- `02_bit_mi.py` holds two 16384² float32 matrices (~2.1 GB). Fine in 15 GB.
- Co-occurrence is accumulated in float32; counts ≤ 518,901 < 2²⁴ so this is exact.
- `03` mirrors `RankingSet.batched_rank` exactly: rank = #{sim ≥ sim(query,truth)} − 1,
  with the weighting applied to rankingset rows, query, *and* truth.
- 141 rankingset rows have zero bits set; they are handled but never retrievable.
