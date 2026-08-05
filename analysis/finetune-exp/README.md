# finetune-exp

Isolated analysis of experimental spectra sources as a MARINA finetuning set.
Independent of MARINA code — own pixi env, reads MARINA1 read-only from
`$MARINA_DATA_ROOT/Datasets/MARINA1/` (`MARINA_DATA_ROOT` defaults to `/home/user/atong`).

Write-up: [`wiki/experiments/experimental-finetuning-dataset.md`](../../../wiki/experiments/experimental-finetuning-dataset.md)

## Run order

Scripts resolve `raw/` and `results/` from their own location, so the working
directory does not matter.

```bash
pixi run python scripts/inspect_sources.py        # schema probes, prints only

# NMR / JEOL provenance
pixi run python scripts/discriminate_hsqc.py      # which signature means what
pixi run python scripts/marina1_hsqc_classify.py  # authoritative provenance table

# MS/MS libraries
pixi run python scripts/msms_overlap.py           # MassSpecGym (already local)
pixi run python scripts/massbank_overlap.py       # MassBank  -> negative mode
pixi run python scripts/gnps_overlap.py           # GNPS      -> largest
pixi run python scripts/union_coverage.py         # de-duplicated union
```

`hsqc_provenance.py` and `validate_provenance.py` are kept for the record:
the first joins MARINA1 back to SPECTRE (superseded by `marina1_hsqc_classify.py`,
which avoids a mislabelling bug when one SMILES has several upstream sources),
the second is the peak-count test that returned null.

## Headline numbers

MARINA1 HSQC, 486,208 molecules — **98.0% simulated**:

| Provenance | Molecules | Share |
|---|---|---|
| Mnova (simulated) | 380,552 | 78.27% |
| ACD/Labs (simulated) | 95,954 | 19.74% |
| JEOL (experimental) | 9,702 | 2.00% |

Experimental MS/MS available for MARINA1 molecules (its own MS/MS is 100% ICEBERG-simulated):

| Source | Matched molecules | % of MARINA1 | Negative mode |
|---|---|---|---|
| GNPS | 22,841 | 4.69% | 10,076 |
| MassSpecGym | 16,565 | 3.40% | 0 (positive only) |
| MassBank | 7,007 | 1.44% | 3,691 |
| **Union** | **25,455** | **5.23%** | **10,770** |

## Re-fetching raw data

`raw/` is not in git. To rebuild, from this directory:

```bash
# GNPS (4.5 GB)
curl -L -o raw/gnps/ALL_GNPS.mgf https://external.gnps2.org/gnpslibrary/ALL_GNPS.mgf

# MassBank 2026.03 (137 MB)
curl -L -o raw/massbank/MassBank_NISTformat.msp \
  https://github.com/MassBank/MassBank-data/releases/download/2026.03/MassBank_NISTformat.msp

# SPECTRE index + HSQC stores, from the local MoonshotDatasetv3 zip
unzip -o "${MARINA_DATA_ROOT:-/home/user/atong}/Datasets/MoonshotDatasetv3.zip" \
  "index.pkl" "_lmdb/*/HSQC_NMR.lmdb/*" -d raw/spectre/
```

MassSpecGym is read in place from `../domain-compare/raw/massspecgym/`.
