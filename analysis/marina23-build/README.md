# marina23-build

Builds the MARINA2, MARINA3 and MARINA4 datasets by transforming MARINA1: substituting
experimental spectra in (2), keeping only the experimental slice (3), and collapsing
¹³C/¹H to per-peak (4).

Write-up: `wiki/infrastructure/marina2-dataset.md`.

**Environment: the `finetune-exp` pixi env.** That is what every run of these scripts has
used. `pixi.toml` here mirrors it and is not installed — either `pixi install` it, or keep
using finetune-exp's as shown below.

**This is the one analysis directory that is not read-only.** MARINA1 is read read-only,
but `03_build.py` and `09_build_marina4.py` *write sibling datasets* — see the warning
under Run order.

`Datasets/` below always means `$MARINA_DATA_ROOT/Datasets/`, which lives **outside** this
repo; `MARINA_DATA_ROOT` defaults to `/home/user/atong`. Paths inside the repo — this
directory and the two sibling analysis dirs it reads — are resolved from `__file__`.

## Prerequisites

~5.4 GB of `raw/` data owned by two other directories. Nothing here fetches it.

| File | Owner |
|---|---|
| `gnps/ALL_GNPS.mgf` (4.3 G) | `../finetune-exp/raw/` |
| `massbank/MassBank_NISTformat.msp` (131 M) | `../finetune-exp/raw/` |
| `nmrexp/NMRexp_10to24_1_1004.parquet` (631 M) | `../domain-compare/raw/` |
| `nmrshiftdb2/data/nmrshiftdb2_2024/mol_nmrshift_nmrshiftdb2_2024_all.pkl` (130 M) | `../domain-compare/raw/` |
| `massspecgym/MassSpecGym.tsv` (250 M) | `../domain-compare/raw/` |

`../finetune-exp/README.md` has the re-fetch commands for GNPS and MassBank. The
three domain-compare files have no recorded re-fetch commands; their provenance is in
`wiki/experiments/spectroscopic-dataset-domain.md`.

`03_build.py` also needs `../finetune-exp/results/marina1_hsqc_classified.parquet`
— the JEOL/Mnova/ACD HSQC provenance table that decides which HSQC rows MARINA3 keeps.

## Run order

```bash
# runs under finetune-exp's env; scripts resolve their own paths, so cwd does not matter
cd /home/user/atong/MARINA/analysis/finetune-exp
B=/home/user/atong/MARINA/analysis/marina23-build/scripts

pixi run python $B/01_extract_msms.py     # streams the 4.3 G GNPS MGF
pixi run python $B/02_extract_nmr.py
pixi run python $B/03_build.py            # WRITES Datasets/MARINA2 and MARINA3
pixi run python $B/04_verify.py           # PASS/FAIL to stdout, exit 1 on any failure

# analysis of what was built -- any order after 03
pixi run python $B/05_nmrexp_agreement.py
pixi run python $B/06_agreement_dedup.py
pixi run python $B/07_convention_check.py
pixi run python $B/08_dropout_analysis.py

# independent of 01-08; needs only MARINA1
pixi run python $B/09_build_marina4.py    # WRITES Datasets/MARINA4
```

> **Destructive.** `03_build.py` does `shutil.rmtree` on `Datasets/MARINA2/` and
> `Datasets/MARINA3/` before rebuilding; `09_build_marina4.py` does the same to
> `Datasets/MARINA4/`. Re-running throws away those sibling datasets. The `MARINA*.zip`
> snapshots beside them are not touched, and the copies on the `smart-datasets` PVC are
> not either — so a botched rebuild is recoverable from the zips, not from disk.

| script | question | output |
|---|---|---|
| `01_extract_msms.py` | — | `results/msms_experimental.pkl` (best real positive-mode spectrum per molecule) |
| `02_extract_nmr.py` | — | `results/nmr_experimental.pkl`, `results/nmr_experimental_summary.json` |
| `03_build.py` | — | `Datasets/MARINA2/`, `Datasets/MARINA3/`, `results/build_report.json` |
| `04_verify.py` | did the build preserve idx/splits/retrieval and substitute what it claimed? | stdout PASS/FAIL |
| `05_nmrexp_agreement.py` | does NMRexp agree with nmrshiftdb2 where both cover a molecule? | `results/nmrexp_agreement.json` |
| `06_agreement_dedup.py` | how much of that disagreement was the peak convention? | `results/nmrexp_agreement_dedup.json` |
| `07_convention_check.py` | is MARINA1's NMR per-atom or per-peak? | `results/convention_check.json` |
| `08_dropout_analysis.py` | what modality presence does dropout actually produce? | `results/dropout_analysis.json` |
| `09_build_marina4.py` | — | `Datasets/MARINA4/`, `results/marina4_build.json` |

## Headline numbers

MARINA1 is 487,028 molecules; MARINA2 keeps every one of them.

- MARINA2 substitutions: ¹³C **16,708 replaced + 9 added**, ¹H **14,500 + 3**,
  MS/MS **24,215 + 1,680**. The 1,680 are molecules with no MS/MS at all in MARINA1,
  so MS/MS coverage rises 475,871 → **477,551**.
- MARINA3 = **48,108** molecules (train 43,267 / val 2,448 / test 2,393); HSQC 9,702,
  ¹³C 16,717, ¹H 14,503, MS/MS 25,895.
- MARINA4 collapse: ¹³C train entries 11,151,219 → 10,224,618 (**91.7% retained**),
  ¹H 11,013,844 → 9,744,706 (**88.5%**).
- NMRexp vs nmrshiftdb2 ¹³C: naive gross disagreement **20.3%** → **4.8%** once both sides
  are collapsed to unique peaks; median deviation 0.595 → **0.295 ppm**. ~76% of the
  apparent disagreement was the convention artifact.
- **Dropout does not hit its target.** `compute_drop_percentage` ignores the `always_keep`
  exemption, so MARINA1/MARINA2/MARINA4 train at **62.6% modality presence, not 50%**
  (+12.6 pp). The drop_p that would actually give 50% is **0.668**, not 0.500. This
  applies to every MARINA run to date.
- On MARINA3 the target is unreachable: HSQC/¹³C/¹H cap out at 0.202 / 0.348 / 0.302, and
  `mass_spec`'s always-keep floor (0.467) sits above HSQC's ceiling (0.202), so no dropout
  schedule can balance them.

## Notes

- **`06` supersedes `05` for ¹³C, not for ¹H.** `05` compared nmrshiftdb2's per-*atom*
  lists against NMRexp's per-*peak* lists, which force-matches surplus symmetry-equivalent
  atoms onto unrelated peaks; its 20.3% ¹³C figure is that artifact. Both outputs are kept
  deliberately so the size of the artifact stays visible. For ¹H the integration-expanded
  comparison in `05` (median 0.040 ppm, 5.6% gross) is the better-matched one — collapsing
  per-atom proton shifts does not reproduce multiplet grouping.
- `results/dropout_analysis.json` covers MARINA1/2/3 only: MARINA4 did not exist when `08`
  ran. MARINA4's `index.pkl` is byte-identical to MARINA1's (`09` copies it verbatim, and
  collapsing shifts changes no `has_*` flag), so its numbers are MARINA1's exactly.
- `04_verify.py` exits 1 on any failure, so it is safe to chain — but its MARINA2
  "untouched rows byte-identical" check is a 2,000-row spot check per split, not exhaustive.
- Negative-mode MS/MS is parsed and counted in `01` but deliberately not emitted; it needs
  a modality slot the schema does not have.
- Substitution perturbs peak counts, and not symmetrically: ¹³C lists get shorter for 29.9%
  of substituted val molecules, ¹H lists longer for 64.3%. A MARINA1-vs-MARINA2 delta is
  therefore not purely "simulated vs experimental shifts".
- These scripts hold no absolute paths. Their own directory comes from
  `Path(__file__).resolve().parent.parent`; `domain-compare/raw/` and `finetune-exp/` come
  from `ROOT.parent`, so the whole `analysis/` tree moves as a unit; and `Datasets/` comes
  from `$MARINA_DATA_ROOT` (default `/home/user/atong`), the only thing to re-point if the
  datasets move.
