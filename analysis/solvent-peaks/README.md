# Solvent residual peaks in MARINA training data (2026-09-30)

Question: do MARINA's training spectra contain solvent residual peaks (so should users include them at inference)?

    cd ~/Workspace/MARINA
    DATASET_ROOT=/tmp pixi run python analysis/solvent-peaks/train_solvent_peaks.py ../Datasets/MARINA-DB MARINA-DB
    DATASET_ROOT=/tmp pixi run python analysis/solvent-peaks/hsqc_solvent.py ../Datasets/MARINA-DB MARINA-DB
    (same with ../Datasets/MARINA-DB-OPEN; it splits by source via nmr_sources.parquet)

- `train_solvent_peaks.py`: share of molecules with a 1D peak at a solvent line vs at nearby control positions. Not
  decisive on its own: Mnova (which cannot contain solvent) shows the same line/control ratios (77.16: 1.51; 7.26: 1.64;
  2.50: 0.65), i.e. the natural shape of the shift distribution.
- `hsqc_solvent.py` (decisive): HSQC rows at residual-solvent cross-peaks. CHCl3 (77.16, 7.26): 0 of 460k Mnova, 2 of 96k
  ACD, 1 of 29k CH-NMR-NP molecules. The DMSO / CD3OD / pyridine positions are hit at the same per-molecule rate in Mnova
  as in the other sources, i.e. real CH groups, not solvent.
- CH-NMR-NP row structure: 727k 13C values, all but 18 attached to a numbered atom (the 18 are ordinary unlabelled atoms);
  no solvent rows (the only "solvent" mentions: a real C-21 proton "overlapped with solvent", and an "NH in CDCl3" note).

Conclusion: no source (Mnova, ACD/Labs, SPECTRE-predicted 1D, CH-NMR-NP/"JEOL") contains solvent peaks; all are per-atom.
