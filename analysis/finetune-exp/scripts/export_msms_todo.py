"""Audit MARINA1 MS/MS coverage and export SMILES lists for spectrum simulation.

MARINA1's MS/MS is ICEBERG-simulated, positive mode, fixed 20 eV -- and it does not
cover the whole dataset. This writes two lists:

  msms_todo_positive.smi  unique SMILES lacking any MS/MS, i.e. the positive-mode gap
  msms_all_negative.smi   every unique SMILES in the dataset, for a negative-mode pass

Both are validity-filtered with RDKit, since index.pkl carries a few degenerate
entries that would fail or waste a simulation slot. Rejects are written alongside so
the exclusion is auditable rather than silent.
"""
import os
import pickle
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

DATA_ROOT = os.environ.get("MARINA_DATA_ROOT", "/home/user/atong")
D = Path(DATA_ROOT) / "Datasets/MARINA1"
OUT = Path(__file__).resolve().parent.parent / "results"
MIN_HEAVY = 3  # below this a fragmentation spectrum is not meaningful


def classify(smiles):
    """-> ('ok'|'unparseable'|'too_small', n_heavy)"""
    if not isinstance(smiles, str) or not smiles.strip():
        return "unparseable", 0
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "unparseable", 0
    n = mol.GetNumHeavyAtoms()
    return ("ok" if n >= MIN_HEAVY else "too_small"), n


df = pd.DataFrame.from_dict(pickle.load(open(D / "index.pkl", "rb")), orient="index")
missing = df[~df.has_mass_spec]

print(f"index.pkl entries              : {len(df):,}")
print(f"  with MS/MS                   : {int(df.has_mass_spec.sum()):,}")
print(f"  MISSING MS/MS                : {len(missing):,}  ({len(missing)/len(df):.2%})")
print(f"  missing by split             : {missing.split.value_counts().to_dict()}")
print()
print(f"unique SMILES, whole dataset   : {df.smiles.nunique():,}")
print(f"unique SMILES, missing MS/MS   : {missing.smiles.nunique():,}")
print()

# One RDKit pass over the whole vocabulary, reused for both exports.
verdicts = {s: classify(s) for s in df.smiles.unique()}
counts = pd.Series([v[0] for v in verdicts.values()]).value_counts()
print("RDKit audit over unique SMILES :")
for k, v in counts.items():
    print(f"  {k:14s}: {v:,}")

rejects = {s: v for s, (v, _) in verdicts.items() if v != "ok"}
if rejects:
    print("\n  rejected SMILES:")
    for s, why in sorted(rejects.items(), key=lambda kv: kv[1]):
        print(f"    [{why}] {s!r}  (heavy={verdicts[s][1]})")

# Preserve dataset order (by idx) rather than set order, so the lists are stable.
seen_pos, pos = set(), []
for s in missing.sort_index().smiles:
    if s not in seen_pos and verdicts[s][0] == "ok":
        seen_pos.add(s)
        pos.append(s)

seen_all, allsmi = set(), []
for s in df.sort_index().smiles:
    if s not in seen_all and verdicts[s][0] == "ok":
        seen_all.add(s)
        allsmi.append(s)

OUT.mkdir(exist_ok=True)
(OUT / "msms_todo_positive.smi").write_text("\n".join(pos) + "\n")
(OUT / "msms_all_negative.smi").write_text("\n".join(allsmi) + "\n")
if rejects:
    (OUT / "msms_rejected.smi").write_text(
        "\n".join(f"{why}\t{s!r}" for s, why in sorted(rejects.items())) + "\n"
    )

print()
print(f"wrote {OUT/'msms_todo_positive.smi'}  ({len(pos):,} SMILES)")
print(f"wrote {OUT/'msms_all_negative.smi'}   ({len(allsmi):,} SMILES)")
if rejects:
    print(f"wrote {OUT/'msms_rejected.smi'}        ({len(rejects)} excluded)")
