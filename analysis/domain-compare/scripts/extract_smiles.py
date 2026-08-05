"""Extract canonical SMILES from each spectroscopic dataset under MARINA's canonicalization rule.

One output file per source: smiles/<source>.txt, one canonical SMILES per line, deduped.
Counts are written to smiles/<source>.json.

MARINA rule (see wiki/infrastructure/marina-datasets.md): double-pass canonicalization
with isomericSmiles=False. Applied to every source so the sets are comparable.
"""

import glob
import json
import os
import pickle
import sys
from multiprocessing import Pool

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("MARINA_DATA_ROOT", "/home/user/atong")
RAW = os.path.join(ROOT, "raw")
OUT = os.path.join(ROOT, "smiles")


def canonicalize(smiles):
    """MARINA's exact rule: canonicalize twice, drop stereochemistry."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)


def read_marina1():
    d = pickle.load(open(os.path.join(DATA_ROOT, "Datasets/MARINA1/index.pkl"), "rb"))
    return [v["smiles"] for v in d.values() if v.get("smiles")]


def read_mmsd():
    import pyarrow.parquet as pq

    out = []
    for p in sorted(glob.glob(os.path.join(RAW, "mmsd", "*.parquet"))):
        out.extend(pq.read_table(p, columns=["smiles"]).column("smiles").to_pylist())
    return out


def read_nmrexp():
    import pyarrow.parquet as pq

    p = os.path.join(RAW, "nmrexp", "NMRexp_10to24_1_1004.parquet")
    # 3.37M rows but only ~1.4M unique compounds -- dedupe raw before canonicalizing
    col = pq.read_table(p, columns=["SMILES"]).column("SMILES").to_pylist()
    return col


def read_nmrshiftdb2():
    p = os.path.join(
        RAW, "nmrshiftdb2", "data", "nmrshiftdb2_2024",
        "mol_nmrshift_nmrshiftdb2_2024_all.pkl",
    )
    return pickle.load(open(p, "rb"))["smiles"]


def read_massspecgym():
    import csv

    p = os.path.join(RAW, "massspecgym", "MassSpecGym.tsv")
    with open(p, newline="") as fh:
        return [row["smiles"] for row in csv.DictReader(fh, delimiter="\t")]


READERS = {
    "marina1": read_marina1,
    "mmsd": read_mmsd,
    "nmrexp": read_nmrexp,
    "nmrshiftdb2": read_nmrshiftdb2,
    "massspecgym": read_massspecgym,
}


def main():
    os.makedirs(OUT, exist_ok=True)
    sources = sys.argv[1:] or list(READERS)

    for name in sources:
        print(f"[{name}] reading...", flush=True)
        raw = READERS[name]()
        n_raw = len(raw)

        # dedupe on the raw string first -- cheap, and NMRexp repeats molecules across papers
        uniq_raw = {s for s in raw if s}
        print(f"[{name}] {n_raw} rows -> {len(uniq_raw)} unique raw strings, canonicalizing...", flush=True)

        with Pool(os.cpu_count()) as pool:
            canon = pool.map(canonicalize, list(uniq_raw), chunksize=2000)

        good = {c for c in canon if c}
        stats = {
            "source": name,
            "rows_read": n_raw,
            "unique_raw_strings": len(uniq_raw),
            "failed_canonicalization": len(uniq_raw) - sum(1 for c in canon if c),
            "unique_canonical": len(good),
        }

        with open(os.path.join(OUT, f"{name}.txt"), "w") as fh:
            fh.write("\n".join(sorted(good)) + "\n")
        json.dump(stats, open(os.path.join(OUT, f"{name}.json"), "w"), indent=2)
        print(f"[{name}] {json.dumps(stats)}", flush=True)


if __name__ == "__main__":
    main()
