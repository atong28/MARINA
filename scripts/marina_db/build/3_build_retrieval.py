"""3_build_retrieval.py -- build the retrieval set from structures only (no MS/MS).

Retrieval = smiles_dict (structure DBs) union journal union SPECTRE-corpus structures
union Mnova structures union the SPECTRE retrieval bank (candidate pool), all fixed-point
canonicalized (2D). Folding in SPECTRE's bank makes MARINA-DB retrieval a strict superset
of it. Deliberately independent of the MS/MS predictions -- positive is being re-predicted,
negative is now folded in too (mass_spec_neg) -- so the retrieval set and the rankingsets built on it
(stage 4) are NOT blocked on spectral data. This is the structure track; the spectral track
(stage 6+) can lag.

MS/MS predictions are ICEBERG runs over molecules already sourced from the structure
DBs / SPECTRE / Mnova, so they are *expected* to introduce no new structures. That is
not assumed silently: 12_verify.py asserts every index molecule is present in retrieval,
which will confirm it once positive MS/MS is folded into the index.

Writes config.RETRIEVAL_PKL: {idx -> {'smiles': canonical_2d}} in sorted-SMILES order.
"""
import glob
import json
import os
import pickle
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))   # scripts/marina_db -> config
sys.path.insert(0, str(_HERE.parents[3]))   # repo root -> src
from config import DATA_RAW, DATA_CLEANED, SMILES_DICT, RETRIEVAL_PKL, BENCH_JOURNAL, SPECTRE_RETRIEVAL
from src.modules.data.smiles import canonicalize_smiles


def structure_dict_smiles() -> set:
    """Canonical SMILES from the merged structure DBs (already canonical from stage 2)."""
    with open(SMILES_DICT) as f:
        return set(json.load(f).keys())


def journal_smiles() -> set:
    """Canonical (2D) SMILES of the frozen Journal benchmark -- gold rows must be
    retrievable so rank@k is well-defined for all 467 (subsumes the old augment stage)."""
    journal = pickle.load(open(BENCH_JOURNAL, "rb"))
    return {c for c in (canonicalize_smiles(e["smiles"])
                        for e in journal.values() if e.get("smiles")) if c}


def spectre_smiles() -> set:
    """Canonical SMILES of every SPECTRE-corpus molecule (data/raw/index.pkl)."""
    index = pickle.load(open(DATA_RAW / "index.pkl", "rb"))
    return {c for c in (canonicalize_smiles(e["smiles"])
                        for e in index.values() if e.get("smiles")) if c}


def spectre_retrieval_smiles() -> set:
    """Canonical SMILES of SPECTRE's retrieval bank (data/raw/spectre_retrieval.pkl,
    extracted from SPECTRE inference metadata, 526,316 SMILES). Folded in so MARINA-DB
    retrieval is a strict superset of SPECTRE's candidate pool -- recovering the NP-like
    structure-only candidates SPECTRE carries that the fresh NP dumps alone do not."""
    bank = pickle.load(open(SPECTRE_RETRIEVAL, "rb"))
    return {c for c in (canonicalize_smiles(s) for s in bank if s) if c}


def mnova_smiles() -> set:
    """Canonical SMILES of Mnova-predicted molecules with a usable HSQC prediction.

    Same SUCCESS + non-empty filter as the spectral assembly (6_build_index.py), so the
    retrieval structures match the molecules that would enter the mapping from Mnova."""
    out = set()
    for file in sorted(glob.glob(str(DATA_RAW / "mnova_predictions" / "*.jsonl"))):
        with open(file) as f:
            for line in f:
                d = json.loads(line)
                if d["predictions"]["hsqc"]["status"] != "SUCCESS":
                    continue
                h = d["predictions"]["hsqc"]["H"]
                c = d["predictions"]["hsqc"]["C"]
                if not h or not c:
                    continue
                cs = canonicalize_smiles(d["smiles"])
                if cs:
                    out.add(cs)
    return out


def main():
    os.makedirs(DATA_CLEANED, exist_ok=True)
    parts = {
        "smiles_dict": structure_dict_smiles(),
        "journal": journal_smiles(),
        "spectre": spectre_smiles(),
        "mnova": mnova_smiles(),
        "spectre_retrieval": spectre_retrieval_smiles(),
    }
    all_smiles = sorted(s for s in set().union(*parts.values()) if s)
    retrieval = {i: {"smiles": s} for i, s in enumerate(all_smiles)}

    for k, v in parts.items():
        print(f"  {k}: {len(v)}")
    print(f"retrieval union: {len(retrieval)} (MS/MS excluded by design)")
    with open(RETRIEVAL_PKL, "wb") as f:
        pickle.dump(retrieval, f)
    print(f"wrote {RETRIEVAL_PKL}")


if __name__ == "__main__":
    main()
