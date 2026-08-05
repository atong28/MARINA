"""Compare the molecular domain of public spectroscopic datasets against MARINA1.

Two passes:
  1. Overlap, on the FULL sets -- exact canonical SMILES and Bemis-Murcko scaffold
     intersection with MARINA1. Cheap, and the counts are themselves a result.
  2. Descriptors + Sherlock-FP out-of-vocabulary fraction, on a fixed subsample per
     source (large sets would otherwise dominate wall time on 4 cores).

Writes results/overlap.json and results/descriptors.parquet.

The sFP OOV fraction is the actionable number: the share of a molecule's radius<=6
circular substructures that have no bit in MARINA's 16,384-entry vocabulary, i.e. the
share of that molecule MARINA's prediction target literally cannot express.
Bit-key construction mirrors MARINA/src/modules/data/fp_utils.py:get_bitinfos.
"""

import json
import os
import pickle
import random
import sys
from multiprocessing import Pool

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("MARINA_DATA_ROOT", "/home/user/atong")
SMILES_DIR = os.path.join(ROOT, "smiles")
RESULTS = os.path.join(ROOT, "results")
VOCAB = os.path.join(DATA_ROOT, "Datasets/MARINA1/RankingEntropy/bitinfo_to_idx.pkl")

SUBSAMPLE = 30_000
SEED = 0
SFP_RADIUS = 6

# The filter Alberts et al. applied and SpecX inherited verbatim.
ALBERTS_ELEMENTS = {"C", "H", "O", "N", "S", "P", "Si", "B", "F", "Cl", "Br", "I"}

_VOCAB_KEYS = None
_NP_MODEL = None


def _init_worker():
    global _VOCAB_KEYS, _NP_MODEL
    _VOCAB_KEYS = set(pickle.load(open(VOCAB, "rb")).keys())
    sys.path.insert(0, os.path.join(ROOT, "vendor"))
    import npscorer

    _NP_MODEL = npscorer.readNPModel(os.path.join(ROOT, "vendor", "publicnp.model.gz"))


def sfp_oov_fraction(mol):
    """Fraction of radius<=6 circular substructures absent from MARINA's sFP vocabulary."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=SFP_RADIUS)
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateBitInfoMap()
    gen.GetSparseFingerprint(mol, additionalOutput=ao)

    keys = set()
    for bit_id, atom_envs in ao.GetBitInfoMap().items():
        for atom_idx, curr_radius in atom_envs:
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, curr_radius, atom_idx)
            submol = Chem.PathToSubmol(mol, env)
            frag = Chem.MolToSmiles(submol, canonical=True)
            sym = mol.GetAtomWithIdx(atom_idx).GetSymbol()
            keys.add((bit_id, sym, frag, curr_radius))

    if not keys:
        return None, 0
    missing = sum(1 for k in keys if k not in _VOCAB_KEYS)
    return missing / len(keys), len(keys)


def describe(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    ring_sizes = [len(r) for r in mol.GetRingInfo().AtomRings()]
    elements = {a.GetSymbol() for a in mol.GetAtoms()}
    oov, n_bits = sfp_oov_fraction(mol)

    import npscorer

    return {
        "smiles": smiles,
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "mw": Descriptors.ExactMolWt(mol),
        "fsp3": rdMolDescriptors.CalcFractionCSP3(mol),
        "n_rings": rdMolDescriptors.CalcNumRings(mol),
        "max_ring_size": max(ring_sizes) if ring_sizes else 0,
        "n_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "n_stereocenters": len(
            Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)
        ),
        "n_rotatable": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "n_O": sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "O"),
        "n_N": sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "N"),
        "np_score": npscorer.scoreMol(mol, _NP_MODEL),
        "sfp_oov_frac": oov,
        "n_sfp_substructs": n_bits,
        # Alberts' filter is inclusive: 5 <= n <= 35. Was strict until 2026-08-04,
        # which under-counted MMSD by 2.3 points.
        "in_alberts_window": 5 <= mol.GetNumHeavyAtoms() <= 35,
        "alberts_elements_ok": elements <= ALBERTS_ELEMENTS,
    }


def murcko(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaf, canonical=True) if scaf.GetNumAtoms() else None
    except Exception:
        return None


def load(source):
    with open(os.path.join(SMILES_DIR, f"{source}.txt")) as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def main():
    os.makedirs(RESULTS, exist_ok=True)
    sources = sys.argv[1:] or ["marina1", "mmsd", "nmrexp", "nmrshiftdb2", "massspecgym"]
    sets = {s: load(s) for s in sources}
    for s, v in sets.items():
        print(f"{s}: {len(v)} molecules", flush=True)

    # ---- pass 1: overlap against MARINA1, full sets ----
    marina = set(sets["marina1"])
    print("\ncomputing Murcko scaffolds (full sets)...", flush=True)
    scaffolds = {}
    with Pool(os.cpu_count()) as pool:
        for s, v in sets.items():
            sc = pool.map(murcko, v, chunksize=2000)
            scaffolds[s] = {x for x in sc if x}
            print(f"  {s}: {len(scaffolds[s])} unique scaffolds", flush=True)

    overlap = {}
    for s, v in sets.items():
        if s == "marina1":
            continue
        vs = set(v)
        overlap[s] = {
            "n_molecules": len(vs),
            "exact_shared_with_marina1": len(vs & marina),
            "exact_shared_pct_of_source": round(100 * len(vs & marina) / len(vs), 3),
            "n_scaffolds": len(scaffolds[s]),
            "scaffolds_shared_with_marina1": len(scaffolds[s] & scaffolds["marina1"]),
            "scaffolds_shared_pct_of_source": round(
                100 * len(scaffolds[s] & scaffolds["marina1"]) / len(scaffolds[s]), 3
            ),
        }
    overlap["marina1"] = {
        "n_molecules": len(marina),
        "n_scaffolds": len(scaffolds["marina1"]),
    }
    json.dump(overlap, open(os.path.join(RESULTS, "overlap.json"), "w"), indent=2)
    print("\n" + json.dumps(overlap, indent=2), flush=True)

    # ---- pass 2: descriptors + sFP OOV, subsampled ----
    rng = random.Random(SEED)
    frames = []
    for s, v in sets.items():
        sample = v if len(v) <= SUBSAMPLE else rng.sample(v, SUBSAMPLE)
        print(f"\ndescribing {s} (n={len(sample)})...", flush=True)
        with Pool(os.cpu_count(), initializer=_init_worker) as pool:
            rows = pool.map(describe, sample, chunksize=200)
        df = pd.DataFrame([r for r in rows if r])
        df["source"] = s
        frames.append(df)
        print(f"  {s}: {len(df)} described", flush=True)

    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(os.path.join(RESULTS, "descriptors.parquet"), index=False)
    print(f"\nwrote {len(out)} rows to results/descriptors.parquet", flush=True)


if __name__ == "__main__":
    main()
