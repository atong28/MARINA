"""UMAP-embed all five sets in two fingerprint spaces and save 2D coordinates.

Two representations, chosen as a contrast pair:
  ECFP4  -- vocabulary-free Morgan r=2, 2048 bit. Comparable to the prior literature.
  sFP    -- MARINA's own 16,384-bit Sherlock vocabulary. Vocabulary-BASED: substructures
            absent from the NP-mined vocabulary have no bit and are silently dropped, so
            non-NP molecules appear artificially close to NPs. The map therefore cannot
            be read as evidence of NP-likeness -- divergence from the ECFP4 map is the signal.

UMAP's `jaccard` metric on sparse binary input is exactly Tanimoto, so no O(n^2) distance
matrix is needed. Sets are balanced-subsampled because the largest set would otherwise
dominate the manifold.
"""

import json
import os
import pickle
import random
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy.sparse import csr_matrix, vstack

RDLogger.DisableLog("rdApp.*")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("MARINA_DATA_ROOT", "/home/user/atong")
SMILES_DIR = os.path.join(ROOT, "smiles")
RESULTS = os.path.join(ROOT, "results")
VOCAB = os.path.join(DATA_ROOT, "Datasets/MARINA1/RankingEntropy/bitinfo_to_idx.pkl")

PER_SET = 15_000
SEED = 0
SOURCES = ["marina1", "mmsd", "nmrexp", "nmrshiftdb2", "massspecgym"]


def ecfp4_matrix(smiles_list):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    rows, cols = [], []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for b in gen.GetFingerprint(mol).GetOnBits():
            rows.append(i)
            cols.append(b)
    data = np.ones(len(rows), dtype=np.uint8)
    return csr_matrix((data, (rows, cols)), shape=(len(smiles_list), 2048))


_VOCAB = None


def _init_sfp():
    global _VOCAB
    _VOCAB = pickle.load(open(VOCAB, "rb"))


def _sfp_cols(smi):
    """Sherlock-FP column indices present in one molecule (empty if unparseable)."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return []
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=6)
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateBitInfoMap()
    gen.GetSparseFingerprint(mol, additionalOutput=ao)
    seen = set()
    for bit_id, envs in ao.GetBitInfoMap().items():
        for atom_idx, r in envs:
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, r, atom_idx)
            frag = Chem.MolToSmiles(Chem.PathToSubmol(mol, env), canonical=True)
            key = (bit_id, mol.GetAtomWithIdx(atom_idx).GetSymbol(), frag, r)
            col = _VOCAB.get(key)
            if col is not None:
                seen.add(col)
    return sorted(seen)


def sfp_matrix(smiles_list):
    from multiprocessing import Pool

    with Pool(os.cpu_count(), initializer=_init_sfp) as pool:
        per_mol = pool.map(_sfp_cols, smiles_list, chunksize=200)
    rows, cols = [], []
    for i, cs in enumerate(per_mol):
        rows.extend([i] * len(cs))
        cols.extend(cs)
    data = np.ones(len(rows), dtype=np.uint8)
    return csr_matrix((data, (rows, cols)), shape=(len(smiles_list), 16384))


def main():
    import umap

    os.makedirs(RESULTS, exist_ok=True)
    rng = random.Random(SEED)

    labels, smiles = [], []
    for s in SOURCES:
        with open(os.path.join(SMILES_DIR, f"{s}.txt")) as fh:
            v = [ln.strip() for ln in fh if ln.strip()]
        sample = v if len(v) <= PER_SET else rng.sample(v, PER_SET)
        smiles.extend(sample)
        labels.extend([s] * len(sample))
        print(f"{s}: {len(sample)} sampled", flush=True)

    for rep, fn in [("ecfp4", ecfp4_matrix), ("sfp", sfp_matrix)]:
        print(f"\nbuilding {rep} matrix...", flush=True)
        X = fn(smiles)
        density = X.nnz / X.shape[0]
        print(f"  shape {X.shape}, mean on-bits/molecule {density:.1f}", flush=True)
        empty = int((np.asarray(X.sum(axis=1)).ravel() == 0).sum())
        print(f"  molecules with all-zero vector: {empty}", flush=True)

        print(f"  running UMAP (jaccard) on {rep}...", flush=True)
        reducer = umap.UMAP(
            n_neighbors=25, min_dist=0.1, metric="jaccard", random_state=SEED, verbose=True
        )
        emb = reducer.fit_transform(X)

        df = pd.DataFrame({"source": labels, "smiles": smiles, "x": emb[:, 0], "y": emb[:, 1]})
        df.to_parquet(os.path.join(RESULTS, f"umap_{rep}.parquet"), index=False)
        print(f"  wrote results/umap_{rep}.parquet", flush=True)


if __name__ == "__main__":
    main()
