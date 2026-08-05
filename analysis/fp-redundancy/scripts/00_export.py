#!/usr/bin/env python3
"""Export the MARINA1 rankingset + bit metadata into torch-free .npz form.

Run once with MARINA's env (it is the only thing here that needs torch), from the
MARINA repo root:

    pixi run python3 analysis/fp-redundancy/scripts/00_export.py

The rankingset CSR stores L2-normalized values; only its sparsity pattern
matters, so we keep just indptr/indices.
"""
import os
import pickle
import sys

import numpy as np
import torch

DATA_ROOT = os.environ.get("MARINA_DATA_ROOT", "/home/user/atong")
DATASET = os.environ.get("DATASET_ROOT", os.path.join(DATA_ROOT, "Datasets/MARINA1"))
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "results", "rankingset.npz")

rs = torch.load(os.path.join(DATASET, "RankingEntropy", "rankingset.pt"), weights_only=True)
N, B = rs.shape
indptr = rs.crow_indices().numpy().astype(np.int64)
indices = rs.col_indices().numpy().astype(np.int32)
print(f"rankingset: {N} x {B}, {len(indices)} nonzeros", flush=True)

with open(os.path.join(DATASET, "RankingEntropy", "bitinfo_to_idx.pkl"), "rb") as f:
    bitinfo_to_col = pickle.load(f)
# BitInfo = (bit_id, atom_symbol, frag_smiles, radius); radius-0 bits carry an
# empty frag_smiles and are identified by atom_symbol alone.
frags = np.empty(B, dtype=object)
atoms = np.empty(B, dtype=object)
radii = np.zeros(B, dtype=np.int8)
for bi, colx in bitinfo_to_col.items():
    atoms[colx] = bi[1]
    frags[colx] = bi[2]
    radii[colx] = bi[3]

os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez_compressed(OUT, indptr=indptr, indices=indices, shape=np.array([N, B]),
                    frags=frags, atoms=atoms, radii=radii)
print(f"wrote {OUT}", flush=True)
