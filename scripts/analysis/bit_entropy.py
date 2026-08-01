"""Compute per-bit entropy for the 16384-bit sherlock (RankingEntropy) fingerprint.

Reproduces the selection criterion in EntropyFPLoader.setup: binary entropy of
each circular-substructure feature's presence rate over the retrieval set.
Cross-checks against the column sums of the prebuilt rankingset CSR.
"""
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.modules.data.fp_utils import compute_entropy, load_smiles_index

ROOT = os.environ["DATASET_ROOT"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bit_entropy.npz")
MAX_RADIUS = 6

n_retrieval = len(load_smiles_index(os.path.join(ROOT, "retrieval.pkl")))
print(f"retrieval size: {n_retrieval}", flush=True)

with open(os.path.join(ROOT, "RankingEntropy", "bitinfo_to_idx.pkl"), "rb") as f:
    bitinfo_to_col = pickle.load(f)
print(f"selected bits: {len(bitinfo_to_col)}", flush=True)

with open(os.path.join(ROOT, f"count_hashes_under_radius_{MAX_RADIUS}.pkl"), "rb") as f:
    counts_map = pickle.load(f)
print(f"total features in counts file: {len(counts_map)}", flush=True)

# Candidate pool: every feature with radius <= MAX_RADIUS (the `filtered` step in setup)
cand_bitinfos, cand_counts = zip(*[(bi, c) for bi, c in counts_map.items() if bi[3] <= MAX_RADIUS])
cand_counts = np.asarray(cand_counts, dtype=np.int64)
cand_entropy = compute_entropy(cand_counts, total_dataset_size=n_retrieval)
print(f"candidate features (radius<={MAX_RADIUS}): {len(cand_bitinfos)}", flush=True)

# Selected bits, ordered by fingerprint column index
n_bits = len(bitinfo_to_col)
sel_counts = np.zeros(n_bits, dtype=np.int64)
sel_radius = np.zeros(n_bits, dtype=np.int8)
missing = 0
for bi, col in bitinfo_to_col.items():
    c = counts_map.get(bi)
    if c is None:
        missing += 1
        continue
    sel_counts[col] = c
    sel_radius[col] = bi[3]
print(f"selected bits missing from counts file: {missing}", flush=True)

sel_entropy = compute_entropy(sel_counts, total_dataset_size=n_retrieval)

# Cross-check: per-column occupancy of the rankingset CSR over the same retrieval
# molecules. CSR values are L2-normalized per row, so count column indices rather
# than summing values.
csr = torch.load(os.path.join(ROOT, "RankingEntropy", "rankingset.pt"), weights_only=True)
print(f"rankingset shape: {tuple(csr.shape)}", flush=True)
csr_counts = np.bincount(csr.col_indices().numpy(), minlength=n_bits).astype(np.int64)
n_disagree = int((csr_counts != sel_counts).sum())
print(f"bits where rankingset colsum != counts-file count: {n_disagree}", flush=True)
if n_disagree:
    d = np.abs(csr_counts - sel_counts)
    print(f"  max abs diff: {d.max()}, mean abs diff: {d.mean():.4f}", flush=True)

np.savez_compressed(
    OUT,
    n_retrieval=n_retrieval,
    sel_counts=sel_counts,
    sel_entropy=sel_entropy,
    sel_radius=sel_radius,
    csr_counts=csr_counts,
    cand_counts=cand_counts,
    cand_entropy=cand_entropy,
)
print(f"wrote {OUT}", flush=True)

p = sel_counts / n_retrieval
print("\n--- selected-bit entropy ---", flush=True)
for q in (0, 1, 5, 25, 50, 75, 95, 99, 100):
    print(f"  p{q:<3d} {np.percentile(sel_entropy, q):.6f}   (freq {np.percentile(p, q):.6f})", flush=True)
print(f"  mean {sel_entropy.mean():.6f}  std {sel_entropy.std():.6f}", flush=True)
print(f"  total bit-entropy (sum): {sel_entropy.sum():.1f} bits", flush=True)
