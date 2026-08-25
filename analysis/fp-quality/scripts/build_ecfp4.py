"""Build a hash-folded ECFP rankingset (Morgan, folded binary) over retrieval.pkl.

The off-the-shelf baseline (Rogers & Hahn). Output matches the CSR rankingset.pt contract the
other fingerprints use: torch sparse_csr with values 1/sqrt(nnz) per row (so cosine == the
deployed metric) and col indices = set bits. Row order = retrieval.pkl integer keys 0..N-1.

Usage: python build_ecfp4.py <retrieval.pkl> <out_dir> [n_bits=2048] [radius=2] [workers=16] [name=RankingEntropyECFP4]
Writes <out_dir>/<name>/rankingset.pt
  - ecfp4 baseline:            n_bits=2048  radius=2  name=RankingEntropyECFP4
  - width+radius-matched fair: n_bits=16384 radius=6  name=RankingEntropyECFPr6
"""
import os, pickle, sys, time
from multiprocessing import Pool

import numpy as np


def _onbits(smi):
    from rdkit import Chem
    m = Chem.MolFromSmiles(smi) if smi else None
    if m is None:
        return []
    bv = _GEN.GetFingerprint(m)
    return list(bv.GetOnBits())


def _init(radius, nbits):
    global _GEN
    from rdkit import RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    _GEN = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)


def main():
    retrieval = sys.argv[1]
    out_dir = sys.argv[2]
    nbits = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
    radius = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    workers = int(sys.argv[5]) if len(sys.argv) > 5 else 16
    name = sys.argv[6] if len(sys.argv) > 6 else "RankingEntropyECFP4"
    import torch

    with open(retrieval, "rb") as f:
        R = pickle.load(f)
    N = len(R)
    smiles = [(R[i]["smiles"] if isinstance(R[i], dict) else R[i]) for i in range(N)]
    print(f"building ECFP4 r={radius} nbits={nbits} for {N} molecules on {workers} workers", flush=True)

    crow = [0]
    cols = []
    vals = []
    t0 = time.time()
    with Pool(workers, initializer=_init, initargs=(radius, nbits)) as pool:
        for i, ob in enumerate(pool.imap(_onbits, smiles, chunksize=500)):
            ob = sorted(set(ob))
            cols.extend(ob)
            v = 1.0 / np.sqrt(len(ob)) if ob else 0.0
            vals.extend([v] * len(ob))
            crow.append(len(cols))
            if i % 50000 == 0:
                print(f"  {i}/{N}  {time.time()-t0:.0f}s", flush=True)

    csr = torch.sparse_csr_tensor(
        torch.tensor(crow, dtype=torch.int64),
        torch.tensor(cols, dtype=torch.int64),
        torch.tensor(vals, dtype=torch.float32),
        size=(N, nbits),
    )
    dst = os.path.join(out_dir, name)
    os.makedirs(dst, exist_ok=True)
    out = os.path.join(dst, "rankingset.pt")
    torch.save(csr, out)
    nnz = np.diff(crow)
    print(f"saved {out}  N={N} D={nbits}  on-bits mean={nnz.mean():.2f} "
          f"min={nnz.min()} max={nnz.max()}", flush=True)


if __name__ == "__main__":
    main()
