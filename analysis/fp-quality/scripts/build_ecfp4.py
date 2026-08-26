"""Build a hash-folded RDKit fingerprint rankingset (folded binary) over retrieval.pkl.

The off-the-shelf baselines: Morgan/ECFP (Rogers & Hahn) and Atom Pair. Output matches the CSR
rankingset.pt contract the other fingerprints use: torch sparse_csr with values 1/sqrt(nnz) per
row (so cosine == the deployed metric) and col indices = set bits. Row order = retrieval.pkl
integer keys 0..N-1.

Usage: python build_ecfp4.py --retrieval <retrieval.pkl> --out_dir <dir> \
           [--fp-type morgan|atompair] [--nbits 2048] [--radius 2] [--workers 16] [--name ...]
Writes <out_dir>/<name>/rankingset.pt
  - ecfp4 baseline:            --fp-type morgan   --nbits 2048  --radius 2  --name ECFP4_2048
  - width+radius-matched fair: --fp-type morgan   --nbits 16384 --radius 10 --name ECFP4_16384
  - atom pair (radius ignored): --fp-type atompair --nbits 2048             --name AtomPair_2048

Deps: torch + rdkit + numpy. Run under the ~/Workspace master pixi env or the Nautilus image.
"""
import argparse, os, pickle, time
from multiprocessing import Pool

import numpy as np


def _onbits(smi):
    from rdkit import Chem
    m = Chem.MolFromSmiles(smi) if smi else None
    if m is None:
        return []
    bv = _GEN.GetFingerprint(m)
    return list(bv.GetOnBits())


def _init(fp_type, radius, nbits):
    global _GEN
    from rdkit import RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    if fp_type == "morgan":
        _GEN = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    elif fp_type == "atompair":
        _GEN = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=nbits)
    else:
        raise ValueError(f"unknown fp-type {fp_type}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fp-type", choices=["morgan", "atompair"], default="morgan")
    ap.add_argument("--nbits", type=int, default=2048)
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--name", default="RankingEntropyECFP4")
    a = ap.parse_args()
    retrieval, out_dir, fp_type, nbits, radius, workers, name = (
        a.retrieval, a.out_dir, a.fp_type, a.nbits, a.radius, a.workers, a.name)
    import torch

    with open(retrieval, "rb") as f:
        R = pickle.load(f)
    N = len(R)
    smiles = [(R[i]["smiles"] if isinstance(R[i], dict) else R[i]) for i in range(N)]
    print(f"building {fp_type} r={radius} nbits={nbits} for {N} molecules on {workers} workers", flush=True)

    crow = [0]
    cols = []
    vals = []
    t0 = time.time()
    with Pool(workers, initializer=_init, initargs=(fp_type, radius, nbits)) as pool:
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
