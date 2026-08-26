"""Build a folded-binary MAP4 rankingset over retrieval.pkl.

MAP4 (Capecchi et al.) is natively a MinHash fingerprint; Count Your Bits benchmarks the FOLDED
BINARY variant. We enumerate MAP4's atom-pair shingle set (the `map4` package's shingle code,
which is pure RDKit — no tmap) and fold each shingle to a bit via a deterministic hash
(bit = blake2b(shingle) % nbits), taking the set-union. That yields a binary column-set on which
Tanimoto is the valid metric — NOT the MinHash signature, whose Jaccard != column-set Tanimoto
(the trap the fp-quality hand-off calls out).

blake2b (not builtin hash()) so the fold is identical across Pool workers regardless of
PYTHONHASHSEED. Output matches the CSR rankingset.pt contract build_ecfp4.py writes:
torch.sparse_csr, vals 1/sqrt(nnz) per row, col = on-bits, row order = retrieval.pkl keys 0..N-1.

Usage: python build_map4.py --retrieval <pkl> --out_dir <dir> [--nbits 2048] [--radius 2] \
           [--workers 16] [--name MAP4_2048]

Deps: torch + rdkit + numpy + the `map4` package (only its shingle enumeration is used; no tmap).
"""
import argparse, hashlib, os, pickle, time
from multiprocessing import Pool

import numpy as np

_CALC = None
_NBITS = None


def _fold_bit(shingle, nbits):
    return int.from_bytes(hashlib.blake2b(shingle.encode(), digest_size=8).digest(), "big") % nbits


def _onbits(smi):
    from rdkit import Chem
    m = Chem.MolFromSmiles(smi) if smi else None
    if m is None:
        return []
    try:
        shingles = _CALC._calculate(m)   # Set[str] of MAP4 atom-pair shingles (pure rdkit)
    except Exception:
        return []
    return sorted({_fold_bit(s, _NBITS) for s in shingles})


def _init(radius, nbits):
    global _CALC, _NBITS
    from rdkit import RDLogger
    from map4.map4 import MAP4Calculator
    RDLogger.DisableLog("rdApp.*")
    # __new__ bypasses MAP4Calculator.__init__'s MHFPEncoder (which needs mhfp); we only use the
    # shingle enumeration, which reads self.radius / self.include_duplicated_shingles.
    c = MAP4Calculator.__new__(MAP4Calculator)
    c.radius = radius
    c.include_duplicated_shingles = False
    _CALC, _NBITS = c, nbits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--nbits", type=int, default=2048)
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--name", default="MAP4_2048")
    a = ap.parse_args()
    import torch

    with open(a.retrieval, "rb") as f:
        R = pickle.load(f)
    N = len(R)
    smiles = [(R[i]["smiles"] if isinstance(R[i], dict) else R[i]) for i in range(N)]
    print(f"building folded MAP4 r={a.radius} nbits={a.nbits} for {N} molecules on {a.workers} workers", flush=True)

    crow = [0]
    cols = []
    vals = []
    t0 = time.time()
    with Pool(a.workers, initializer=_init, initargs=(a.radius, a.nbits)) as pool:
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
        size=(N, a.nbits),
    )
    dst = os.path.join(a.out_dir, a.name)
    os.makedirs(dst, exist_ok=True)
    out = os.path.join(dst, "rankingset.pt")
    torch.save(csr, out)
    nnz = np.diff(crow)
    print(f"saved {out}  N={N} D={a.nbits}  on-bits mean={nnz.mean():.2f} "
          f"min={nnz.min()} max={nnz.max()}", flush=True)


if __name__ == "__main__":
    main()
