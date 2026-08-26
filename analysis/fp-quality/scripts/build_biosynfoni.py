"""Build a binarized Biosynfoni rankingset over retrieval.pkl.

Biosynfoni (Meijer et al.) is a fixed dictionary of ~39 biosynthesis-informed substructure keys —
no folding, so a single width D=39. We compute the per-molecule count vector and BINARIZE it
(nonzero -> on-bit), matching the binary column-set contract the other fingerprints use. Its high
collision rate is expected; it is the intended NP-domain counter-example in the table.

Output matches the CSR rankingset.pt contract build_ecfp4.py writes: torch.sparse_csr, vals
1/sqrt(nnz) per row, col = on-bits, row order = retrieval.pkl keys 0..N-1.

Usage: python build_biosynfoni.py --retrieval <pkl> --out_dir <dir> [--workers 16] [--name Biosynfoni]

Deps: torch + rdkit + numpy + the `biosynfoni` package.
"""
import argparse, os, pickle, time
from multiprocessing import Pool

import numpy as np

_D = None


def _counts(mol):
    """39-length count vector from whichever biosynfoni API this version exposes."""
    from biosynfoni import Biosynfoni
    bs = Biosynfoni(mol)
    for attr in ("get_biosynfoni", "fingerprint", "counted_fp"):
        v = getattr(bs, attr, None)
        vec = v() if callable(v) else v
        if vec is not None:
            return list(vec)
    raise AttributeError("biosynfoni: no count-vector accessor found")


def _onbits(smi):
    from rdkit import Chem
    m = Chem.MolFromSmiles(smi) if smi else None
    if m is None:
        return []
    try:
        vec = _counts(m)
    except Exception:
        return []
    return [k for k, c in enumerate(vec) if c]


def _init():
    global _D
    from rdkit import RDLogger, Chem
    RDLogger.DisableLog("rdApp.*")
    _D = len(_counts(Chem.MolFromSmiles("CCO")))  # fix D from the fixed dictionary length


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--name", default="Biosynfoni")
    a = ap.parse_args()
    import torch

    with open(a.retrieval, "rb") as f:
        R = pickle.load(f)
    N = len(R)
    smiles = [(R[i]["smiles"] if isinstance(R[i], dict) else R[i]) for i in range(N)]

    from rdkit import Chem
    D = len(_counts(Chem.MolFromSmiles("CCO")))
    print(f"building Biosynfoni (fixed D={D}) for {N} molecules on {a.workers} workers", flush=True)

    crow = [0]
    cols = []
    vals = []
    t0 = time.time()
    with Pool(a.workers, initializer=_init) as pool:
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
        size=(N, D),
    )
    dst = os.path.join(a.out_dir, a.name)
    os.makedirs(dst, exist_ok=True)
    out = os.path.join(dst, "rankingset.pt")
    torch.save(csr, out)
    nnz = np.diff(crow)
    print(f"saved {out}  N={N} D={D}  on-bits mean={nnz.mean():.2f} "
          f"min={nnz.min()} max={nnz.max()}", flush=True)


if __name__ == "__main__":
    main()
