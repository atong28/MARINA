"""Build a folded-binary MAP4 rankingset over retrieval.pkl.

MAP4 (Capecchi et al.) is natively a MinHash fingerprint; Count Your Bits benchmarks the FOLDED
BINARY variant. We enumerate MAP4's atom-pair shingle set and fold each shingle to a bit via a
deterministic hash (bit = blake2b(shingle) % nbits), taking the set-union. That yields a binary
column-set on which Tanimoto is the valid metric — NOT the MinHash signature, whose Jaccard !=
column-set Tanimoto (the trap the fp-quality hand-off calls out).

The shingle enumeration (_get_atom_envs / _all_pairs) is inlined from the `map4` package (Capecchi
et al.) — pure RDKit — rather than imported, because that package's top-level imports pull sklearn/
matplotlib/mhfp (for its t-SNE viz + MinHash paths) which we do not need and cannot rely on in the
build env. blake2b (not builtin hash()) so the fold is identical across Pool workers regardless of
PYTHONHASHSEED. Output matches the CSR rankingset.pt contract build_ecfp4.py writes.

Usage: python build_map4.py --retrieval <pkl> --out_dir <dir> [--nbits 2048] [--radius 2] \
           [--workers 16] [--name MAP4_2048]

Deps: torch + rdkit + numpy. No external fingerprint package.
"""
import argparse, hashlib, itertools, os, pickle, time
from multiprocessing import Pool

import numpy as np

_RADIUS = None
_NBITS = None


def _find_env(mol, atom_idx, radius):
    """Canonical SMILES of the circular environment of `radius` around `atom_idx` ("" if empty)."""
    from rdkit.Chem import FindAtomEnvironmentOfRadiusN, PathToSubmol, MolToSmiles
    env = FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
    amap = {}
    submol = PathToSubmol(mol, env, atomMap=amap)
    if atom_idx in amap:
        return MolToSmiles(submol, rootedAtAtom=amap[atom_idx], canonical=True, isomericSmiles=False)
    return ""


def _shingles(mol):
    """MAP4 atom-pair shingle set: '{smaller_env}|{topodist}|{larger_env}' over all atom pairs and
    radii 1..R (inlined from map4._get_atom_envs / _all_pairs)."""
    from rdkit.Chem.rdmolops import GetDistanceMatrix
    n = mol.GetNumAtoms()
    envs = {a: [_find_env(mol, a, r) for r in range(1, _RADIUS + 1)] for a in range(n)}
    dm = GetDistanceMatrix(mol)
    out = set()
    for i, j in itertools.combinations(range(n), 2):
        dist = str(int(dm[i][j]))
        for r in range(_RADIUS):
            ea, eb = envs[i][r], envs[j][r]
            smaller, larger = (eb, ea) if len(ea) > len(eb) else (ea, eb)
            out.add(f"{smaller}|{dist}|{larger}")
    return out


def _fold_bit(shingle, nbits):
    return int.from_bytes(hashlib.blake2b(shingle.encode(), digest_size=8).digest(), "big") % nbits


def _onbits(smi):
    from rdkit import Chem
    m = Chem.MolFromSmiles(smi) if smi else None
    if m is None:
        return []
    try:
        return sorted({_fold_bit(s, _NBITS) for s in _shingles(m)})
    except Exception:
        return []


def _init(radius, nbits):
    global _RADIUS, _NBITS
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    _RADIUS, _NBITS = radius, nbits


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
        for i, ob in enumerate(pool.imap(_onbits, smiles, chunksize=200)):
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
