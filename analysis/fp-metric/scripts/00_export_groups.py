#!/usr/bin/env python3
"""Export the substructure grouping of the deployed Morgan 16k vocabulary, plus the
per-molecule set of substructures actually present in each cached test molecule.

This is the one script here that needs MARINA's env (RDKit); run it from the MARINA
repo root:

    pixi run python3 analysis/fp-metric/scripts/00_export_groups.py

Two things get written to results/groups.npz:

  1. `gid_<variant>`: (16384,) group id per fingerprint column. Two bits share a group
     iff they describe the same substructure. Radius-0 Morgan features all serialize to
     frag_smiles == '' (MolFragmentToSmiles on an empty bond environment), so grouping
     naively on that field would merge 425 unrelated atom-type bits into one bucket.
     Two variants bracket the choice:
       strict : radius-0 features are singleton groups     (conservative -- understates
                within-group error, never inflates it)
       atom   : radius-0 features grouped by atom symbol   (aggressive)

  2. `present_<variant>_{idx,ptr}`: per test molecule, the group ids whose substructure
     occurs *anywhere* in the molecule, over the full unrestricted feature enumeration
     rather than only the selected 16384. This is what separates "model predicted
     chemistry the molecule really has, but the target codes it in a slot the entropy
     selection dropped" from "model predicted chemistry that isn't there".
"""
import argparse
import os
import pickle
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MARINA_ROOT = os.path.dirname(os.path.dirname(HERE))     # HERE = analysis/<dir>, up 2
DATA_ROOT = os.environ.get("MARINA_DATA_ROOT", "/home/user/atong")
os.environ.setdefault("DATASET_ROOT", os.path.join(DATA_ROOT, "Datasets/MARINA1"))
sys.path.insert(0, MARINA_ROOT)

import numpy as np

from src.modules.data.fp_utils import MORGAN, extract_features

REDUNDANCY = os.path.join(os.path.dirname(HERE), "fp-redundancy")
OUT = os.path.join(HERE, "results", "groups.npz")

ap = argparse.ArgumentParser()
ap.add_argument("--preds", default=os.path.join(REDUNDANCY, "results", "preds.npz"),
                help="cached predictions from fp-redundancy/scripts/01_predict.py")
ap.add_argument("--dataset_root", default=os.environ["DATASET_ROOT"])
ap.add_argument("--fp_type", default="RankingEntropy")
ap.add_argument("--radius", type=int, default=0,
                help="0 = infer from the vocabulary's own max radius")
cli = ap.parse_args()

VARIANTS = ("strict", "atom")


def group_key(bitinfo, variant):
    """Substructure identity of a Morgan feature key (bit_id, atom_symbol, frag, radius)."""
    bit_id, sym, frag, rad = bitinfo
    if rad == 0:
        return ("atom0", sym) if variant == "atom" else ("bit0", bit_id, sym)
    return ("frag", frag)


# ---------------------------------------------------------------- vocabulary
map_path = os.path.join(cli.dataset_root, cli.fp_type, "bitinfo_to_idx.pkl")
with open(map_path, "rb") as f:
    bitinfo_to_col = pickle.load(f)
B = len(bitinfo_to_col)
col_to_bitinfo = {v: k for k, v in bitinfo_to_col.items()}
assert len(col_to_bitinfo) == B, "column ids are not unique"
assert sorted(col_to_bitinfo) == list(range(B)), "columns are not a dense 0..B-1 range"

radii = np.array([col_to_bitinfo[j][3] for j in range(B)], dtype=np.int32)
frags = [col_to_bitinfo[j][2] for j in range(B)]
# The '' fragment must be exactly the radius-0 features, or the grouping below is wrong.
empty = np.array([f == "" for f in frags])
assert np.array_equal(empty, radii == 0), \
    "empty frag_smiles does not coincide with radius 0; group_key needs revisiting"
max_radius = int(radii.max()) if cli.radius == 0 else cli.radius
print(f"vocabulary: {B} bits, radius <= {max_radius}, {int((radii == 0).sum())} at radius 0")

gid, key_to_gid, vocab_keys = {}, {}, {}
for variant in VARIANTS:
    keys = [group_key(col_to_bitinfo[j], variant) for j in range(B)]
    uniq = {k: i for i, k in enumerate(sorted(set(keys), key=repr))}
    gid[variant] = np.array([uniq[k] for k in keys], dtype=np.int32)
    key_to_gid[variant] = uniq
    vocab_keys[variant] = uniq.keys()
    sizes = np.bincount(gid[variant])
    biggest = max(uniq, key=lambda k: sizes[uniq[k]])
    print(f"  {variant:6s}: {len(uniq):5d} groups, largest = {repr(biggest)} "
          f"({sizes.max()} bits), {int((sizes == 1).sum())} singletons")

# ---------------------------------------------------------------- molecules
mol_idx = np.load(cli.preds, allow_pickle=True)["mol_idx"]
with open(os.path.join(cli.dataset_root, "index.pkl"), "rb") as f:
    index = pickle.load(f)
smiles = [index[int(i)]["smiles"] for i in mol_idx]
print(f"{len(smiles)} test molecules from {cli.preds}")

present = {v: ([], [0]) for v in VARIANTS}
t0 = time.time()
for n, smi in enumerate(smiles):
    # Unrestricted enumeration: every circular feature of the molecule, not just the
    # ones the entropy selection kept.
    feats = extract_features(smi, radius=max_radius, kind=MORGAN)
    for variant in VARIANTS:
        ids = {key_to_gid[variant][k] for k in
               (group_key(bi, variant) for bi in feats)
               if k in key_to_gid[variant]}
        idx_list, ptr = present[variant]
        idx_list.append(np.fromiter(sorted(ids), dtype=np.int32, count=len(ids)))
        ptr.append(ptr[-1] + len(ids))
    if (n + 1) % 500 == 0:
        rate = (n + 1) / (time.time() - t0)
        print(f"  {n+1}/{len(smiles)}  ({rate:.0f} mol/s)", flush=True)

payload = {"mol_idx": mol_idx, "radii": radii, "max_radius": max_radius, "n_bits": B}
for variant in VARIANTS:
    idx_list, ptr = present[variant]
    payload[f"gid_{variant}"] = gid[variant]
    payload[f"n_groups_{variant}"] = len(key_to_gid[variant])
    payload[f"present_{variant}_idx"] = np.concatenate(idx_list)
    payload[f"present_{variant}_ptr"] = np.array(ptr, dtype=np.int64)
    print(f"{variant:6s}: mean vocab groups present per molecule = {ptr[-1]/len(smiles):.1f}")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez_compressed(OUT, **payload)
print(f"wrote {OUT}  ({time.time()-t0:.1f}s)")
