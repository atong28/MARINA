#!/usr/bin/env python3
"""Split the exact-duplicate pairs into their two possible causes.

  nesting     : one fragment is a substructure of the other (radius r vs r+k on the
                same atom environment) -- an artifact of the ECFP feature family
  congeneric  : neither contains the other, so the columns coincide only because the
                dataset holds a congeneric series sharing both fragments

Needs RDKit, so run under MARINA's env, from the MARINA repo root:

    pixi run python3 analysis/fp-redundancy/scripts/08_dup_mechanism.py
"""
import os

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
z = np.load(os.path.join(HERE, "results", "rankingset.npz"), allow_pickle=True)
N, B = (int(v) for v in z["shape"])
frags, radii = z["frags"], z["radii"]
counts = np.bincount(z["indices"], minlength=B).astype(np.float64)
root = np.load(os.path.join(HERE, "results", "bit_mi.npz"))["dup_root"]

pi, pj = [], []
for r in np.unique(root):
    m = np.nonzero(root == r)[0]
    if len(m) > 1:
        a, b = np.triu_indices(len(m), 1)
        pi.append(m[a]); pj.append(m[b])
pi, pj = np.concatenate(pi), np.concatenate(pj)

# ECFP environment fragments are open-valence and often not valid molecules;
# SMARTS parsing tolerates that and is the right semantics for containment.
mols = {}
for i in set(pi.tolist()) | set(pj.tolist()):
    s = frags[i]
    mols[i] = Chem.MolFromSmarts(s) if s else None

nesting = np.zeros(len(pi), dtype=bool)
unparsed = 0
for k, (a, b) in enumerate(zip(pi, pj)):
    ma, mb = mols[int(a)], mols[int(b)]
    if ma is None or mb is None:
        unparsed += 1
        continue
    try:
        nesting[k] = mb.HasSubstructMatch(ma) or ma.HasSubstructMatch(mb)
    except Exception:
        unparsed += 1

p = counts[pi] / N
H = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))   # identical support => MI = H_i = H_j
ok = np.ones(len(pi), dtype=bool)

print(f"exact-duplicate pairs: {len(pi)}  (unparsed fragments: {unparsed})")
print(f"  nesting (one contains the other): {nesting.sum()} "
      f"({100*nesting.mean():.1f}% of pairs, {100*H[nesting].sum()/H.sum():.1f}% of duplicated entropy)")
print(f"  congeneric (neither contains)   : {(~nesting).sum()} "
      f"({100*(~nesting).mean():.1f}% of pairs, {100*H[~nesting].sum()/H.sum():.1f}% of duplicated entropy)")
top = np.argsort(-H)[:50]
print(f"  among top-50 by duplicated entropy: {nesting[top].sum()} nesting, "
      f"{50-nesting[top].sum()} congeneric")
print(f"  median molecules: nesting {np.median(counts[pi[nesting]]):.0f}, "
      f"congeneric {np.median(counts[pi[~nesting]]):.0f}")

# how well does the radius-difference proxy track the real answer?
diff = radii[pi] != radii[pj]
tp = (diff & nesting).sum()
print(f"\n  radius-differs proxy vs true nesting: "
      f"precision {tp/max(diff.sum(),1):.3f}, recall {tp/max(nesting.sum(),1):.3f}")
