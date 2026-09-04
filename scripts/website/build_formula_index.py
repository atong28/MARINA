#!/usr/bin/env python3
"""
Precompute formula_index.json for a model directory: per rankingset row, the
molecule's element counts ({"C": 20, "H": 24, ...}) or null when its structure
could not be parsed. Aligned to metadata.json rows exactly like mw_index.json, so
the backend's atom-count filter (session.indices_matching_formula) can map kept
rows back to global indices.

Counts come from RDKit's CalcMolFormula (Hill notation, implicit H included);
charge signs are ignored. The backend builds nothing lazily for this filter, so
run this after assembling a model dir if you want the formula filter available.

    python scripts/website/build_formula_index.py /path/to/model_root
"""
from __future__ import annotations

import argparse
import json
import os
import re

_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")


def counts_from_formula(formula: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sym, num in _TOKEN.findall(formula):
        if sym:
            counts[sym] = counts.get(sym, 0) + (int(num) if num else 1)
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_root", help="Model directory holding metadata.json")
    ap.add_argument("--force", action="store_true", help="Rebuild even if the index exists")
    args = ap.parse_args()

    out_path = os.path.join(args.model_root, "formula_index.json")
    if os.path.exists(out_path) and not args.force:
        print(f"{out_path} already exists; pass --force to rebuild.")
        return 0

    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    meta_path = os.path.join(args.model_root, "metadata.json")
    print(f"Reading {meta_path}…")
    with open(meta_path) as fh:
        metadata = json.load(fh)

    n = len(metadata)
    print(f"Deriving element counts for {n} molecules…")
    out: list[dict | None] = [None] * n
    failed = 0
    for i in range(n):
        entry = metadata.get(str(i))
        smi = entry and (entry.get("canonical_2d_smiles") or entry.get("smiles"))
        counts = None
        if smi:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                counts = counts_from_formula(rdMolDescriptors.CalcMolFormula(mol))
        if counts is None:
            failed += 1
        out[i] = counts
        if (i + 1) % 25_000 == 0:
            print(f"  {i + 1}/{n}")

    with open(out_path, "w") as fh:
        json.dump(out, fh)
    print(f"Wrote {out_path} ({n - failed} parsed, {failed} unparseable)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
