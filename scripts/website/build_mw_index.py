#!/usr/bin/env python3
"""
Precompute the monoisotopic mass index a model directory needs for MW filtering.

The backend derives these masses on first use and caches them, but that pass is
minutes long for a full database and lands on whichever request happens to be
first. Running this after downloading a model keeps that cost offline.

Writes mw_index.json: one entry per rankingset row, null where the structure
could not be parsed. Masses are monoisotopic (Descriptors.ExactMolWt), matching
the "Exact mass" shown on result cards.

    python scripts/website/build_mw_index.py data/marina_best
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_root", help="Model directory holding metadata.json")
    ap.add_argument("--force", action="store_true", help="Rebuild even if the index exists")
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from backend.app.session import MW_INDEX_FILENAME, _mw_from_entry

    out_path = os.path.join(args.model_root, MW_INDEX_FILENAME)
    if os.path.exists(out_path) and not args.force:
        print(f"{out_path} already exists; pass --force to rebuild.")
        return 0

    meta_path = os.path.join(args.model_root, "metadata.json")
    print(f"Reading {meta_path}…")
    with open(meta_path) as fh:
        metadata = json.load(fh)

    n = len(metadata)
    print(f"Deriving {n} monoisotopic masses…")
    masses = []
    failed = 0
    for i in range(n):
        mw = _mw_from_entry(metadata.get(str(i)))
        if mw is None:
            failed += 1
        masses.append(mw)
        if (i + 1) % 25_000 == 0:
            print(f"  {i + 1}/{n}")

    with open(out_path, "w") as fh:
        json.dump(masses, fh)

    print(f"Wrote {out_path} ({n - failed} masses, {failed} unparseable)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
