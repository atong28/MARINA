#!/usr/bin/env python3
"""Copy Benchmark/papers/<NPID>.txt to Benchmark/filtered/<NPID>/info.txt for every NPID dir."""
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent
PAPERS = BASE / "papers"
FILTERED = BASE / "filtered"

copied, missing = 0, []
for npid_dir in sorted(FILTERED.iterdir()):
    if not npid_dir.is_dir():
        continue
    npid = npid_dir.name
    src = PAPERS / f"{npid}.txt"
    if not src.exists():
        missing.append(npid)
        continue
    shutil.copyfile(src, npid_dir / "info.txt")
    copied += 1

print(f"Copied {copied} info.txt files.")
if missing:
    print(f"Missing source .txt for: {missing}")
