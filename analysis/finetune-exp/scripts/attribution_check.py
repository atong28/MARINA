"""Is the rounded/+-1 HSQC population really JEOL, or is it NP-MRD?

marina1_hsqc_classify.py measures that 9,702 molecules carry literature-transcribed
HSQC (rounded shifts, +/-1 multiplicity). Calling that "JEOL" is an *attribution*,
resting only on SPECTRE naming the JEOL database as its experimental HSQC source.

NP-MRD is the obvious rival: experimental, NP-specific, and already the source of
MARINA1's benchmark.pkl. MARINA1's metadata.json carries per-molecule cross-DB IDs
(npmrd / lotus / coconut), so the two hypotheses make different predictions:

  JEOL   -> experimental class should NOT be especially NP-MRD-linked;
            its linkage rate should look like the simulated classes.
  NP-MRD -> experimental class should be strongly enriched for npmrd IDs.

Writes results/attribution_check.json.
"""
import json
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
RESULTS = HERE / "results"
DATA_ROOT = os.environ.get("MARINA_DATA_ROOT", "/home/user/atong")
MARINA1 = os.path.join(DATA_ROOT, "Datasets/MARINA1")

prov = pd.read_parquet(RESULTS / "marina1_hsqc_classified.parquet")
prov_by_idx = dict(zip(prov["idx"], prov["provenance"]))
print(f"classified molecules: {len(prov_by_idx):,}")

print("loading metadata.json (554 MB) ...")
meta = json.load(open(f"{MARINA1}/metadata.json"))
print(f"metadata entries: {len(meta):,}")

DBS = ("npmrd", "lotus", "coconut")
counts = defaultdict(lambda: defaultdict(int))
totals = defaultdict(int)

for idx_str, entry in meta.items():
    try:
        idx = int(idx_str)
    except (TypeError, ValueError):
        continue
    p = prov_by_idx.get(idx)
    if p is None:
        continue
    totals[p] += 1
    for db in DBS:
        v = entry.get(db)
        if isinstance(v, dict) and any(x is not None for x in v.values()):
            counts[p][db] += 1
        elif v not in (None, {}, ""):
            counts[p][db] += 1
    if not any(counts[p].get(db) for db in DBS):
        pass

summary = {}
for p, tot in totals.items():
    summary[p] = {
        "molecules_with_metadata": tot,
        **{f"{db}_linked": counts[p][db] for db in DBS},
        **{f"{db}_pct": round(100 * counts[p][db] / tot, 2) for db in DBS},
    }

print("\nDatabase linkage by HSQC provenance:")
hdr = f"{'provenance':22s} {'n':>8s} " + "".join(f"{db:>14s}" for db in DBS)
print(hdr)
for p, s in sorted(summary.items()):
    row = f"{p:22s} {s['molecules_with_metadata']:>8,} "
    row += "".join(f"{s[f'{db}_pct']:>13.2f}%" for db in DBS)
    print(row)

with open(RESULTS / "attribution_check.json", "w") as f:
    json.dump(summary, f, indent=2)
