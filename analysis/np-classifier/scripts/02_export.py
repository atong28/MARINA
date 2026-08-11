"""
Turn the raw fetch log into the two artifacts that get consumed downstream.

    cd /home/user/atong/MARINA
    pixi run python3 analysis/np-classifier/scripts/02_export.py

  results/npclassifier.json     -> the backend, keyed by str(global_idx), same key space
                                   as metadata.json. Label strings are interned into a
                                   shared table; spelled out in full the file is several
                                   times larger for no added information.
  results/npclassifier.parquet  -> analysis, one row per molecule with list columns.

Reports how many molecules came back with no labels at all. That is NPClassifier
declining to classify, not a fetch failure, and it bounds every downstream analysis that
uses these labels as ground truth.
"""
from __future__ import annotations

import json
import os
from collections import Counter

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(HERE), "results")
JSONL = os.path.join(OUT_DIR, "npclassifier.jsonl")

TIERS = ["pathway", "superclass", "class"]
KEY = {"pathway": "pathway_results", "superclass": "superclass_results", "class": "class_results"}


def main() -> None:
    records, errors = {}, {}
    with open(JSONL) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            idx = r["idx"]
            if "error" in r:
                # Only keep an error if no good record exists; a retry appends a later line.
                errors.setdefault(idx, r["error"])
            else:
                records[idx] = r
                errors.pop(idx, None)

    n = len(records)
    print(f"classified={n}  errors={len(errors)}")
    if errors:
        print("  error kinds:", Counter(errors.values()).most_common(5))

    # ── interned JSON for the backend ────────────────────────────────────────
    tables = {t: {} for t in TIERS}          # label -> id
    entries = {}
    unlabelled = 0
    for idx, r in records.items():
        row = []
        for t in TIERS:
            ids = []
            for lab in r[KEY[t]]:
                ids.append(tables[t].setdefault(lab, len(tables[t])))
            row.append(ids)
        if not any(row):
            unlabelled += 1
        row.append(1 if r["isglycoside"] else 0)
        entries[str(idx)] = row

    payload = {
        "schema": "npclassifier/1",
        "tiers": TIERS,
        "labels": {t: list(tables[t]) for t in TIERS},
        "entries": entries,
    }
    out_json = os.path.join(OUT_DIR, "npclassifier.json")
    with open(out_json, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"wrote {out_json}  ({os.path.getsize(out_json)/1e6:.1f} MB)")
    for t in TIERS:
        print(f"  distinct {t:11s} {len(tables[t])}")
    print(f"  no labels at any tier: {unlabelled} ({100*unlabelled/max(n,1):.2f}%)")

    # ── parquet for analysis ─────────────────────────────────────────────────
    df = pd.DataFrame({
        "idx": list(records),
        "smiles": [records[i]["smiles"] for i in records],
        **{t: [records[i][KEY[t]] for i in records] for t in TIERS},
        "isglycoside": [records[i]["isglycoside"] for i in records],
    }).sort_values("idx").reset_index(drop=True)
    out_pq = os.path.join(OUT_DIR, "npclassifier.parquet")
    df.to_parquet(out_pq, index=False)
    print(f"wrote {out_pq}  ({os.path.getsize(out_pq)/1e6:.1f} MB)")

    coverage = {
        "n_classified": n,
        "n_errors": len(errors),
        "n_unlabelled": unlabelled,
        "distinct": {t: len(tables[t]) for t in TIERS},
        "labelled_at_tier": {t: int(sum(len(v) > 0 for v in df[t])) for t in TIERS},
        "n_glycoside": int(df["isglycoside"].sum()),
        "error_indices": sorted(errors)[:1000],
    }
    with open(os.path.join(OUT_DIR, "coverage.json"), "w") as f:
        json.dump(coverage, f, indent=2)
    for t in TIERS:
        print(f"  labelled at {t:11s} {coverage['labelled_at_tier'][t]} "
              f"({100*coverage['labelled_at_tier'][t]/max(n,1):.1f}%)")


if __name__ == "__main__":
    main()
