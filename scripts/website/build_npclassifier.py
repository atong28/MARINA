#!/usr/bin/env python3
"""
Build npclassifier.json for a deployment model directory.

Two sources, cheapest first:
  1. RECYCLE — copy annotations from an existing model's npclassifier.json for
     any molecule that also appears there (matched by canonical_2d_smiles), so we
     don't re-ask the API for structures already classified (e.g. MARINA1 ⊂ MARINA-DB).
  2. REGENERATE — for the remaining molecules, query the NPClassifier API
     (https://npclassifier.gnps2.org/classify?smiles=...).

Writes the interned schema the backend expects (see backend/app/session.get_npclassifier):
    {"schema":"npclassifier/1", "tiers":["pathway","superclass","class"],
     "labels":{tier:[...strings]}, "entries":{"<idx>":[[ids],[ids],[ids], isglycoside]}}

A molecule the API returns but declines to classify gets a record with empty tiers
(a real "classified, no labels" fact). A molecule whose API call *errors* gets no
record and is left for a later run — the API cache is checkpointed so reruns resume.

Usage:
  python scripts/website/build_npclassifier.py \
      --model-root  /path/to/target_model \
      --recycle-from /path/to/reference_model \
      [--workers 12] [--cache <path>] [--limit N] [--recycle-only]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

TIERS = ["pathway", "superclass", "class"]
API_URL = "https://npclassifier.gnps2.org/classify"
# API response field per tier.
_API_FIELD = {"pathway": "pathway_results", "superclass": "superclass_results", "class": "class_results"}


def _smiles_of(entry: dict) -> str | None:
    """The 2D canonical SMILES is the recycle/query key (same convention both sides)."""
    return entry.get("canonical_2d_smiles") or entry.get("smiles")


def build_recycle_map(reference_root: str) -> dict[str, dict]:
    """canonical_2d_smiles -> {pathway:[...], superclass:[...], class:[...], isglycoside:bool}."""
    npc_path = os.path.join(reference_root, "npclassifier.json")
    meta_path = os.path.join(reference_root, "metadata.json")
    if not (os.path.isfile(npc_path) and os.path.isfile(meta_path)):
        print(f"[recycle] reference lacks npclassifier.json/metadata.json at {reference_root}; skipping recycle")
        return {}

    print(f"[recycle] reading {npc_path}")
    with open(npc_path) as fh:
        npc = json.load(fh)
    tiers, labels, entries = npc["tiers"], npc["labels"], npc["entries"]

    print(f"[recycle] reading {meta_path} (large)…")
    with open(meta_path) as fh:
        ref_meta = json.load(fh)

    out: dict[str, dict] = {}
    for idx, row in entries.items():
        meta = ref_meta.get(idx)
        if not meta:
            continue
        smi = _smiles_of(meta)
        if not smi:
            continue
        ann = {tier: [labels[tier][i] for i in ids if 0 <= i < len(labels[tier])]
               for tier, ids in zip(tiers, row)}
        ann["isglycoside"] = bool(row[len(tiers)])
        out[smi] = ann
    del ref_meta
    print(f"[recycle] built map of {len(out)} annotated structures")
    return out


def _parse_api(resp: dict) -> dict:
    """NPClassifier API JSON -> annotation dict (empty tiers if it declined)."""
    return {
        "pathway":    list(resp.get(_API_FIELD["pathway"]) or []),
        "superclass": list(resp.get(_API_FIELD["superclass"]) or []),
        "class":      list(resp.get(_API_FIELD["class"]) or []),
        "isglycoside": bool(resp.get("isglycoside", False)),
    }


def query_api(smiles_list: list[str], workers: int, cache: dict, cache_path: str) -> None:
    """Fill `cache[smiles]` with an annotation dict for each smiles (skips cached)."""
    todo = [s for s in smiles_list if s not in cache]
    print(f"[api] {len(todo)} to query ({len(smiles_list) - len(todo)} already cached), {workers} workers")
    if not todo:
        return

    lock = threading.Lock()
    done = [0]
    tl = threading.local()

    def session() -> requests.Session:
        if not hasattr(tl, "s"):
            tl.s = requests.Session()
        return tl.s

    def fetch(smi: str):
        for attempt in range(3):
            try:
                r = session().get(API_URL, params={"smiles": smi}, timeout=15)
                if r.status_code == 200:
                    return smi, _parse_api(r.json())
                # 4xx on a bad SMILES: a real "cannot classify", record empty rather than retry forever.
                if 400 <= r.status_code < 500:
                    return smi, {"pathway": [], "superclass": [], "class": [], "isglycoside": False}
            except Exception:
                pass
            time.sleep(0.5 * (attempt + 1))
        return smi, None            # transient failure: leave uncached for a later run

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(fetch, s) for s in todo]
        for fut in as_completed(futures):
            smi, ann = fut.result()
            with lock:
                if ann is not None:
                    cache[smi] = ann
                done[0] += 1
                if done[0] % 2000 == 0:
                    with open(cache_path, "wb") as fh:
                        pickle.dump(cache, fh)
                    print(f"[api] {done[0]}/{len(todo)} (cached {len(cache)})")
    with open(cache_path, "wb") as fh:
        pickle.dump(cache, fh)
    print(f"[api] done; cache has {len(cache)} entries")


def intern_and_write(model_root: str, ann_by_idx: dict[str, dict]) -> None:
    """Intern label strings and write npclassifier.json."""
    labels: dict[str, list[str]] = {t: [] for t in TIERS}
    label_id: dict[str, dict[str, int]] = {t: {} for t in TIERS}

    def id_of(tier: str, label: str) -> int:
        table = label_id[tier]
        if label not in table:
            table[label] = len(labels[tier])
            labels[tier].append(label)
        return table[label]

    entries: dict[str, list] = {}
    for idx, ann in ann_by_idx.items():
        row = [[id_of(t, lbl) for lbl in ann.get(t, [])] for t in TIERS]
        row.append(bool(ann.get("isglycoside", False)))
        entries[str(idx)] = row

    out = {"schema": "npclassifier/1", "tiers": TIERS, "labels": labels, "entries": entries}
    path = os.path.join(model_root, "npclassifier.json")
    with open(path, "w") as fh:
        json.dump(out, fh)
    print(f"[write] {path}: {len(entries)} entries, "
          f"labels {{{', '.join(f'{t}:{len(labels[t])}' for t in TIERS)}}}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-root", required=True, help="Target model dir (holds metadata.json)")
    ap.add_argument("--recycle-from", default=None, help="Reference model dir to recycle annotations from")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--cache", default=None, help="API cache pkl (default: <model-root>/.npclassifier_api_cache.pkl)")
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N molecules (debug)")
    ap.add_argument("--recycle-only", action="store_true", help="Skip the API; recycle matches only")
    a = ap.parse_args()

    meta_path = os.path.join(a.model_root, "metadata.json")
    print(f"[target] reading {meta_path} (large)…")
    with open(meta_path) as fh:
        meta = json.load(fh)
    items = list(meta.items())
    if a.limit:
        items = items[:a.limit]
    print(f"[target] {len(items)} molecules")

    recycle = build_recycle_map(a.recycle_from) if a.recycle_from else {}

    # Partition by canonical_2d_smiles: recyclable vs needs-API.
    idx_smiles = {idx: _smiles_of(m) for idx, m in items}
    recyclable = {idx for idx, smi in idx_smiles.items() if smi in recycle}
    to_query = sorted({smi for idx, smi in idx_smiles.items() if smi and smi not in recycle})
    print(f"[sizing] total={len(items)}  recyclable={len(recyclable)}  "
          f"unique_smiles_needing_api={len(to_query)}")

    cache_path = a.cache or os.path.join(a.model_root, ".npclassifier_api_cache.pkl")
    cache: dict[str, dict] = {}
    if os.path.isfile(cache_path):
        with open(cache_path, "rb") as fh:
            cache = pickle.load(fh)
        print(f"[api] loaded cache with {len(cache)} entries from {cache_path}")

    if not a.recycle_only:
        query_api(to_query, a.workers, cache, cache_path)

    # Assemble per-index annotations (recycle first, then API cache).
    ann_by_idx: dict[str, dict] = {}
    for idx, smi in idx_smiles.items():
        if smi in recycle:
            ann_by_idx[idx] = recycle[smi]
        elif smi in cache:
            ann_by_idx[idx] = cache[smi]
        # else: no annotation available (API error / not run) → no entry
    print(f"[assemble] {len(ann_by_idx)}/{len(items)} molecules have annotations "
          f"({len(recyclable)} recycled, {len(ann_by_idx) - len(recyclable)} from API)")

    intern_and_write(a.model_root, ann_by_idx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
