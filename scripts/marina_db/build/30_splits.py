"""30_splits.py -- unify three divergent split policies into one (decisions D7 + D9).

Sources merged:
  - scripts/dataset/generate_dataset.py : random 90/5/5, benchmark.pkl -> test
  - analysis/spectre-split/scripts/01_spectre_splits.py : SPECTRE val/test + wide train set
  - analysis/spectre-split/scripts/02_match.py : SPECTRE membership -> split, benchmark guard

Unified policy, applied to config.INDEX_PKL (written by 20_build_index.py with split=None):
  1. FORCED assignments by fixed-point canonical-SMILES membership (canonicalize BOTH
     sides with src.modules.data.smiles.canonicalize_smiles):
       benchmark molecules -> test   (D7: benchmark.pkl AND the FULL 467-entry
                                       benchmark-journal-prepared.pkl, not just annotated)
       SPECTRE test -> test ; SPECTRE val -> val ; SPECTRE train -> train
     Precedence: benchmark->test beats everything (a benchmark molecule must never
     train); then SPECTRE test > val > train.
  2. FREE POOL (everything else): drawn randomly (seed=config.SEED) so the GLOBAL
     dataset totals land on config.SPLIT_WEIGHTS 90/5/5 (D9) -- target-minus-forced,
     NOT 90/5/5 of the remainder.
Rewrites config.INDEX_PKL in place with the per-entry 'split'; writes splits_report.json.
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))
sys.path.insert(0, str(_HERE.parents[3]))
from config import INDEX_PKL, BENCH_ANNOTATED, BENCH_JOURNAL_PREPARED, SPECTRE_SPLITS_PKL, SPLIT_WEIGHTS, SEED
from src.modules.data.smiles import canonicalize_smiles

import argparse
import json
import pickle
import random
import subprocess
from collections import Counter
from multiprocessing import Pool, cpu_count

REPO_ROOT = _HERE.parents[3]
# 01's output location, used as a fallback and as the derive target.
_SPECTRE_DERIVE = REPO_ROOT / "analysis" / "spectre-split" / "scripts" / "01_spectre_splits.py"
_SPECTRE_RESULT = REPO_ROOT / "analysis" / "spectre-split" / "results" / "spectre_splits.pkl"

# Expected spectre_splits.pkl keys (see 01_spectre_splits.py):
#   train/val/test -> sets of canonical SMILES (train = union of SPECTRE's 2D + 1D
#                     train branches, minus val/test); val_idx/test_idx unused here.
SPECTRE_KEYS = ("train", "val", "test")


def _canon_list(smiles, workers):
    """Fixed-point canonicalize a list; parallel for large inputs. None on parse failure."""
    if workers > 1 and len(smiles) > 2000:
        with Pool(workers) as pool:
            return pool.map(canonicalize_smiles, smiles, chunksize=500)
    return [canonicalize_smiles(s) for s in smiles]


def _canon_set(smiles, workers):
    return {c for c in _canon_list(list(smiles), workers) if c}


def load_spectre_splits(path, workers):
    """Prefer the prebuilt pkl (config path, then 01's results dir); else run 01 to derive.

    Returns re-canonicalized (via the shared fn) train/val/test SMILES sets so membership
    lines up with the index side. 01's canon differs slightly (no largest-fragment strip).
    """
    candidates = [Path(path), SPECTRE_SPLITS_PKL, _SPECTRE_RESULT]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        # Derive it the way 01_spectre_splits.py does (needs the SPECTRE trees/zip it reads).
        print(f"No spectre_splits.pkl found; deriving via {_SPECTRE_DERIVE}")
        subprocess.run([sys.executable, str(_SPECTRE_DERIVE)], check=True)
        if not _SPECTRE_RESULT.exists():
            raise FileNotFoundError(f"derivation did not produce {_SPECTRE_RESULT}")
        src = _SPECTRE_RESULT
    print(f"Loading SPECTRE splits from {src}")
    spec = pickle.load(open(src, "rb"))
    missing = [k for k in SPECTRE_KEYS if k not in spec]
    if missing:
        raise KeyError(f"{src} missing expected keys {missing}; has {list(spec)}")
    return {k: _canon_set(spec[k], workers) for k in SPECTRE_KEYS}


def load_benchmark_smiles(bench_pkl, journal_pkl, workers):
    """Canonical SMILES of ALL benchmark molecules: benchmark.pkl + FULL journal set (D7)."""
    raw = []
    for p in (bench_pkl, journal_pkl):
        data = pickle.load(open(p, "rb"))
        raw += [e["smiles"] for e in data.values() if e.get("smiles")]
    return _canon_set(raw, workers)


def main():
    ap = argparse.ArgumentParser(description="Unified MARINA-DB split assignment (D7 + D9).")
    ap.add_argument("--index", type=Path, default=INDEX_PKL, help="index.pkl to assign (in place)")
    ap.add_argument("--spectre", type=Path, default=SPECTRE_SPLITS_PKL, help="spectre_splits.pkl")
    ap.add_argument("--workers", type=int, default=cpu_count(), help="canonicalization workers")
    args = ap.parse_args()

    index = pickle.load(open(args.index, "rb"))
    idxs = sorted(index)
    n = len(idxs)
    print(f"Loaded {n} entries from {args.index}")

    # --- forced membership sets (all fixed-point canonical) ---
    bench_set = load_benchmark_smiles(BENCH_ANNOTATED, BENCH_JOURNAL_PREPARED, args.workers)
    spec = load_spectre_splits(args.spectre, args.workers)
    print(f"benchmark: {len(bench_set)} canonical SMILES | "
          f"SPECTRE train/val/test: {len(spec['train'])}/{len(spec['val'])}/{len(spec['test'])}")

    canoned = _canon_list([index[i]["smiles"] for i in idxs], args.workers)
    canon_failed = sum(c is None for c in canoned)

    # --- assign forced; collect the free pool ---
    # Precedence: benchmark->test wins over everything; then SPECTRE test > val > train.
    assign = {}
    free_idxs = []
    rule = Counter()
    for i, c in zip(idxs, canoned):
        if c is None:
            free_idxs.append(i)          # unparseable -> unmatchable -> free pool
        elif c in bench_set:
            assign[i] = "test"; rule["benchmark_test"] += 1
        elif c in spec["test"]:
            assign[i] = "test"; rule["spectre_test"] += 1
        elif c in spec["val"]:
            assign[i] = "val"; rule["spectre_val"] += 1
        elif c in spec["train"]:
            assign[i] = "train"; rule["spectre_train"] += 1
        else:
            free_idxs.append(i)
    forced = Counter(assign.values())

    # --- free-pool allocation to hit GLOBAL SPLIT_WEIGHTS (D9) ---
    tw, vw, _ = SPLIT_WEIGHTS
    t_train, t_val = round(tw * n), round(vw * n)
    targets = {"train": t_train, "val": t_val, "test": n - t_train - t_val}
    # free share = target - already-forced, clamped at 0 (forced may exceed a target)
    needs = {s: max(0, targets[s] - forced.get(s, 0)) for s in targets}
    # reconcile to exactly the free-pool size; train is the sink, cascade to test then val
    delta = len(free_idxs) - sum(needs.values())
    for s in ("train", "test", "val"):
        if delta == 0:
            break
        adj = delta if delta > 0 else max(-needs[s], delta)
        needs[s] += adj
        delta -= adj
    assert delta == 0 and sum(needs.values()) == len(free_idxs), "free-pool reconcile failed"

    rng = random.Random(SEED)
    rng.shuffle(free_idxs)
    pos = 0
    for s in ("train", "val", "test"):
        for i in free_idxs[pos:pos + needs[s]]:
            assign[i] = s
        pos += needs[s]

    # --- rewrite index in place ---
    for i in idxs:
        index[i]["split"] = assign[i]
    pickle.dump(index, open(args.index, "wb"))

    final = Counter(assign.values())
    report = {
        "total_entries": n,
        "canon_failed": canon_failed,
        "target_counts": targets,
        "forced_by_rule": dict(rule),
        "forced_by_split": dict(forced),
        "free_pool_size": len(free_idxs),
        "free_pool_allocation": needs,
        "final_counts": {s: final[s] for s in ("train", "val", "test")},
        "realized_fractions": {s: round(final[s] / n, 4) for s in ("train", "val", "test")},
        "benchmark_canonical": len(bench_set),
        "spectre_sizes": {s: len(spec[s]) for s in SPECTRE_KEYS},
    }
    report_path = args.index.parent / "splits_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote {args.index} and {report_path}")
    for s in ("train", "val", "test"):
        print(f"  {s:5s} {final[s]:>8d}  ({report['realized_fractions'][s]:.4f})")


if __name__ == "__main__":
    main()
