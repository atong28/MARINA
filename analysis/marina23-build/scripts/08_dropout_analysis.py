"""Exact per-modality presence rates under MARINADataset's dropout, per dataset.

`compute_drop_percentage` sets drop_p = 1 - 0.5/availability, which would give 50%
presence if dropping were independent. It is not: `__getitem__` picks one uniformly
random *available* modality as `always_keep` and exempts it. For a molecule with k
available modalities that lifts each one's survival probability by 1/k.

Exact marginal presence for modality m:

    P(m) = (1/N) * sum over molecules having m of [ 1/k_i + (1 - 1/k_i)(1 - p_m) ]

with k_i the number of available spectral modalities for molecule i. Writing
    c_m = (1/N) sum_{i has m} 1/k_i        (always-keep contribution)
    d_m = (1/N) sum_{i has m} (1 - 1/k_i)  (droppable contribution)
gives P(m) = c_m + d_m(1 - p_m), so the p hitting a 50% target is
    p* = 1 - (0.5 - c_m)/d_m
which is only feasible when c_m <= 0.5 (else always-keep alone overshoots) and
availability = c_m + d_m >= 0.5 (else even p=0 undershoots).
"""
import json
import os
import pickle
from pathlib import Path

DATA_ROOT = Path(os.environ.get("MARINA_DATA_ROOT", "/home/user/atong"))
DATASETS = DATA_ROOT / "Datasets"
OUT = Path(__file__).resolve().parent.parent / "results"
SPECTRAL = ("hsqc", "c_nmr", "h_nmr", "mass_spec")


def analyse(name):
    path = DATASETS / name / "index.pkl"
    if not path.exists():
        return None
    index = pickle.load(open(path, "rb"))
    # MARINADataset.__init__ keeps only entries with >=1 spectral modality, and
    # compute_drop_percentage runs on that filtered set -- match it exactly.
    train = [v for v in index.values()
             if v["split"] == "train" and any(v[f"has_{m}"] for m in SPECTRAL)]
    n = len(train)

    avail = {m: sum(1 for v in train if v[f"has_{m}"]) / n for m in SPECTRAL}
    # what the code currently sets
    drop_p = {m: (1 - 0.5 / avail[m]) if avail[m] > 0.5 else 0.0 for m in SPECTRAL}

    c = {m: 0.0 for m in SPECTRAL}
    d = {m: 0.0 for m in SPECTRAL}
    k_hist = {}
    for v in train:
        present = [m for m in SPECTRAL if v[f"has_{m}"]]
        k = len(present)
        k_hist[k] = k_hist.get(k, 0) + 1
        if k == 0:
            continue
        for m in present:
            c[m] += 1.0 / k
            d[m] += 1.0 - 1.0 / k
    c = {m: c[m] / n for m in SPECTRAL}
    d = {m: d[m] / n for m in SPECTRAL}

    out = {"train_molecules": n, "k_distribution": dict(sorted(k_hist.items())), "modalities": {}}
    for m in SPECTRAL:
        actual = c[m] + d[m] * (1 - drop_p[m])
        if c[m] > 0.5:
            fix, status = None, "IMPOSSIBLE: always-keep alone exceeds 50%"
        elif avail[m] < 0.5:
            fix, status = None, f"IMPOSSIBLE: only {avail[m]:.1%} of molecules have it"
        elif d[m] <= 0:
            fix, status = None, "IMPOSSIBLE: nothing droppable"
        else:
            p = 1 - (0.5 - c[m]) / d[m]
            fix, status = (round(p, 4), "ok") if 0 <= p <= 1 else (None, "IMPOSSIBLE: p out of range")
        out["modalities"][m] = {
            "availability": round(avail[m], 4),
            "current_drop_p": round(drop_p[m], 4),
            "ACTUAL_presence_now": round(actual, 4),
            "always_keep_floor_c": round(c[m], 4),
            "drop_p_for_50pct": fix,
            "status": status,
        }
    return out


report = {name: analyse(name) for name in ("MARINA1", "MARINA2", "MARINA3", "MARINA4")}
report = {k: v for k, v in report.items() if v}

OUT.mkdir(exist_ok=True)
(OUT / "dropout_analysis.json").write_text(json.dumps(report, indent=2))

for name, r in report.items():
    print(f"\n=== {name} ===  train={r['train_molecules']:,}  "
          f"modalities-per-molecule {r['k_distribution']}")
    print(f"  {'modality':10s} {'avail':>7s} {'drop_p':>7s} {'ACTUAL':>8s} {'target=0.5 p':>13s}  status")
    for m, s in r["modalities"].items():
        fix = "-" if s["drop_p_for_50pct"] is None else f"{s['drop_p_for_50pct']:.4f}"
        print(f"  {m:10s} {s['availability']:7.3f} {s['current_drop_p']:7.3f} "
              f"{s['ACTUAL_presence_now']:8.3f} {fix:>13s}  {s['status']}")
