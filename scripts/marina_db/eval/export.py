#!/usr/bin/env python3
"""
Reporting for the marina_db eval stage: one script, four subcommands, merged from
the legacy reporting quartet.

  xlsx          per-seed benchmark results (…_benchmark_results.pkl) -> NP-MRD /
                Journal sheets  (was benchmark_export.py)
  plots         cosine-similarity histograms + top-k dereplication curves across
                seeds  (was benchmark_plots.py)
  comparison    Annotated-vs-Journal per-compound flip spreadsheet from a
                compare_benchmarks-style JSON  (was build_comparison_xlsx.py)
  cosine-delta  per-compound Annotated->Journal cosine delta over seeds
                (was build_cosine_delta_xlsx.py)

Defaults resolve from config (BENCH_ROOT); override with flags. No hardcoded
absolute paths.
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))   # scripts/marina_db  (config)
sys.path.insert(0, str(_HERE.parents[3]))   # repo root

import argparse
import json
import math
import os
import pickle
import statistics as st

import pandas as pd

from config import BENCH_ROOT, REPO_ROOT

BENCHMARKS_DIR = os.path.join(BENCH_ROOT, "benchmarks")
SEEDS = [0, 1, 2]
COLORS = ["#2196F3", "#FF9800", "#4CAF50"]
BENCHMARKS = {"NP-MRD": "benchmark_results", "Journal": "benchmark_journal_results"}


# ------------------------------------------------------------------ xlsx --------
def _format_results(results):
    rows = []
    for k, v in results.items():
        topk = v["predictions"]["dereplication_topk"]
        rows.append({
            "npid": v.get("npid", k),
            "dereplication_rank": (-1 if not any(topk.values())
                                   else next(r for r, hit in topk.items() if hit)),
            "cosine_sim": v["predictions"]["cosine_sim"].item(),
            "h_nmr": "\n".join(f"{r[0]:.3f}" for r in v["input"]["h_nmr"].tolist()),
            "c_nmr": "\n".join(f"{r[0]:.3f}" for r in v["input"]["c_nmr"].tolist()),
            "hsqc": "\n".join(f"{r[1]:.3f}\t{r[0]:.3f}\t{int(r[2])}"
                              for r in v["input"]["hsqc"].tolist()),
            "mw": round(float(v["input"]["mw"]), 5),
            "smiles": v["smiles"],
        })
    return pd.DataFrame(rows)


def cmd_xlsx(a):
    for seed in a.seeds:
        out_path = os.path.join(a.out_dir, f"marina-final-run-seed-{seed}_benchmark.xlsx")
        with pd.ExcelWriter(out_path, engine="openpyxl") as w:
            nm = os.path.join(BENCHMARKS_DIR, f"marina-final-run-seed-{seed}_benchmark_results.pkl")
            _format_results(pickle.load(open(nm, "rb"))).to_excel(w, sheet_name="NP-MRD", index=False)
            jn = os.path.join(BENCHMARKS_DIR, f"marina-final-run-seed-{seed}_benchmark_journal_results.pkl")
            _format_results(pickle.load(open(jn, "rb"))).to_excel(w, sheet_name="Journal", index=False)
        print(f"Wrote {out_path}")


# ------------------------------------------------------------------ plots -------
def _load_all(suffix, seeds):
    return [pickle.load(open(os.path.join(
        BENCHMARKS_DIR, f"marina-final-run-seed-{s}_{suffix}.pkl"), "rb")) for s in seeds]


def cmd_plots(a):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(a.out_dir, exist_ok=True)
    seeds = a.seeds

    # Plot 1: cosine-similarity histograms
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    bins = np.linspace(0, 1, 25)
    for ax, (label, suffix) in zip(axes, BENCHMARKS.items()):
        for s, data, color in zip(seeds, _load_all(suffix, seeds), COLORS):
            sims = [v["predictions"]["cosine_sim"].item() for v in data.values()]
            m = np.mean(sims)
            ax.hist(sims, bins=bins, alpha=0.5, color=color, label=f"Seed {s} (μ={m:.3f})")
            ax.axvline(m, color=color, linestyle="--", linewidth=1)
        ax.set_xlabel("Cosine Similarity"); ax.set_ylabel("Count")
        ax.set_title(f"{label} Benchmark"); ax.legend(fontsize=8)
    fig.suptitle("Distribution of Predicted Fingerprint Cosine Similarity", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(a.out_dir, "cosine_similarity_hist.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved cosine_similarity_hist.png")

    # Plot 2: top-k dereplication curves
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (label, suffix) in zip(axes, BENCHMARKS.items()):
        datasets = _load_all(suffix, seeds)
        n = len(datasets[0])
        all_topk = np.array([[sum(v["predictions"]["dereplication_topk"][k]
                                  for v in d.values()) / len(d) * 100
                              for k in range(1, 11)] for d in datasets])
        mean_topk, std_topk = all_topk.mean(axis=0), all_topk.std(axis=0)
        ks = range(1, 11)
        for s, hits, color in zip(seeds, all_topk, COLORS):
            ax.plot(ks, hits, color=color, alpha=0.4, linewidth=1)
            ax.scatter(ks, hits, color=color, s=20, alpha=0.6)
        ax.plot(ks, mean_topk, color="black", linewidth=2, label="Mean")
        ax.fill_between(ks, mean_topk - std_topk, mean_topk + std_topk, color="black", alpha=0.1)
        ax.axhline(100, color="gray", linestyle=":", linewidth=1, label="Theoretical max")
        for s, color in zip(seeds, COLORS):
            ax.plot([], [], color=color, label=f"Seed {s}")
        ax.set_xlabel("Top-K"); ax.set_ylabel("Dereplication Rate (%)")
        ax.set_title(f"{label} Benchmark (n={n})"); ax.set_xticks(range(1, 11)); ax.legend(fontsize=8)
    fig.suptitle("Top-K Dereplication Rate", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(a.out_dir, "topk_dereplication.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved topk_dereplication.png")


# ------------------------------------------------------- comparison (flips) -----
def _fmt_hsqc(rows):
    return "\n".join(f"{h:.3f}\t{c:.3f}\t{p}" for h, c, p in rows)


def _fmt_1d(vals):
    return "\n".join(f"{v:.3f}" for v in vals)


def _hit(rec, k):
    tk = rec["topk"]
    return tk[str(k)] if str(k) in tk else tk[k]


def _compare_row(npid, a, j, names):
    return {
        "npid": npid, "name": names.get(npid, ""), "mw": a.get("mw"),
        "ann_rank": a["dereplication_rank"], "jrnl_rank": j["dereplication_rank"],
        "ann_hit@1": _hit(a, 1), "jrnl_hit@1": _hit(j, 1),
        "ann_hit@10": _hit(a, 10), "jrnl_hit@10": _hit(j, 10),
        "ann_cosine": a["cosine_sim"], "jrnl_cosine": j["cosine_sim"],
        "ann_top1_cos_sfp": a["top1_cos_sfp"], "jrnl_top1_cos_sfp": j["top1_cos_sfp"],
        "ann_top1_smiles": a["top1_smiles"], "jrnl_top1_smiles": j["top1_smiles"],
        "ann_n_hsqc": a["n_hsqc"], "jrnl_n_hsqc": j["n_hsqc"],
        "ann_n_c_nmr": a["n_c_nmr"], "jrnl_n_c_nmr": j["n_c_nmr"],
        "ann_n_h_nmr": a["n_h_nmr"], "jrnl_n_h_nmr": j["n_h_nmr"],
        "smiles": a["smiles"],
        "ann_hsqc": _fmt_hsqc(a["hsqc"]), "jrnl_hsqc": _fmt_hsqc(j["hsqc"]),
        "ann_c_nmr": _fmt_1d(a["c_nmr"]), "jrnl_c_nmr": _fmt_1d(j["c_nmr"]),
        "ann_h_nmr": _fmt_1d(a["h_nmr"]), "jrnl_h_nmr": _fmt_1d(j["h_nmr"]),
    }


def cmd_comparison(a):
    data = json.load(open(a.compare))
    names = json.load(open(a.names)) if a.names and os.path.exists(a.names) else {}
    summary_rows, sheets = [], {}
    for split in ("val", "test"):
        ann, jrn = data["annotated"][split], data["journal"][split]
        shared = [n for n in ann if n in jrn]   # journal NPIDs are a subset of annotated
        sheets[f"{split}_all"] = pd.DataFrame(
            [_compare_row(n, ann[n], jrn[n], names) for n in shared]
        ).sort_values(["ann_rank", "npid"])
        for k in (1, 10):
            flips = [n for n in shared if (not _hit(ann[n], k)) and _hit(jrn[n], k)]
            df = pd.DataFrame([_compare_row(n, ann[n], jrn[n], names) for n in flips])
            if not df.empty:
                df = df.sort_values(["jrnl_rank", "npid"])
            sheets[f"{split}_flips_top{k}"] = df
            ann_hits = sum(_hit(ann[n], k) for n in shared)
            jrn_hits = sum(_hit(jrn[n], k) for n in shared)
            summary_rows.append({
                "split": split, "k": k, "n_shared": len(shared),
                "annotated_hits": ann_hits, "annotated_hit_pct": round(100 * ann_hits / len(shared), 2),
                "journal_hits": jrn_hits, "journal_hit_pct": round(100 * jrn_hits / len(shared), 2),
                "flips_wrong_ann_correct_jrnl": len(flips),
                "reverse_flips_correct_ann_wrong_jrnl": sum(
                    1 for n in shared if _hit(ann[n], k) and not _hit(jrn[n], k)),
            })
    order = ["val_flips_top1", "val_flips_top10", "test_flips_top1", "test_flips_top10",
             "val_all", "test_all"]
    with pd.ExcelWriter(a.out, engine="openpyxl") as w:
        pd.DataFrame(summary_rows).to_excel(w, sheet_name="summary", index=False)
        for name in order:
            sheets[name].to_excel(w, sheet_name=name, index=False)
    print("Wrote", a.out)


# ------------------------------------------------------- cosine-delta -----------
def _norm_two_sided_p(t):
    return 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))


def cmd_cosine_delta(a):
    seed_jsons = a.compare.split(",")
    seeds = [json.load(open(p)) for p in seed_jsons]
    names = json.load(open(a.names)) if a.names and os.path.exists(a.names) else {}
    NS = len(seeds)

    rows = []
    for split in ("val", "test"):
        shared = [n for n in seeds[0]["annotated"][split] if n in seeds[0]["journal"][split]]
        for npid in shared:
            ann = [s["annotated"][split][npid]["cosine_sim"] for s in seeds]
            jrn = [s["journal"][split][npid]["cosine_sim"] for s in seeds]
            deltas = [j - x for x, j in zip(ann, jrn)]
            a0, j0 = seeds[0]["annotated"][split][npid], seeds[0]["journal"][split][npid]
            row = {
                "npid": npid, "name": names.get(npid, ""), "split": split, "mw": a0.get("mw"),
                "delta_mean": round(st.mean(deltas), 4),
                "delta_std": round(st.stdev(deltas), 4) if NS > 1 else 0.0,
                "ann_cos_mean": round(st.mean(ann), 4), "jrnl_cos_mean": round(st.mean(jrn), 4),
            }
            for i in range(NS):
                row[f"ann_cos_s{i}"] = round(ann[i], 4)
            for i in range(NS):
                row[f"jrnl_cos_s{i}"] = round(jrn[i], 4)
            for i in range(NS):
                row[f"delta_s{i}"] = round(deltas[i], 4)
            row.update({
                "ann_n_hsqc": a0["n_hsqc"], "jrnl_n_hsqc": j0["n_hsqc"], "d_n_hsqc": j0["n_hsqc"] - a0["n_hsqc"],
                "ann_n_c_nmr": a0["n_c_nmr"], "jrnl_n_c_nmr": j0["n_c_nmr"], "d_n_c_nmr": j0["n_c_nmr"] - a0["n_c_nmr"],
                "ann_n_h_nmr": a0["n_h_nmr"], "jrnl_n_h_nmr": j0["n_h_nmr"], "d_n_h_nmr": j0["n_h_nmr"] - a0["n_h_nmr"],
                "smiles": a0["smiles"],
            })
            rows.append(row)

    df = pd.DataFrame(rows)
    col_order = (["npid", "name", "split", "mw", "delta_mean", "delta_std",
                  "ann_cos_mean", "jrnl_cos_mean"]
                 + [f"ann_cos_s{i}" for i in range(NS)]
                 + [f"jrnl_cos_s{i}" for i in range(NS)]
                 + [f"delta_s{i}" for i in range(NS)]
                 + ["ann_n_hsqc", "jrnl_n_hsqc", "d_n_hsqc",
                    "ann_n_c_nmr", "jrnl_n_c_nmr", "d_n_c_nmr",
                    "ann_n_h_nmr", "jrnl_n_h_nmr", "d_n_h_nmr", "smiles"])
    df = df[col_order]

    srows = []
    for label, sub in [("all", df), ("val", df[df.split == "val"]), ("test", df[df.split == "test"])]:
        d = sub["delta_mean"].tolist()
        n = len(d)
        mean = st.mean(d); sd = st.stdev(d); se = sd / math.sqrt(n); t = mean / se
        srows.append({
            "subset": label, "n_compounds": n,
            "mean_ann_cos": round(sub["ann_cos_mean"].mean(), 4),
            "mean_jrnl_cos": round(sub["jrnl_cos_mean"].mean(), 4),
            "mean_delta": round(mean, 4), "std_delta": round(sd, 4), "se_delta": round(se, 4),
            "paired_t": round(t, 2), "approx_p_two_sided": f"{_norm_two_sided_p(t):.2e}",
            "n_delta>0": int((sub["delta_mean"] > 0).sum()),
            "n_delta<0": int((sub["delta_mean"] < 0).sum()),
            "n_delta==0": int((sub["delta_mean"] == 0).sum()),
        })

    df_sorted = df.sort_values("delta_mean", ascending=False)
    with pd.ExcelWriter(a.out, engine="openpyxl") as w:
        pd.DataFrame(srows).to_excel(w, sheet_name="summary", index=False)
        df_sorted.to_excel(w, sheet_name="all_compounds", index=False)
        df_sorted[df_sorted.split == "val"].to_excel(w, sheet_name="val", index=False)
        df_sorted[df_sorted.split == "test"].to_excel(w, sheet_name="test", index=False)
    print("Wrote", a.out)


def _seeds(s):
    return [int(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("xlsx", help="per-seed benchmark results -> xlsx")
    p.add_argument("--seeds", type=_seeds, default=SEEDS)
    p.add_argument("--out_dir", default=str(BENCH_ROOT))
    p.set_defaults(func=cmd_xlsx)

    p = sub.add_parser("plots", help="cosine-hist + top-k dereplication plots")
    p.add_argument("--seeds", type=_seeds, default=SEEDS)
    p.add_argument("--out_dir", default=str(REPO_ROOT / "wiki" / "natural-products-chemistry" / "marina-benchmark"))
    p.set_defaults(func=cmd_plots)

    p = sub.add_parser("comparison", help="Annotated-vs-Journal flip spreadsheet")
    p.add_argument("--compare", required=True, help="compare_benchmarks JSON")
    p.add_argument("--names", default=None, help="npid->name JSON (optional)")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_comparison)

    p = sub.add_parser("cosine-delta", help="Annotated->Journal cosine delta over seeds")
    p.add_argument("--compare", required=True, help="comma-separated per-seed compare JSONs")
    p.add_argument("--names", default=None, help="npid->name JSON (optional)")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_cosine_delta)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
