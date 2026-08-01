#!/usr/bin/env python3
"""
Per-compound Annotated->Journal cosine-similarity delta, averaged over the three
marina-final-run seeds.

For every compound present in both benchmarks, and each seed, delta = journal
cosine - annotated cosine (cosine = predicted vs ground-truth sparse FP). The
headline column is delta_mean = mean over the 3 seeds. Peak counts are
model-independent (same input across seeds), shown once per benchmark.

Inputs: compare_seed{0,1,2}.json (from compare_benchmarks.py) + npid_names.json.
Output: xlsx with a summary sheet, an all-compounds sheet, and per-split sheets.
"""
import json
import math
import os
import statistics as st
import pandas as pd

SEED_JSONS = os.environ.get("SEED_JSONS", "/tmp/compare_seed0.json,/tmp/compare_seed1.json,/tmp/compare_seed2.json").split(",")
NAMES = os.environ.get("NAMES_JSON", "/tmp/npid_names.json")
OUT = os.environ.get("OUT_XLSX", "/home/user/atong/Benchmark/marina-final-run_cosine-delta_annotated-vs-journal.xlsx")

seeds = [json.load(open(p)) for p in SEED_JSONS]
names = json.load(open(NAMES))
NS = len(seeds)


def norm_two_sided_p(t):
    # two-sided p from a z/t statistic via normal approx (df large here)
    return 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))


rows = []
for split in ("val", "test"):
    # NPIDs present in both benchmarks for all seeds (identical sets across seeds)
    shared = [n for n in seeds[0]["annotated"][split] if n in seeds[0]["journal"][split]]
    for npid in shared:
        ann = [s["annotated"][split][npid]["cosine_sim"] for s in seeds]
        jrn = [s["journal"][split][npid]["cosine_sim"] for s in seeds]
        deltas = [j - a for a, j in zip(ann, jrn)]
        a0 = seeds[0]["annotated"][split][npid]
        j0 = seeds[0]["journal"][split][npid]
        row = {
            "npid": npid, "name": names.get(npid, ""), "split": split,
            "mw": a0.get("mw"),
            "delta_mean": round(st.mean(deltas), 4),
            "delta_std": round(st.stdev(deltas), 4) if NS > 1 else 0.0,
            "ann_cos_mean": round(st.mean(ann), 4),
            "jrnl_cos_mean": round(st.mean(jrn), 4),
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

# ---- summary with paired test on per-compound delta_mean ----
srows = []
for label, sub in [("all", df), ("val", df[df.split == "val"]), ("test", df[df.split == "test"])]:
    d = sub["delta_mean"].tolist()
    n = len(d)
    mean = st.mean(d); sd = st.stdev(d); se = sd / math.sqrt(n)
    t = mean / se
    srows.append({
        "subset": label, "n_compounds": n,
        "mean_ann_cos": round(sub["ann_cos_mean"].mean(), 4),
        "mean_jrnl_cos": round(sub["jrnl_cos_mean"].mean(), 4),
        "mean_delta": round(mean, 4), "std_delta": round(sd, 4), "se_delta": round(se, 4),
        "paired_t": round(t, 2), "approx_p_two_sided": f"{norm_two_sided_p(t):.2e}",
        "n_delta>0": int((sub["delta_mean"] > 0).sum()),
        "n_delta<0": int((sub["delta_mean"] < 0).sum()),
        "n_delta==0": int((sub["delta_mean"] == 0).sum()),
    })
summary = pd.DataFrame(srows)

df_sorted = df.sort_values("delta_mean", ascending=False)
with pd.ExcelWriter(OUT, engine="openpyxl") as w:
    summary.to_excel(w, sheet_name="summary", index=False)
    df_sorted.to_excel(w, sheet_name="all_compounds", index=False)
    df_sorted[df_sorted.split == "val"].to_excel(w, sheet_name="val", index=False)
    df_sorted[df_sorted.split == "test"].to_excel(w, sheet_name="test", index=False)

print("Wrote", OUT)
print(summary.to_string(index=False))
