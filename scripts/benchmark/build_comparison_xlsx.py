#!/usr/bin/env python3
"""
Build the Annotated-vs-Journal error-analysis spreadsheet for one model
(marina-final-run-seed-2) from the per-compound JSON emitted by
compare_benchmarks.py.

Focus: molecules WRONG on Annotated (benchmark.pkl) but CORRECT on Journal
(benchmark-journal.pkl), where wrong/correct == miss/hit within top-k, for
k = 1 and k = 10, in each of the val and test splits.

Sheets:
  summary                       counts + hit rates
  val_flips_top1 / _top10       val compounds: ann miss@k & journal hit@k
  test_flips_top1 / _top10      test compounds: ann miss@k & journal hit@k
  val_all / test_all            every shared compound with both-benchmark status
"""
import json
import os
import pandas as pd

COMPARE = os.environ.get("COMPARE_JSON", "/tmp/compare_results.json")
NAMES = os.environ.get("NAMES_JSON", "/tmp/npid_names.json")
OUT = os.environ.get("OUT_XLSX", "/home/user/atong/Benchmark/marina-final-run-seed-2_benchmark_comparison.xlsx")

data = json.load(open(COMPARE))
names = json.load(open(NAMES))


def fmt_hsqc(rows):
    return "\n".join(f"{h:.3f}\t{c:.3f}\t{p}" for h, c, p in rows)


def fmt_1d(vals):
    return "\n".join(f"{v:.3f}" for v in vals)


def compare_row(npid, a, j):
    """a, j are the annotated / journal per-compound records for the same npid."""
    return {
        "npid": npid,
        "name": names.get(npid, ""),
        "mw": a.get("mw"),
        # retrieval outcome
        "ann_rank": a["dereplication_rank"],
        "jrnl_rank": j["dereplication_rank"],
        "ann_hit@1": a["topk"]["1"] if "1" in a["topk"] else a["topk"][1],
        "jrnl_hit@1": j["topk"]["1"] if "1" in j["topk"] else j["topk"][1],
        "ann_hit@10": a["topk"]["10"] if "10" in a["topk"] else a["topk"][10],
        "jrnl_hit@10": j["topk"]["10"] if "10" in j["topk"] else j["topk"][10],
        "ann_cosine": a["cosine_sim"],
        "jrnl_cosine": j["cosine_sim"],
        "ann_top1_cos_sfp": a["top1_cos_sfp"],
        "jrnl_top1_cos_sfp": j["top1_cos_sfp"],
        "ann_top1_smiles": a["top1_smiles"],
        "jrnl_top1_smiles": j["top1_smiles"],
        # input-peak differences that drive the flip
        "ann_n_hsqc": a["n_hsqc"], "jrnl_n_hsqc": j["n_hsqc"],
        "ann_n_c_nmr": a["n_c_nmr"], "jrnl_n_c_nmr": j["n_c_nmr"],
        "ann_n_h_nmr": a["n_h_nmr"], "jrnl_n_h_nmr": j["n_h_nmr"],
        "smiles": a["smiles"],
        "ann_hsqc": fmt_hsqc(a["hsqc"]), "jrnl_hsqc": fmt_hsqc(j["hsqc"]),
        "ann_c_nmr": fmt_1d(a["c_nmr"]), "jrnl_c_nmr": fmt_1d(j["c_nmr"]),
        "ann_h_nmr": fmt_1d(a["h_nmr"]), "jrnl_h_nmr": fmt_1d(j["h_nmr"]),
    }


def hit(rec, k):
    tk = rec["topk"]
    return tk[str(k)] if str(k) in tk else tk[k]


summary_rows = []
sheets = {}

for split in ("val", "test"):
    ann = data["annotated"][split]
    jrn = data["journal"][split]
    shared = [n for n in ann if n in jrn]  # journal ⊂ annotated NPIDs
    rows_all = [compare_row(n, ann[n], jrn[n]) for n in shared]
    df_all = pd.DataFrame(rows_all).sort_values(["ann_rank", "npid"])
    sheets[f"{split}_all"] = df_all

    for k in (1, 10):
        flips = [n for n in shared if (not hit(ann[n], k)) and hit(jrn[n], k)]
        df = pd.DataFrame([compare_row(n, ann[n], jrn[n]) for n in flips])
        if not df.empty:
            df = df.sort_values(["jrnl_rank", "npid"])
        sheets[f"{split}_flips_top{k}"] = df

        ann_hits = sum(hit(ann[n], k) for n in shared)
        jrn_hits = sum(hit(jrn[n], k) for n in shared)
        summary_rows.append({
            "split": split, "k": k, "n_shared": len(shared),
            "annotated_hits": ann_hits, "annotated_hit_pct": round(100 * ann_hits / len(shared), 2),
            "journal_hits": jrn_hits, "journal_hit_pct": round(100 * jrn_hits / len(shared), 2),
            "flips_wrong_ann_correct_jrnl": len(flips),
            "reverse_flips_correct_ann_wrong_jrnl": sum(
                1 for n in shared if hit(ann[n], k) and not hit(jrn[n], k)),
        })

summary = pd.DataFrame(summary_rows)

order = ["summary",
         "val_flips_top1", "val_flips_top10", "test_flips_top1", "test_flips_top10",
         "val_all", "test_all"]
with pd.ExcelWriter(OUT, engine="openpyxl") as w:
    summary.to_excel(w, sheet_name="summary", index=False)
    for name in order[1:]:
        sheets[name].to_excel(w, sheet_name=name, index=False)

print("Wrote", OUT)
print(summary.to_string(index=False))
