"""
Peak-aware 13C similarity between the three MARINA benchmark spectral sources
(Annotated / Journal / Simulated) for the same molecules, using NMRPeak's
CNMRSimilarityScorer. Model-independent measurement of the spectral domain gap.

Isolated analysis — does not touch MARINA code. Reads benchmark pkls read-only.
"""
import os
import pickle
import csv
import statistics as st
from spectrum_similarity_scorer import CNMRSimilarityScorer

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("MARINA_DATA_ROOT", "/home/user/atong")

PATHS = {
    "annotated": os.path.join(DATA_ROOT, "Datasets/MARINA1/benchmark.pkl"),
    "journal":   os.path.join(DATA_ROOT, "Benchmark/benchmark-journal.pkl"),
    "sim":       os.path.join(DATA_ROOT, "Benchmark/benchmark-sim.pkl"),
}

def load(p):
    with open(p, "rb") as f:
        return pickle.load(f)

def cnmr_shifts(entry):
    """Extract 13C shift list from a benchmark entry, or None if absent."""
    t = entry.get("input", {}).get("c_nmr", None)
    if t is None:
        return None
    return [float(x) for x in t.flatten().tolist()]

def main():
    ann = {v["npid"]: v for v in load(PATHS["annotated"]).values()}
    jrn = load(PATHS["journal"])
    sim = load(PATHS["sim"])

    common = sorted(set(ann) & set(jrn) & set(sim))
    print(f"Common NPIDs across all three: {len(common)}")

    scorer = CNMRSimilarityScorer(scale=5.0, tolerance=20.0)  # paper defaults for 13C

    pairs = [("ann_vs_sim", ann, sim),
             ("jrn_vs_sim", jrn, sim),
             ("ann_vs_jrn", ann, jrn)]

    rows = []
    for npid in common:
        split = ann[npid].get("split", "?")
        row = {"npid": npid, "split": split}
        cshifts = {"ann": cnmr_shifts(ann[npid]),
                   "jrn": cnmr_shifts(jrn[npid]),
                   "sim": cnmr_shifts(sim[npid])}
        row["n_ann"] = len(cshifts["ann"]) if cshifts["ann"] else 0
        row["n_jrn"] = len(cshifts["jrn"]) if cshifts["jrn"] else 0
        row["n_sim"] = len(cshifts["sim"]) if cshifts["sim"] else 0
        for label, dA, dB in pairs:
            a = cshifts[label[:3]]
            b = cshifts[label.split("_")[-1][:3]]
            if not a or not b:
                row[f"{label}_sim"] = None
                continue
            res = scorer.calculate_similarity_score(a, b)
            row[f"{label}_sim"] = res["similarity"]
            row[f"{label}_avgpk"] = res["avg_peak_similarity"]
            row[f"{label}_pen"] = res["penalty_coefficient"]
        rows.append(row)

    # write per-molecule CSV
    csv_path = os.path.join(ROOT, "cnmr_similarity_per_molecule.csv")
    fields = ["npid", "split", "n_ann", "n_jrn", "n_sim",
              "ann_vs_sim_sim", "ann_vs_sim_avgpk", "ann_vs_sim_pen",
              "jrn_vs_sim_sim", "jrn_vs_sim_avgpk", "jrn_vs_sim_pen",
              "ann_vs_jrn_sim", "ann_vs_jrn_avgpk", "ann_vs_jrn_pen"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    print(f"Per-molecule CSV -> {csv_path}\n")

    def summarize(key, subset=None):
        vals = [r[key] for r in rows
                if r.get(key) is not None and (subset is None or r["split"] == subset)]
        if not vals:
            return None
        vals_sorted = sorted(vals)
        return {
            "n": len(vals),
            "mean": st.mean(vals),
            "std": st.pstdev(vals),
            "median": st.median(vals),
            "p25": vals_sorted[len(vals)//4],
            "p75": vals_sorted[(3*len(vals))//4],
            "min": vals_sorted[0],
        }

    print("=" * 78)
    print("13C PEAK-AWARE SIMILARITY  (NMRPeak metric, scale=5.0 ppm, tol=20.0 ppm)")
    print("Higher = the two spectral sources agree more for the same molecule.")
    print("=" * 78)
    for split in [None, "val", "test"]:
        tag = "ALL" if split is None else split.upper()
        print(f"\n--- split = {tag} ---")
        print(f"{'pair':<12} {'n':>4} {'mean':>7} {'std':>6} {'median':>7} "
              f"{'p25':>6} {'p75':>6} {'min':>6}")
        for label, _, _ in pairs:
            s = summarize(f"{label}_sim", split)
            if s:
                print(f"{label:<12} {s['n']:>4} {s['mean']:>7.3f} {s['std']:>6.3f} "
                      f"{s['median']:>7.3f} {s['p25']:>6.3f} {s['p75']:>6.3f} {s['min']:>6.3f}")

    # decomposition: shift-agreement (avgpk) vs peak-count penalty (pen)
    print("\n" + "=" * 78)
    print("DECOMPOSITION (ALL): avg peak similarity (shift agreement) vs penalty (count)")
    print("=" * 78)
    print(f"{'pair':<12} {'avgpk_mean':>11} {'pen_mean':>9}")
    for label, _, _ in pairs:
        a = summarize(f"{label}_avgpk")
        p = summarize(f"{label}_pen")
        if a and p:
            print(f"{label:<12} {a['mean']:>11.3f} {p['mean']:>9.3f}")

    # peak-count deltas
    print("\n" + "=" * 78)
    print("PEAK COUNTS (13C, ALL molecules)")
    print("=" * 78)
    for k in ("n_ann", "n_jrn", "n_sim"):
        vals = [r[k] for r in rows]
        print(f"  {k}: mean={st.mean(vals):.1f}  median={st.median(vals)}  "
              f"min={min(vals)}  max={max(vals)}")
    d_as = [r["n_sim"] - r["n_ann"] for r in rows]
    d_js = [r["n_sim"] - r["n_jrn"] for r in rows]
    print(f"  (n_sim - n_ann): mean={st.mean(d_as):+.2f}  "
          f"(#mols where ann has FEWER 13C peaks than sim = "
          f"{sum(1 for x in d_as if x>0)}/{len(d_as)})")
    print(f"  (n_sim - n_jrn): mean={st.mean(d_js):+.2f}  "
          f"(#mols where jrn has FEWER 13C peaks than sim = "
          f"{sum(1 for x in d_js if x>0)}/{len(d_js)})")

if __name__ == "__main__":
    main()
