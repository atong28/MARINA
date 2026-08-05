"""
Task 1: When Annotated and Journal disagree strongly on 13C, is it because peaks
are MISSING (count mismatch, shared peaks still agree) or because peaks are at
genuinely DIFFERENT positions (large matched-pair shift distances -> possible
mis-referencing, mis-extraction, or wrong compound)?

For the worst ann_vs_jrn molecules we inspect the Stage-I one-to-one matches:
distribution of matched-pair distances, and the peak-count delta.
"""
import os, pickle
from spectrum_similarity_scorer import CNMRSimilarityScorer

DATA_ROOT = os.environ.get("MARINA_DATA_ROOT", "/home/user/atong")

PATHS = {
    "annotated": os.path.join(DATA_ROOT, "Datasets/MARINA1/benchmark.pkl"),
    "journal":   os.path.join(DATA_ROOT, "Benchmark/benchmark-journal.pkl"),
}
def load(p):
    with open(p, "rb") as f: return pickle.load(f)
def cshifts(e):
    t = e.get("input", {}).get("c_nmr", None)
    return None if t is None else [float(x) for x in t.flatten().tolist()]

ann = {v["npid"]: v for v in load(PATHS["annotated"]).values()}
jrn = load(PATHS["journal"])
common = sorted(set(ann) & set(jrn))
scorer = CNMRSimilarityScorer(scale=5.0, tolerance=20.0)

# rank by ann_vs_jrn similarity
scored = []
for npid in common:
    a, b = cshifts(ann[npid]), cshifts(jrn[npid])
    if not a or not b: continue
    res = scorer.calculate_similarity_score(a, b)
    scored.append((res["similarity"], npid, a, b, res))
scored.sort()

print("Deep-dive: 8 worst Annotated<->Journal 13C molecules")
print("For each, Stage-I (one-to-one) matched-pair distances tell us if shared "
      "peaks AGREE (small dist) or DIFFER (large dist).\n")
buckets_all = {"<0.5": 0, "0.5-1": 0, "1-2": 0, "2-5": 0, ">5": 0}
n_stage1_all = 0
for sim, npid, a, b, res in scored[:8]:
    stage1 = [m for m in res["matches"] if m[0] == 1]     # (round,i,j,sa,sb,sim,dist,valid)
    dists = sorted(m[6] for m in stage1)
    n1 = len(stage1)
    b_ = {"<0.5": 0, "0.5-1": 0, "1-2": 0, "2-5": 0, ">5": 0}
    for d in dists:
        k = "<0.5" if d < 0.5 else "0.5-1" if d < 1 else "1-2" if d < 2 else "2-5" if d < 5 else ">5"
        b_[k] += 1; buckets_all[k] += 1
    n_stage1_all += n1
    med = dists[len(dists)//2] if dists else 0
    print(f"{npid}  sim={sim:.3f}  n_ann={len(a)} n_jrn={len(b)} (Δ={abs(len(a)-len(b))})  "
          f"stage1_matches={n1}  median_dist={med:.2f}ppm  max_dist={max(dists):.2f}")
    print(f"    matched-dist buckets: {b_}")
    # show the >5 ppm matched pairs (genuine peak disagreements)
    big = [(round(m[3],2), round(m[4],2), round(m[6],2)) for m in stage1 if m[6] > 5]
    if big:
        print(f"    LARGE-disparity matched pairs (ann_shift, jrn_shift, dist): {big[:8]}")

print("\nAggregate over the 8 worst molecules:")
print(f"  total Stage-I matched pairs: {n_stage1_all}")
print(f"  distance buckets: {buckets_all}")
frac_close = (buckets_all['<0.5'] + buckets_all['0.5-1']) / max(1, n_stage1_all)
print(f"  fraction of matched pairs within 1 ppm: {frac_close:.1%}")
print("  => if this fraction is high, the low similarity is driven by MISSING peaks (count),")
print("     not by shared peaks being at different positions.")
