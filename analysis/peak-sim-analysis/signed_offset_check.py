"""Task 1 refinement: for the worst ann-vs-jrn 13C molecules, are the ~1-2 ppm
matched-pair gaps a SYSTEMATIC signed offset (referencing/rounding -> same peaks,
shifted) or scattered (genuinely different peaks)? Report mean SIGNED diff and its
std over Stage-I matched pairs."""
import os, pickle, statistics as st
from spectrum_similarity_scorer import CNMRSimilarityScorer
DATA_ROOT = os.environ.get("MARINA_DATA_ROOT", "/home/user/atong")
def load(p):
    with open(p,"rb") as f: return pickle.load(f)
def cs(e):
    t=e.get("input",{}).get("c_nmr"); return None if t is None else [float(x) for x in t.flatten().tolist()]
ann={v["npid"]:v for v in load(os.path.join(DATA_ROOT,"Datasets/MARINA1/benchmark.pkl")).values()}
jrn=load(os.path.join(DATA_ROOT,"Benchmark/benchmark-journal.pkl"))
sc=CNMRSimilarityScorer(scale=5.0,tolerance=20.0)
targets=["NP0331384","NP0332307","NP0331385","NP0333455","NP0332323","NP0331377","NP0331325","NP0331323"]
print(f"{'npid':<12}{'n_ann':>6}{'n_jrn':>6}{'signed_mean':>12}{'signed_std':>11}  interpretation")
for npid in targets:
    a,b=cs(ann[npid]),cs(jrn[npid])
    res=sc.calculate_similarity_score(a,b)
    # signed diff = ann_shift - jrn_shift for Stage-I matched pairs within 5 ppm (exclude forced far pairs)
    diffs=[m[3]-m[4] for m in res["matches"] if m[0]==1 and m[6]<=5]
    if not diffs:
        print(f"{npid:<12}{len(a):>6}{len(b):>6}   (no close matches)"); continue
    mu=st.mean(diffs); sd=st.pstdev(diffs)
    interp = "SYSTEMATIC offset (same peaks, shifted)" if sd < 0.4 and abs(mu) > 0.5 \
             else "tight agreement" if abs(mu) < 0.3 and sd < 0.4 \
             else "scattered / mixed"
    print(f"{npid:<12}{len(a):>6}{len(b):>6}{mu:>12.2f}{sd:>11.2f}  {interp}")
