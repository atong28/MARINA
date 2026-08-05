"""
Peak-aware spectral-source similarity across all three MARINA benchmark sources
(Annotated / Journal / Simulated) for 3 modalities:
  - 13C   : NMRPeak CNMRSimilarityScorer (scale=5, tol=20)
  - 1H    : shift-only, CNMRSimilarityScorer with 1H params (scale=1, tol=2)
  - HSQC  : prototype 2D scorer on (13C, 1H) cross-peaks
Isolated analysis; reads benchmark pkls read-only.
"""
import os, pickle, csv, statistics as st
from spectrum_similarity_scorer import CNMRSimilarityScorer
from hsqc_similarity import HSQC2DSimilarityScorer

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("MARINA_DATA_ROOT", "/home/user/atong")

PATHS = {
    "annotated": os.path.join(DATA_ROOT, "Datasets/MARINA1/benchmark.pkl"),
    "journal":   os.path.join(DATA_ROOT, "Benchmark/benchmark-journal.pkl"),
    "sim":       os.path.join(DATA_ROOT, "Benchmark/benchmark-sim.pkl"),
}
def load(p):
    with open(p, "rb") as f: return pickle.load(f)

def shifts1d(entry, mod):
    t = entry.get("input", {}).get(mod, None)
    return None if t is None else [float(x) for x in t.flatten().tolist()]

def hsqc_pts(entry):
    t = entry.get("input", {}).get("hsqc", None)
    if t is None: return None
    # (13C, 1H, phase-sign); only the SIGN of col2 is meaningful (CH/CH3=+ , CH2=-)
    return [(float(r[0]), float(r[1]), 1 if float(r[2]) >= 0 else -1) for r in t.tolist()]

def main():
    ann = {v["npid"]: v for v in load(PATHS["annotated"]).values()}
    jrn = load(PATHS["journal"]); sim = load(PATHS["sim"])
    common = sorted(set(ann) & set(jrn) & set(sim))
    print(f"Common NPIDs: {len(common)}")

    c_scorer = CNMRSimilarityScorer(scale=5.0, tolerance=20.0)
    h_scorer = CNMRSimilarityScorer(scale=1.0, tolerance=2.0)   # shift-only 1H
    q_scorer  = HSQC2DSimilarityScorer(scale_c=5.0, scale_h=1.0, tol_c=20.0, tol_h=2.0, use_phase=True)
    q0_scorer = HSQC2DSimilarityScorer(scale_c=5.0, scale_h=1.0, tol_c=20.0, tol_h=2.0, use_phase=False)

    pairs = [("ann_vs_sim", ann, sim), ("jrn_vs_sim", jrn, sim), ("ann_vs_jrn", ann, jrn)]
    rows = []
    for npid in common:
        r = {"npid": npid, "split": ann[npid].get("split", "?")}
        feats = {}
        for tag, src in [("ann", ann), ("jrn", jrn), ("sim", sim)]:
            feats[tag] = {"c": shifts1d(src[npid], "c_nmr"),
                          "h": shifts1d(src[npid], "h_nmr"),
                          "q": hsqc_pts(src[npid])}
        for label, dA, dB in pairs:
            a, b = label[:3], label.split("_")[-1][:3]
            for mod, scorer in [("c", c_scorer), ("h", h_scorer),
                                ("q", q_scorer), ("q0", q0_scorer)]:
                A, B = feats[a][mod.rstrip("0")], feats[b][mod.rstrip("0")]
                r[f"{label}_{mod}"] = scorer.calculate_similarity_score(A, B)["similarity"] if A and B else None
        rows.append(r)

    # CSV
    fields = ["npid", "split"] + [f"{p}_{m}" for p in ("ann_vs_sim","jrn_vs_sim","ann_vs_jrn") for m in ("c","h","q","q0")]
    csv_path = os.path.join(ROOT, "all_modalities_per_molecule.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows: w.writerow({k: r.get(k) for k in fields})

    def summ(key, split=None):
        v = [r[key] for r in rows if r.get(key) is not None and (split is None or r["split"] == split)]
        return None if not v else (len(v), st.mean(v), st.pstdev(v), st.median(v), min(v))

    modname = {"c": "13C", "h": "1H (shift-only)",
               "q": "HSQC 2D (phase-aware)", "q0": "HSQC 2D (shift-only)"}
    print("\n" + "="*72)
    print("PEAK-AWARE SIMILARITY BY MODALITY AND SOURCE PAIR  (mean / median / std / min)")
    print("="*72)
    for mod in ("c", "h", "q", "q0"):
        print(f"\n### {modname[mod]}  (ALL n=152)")
        print(f"{'pair':<12}{'mean':>8}{'median':>8}{'std':>7}{'min':>7}")
        for label, _, _ in pairs:
            s = summ(f"{label}_{mod}")
            if s: print(f"{label:<12}{s[1]:>8.3f}{s[3]:>8.3f}{s[2]:>7.3f}{s[4]:>7.3f}")

    # compact test-split cross-modal table (mirrors model eval which reports test split)
    print("\n" + "="*72)
    print("TEST SPLIT — mean similarity, all modalities  (rows=source pair)")
    print("="*72)
    print(f"{'pair':<12}{'13C':>8}{'1H':>8}{'HSQC':>10}{'HSQC-noph':>11}")
    for label, _, _ in pairs:
        c = summ(f"{label}_c", 'test'); h = summ(f"{label}_h", 'test')
        q = summ(f"{label}_q", 'test'); q0 = summ(f"{label}_q0", 'test')
        print(f"{label:<12}{c[1]:>8.3f}{h[1]:>8.3f}{q[1]:>10.3f}{q0[1]:>11.3f}")
    print(f"\nWrote {csv_path}")

if __name__ == "__main__":
    main()
