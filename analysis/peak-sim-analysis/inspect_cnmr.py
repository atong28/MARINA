"""Inspect c_nmr / h_nmr format and NPID overlap across the three benchmarks."""
import os, pickle

DATA_ROOT = os.environ.get("MARINA_DATA_ROOT", "/home/user/atong")

def load(p):
    with open(p, "rb") as f:
        return pickle.load(f)

ann = load(os.path.join(DATA_ROOT, "Datasets/MARINA1/benchmark.pkl"))
jrn = load(os.path.join(DATA_ROOT, "Benchmark/benchmark-journal.pkl"))
sim = load(os.path.join(DATA_ROOT, "Benchmark/benchmark-sim.pkl"))

# Re-key annotated by npid
ann_by_npid = {v["npid"]: v for v in ann.values()}

print("counts:", "ann", len(ann_by_npid), "jrn", len(jrn), "sim", len(sim))
A, J, S = set(ann_by_npid), set(jrn), set(sim)
print("NPID overlap all three:", len(A & J & S))
print("ann∩sim:", len(A & S), "ann∩jrn:", len(A & J), "jrn∩sim:", len(J & S))
print("in ann not jrn:", sorted(A - J)[:10])

# Look at one common NPID's c_nmr / h_nmr across the three
common = sorted(A & J & S)
npid = common[0]
print(f"\n=== example NPID {npid} ===")
for name, d in [("ann", ann_by_npid), ("jrn", jrn), ("sim", sim)]:
    inp = d[npid]["input"]
    print(f"\n[{name}] input keys: {list(inp)}")
    for mod in ("c_nmr", "h_nmr"):
        if mod in inp and inp[mod] is not None:
            t = inp[mod]
            shape = getattr(t, "shape", None)
            try:
                vals = t.flatten().tolist()
            except Exception:
                vals = t
            print(f"  {mod}: type={type(t).__name__} shape={shape} n={len(vals)}")
            print(f"    vals={[round(x,2) for x in vals][:40]}")
        else:
            print(f"  {mod}: MISSING/None")
    # hsqc shape only
    if "hsqc" in inp and inp["hsqc"] is not None:
        print(f"  hsqc: shape={getattr(inp['hsqc'],'shape',None)}")
