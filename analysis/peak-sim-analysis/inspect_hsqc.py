"""Inspect HSQC [N,3] column semantics + h_nmr across the three benchmarks."""
import os, pickle
DATA_ROOT = os.environ.get("MARINA_DATA_ROOT", "/home/user/atong")
def load(p):
    with open(p,"rb") as f: return pickle.load(f)
ann={v["npid"]:v for v in load(os.path.join(DATA_ROOT,"Datasets/MARINA1/benchmark.pkl")).values()}
jrn=load(os.path.join(DATA_ROOT,"Benchmark/benchmark-journal.pkl"))
sim=load(os.path.join(DATA_ROOT,"Benchmark/benchmark-sim.pkl"))

npid=sorted(set(ann)&set(jrn)&set(sim))[0]
print("NPID",npid)
for name,d in [("ann",ann),("jrn",jrn),("sim",sim)]:
    h=d[npid]["input"]["hsqc"]
    print(f"\n[{name}] hsqc shape={tuple(h.shape)}")
    rows=h.tolist()
    # column ranges
    import statistics
    for c in range(h.shape[1]):
        col=[r[c] for r in rows]
        print(f"  col{c}: min={min(col):.2f} max={max(col):.2f} distinct3={sorted(set(round(x,1) for x in col))[:6]}")
    print("  first 5 rows:", [[round(x,2) for x in r] for r in rows[:5]])
