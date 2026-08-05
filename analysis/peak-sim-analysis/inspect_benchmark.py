"""Probe the schema of the three MARINA benchmark pickles (read-only)."""
import os, pickle, sys

DATA_ROOT = os.environ.get("MARINA_DATA_ROOT", "/home/user/atong")

paths = {
    "annotated": os.path.join(DATA_ROOT, "Datasets/MARINA1/benchmark.pkl"),
    "journal":   os.path.join(DATA_ROOT, "Benchmark/benchmark-journal.pkl"),
    "sim":       os.path.join(DATA_ROOT, "Benchmark/benchmark-sim.pkl"),
}

def describe(obj, depth=0, maxdepth=3):
    pad = "  " * depth
    t = type(obj).__name__
    if isinstance(obj, dict):
        print(f"{pad}dict[{len(obj)}] keys sample: {list(obj)[:5]}")
        if obj and depth < maxdepth:
            k = next(iter(obj))
            print(f"{pad}-> value for key {k!r}:")
            describe(obj[k], depth+1, maxdepth)
    elif isinstance(obj, (list, tuple)):
        print(f"{pad}{t}[{len(obj)}]")
        if obj and depth < maxdepth:
            describe(obj[0], depth+1, maxdepth)
    else:
        r = repr(obj)
        if len(r) > 200: r = r[:200] + "..."
        print(f"{pad}{t}: {r}")

for name, p in paths.items():
    print(f"\n===== {name}: {p} =====")
    try:
        with open(p, "rb") as f:
            obj = pickle.load(f)
        describe(obj)
    except Exception as e:
        print(f"  LOAD FAILED: {type(e).__name__}: {e}")
