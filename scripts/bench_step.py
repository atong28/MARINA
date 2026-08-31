"""Per-step cost attribution for the MARINA-DB slowdown.

Drives the real MARINA model with synthetic batches whose per-modality sequence lengths are
sampled from the true train-split distributions (exported from the parquet 'shape' columns),
padded to the batch max exactly as the collate does. Times forward+backward under fp16 autocast
with an optimizer step every `accum` micro-steps, mirroring --precision 16-mixed / accum 4.

One variable changes per config, so the ms/step deltas attribute directly.
"""
import json, time, sys, argparse
import numpy as np
import torch

sys.path.insert(0, "/code")
from src.modules.marina.args import MARINAArgs
from src.modules.marina.model import MARINA
from src.modules.core.const import FORMULA_ELEMENTS

D_IN = {"hsqc": 3, "c_nmr": 1, "h_nmr": 1, "mass_spec": 2, "mass_spec_neg": 2, "mw": 1}
PARQ = {"hsqc": "HSQC_NMR", "c_nmr": "C_NMR", "h_nmr": "H_NMR",
        "mass_spec": "MassSpec", "mass_spec_neg": "MassSpecNeg"}

def make_batch(mods, lens, B, rng, device):
    batch = {}
    for m in mods:
        if m == "mw":
            batch["mw"] = torch.randn(B, 1, 1, device=device) * 100 + 400
            continue
        if m == "formula":
            f = torch.zeros(B, 1, len(FORMULA_ELEMENTS), device=device)
            f[:, 0, :6] = torch.randint(1, 40, (B, 6), device=device).float()
            batch["formula"] = f
            continue
        pool = lens[PARQ[m]]
        Ls = rng.choice(pool, B)
        Lmax = int(Ls.max())
        x = torch.zeros(B, Lmax, D_IN[m], device=device)
        for i, L in enumerate(Ls):
            x[i, :int(L)] = torch.randn(int(L), D_IN[m], device=device).abs() + 0.5
        batch[m] = x
    return batch

def bench(name, lens, mods, layers, lam, B=32, accum=4, warmup=12, iters=48, device="cuda"):
    args = MARINAArgs()
    args.input_types = list(mods)
    args.layers = layers
    args.lambda_hybrid = lam
    args.batch_size = B
    model = MARINA(args, fp_loader=None).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    scaler = torch.amp.GradScaler("cuda")
    rng = np.random.RandomState(0)
    nparam = sum(p.numel() for p in model.parameters())

    def one(i):
        batch = make_batch(mods, lens, B, rng, device)
        fps = (torch.rand(B, args.out_dim, device=device) < 0.006).float()
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model.loss(model(batch), fps) / accum
        scaler.scale(loss).backward()
        if (i + 1) % accum == 0:
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)

    for i in range(warmup):
        one(i)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(iters):
        one(i)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1000
    del model, opt
    torch.cuda.empty_cache()
    return {"config": name, "ms_per_step": round(ms, 2), "layers": layers,
            "modalities": len(mods), "lambda": lam, "params_M": round(nparam / 1e6, 1)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqlens", default="/root/gurusmart/seqlens.json")
    a = ap.parse_args()
    SL = json.load(open(a.seqlens))
    m1, db = SL["MARINA1"], SL["MARINA-DB"]
    OLD = ["hsqc", "c_nmr", "h_nmr", "mass_spec", "mw"]

    configs = [
        ("1 old (MARINA1, 5mod, L16, lam0)",        m1, OLD,                                    16, 0.0),
        ("2 + MARINA-DB lengths (denser MS)",       db, OLD,                                    16, 0.0),
        ("3 + mass_spec_neg",                       db, OLD + ["mass_spec_neg"],                16, 0.0),
        ("4 + formula",                             db, OLD + ["mass_spec_neg", "formula"],     16, 0.0),
        ("5 + lambda_hybrid 0.5",                   db, OLD + ["mass_spec_neg", "formula"],     16, 0.5),
        ("6 + layers 16->8  == DEPLOYED",           db, OLD + ["mass_spec_neg", "formula"],      8, 0.5),
        ("7 control: old config at layers=8",       m1, OLD,                                     8, 0.0),
    ]
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    rows = []
    for name, lens, mods, layers, lam in configs:
        r = bench(name, lens, mods, layers, lam)
        rows.append(r)
        print(f"{r['config']:42s}  {r['ms_per_step']:7.2f} ms/step   "
              f"L={r['layers']:2d} mods={r['modalities']} lam={r['lambda']} params={r['params_M']}M", flush=True)
    base = rows[0]["ms_per_step"]
    print("\ncumulative vs config 1:")
    for r in rows:
        print(f"  {r['config']:42s}  {r['ms_per_step']/base:5.2f}x")
    print("\nincremental (each step vs the one above):")
    for prev, cur in zip(rows, rows[1:]):
        print(f"  {cur['config']:42s}  {cur['ms_per_step']/prev['ms_per_step']:5.2f}x")
    json.dump(rows, open("/root/gurusmart/bench_step_results.json", "w"), indent=2)
