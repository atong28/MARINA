"""Exp 3 — MCES *within* exact-FP collision buckets ("far collisions").

Question this answers: when two molecules share an IDENTICAL fingerprint, how
structurally different can they actually be? Is same-FP => near-isomorphic (tight
fibers, safe to enumerate the neighborhood of an FP match), or can a same-FP pair
be a scaffold hop (a "far collision" — the failure mode for Moonshot's
enumerate-around-an-exact-match strategy)?

exp1 measures MCES on random/mass-stratified pairs (almost all non-colliding);
exp2 finds the collision buckets but scores badness by mass span as a proxy. This
joins them: it computes graph MCES *inside* each exact-FP tie group, giving the
conditional distribution P(MCES similarity | same FP) and, crucially, its low tail.

Cheap by construction: MCS only ever runs inside tie groups (sum of k-choose-2),
never O(N^2). Embarrassingly parallel across pairs -> --num-shards/--shard-id for
Nautilus fan-out (same deterministic striding as exp1). Each shard writes a partial
parquet to concat later.

Metric: rdFMCS maximum-common-substructure, reported as bond-Tanimoto
sim = commonBonds / (bondsA + bondsB - commonBonds) in [0,1]; LOW sim == far
collision. We do NOT use RascalMCES here: verified that RASCAL silently returns
similarity 0.0 on large natural-product molecules (which dominate MARINA's
collision buckets) — e.g. a taxane benzoate/acetate positional-isomer pair that
rdFMCS scores 0.933, RASCAL scores 0.000. RASCAL is fine for exp1's random
(mostly dissimilar) pairs but produces an all-artifact "far collision" tail inside
same-FP buckets, where the molecules are actually similar. rdFMCS is slower (~1-100s
per large pair) hence the Nautilus fan-out. rdFMCS's own timeout is respected; on
timeout it returns a PARTIAL MCS -> sim is a LOWER bound, flagged in `partial`.

Reuses exp1_mces's NoDaemonPool + per-pair subprocess hard-timeout harness (a
crash/overrun can't wedge the pool) but swaps the compute to rdFMCS.

Deps: torch (read CSR) + rdkit (rdFMCS/mass) + pandas + pyarrow. Run under the
~/Workspace master pixi env locally, or the Nautilus MARINA image on the cluster.
Read-only on the retrieval set.

Usage:
    python exp3_collision_mces.py --retrieval /path/retrieval.pkl \
        --fp /path/RankingEntropyUniqueMultiplicity/rankingset.pt \
        --workers 16 --timeout 60 --max-pairs-per-group 500 --seed 0 \
        --num-shards 1 --shard-id 0 --out results/exp3_collision_mces.parquet

    # buckets-only dry run (no MCES) to size the job before fanning out:
    python exp3_collision_mces.py --retrieval ... --fp ... --dry-run --out /tmp/x.parquet
"""
import argparse, json, os, pickle, time
from collections import defaultdict

import numpy as np
import pandas as pd

# Reuse only the robust plumbing from exp1 (non-daemonic pool so workers can fork a
# per-pair subprocess, and the single-SMILES mass helper). The MCS compute below is
# rdFMCS, not RASCAL — see module docstring for why RASCAL is unusable here.
from exp1_mces import NoDaemonPool, _mass_one

# --- rdFMCS worker (subprocess-isolated hard timeout, mirroring exp1's harness) ---
_SMILES = None
_TIMEOUT = 30
_CHEM = None
_FMCS = None


def _init(smiles, timeout):
    global _SMILES, _TIMEOUT, _CHEM, _FMCS
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFMCS
    RDLogger.DisableLog("rdApp.*")
    _SMILES, _TIMEOUT, _CHEM, _FMCS = smiles, timeout, Chem, rdFMCS


def _fmcs_compute(i, j, q):
    """rdFMCS bond-Tanimoto in a throwaway subprocess so a hang/crash can't wedge the pool."""
    m1 = _CHEM.MolFromSmiles(_SMILES[i])
    m2 = _CHEM.MolFromSmiles(_SMILES[j])
    if m1 is None or m2 is None:
        q.put((i, j, float("nan"), False)); return
    nb1, nb2 = m1.GetNumBonds(), m2.GetNumBonds()
    if nb1 == 0 or nb2 == 0:
        # MCS over edges is undefined for a bondless molecule; sim by shared heavy atom
        q.put((i, j, float("nan"), False)); return
    res = _FMCS.FindMCS(
        [m1, m2], timeout=_TIMEOUT,
        atomCompare=_FMCS.AtomCompare.CompareElements,
        bondCompare=_FMCS.BondCompare.CompareOrderExact,
        ringMatchesRingOnly=True, completeRingsOnly=True,
    )
    common = res.numBonds
    sim = common / (nb1 + nb2 - common) if (nb1 + nb2 - common) else 1.0
    q.put((i, j, float(sim), bool(res.canceled)))   # canceled -> partial MCS, sim is a lower bound


def _fmcs_pair(ij):
    import queue as _queue
    from multiprocessing import Process, Queue
    i, j = ij
    q = Queue()
    p = Process(target=_fmcs_compute, args=(i, j, q))
    p.start()
    try:
        res = q.get(timeout=_TIMEOUT + 30)
    except _queue.Empty:
        res = (i, j, np.nan, True)     # hard hang/segfault -> record NaN/partial
    if p.is_alive():
        p.terminate()
    p.join()
    return res


def load_buckets(path):
    """Group retrieval rows by identical nonzero column-set. Returns
    (N, D, distinct, ties, n_zero) where `ties` is a list of member-index lists
    (each len>=2), excluding the trivial all-zero (no on-bit) group."""
    import torch
    csr = torch.load(path, weights_only=True)
    crow = csr.crow_indices().numpy()
    col = csr.col_indices().numpy()
    N, D = int(csr.shape[0]), int(csr.shape[1])
    groups = defaultdict(list)
    for i in range(N):
        groups[col[crow[i]:crow[i + 1]].tobytes()].append(i)
    n_zero = len(groups.get(b"", []))
    ties = [v for k, v in groups.items() if len(v) > 1 and k != b""]
    return N, D, len(groups), ties, n_zero


def group_pairs(ties, max_pairs_per_group, rng):
    """Within-group unordered pairs, capped per group. For groups whose full pair
    count exceeds the cap we sample pairs directly (no materializing k-choose-2)."""
    pairs, gids = [], []
    for gid, members in enumerate(ties):
        m = sorted(members)
        k = len(m)
        total = k * (k - 1) // 2
        if max_pairs_per_group and total > max_pairs_per_group:
            seen = set()
            while len(seen) < max_pairs_per_group:
                a, b = rng.choice(k, 2, replace=False)
                seen.add((int(min(a, b)), int(max(a, b))))
            gp = [(m[a], m[b]) for a, b in sorted(seen)]
        else:
            gp = [(m[a], m[b]) for a in range(k) for b in range(a + 1, k)]
        pairs.extend(gp)
        gids.extend([gid] * len(gp))
    return pairs, gids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", required=True)
    ap.add_argument("--fp", required=True, help="/path/rankingset.pt (unique-multiplicity)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=30, help="per-pair rdFMCS soft timeout (s)")
    ap.add_argument("--max-pairs-per-group", type=int, default=500,
                    help="cap within-group pairs (0 = all); big symmetric groups otherwise "
                         "dominate; sampling keeps the tail estimate unbiased per group")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1,
                    help="split the pair list into this many independent jobs (MCES is "
                         "embarrassingly parallel); each writes a partial parquet")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true",
                    help="report bucket/pair counts and exit (no MCES)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    N, D, distinct, ties, n_zero = load_buckets(args.fp)
    n_in_tie = sum(len(v) for v in ties)
    largest = max((len(v) for v in ties), default=0)
    print(f"FP: N={N} D={D} distinct={distinct} tie_groups={len(ties)} "
          f"colliding={n_in_tie} ({100*n_in_tie/N:.2f}%) ceiling={100*(N-n_in_tie)/N:.2f}% "
          f"largest_group={largest} all_zero={n_zero}", flush=True)

    pairs, gids = group_pairs(ties, args.max_pairs_per_group, rng)
    # deterministic order so shard slices partition the same pair set across jobs
    order = sorted(range(len(pairs)), key=lambda k: pairs[k])
    pairs = [pairs[k] for k in order]
    gids = [gids[k] for k in order]
    print(f"within-group pairs to score: {len(pairs)} "
          f"(cap {args.max_pairs_per_group}/group)", flush=True)

    if args.dry_run:
        print("dry-run: sizing only, no MCES.", flush=True)
        return

    if args.num_shards > 1:
        sl = slice(args.shard_id, None, args.num_shards)
        pairs, gids = pairs[sl], gids[sl]
        print(f"shard {args.shard_id}/{args.num_shards}: {len(pairs)} pairs this job", flush=True)

    # SMILES aligned to CSR rows (retrieval.pkl int keys 0..N-1; dict or list per exp2)
    with open(args.retrieval, "rb") as f:
        R = pickle.load(f)
    smiles = [(R[i]["smiles"] if isinstance(R[i], dict) else R[i]) for i in range(N)]

    # masses only for touched rows (for the tail readout: is a far collision also a mass jump?)
    touched = sorted({i for p in pairs for i in p})
    with __import__("multiprocessing").Pool(args.workers) as mpool:
        mvals = mpool.map(_mass_one, [smiles[i] for i in touched], chunksize=1000)
    mass = {i: m for i, m in zip(touched, mvals)}

    # MCS ground truth (parallel, the expensive part) — rdFMCS in exp1's hardened pool
    t0 = time.time()
    pos = {p: k for k, p in enumerate(pairs)}
    mcs = np.full(len(pairs), np.nan)
    partial = np.zeros(len(pairs), dtype=bool)
    with NoDaemonPool(args.workers, initializer=_init, initargs=(smiles, args.timeout)) as pool:
        for k, (i, j, sim, pt) in enumerate(pool.imap_unordered(_fmcs_pair, pairs, chunksize=8)):
            mcs[pos[(i, j)]] = sim
            partial[pos[(i, j)]] = pt
            if k % 500 == 0:
                print(f"  mcs {k}/{len(pairs)}  {time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame({
        "gid": gids,
        "i": [p[0] for p in pairs], "j": [p[1] for p in pairs],
        "mcs_sim": mcs, "partial": partial,
        "smiles_i": [smiles[p[0]] for p in pairs],
        "smiles_j": [smiles[p[1]] for p in pairs],
        "mass_i": [mass[p[0]] for p in pairs],
        "mass_j": [mass[p[1]] for p in pairs],
    })
    df["mass_diff"] = (df["mass_i"] - df["mass_j"]).abs()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_parquet(args.out)

    valid = df["mcs_sim"].to_numpy()
    v = valid[np.isfinite(valid)]
    n_nan = int(np.isnan(valid).sum())
    n_part = int(partial.sum())
    print(f"\nMCS done in {time.time()-t0:.0f}s; scored {len(df)} same-FP pairs; "
          f"{n_nan} NaN; {n_part} partial (timed-out, sim is a lower bound)", flush=True)
    if v.size:
        # complete-MCS pairs only for the trustworthy tail (partial sim underestimates)
        comp = df[np.isfinite(df["mcs_sim"]) & (~df["partial"])]["mcs_sim"].to_numpy()
        qs = np.quantile(v, [0.0, 0.05, 0.25, 0.5])
        print("--- P(MCS similarity | same FP) — LOW tail == far collisions ---", flush=True)
        print(f"  all valid (n={v.size}): min={qs[0]:.3f} p5={qs[1]:.3f} "
              f"p25={qs[2]:.3f} median={qs[3]:.3f}", flush=True)
        for thr in (0.3, 0.5, 0.7, 0.9):
            print(f"  far collisions MCS<{thr:.1f}: all={float((v<thr).mean()):.3%}"
                  f"  complete-only={float((comp<thr).mean()) if comp.size else float('nan'):.3%}",
                  flush=True)
        summ = {
            "N": N, "D": D, "tie_groups": len(ties), "colliding": n_in_tie,
            "collision_pct": 100 * n_in_tie / N, "ceiling_pct": 100 * (N - n_in_tie) / N,
            "largest_group": largest, "pairs_scored": int(len(df)),
            "pairs_valid": int(v.size), "pairs_partial": n_part,
            "mcs_min": float(qs[0]), "mcs_p5": float(qs[1]),
            "mcs_p25": float(qs[2]), "mcs_median": float(qs[3]),
            "far_collision_frac_all": {f"<{t}": float((v < t).mean()) for t in (0.3, 0.5, 0.7, 0.9)},
            "far_collision_frac_complete": {f"<{t}": (float((comp < t).mean()) if comp.size else None)
                                            for t in (0.3, 0.5, 0.7, 0.9)},
        }
        with open(os.path.splitext(args.out)[0] + ".summary.json", "w") as f:
            json.dump(summ, f, indent=2)
        # print the 10 worst COMPLETE-MCS far collisions for eyeballing (partials excluded)
        worst = df[np.isfinite(df["mcs_sim"]) & (~df["partial"])].nsmallest(10, "mcs_sim")
        print("\n--- 10 lowest-MCS complete same-FP pairs (worst real far collisions) ---", flush=True)
        for _, r in worst.iterrows():
            print(f"  sim={r['mcs_sim']:.3f} dMass={r['mass_diff']:.1f}  "
                  f"{r['smiles_i']}  ||  {r['smiles_j']}", flush=True)
    print(f"\nsaved {len(df)} pairs -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
