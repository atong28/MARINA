#!/usr/bin/env python3
"""
Stage 60 - build the deployed fingerprint vocabulary + rankingset.

For config.FP_TYPE (default RankingEntropyMultiplicityUncapped) this builds the
uncapped thermometer-multiplicity fingerprint that eval_comprehensive.py and
augment_rankingset.py consume, writing under DATA_DATASET/<FP_TYPE>/:

    bitinfo_to_idx.pkl   feature-key (frag_smiles, k) -> column index
    rankingset.pt        torch.sparse_csr over the retrieval set, rows L2-normalized

It then writes the training-path FragIdx parquets (DATASET_ROOT/arrow/<split>/<loader
FRAGIDX_FILENAME>, e.g. FragIdxMultiplicityUncapped.parquet): per-molecule thermometer
columns so the model's build_mfp(idx) trains on this vocabulary (skip with --no_fragidx).

Design (matches analysis/sfp-report/scripts/build_thermo_fp.py, and produces the same
artifacts as the deployed MultiplicityUncappedEntropyFPLoader on the
substructure-fingerprint branch):
  * A feature is a radius-0..R circular fragment SMILES (Chem.MolFragmentToSmiles,
    isomericSmiles=False), enumerated over a stereo-stripped canonical SMILES.
  * Multiplicity is thermometer-encoded: (frag, k) fires iff the molecule holds
    `frag` with occurrence count >= k. "Uncapped" = the ladder is not truncated at a
    fixed MULTIPLICITY_CAP, so every k up to a fragment's max corpus count is a
    candidate rung.
  * From all candidate rungs we keep the top --out_dim by binary entropy over the
    retrieval corpus -- the same selection criterion (no min-count floor) as
    EntropyFPLoader.setup.

The vocab->rankingset conversion reuses fp_loader / fp_utils: the selected map is
handed to the multiplicity loader's _build_rankingset, which calls the shared
build_rankingset_csr with feature_kind = the loader's MULTIPLICITY_UNCAPPED vocabulary.

    DATASET_ROOT=/workspace pixi run python3 \
        scripts/marina_db/build/60_fp_vocab.py --num_procs 16
"""
import argparse
import multiprocessing as mp
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))
sys.path.insert(0, str(_HERE.parents[3]))

from config import DATA_DATASET, RETRIEVAL_PKL, FP_TYPE, FP_OUT_DIM, FP_RADIUS
from src.modules.data.fp_utils import (
    compute_entropy,
    select_topk_by_entropy,
    count_fragment_occurrences,
    load_smiles_index,
    build_fragidx_parquets,
)
from src.modules.data.fp_loader import FP_LOADERS

_G_RADIUS = None


def _init_hist(radius):
    global _G_RADIUS
    _G_RADIUS = radius


def _chunk_hist(smis):
    """{frag: {count: n_molecules}} for a chunk of SMILES."""
    hist = defaultdict(Counter)
    for smi in smis:
        try:
            occ = count_fragment_occurrences(smi, _G_RADIUS)
        except Exception:
            continue
        for frag, cnt in occ.items():
            hist[frag][cnt] += 1
    return dict(hist)


def build_histogram(smiles_list, radius, num_procs):
    """frag -> Counter{occurrence_count: n_molecules} over the retrieval set."""
    procs = mp.cpu_count() if not num_procs else max(1, int(num_procs))
    csz = 500
    chunks = [smiles_list[i:i + csz] for i in range(0, len(smiles_list), csz)]
    hist = defaultdict(Counter)
    t0 = time.time()
    if procs == 1:
        _init_hist(radius)
        for j, part in enumerate(map(_chunk_hist, chunks)):
            for frag, h in part.items():
                hist[frag].update(h)
    else:
        with mp.Pool(procs, initializer=_init_hist, initargs=(radius,)) as pool:
            for j, part in enumerate(pool.imap_unordered(_chunk_hist, chunks)):
                for frag, h in part.items():
                    hist[frag].update(h)
                if (j + 1) % 100 == 0:
                    print(f"  ~{(j + 1) * csz}/{len(smiles_list)} mols  "
                          f"{time.time() - t0:.0f}s  distinct={len(hist)}", flush=True)
    print(f"enumerated in {time.time() - t0:.0f}s; distinct fragments={len(hist)}",
          flush=True)
    return hist


def select_vocab(hist, retrieval_size, out_dim):
    """Uncapped thermometer candidates -> top-`out_dim` by binary entropy.

    Every rung (frag, k) for k=1..max-corpus-count is a candidate (no min-count
    floor); its presence count is the number of molecules with occurrence(frag) >= k.
    Selection is entropy-only, matching EntropyFPLoader.setup.
    """
    candidates = []          # (frag, k)
    counts = []              # n_molecules with count >= k
    for frag, h in hist.items():
        maxk = max(h)
        suffix = 0
        ge = {}
        for k in range(maxk, 0, -1):     # molecules with count >= k, top-down
            suffix += h.get(k, 0)
            ge[k] = suffix
        for k in range(1, maxk + 1):     # uncapped ladder, every rung is a candidate
            candidates.append((frag, k))
            counts.append(ge[k])

    if not candidates:
        raise RuntimeError("No thermometer candidates found over the retrieval set.")

    counts = np.asarray(counts)
    ent = compute_entropy(counts, total_dataset_size=retrieval_size)

    k = len(candidates) if (out_dim == "inf" or out_dim == float("inf")) else int(out_dim)
    k = min(k, len(candidates))
    topk_sorted = select_topk_by_entropy(ent, candidates, k)

    mapping = {candidates[i]: j for j, i in enumerate(topk_sorted)}
    n_mult = sum(1 for (_, kk) in mapping if kk >= 2)
    print(f"candidate rungs={len(candidates)}  selected={len(mapping)}  "
          f"multiplicity(k>=2)={n_mult} ({100 * n_mult / len(mapping):.1f}%)", flush=True)
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp_type", default=FP_TYPE)
    ap.add_argument("--out_dim", type=int, default=FP_OUT_DIM)
    ap.add_argument("--radius", type=int, default=FP_RADIUS)
    ap.add_argument("--retrieval", default=str(RETRIEVAL_PKL))
    ap.add_argument("--out_dir", default=None,
                    help="defaults to DATA_DATASET/<fp_type>")
    ap.add_argument("--num_procs", type=int, default=0, help="0 = all cores")
    ap.add_argument("--no_fragidx", action="store_true",
                    help="skip building the training-path FragIdx parquets")
    a = ap.parse_args()

    out_dir = Path(a.out_dir) if a.out_dir else Path(DATA_DATASET) / a.fp_type
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{a.fp_type}] retrieval={a.retrieval}  radius={a.radius}  "
          f"out_dim={a.out_dim}  out_dir={out_dir}", flush=True)

    smiles_map = load_smiles_index(a.retrieval)
    smiles_list = list(smiles_map.values())
    retrieval_size = len(smiles_map)
    print(f"[{a.fp_type}] retrieval molecules: {retrieval_size}", flush=True)

    hist = build_histogram(smiles_list, a.radius, a.num_procs)
    mapping = select_vocab(hist, retrieval_size, a.out_dim)

    # Hand the selected vocab to the loader and reuse its CSR writer, which emits
    # both bitinfo_to_idx.pkl and rankingset.pt under out_dir. The loader class is
    # resolved from fp_type (default RankingEntropyMultiplicityUncapped ->
    # MultiplicityUncappedEntropyFPLoader); its FEATURE_KIND drives the CSR's
    # extract_features so the stored (frag, k) columns match the selected vocab.
    loader_class = FP_LOADERS.get(a.fp_type)
    if loader_class is None:
        raise SystemExit(f"[{a.fp_type}] unknown fp_type; FP_LOADERS keys: {list(FP_LOADERS)}")
    loader = loader_class(
        dataset_root=str(out_dir.parent), retrieval_path=a.retrieval)
    loader.bitinfo_to_fp_index_map = mapping
    loader.fp_index_to_bitinfo_map = {v: k for k, v in mapping.items()}
    loader.max_radius = int(a.radius)
    loader.out_dim = len(mapping)
    csr = loader._build_rankingset(out_dir.name, num_procs=a.num_procs)
    print(f"[{a.fp_type}] wrote {out_dir/'bitinfo_to_idx.pkl'} ({len(mapping)} features)",
          flush=True)
    print(f"[{a.fp_type}] wrote {out_dir/'rankingset.pt'} "
          f"({csr.shape[0]} x {csr.shape[1]})", flush=True)

    # Training-path FragIdx: per-molecule thermometer columns for build_mfp(idx), written to
    # the shared arrow tree so the model trains on this vocabulary. Reads the built dataset index.
    dataset_root = out_dir.parent
    index_path = dataset_root / "index.pkl"
    if a.no_fragidx:
        print(f"[{a.fp_type}] skipping FragIdx parquets (--no_fragidx)", flush=True)
    elif not index_path.exists():
        print(f"[{a.fp_type}] WARNING: {index_path} missing; skipping FragIdx parquets "
              f"(run stage 40 first, or pass --no_fragidx)", flush=True)
    else:
        # General FragIdx writer with the loader's multiplicity vocabulary: extract_features
        # keyed on MULTIPLICITY_UNCAPPED yields the same (frag, k) presence keys `mapping`
        # holds, and the loader's FRAGIDX_FILENAME namespaces the shard so build_mfp(idx)
        # reads exactly these columns.
        build_fragidx_parquets(
            index_path=str(index_path),
            out_dir=str(dataset_root),
            bitinfo_to_col=mapping,
            radius=int(a.radius),
            num_procs=a.num_procs,
            feature_kind=loader.FEATURE_KIND,
            filename=loader.FRAGIDX_FILENAME,
        )
        print(f"[{a.fp_type}] wrote multiplicity FragIdx parquets "
              f"({loader.FRAGIDX_FILENAME}) under {dataset_root/'arrow'}", flush=True)


if __name__ == "__main__":
    main()
