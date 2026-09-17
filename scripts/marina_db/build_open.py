#!/usr/bin/env python3
"""build_open.py -- derive MARINA-DB-OPEN from a built MARINA-DB.

MARINA-DB-OPEN is MARINA-DB with every NMR modality (HSQC, 13C, 1H) taken from the Mnova
simulations ONLY -- the SPECTRE corpus (JEOL CH-NMR-NP experimental HSQC + ACD/Labs-predicted
HSQC, CASPRE/PROSPERE 1D) is never consulted. MS/MS is untouched (same SOURCE_PRIORITY as
MARINA-DB). Purpose: measure what the JEOL/ACD data contribute; if nothing, the Mnova-only
dataset is fully releasable.

Everything except the NMR spectra is copied verbatim from the built MARINA-DB, so the two
datasets share idx, SMILES, splits, retrieval bank, every fingerprint vocab/rankingset and
every FragIdx parquet -- results on the two are directly comparable on the identical test set.
Only index.pkl `has_hsqc/has_c_nmr/has_h_nmr` flags and arrow/<split>/{HSQC_NMR,C_NMR,H_NMR}
differ. Molecules with no Mnova NMR keep their idx/split with the flags cleared; MARINADataset
drops any entry with no present spectral modality.

Inputs: config.DATA_DATASET (built MARINA-DB), config.DATA_CLEANED/mapping.json (stage-6 per-source
spectral cache). Output: --out (default ../Datasets/MARINA-DB-OPEN), then stage 11 collapse,
stage 13 pack. Run 12_verify.py --dataset_root <out> afterwards.

    pixi run python scripts/marina_db/build_open.py
"""
import argparse
import importlib
import json
import pickle
import shutil
import sys
from pathlib import Path

from tqdm import tqdm

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))                        # config
sys.path.insert(0, str(_HERE.parent / "build"))              # stage modules
sys.path.insert(0, str(_HERE.parents[1] / "dataset"))        # pack_arrow
sys.path.insert(0, str(_HERE.parents[2]))                    # repo root
from config import DATA_DATASET, DATA_CLEANED, MARINA_DATA_ROOT, SOURCE_PRIORITY

assemble = importlib.import_module("8_assemble_arrow")
collapse = importlib.import_module("11_collapse_peaks")
from pack_arrow import pack_root

SPLITS = ("train", "val", "test")
NMR = ("hsqc", "c_nmr", "h_nmr")
OPEN_PRIORITY = {**SOURCE_PRIORITY, **{m: ["mnova"] for m in NMR}}
FP_DIRS = ("RankingEntropy", "RankingEntropyMultiplicity", "RankingEntropyMultiplicityUncapped",
           "RankingEntropySubstructure", "RankingEntropyUniqueMultiplicity")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=str(DATA_DATASET), help="built MARINA-DB root")
    ap.add_argument("--out", default=str(MARINA_DATA_ROOT / "Datasets" / "MARINA-DB-OPEN"))
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    work = out.parent / (out.name + ".work")
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)

    index = pickle.load(open(src / "index.pkl", "rb"))
    print(f"MARINA-DB index: {len(index)} molecules", flush=True)
    mapping = json.load(open(DATA_CLEANED / "mapping.json"))
    print(f"mapping: {len(mapping)} SMILES", flush=True)

    handles = {s: open(work / f"{s}.jsonl", "w", encoding="utf-8") for s in SPLITS}
    changed = {m: 0 for m in NMR}
    for e in tqdm(index.values(), desc="Materializing Mnova-only jsonl"):
        m = mapping.get(e["smiles"], {})
        row = {"idx": e["idx"], "smiles": e["smiles"]}
        for mod, order in OPEN_PRIORITY.items():
            row[mod] = assemble._priority_access(m.get(mod, {}), order)
        for mod in NMR:
            has = row[mod] != []
            changed[mod] += int(e[f"has_{mod}"] != has)
            e[f"has_{mod}"] = has
        handles[e["split"]].write(json.dumps(row) + "\n")
    for h in handles.values():
        h.close()
    print(f"has_* flags changed vs MARINA-DB: {changed}", flush=True)
    pickle.dump(index, open(work / "index.pkl", "wb"))

    # stage 8 arrow conversion (copies index.pkl from work; spectral parquets from the jsonl)
    assemble.convert_to_arrow(str(work), str(out))

    # everything structural is shared with MARINA-DB byte-for-byte
    for f in ("retrieval.pkl", "metadata.json"):
        shutil.copy2(src / f, out / f)
    for d in FP_DIRS:
        if (out / d).exists():
            shutil.rmtree(out / d)
        shutil.copytree(src / d, out / d)
    for sp in SPLITS:
        for p in sorted((src / "arrow" / sp).glob("FragIdx*.parquet")):
            shutil.copy2(p, out / "arrow" / sp / p.name)
    print("copied retrieval/metadata/FP dirs/FragIdx parquets from MARINA-DB", flush=True)

    collapse.collapse_dataset(out)      # stage 11 (dataset only; journal untouched)
    pack_root(str(out), force=True)     # stage 13
    shutil.rmtree(work)
    print(f"MARINA-DB-OPEN built at {out}; now run 12_verify.py --dataset_root {out}", flush=True)


if __name__ == "__main__":
    main()
