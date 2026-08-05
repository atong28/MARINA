"""Extract experimental 1D 13C/1H peak lists for MARINA1 molecules.

Sources, in precedence order:

  nmrshiftdb2-2024  per-ATOM assigned shifts from a curated database. Matches
                    MARINA1's convention directly -- MARINA1's c_nmr/h_nmr are
                    per-atom Mnova predictions, so equivalent nuclei appear as
                    repeated values in both.
  NMRexp            peak lists mined from ~200k SI documents. Per-PEAK, not
                    per-atom. 1H is recovered to per-proton via the integration
                    ("2H" -> two copies of the shift midpoint); 13C has no
                    integration, so equivalent carbons stay collapsed to one
                    entry. That asymmetry against MARINA1 is inherent to the
                    source and is recorded, not corrected.

Where both sources cover a molecule, nmrshiftdb2 wins. Within a source, the
spectrum with the most peaks wins (solvent is not conditioned on -- MARINA has no
solvent channel -- so multi-solvent duplicates are resolved by informativeness).

Writes results/nmr_experimental.pkl : {smiles: {"c_nmr": [...], "h_nmr": [...]}}
"""
import ast
import json
import os
import pickle
import re
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("MARINA_DATA_ROOT", "/home/user/atong"))
OUT = ROOT / "results"
DOMAIN_RAW = ROOT.parent / "domain-compare" / "raw"
NSDB = DOMAIN_RAW / "nmrshiftdb2/data/nmrshiftdb2_2024/mol_nmrshift_nmrshiftdb2_2024_all.pkl"
NMREXP = DOMAIN_RAW / "nmrexp/NMRexp_10to24_1_1004.parquet"
MARINA1_INDEX = DATA_ROOT / "Datasets/MARINA1/index.pkl"

INTEGRATION = re.compile(r"(\d+(?:\.\d+)?)\s*H", re.IGNORECASE)
# Physically implausible shifts are dropped rather than trusted; SI-mined text
# contains stray coupling constants and page numbers.
C_RANGE = (-30.0, 260.0)
H_RANGE = (-5.0, 20.0)


def canonicalize(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    once = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
    mol = Chem.MolFromSmiles(once)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)


def canon_map(raw, label):
    uniq = sorted({s for s in raw if isinstance(s, str) and s.strip()})
    print(f"[{label}] {len(raw):,} rows -> {len(uniq):,} unique raw, canonicalizing...", flush=True)
    with Pool(4) as pool:
        done = pool.map(canonicalize, uniq, chunksize=2000)
    return {r: c for r, c in zip(uniq, done) if c}


index = pickle.load(open(MARINA1_INDEX, "rb"))
targets = {v["smiles"] for v in index.values()}
print(f"MARINA1 target SMILES: {len(targets):,}", flush=True)

# nucleus -> {smiles: shift list}. Filled lowest-precedence first, then overwritten.
nmrexp_c, nmrexp_h = {}, {}
nsdb_c, nsdb_h = {}, {}
stats = defaultdict(int)

# --------------------------------------------------------------- nmrshiftdb2
nsdb = pickle.load(open(NSDB, "rb"))
nsdb_map = canon_map(nsdb["smiles"], "nmrshiftdb2")
for raw, target, mask in zip(nsdb["smiles"], nsdb["atom_target"], nsdb["atom_mask"]):
    canon = nsdb_map.get(raw)
    if canon is None or canon not in targets:
        continue
    c_shifts = [float(t) for t, m in zip(target, mask) if int(m) == 6]
    h_shifts = [float(t) for t, m in zip(target, mask) if int(m) == 1]
    c_shifts = [s for s in c_shifts if C_RANGE[0] <= s <= C_RANGE[1]]
    h_shifts = [s for s in h_shifts if H_RANGE[0] <= s <= H_RANGE[1]]
    if c_shifts and len(c_shifts) > len(nsdb_c.get(canon, [])):
        nsdb_c[canon] = c_shifts
    if h_shifts and len(h_shifts) > len(nsdb_h.get(canon, [])):
        nsdb_h[canon] = h_shifts
print(f"  nmrshiftdb2 in MARINA1: c_nmr {len(nsdb_c):,}, h_nmr {len(nsdb_h):,}", flush=True)

# -------------------------------------------------------------------- NMRexp
nex = pq.read_table(NMREXP, columns=["SMILES", "NMR_type", "NMR_processed"]).to_pandas()
nex = nex[nex.NMR_type.isin(["13C NMR", "1H NMR"])]
nex_map = canon_map(nex.SMILES.tolist(), "nmrexp")
nex["canon"] = nex.SMILES.map(nex_map)
nex = nex[nex.canon.notna() & nex.canon.isin(targets)]
print(f"  NMRexp rows landing in MARINA1: {len(nex):,}", flush=True)

for canon, nmr_type, processed in tqdm(
        zip(nex.canon, nex.NMR_type, nex.NMR_processed), total=len(nex), unit=" row"):
    try:
        entries = ast.literal_eval(processed) if isinstance(processed, str) else processed
    except (ValueError, SyntaxError):
        stats["parse_fail"] += 1
        continue
    if not entries:
        continue
    shifts = []
    if nmr_type == "13C NMR":
        # (shift, multiplicity, J)
        for e in entries:
            try:
                s = float(e[0])
            except (TypeError, ValueError, IndexError):
                continue
            if C_RANGE[0] <= s <= C_RANGE[1]:
                shifts.append(s)
        target = nmrexp_c
    else:
        # (multiplicity, [J...], integration, shift_start, shift_end)
        for e in entries:
            try:
                lo, hi = float(e[3]), float(e[4])
            except (TypeError, ValueError, IndexError):
                continue
            mid = (lo + hi) / 2.0
            if not (H_RANGE[0] <= mid <= H_RANGE[1]):
                continue
            m = INTEGRATION.search(str(e[2])) if len(e) > 2 and e[2] is not None else None
            n_h = int(float(m.group(1))) if m else 1
            shifts.extend([mid] * max(1, min(n_h, 60)))
        target = nmrexp_h
    if shifts and len(shifts) > len(target.get(canon, [])):
        target[canon] = shifts
print(f"  NMRexp in MARINA1: c_nmr {len(nmrexp_c):,}, h_nmr {len(nmrexp_h):,}", flush=True)

# ------------------------------------------------------- merge by precedence
out = defaultdict(dict)
for smiles, shifts in nmrexp_c.items():
    out[smiles]["c_nmr"] = shifts
for smiles, shifts in nmrexp_h.items():
    out[smiles]["h_nmr"] = shifts
n_c_override = sum(1 for s in nsdb_c if s in nmrexp_c)
n_h_override = sum(1 for s in nsdb_h if s in nmrexp_h)
for smiles, shifts in nsdb_c.items():
    out[smiles]["c_nmr"] = shifts
for smiles, shifts in nsdb_h.items():
    out[smiles]["h_nmr"] = shifts

out = {k: v for k, v in out.items() if v}
OUT.mkdir(exist_ok=True)
with open(OUT / "nmr_experimental.pkl", "wb") as fh:
    pickle.dump(dict(out), fh)

summary = {
    "molecules": len(out),
    "c_nmr": sum(1 for v in out.values() if "c_nmr" in v),
    "h_nmr": sum(1 for v in out.values() if "h_nmr" in v),
    "both": sum(1 for v in out.values() if "c_nmr" in v and "h_nmr" in v),
    "source_nmrshiftdb2": {"c_nmr": len(nsdb_c), "h_nmr": len(nsdb_h)},
    "source_nmrexp": {"c_nmr": len(nmrexp_c), "h_nmr": len(nmrexp_h)},
    "nmrshiftdb2_overrode_nmrexp": {"c_nmr": n_c_override, "h_nmr": n_h_override},
    "nmrexp_parse_failures": stats["parse_fail"],
}
(OUT / "nmr_experimental_summary.json").write_text(json.dumps(summary, indent=2))
print("\n" + json.dumps(summary, indent=2))
print(f"wrote {OUT/'nmr_experimental.pkl'}")
