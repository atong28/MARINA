"""
CH-NMR-NP (JEOL experimental natural-product NMR database) -> MARINA-DB spectra.

Inputs are the scraped mirror (`records.jsonl`, one record per line, built by
CH-NMR-NP/parse.py: meta fields + per-atom `atoms` table + derived `hsqc` cross-peaks)
and a TSV of CONFIRMED structures `chnmr_id <TAB> smiles` (the site carries no SMILES).

Conversion follows the conventions already in MARINA-DB (Mnova / SPECTRE-JEOL rows):
  HSQC   [c, h, -1 if CH2 else +1] per `hsqc` cross-peak, heteroatom-labelled rows
         dropped. A malformed H count (n_h null) makes the edited-HSQC sign unknown, so
         the whole HSQC is rejected.
  13C    every numeric `c_shift` in the atom table (one row per carbon or equivalent group).
  1H     every numeric `h_shift` in the atom table, OH/NH included (Mnova does the same).
Duplicate 1D shifts are merged later by the stage-11 per-peak collapse.
"""
import json
import re
from collections import defaultdict

import numpy as np
from rdkit import Chem

META_FIELDS = ("id", "no", "chs", "name", "reference")
LEADING_NUM = re.compile(r"^-?\d+(?:\.\d+)?")
MNOVA_SENTINEL_FLOOR = -1000.0   # Mnova writes -100000.0 for failed shifts


def shift(s):
    """Verbatim from CH-NMR-NP/parse.py: tolerate typos like '1.04s'."""
    m = LEADING_NUM.match(s)
    return float(m.group()) if m else None


def load_structures(path):
    """[(chnmr_id, smiles)] from the confirmed-structure TSV (header line optional)."""
    out = []
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2 or not parts[0].strip().isdigit():
                continue
            out.append((int(parts[0]), parts[1].strip()))
    return out


def load_records(path, ids):
    """{id: record} for the requested ids only."""
    ids = set(ids)
    recs = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in ids:
                recs[r["id"]] = r
    return recs


def choose_record(recs):
    """Several confirmed records on one structure: most HSQC peaks, tie -> lowest id."""
    return min(recs, key=lambda r: (-len(r["hsqc"]), r["id"]))


def meta(rec):
    return {k: rec[k] for k in META_FIELDS}


def to_spectra(rec):
    """-> (hsqc|None, hsqc_status, c_nmr|None, h_nmr|None).

    hsqc_status: 'ok' | 'nh_null' (sign unknown) | 'empty' (no usable cross-peak)."""
    peaks = [p for p in rec["hsqc"] if not p["het_label"]]
    if any(p["n_h"] is None for p in peaks):
        hsqc, status = None, "nh_null"
    elif not peaks:
        hsqc, status = None, "empty"
    else:
        hsqc = [[p["c"], p["h"], -1 if p["n_h"] == 2 else 1] for p in peaks]
        status = "ok"
    c = [v for v in (shift(a["c_shift"]) for a in rec["atoms"] if a["c_shift"]) if v is not None]
    h = [v for v in (shift(a["h_shift"]) for a in rec["atoms"] if a["h_shift"]) if v is not None]
    return hsqc, status, (c or None), (h or None)


def n_symmetry_distinct_carbons(mol):
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    return len({ranks[a.GetIdx()] for a in mol.GetAtoms() if a.GetSymbol() == "C"})


def c_coverage(c_shifts, mol):
    """Fraction of the molecule's symmetry-distinct carbons with a reported 13C shift."""
    n = n_symmetry_distinct_carbons(mol)
    return (len(c_shifts or []) / n) if n else 1.0


def assignment_mae(a, b, collapse):
    """Mean |a_i - b_j| under the optimal 1:1 assignment of the shorter peak list into the
    longer one (both collapsed first). In 1D with an absolute cost an optimal matching
    preserves sorted order, so an O(m*n) DP is exact. None if either side is empty."""
    a = collapse([v for v in a if v > MNOVA_SENTINEL_FLOOR])
    b = collapse([v for v in b if v > MNOVA_SENTINEL_FLOOR])
    if not a or not b:
        return None
    if len(a) > len(b):
        a, b = b, a
    a, b = np.asarray(a), np.asarray(b)
    m, n = len(a), len(b)
    prev = np.zeros(n + 1)                           # D[0][j] = 0
    for i in range(1, m + 1):
        cand = prev[:-1] + np.abs(a[i - 1] - b)      # D[i-1][j-1] + |a_i - b_j|, j = 1..n
        cur = np.full(n + 1, np.inf)
        cur[1:] = np.minimum.accumulate(cand)        # D[i][j] = min(D[i][j-1], cand_j)
        cur[:i] = np.inf                             # need j >= i
        prev = cur
    return float(prev[n] / m)


def group_confirmed(structures, canon):
    """{canonical smiles: [chnmr_id, ...]} from [(id, smiles)] using `canon` (a list of
    canonical SMILES aligned with `structures`, None on parse failure)."""
    by = defaultdict(list)
    for (cid, _), c in zip(structures, canon):
        if c:
            by[c].append(cid)
    return {c: sorted(set(ids)) for c, ids in by.items()}
