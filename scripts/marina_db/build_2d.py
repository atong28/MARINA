#!/usr/bin/env python3
"""Derive a *-2D dataset: MARINA-DB(-OPEN) + ceiling HMBC/COSY shards built from the Mnova archive.

    cd ~/Workspace && pixi run python3 MARINA/scripts/marina_db/build_2d.py \
        --zip Snapshots/RawData/mnova_predictions.zip \
        --src Datasets/MARINA-DB-OPEN --out Datasets/MARINA-DB-OPEN-2D [--workers 12] [--limit N]

Design: wiki/models/marina2.0-hmbc-cosy.md (decisions D1-D7 as revised by the calibration in
wiki/experiments/marina-experiments/hmbc-cosy-dropout-calibration.md).

The "ceiling" is every correlation the structure could show, using Mnova's per-atom shifts:
  HMBC row  = [dC, dH, ptype, ctype, n_bonds(2|3), |J| or -1]      one row per (carbon, 1H entry) at
              heavy-atom distance 1 (2J) or 2 (3J); 4J is NOT included (observed <1%).
  COSY row  = [dHa, dHb, cls(0=vicinal 3J, 1=geminal), ptype_a, ptype_b, exch, |J| or -1]
              vicinal = 1H entries on bonded heavy atoms; geminal = the two Mnova entries of a
              diastereotopic CH2 (distinct shifts). 4J is NOT included. Stored once per unordered
              pair; the loader emits both orderings.
The model sees only the first two columns; the rest parameterise the training-time dropout
(MARINA/analysis/hmbc-cosy-calibration/results/marina2_dropout_params.json).
  ptype: 0 CH3, 1 CH2, 2 CH, 3 aromatic C-H, 4 olefinic C-H, 5 exchangeable (O-H/N-H/S-H)
  ctype: 0 protonated, 1 quaternary, 2 carbonyl
Mnova atom `number` = RDKit atom index + 1 of the record's own `smiles` (verified 2026-09-15,
analysis/mnova-atom-mapping); rows are exact-deduplicated so symmetry-equivalent atoms (bit-identical
Mnova shifts) collapse to one cross-peak, matching the stage-11 per-peak convention.

Everything not built here is hard-linked from --src (idx / splits / FP targets byte-identical), so
any metric delta between the two datasets is attributable to the two new modalities.
"""
import argparse, collections, json, os, pickle, shutil, subprocess, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1]))            # repo root -> src.modules.data.smiles
sys.path.insert(0, str(_HERE.parent / 'dataset'))   # pack_arrow
os.environ.setdefault('DATASET_ROOT', '/tmp')        # src.modules.core.const needs one at import; unused here
from src.modules.data.smiles import canonicalize_smiles   # noqa: E402
from pack_arrow import pack_root                          # noqa: E402

SPLITS = ('train', 'val', 'test')
PTYPE = {'CH3': 0, 'CH2': 1, 'CH': 2, 'arom': 3, 'olef': 4, 'exch': 5}
CTYPE = {'protonated': 0, 'quaternary': 1, 'carbonyl': 2}
MOD_DIRS = {'hmbc': 'HMBC_NMR', 'cosy': 'COSY_NMR'}


# ----------------------------------------------------------------------------- chemistry
def ptype(atom):
    if atom.GetSymbol() != 'C':
        return PTYPE['exch']
    if atom.GetIsAromatic():
        return PTYPE['arom']
    nh = atom.GetTotalNumHs()
    if nh and any(b.GetBondType() == Chem.BondType.DOUBLE for b in atom.GetBonds()):
        return PTYPE['olef']
    return PTYPE.get({3: 'CH3', 2: 'CH2', 1: 'CH'}.get(nh, 'CH'))


def ctype(atom):
    if atom.GetTotalNumHs() > 0:
        return CTYPE['protonated']
    if any(b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(atom).GetSymbol() == 'O' for b in atom.GetBonds()):
        return CTYPE['carbonyl']
    return CTYPE['quaternary']


def expected_name(atom):
    nh = atom.GetTotalNumHs()
    return atom.GetSymbol() + ('' if nh == 0 else 'H' if nh == 1 else f'H{nh}')


def ceiling(rec):
    """-> (hmbc_rows, cosy_rows, reason) ; rows are lists of float lists; reason set when skipped."""
    p = rec['predictions']['hsqc']
    H, C = p.get('H') or [], p.get('C') or []
    if p['status'] != 'SUCCESS' or not H or not C:
        return None, None, 'no_hc'
    mol = Chem.MolFromSmiles(rec['smiles'])
    if mol is None:
        return None, None, 'rdkit_fail'
    atoms = rec['atoms'] or []
    if len(atoms) != mol.GetNumAtoms():
        return None, None, 'natoms_mismatch'
    for a in atoms:
        i = int(a['number']) - 1
        if a['name'] != expected_name(mol.GetAtomWithIdx(i)):
            return None, None, 'name_mismatch'
    # per-atom shifts and couplings (Mnova index -> RDKit index is number-1)
    c_shift = {}
    for c in C:
        for a in c['atom']:
            c_shift.setdefault(a['index'] - 1, float(c['shift']['value']))
    h_entries = []                                   # (atom, shift)
    jhh = {}
    for h in H:
        for a in h['atom']:
            ai = a['index'] - 1
            h_entries.append((ai, float(h['shift']['value'])))
            for j in h.get('js') or []:
                for b in j['atom']:
                    bi = b['index'] - 1
                    k = (min(ai, bi), max(ai, bi))
                    jhh[k] = max(jhh.get(k, 0.0), abs(float(j['j']['value'])))
    jch = {}
    for c in C:
        for a in c['atom']:
            ci = a['index'] - 1
            for j in c.get('js') or []:
                for b in j['atom']:
                    k = (b['index'] - 1, ci)
                    jch[k] = max(jch.get(k, 0.0), abs(float(j['j']['value'])))
    dm = Chem.GetDistanceMatrix(mol)
    n = mol.GetNumAtoms()
    pt = [ptype(mol.GetAtomWithIdx(i)) for i in range(n)]
    ct = [ctype(mol.GetAtomWithIdx(i)) if mol.GetAtomWithIdx(i).GetSymbol() == 'C' else -1 for i in range(n)]
    carbons = [i for i in range(n) if ct[i] >= 0 and i in c_shift]

    hmbc = set()
    for a, dh in h_entries:
        row_d = dm[a]
        for c in carbons:
            d = int(row_d[c])
            if d == 1 or d == 2:
                hmbc.add((c_shift[c], dh, float(pt[a]), float(ct[c]), float(d + 1), jch.get((a, c), -1.0)))
    cosy = set()
    by_atom = collections.defaultdict(list)
    for a, dh in h_entries:
        by_atom[a].append(dh)
    for i, (a, da) in enumerate(h_entries):
        for b, db in h_entries[i + 1:]:
            if a == b:
                continue
            if int(dm[a][b]) == 1:
                x, y = (da, db) if a < b else (db, da)
                pa_, pb_ = (pt[a], pt[b]) if a < b else (pt[b], pt[a])
                exch = 1.0 if (pt[a] == PTYPE['exch'] or pt[b] == PTYPE['exch']) else 0.0
                cosy.add((x, y, 0.0, float(pa_), float(pb_), exch, jhh.get((min(a, b), max(a, b)), -1.0)))
    for a, shifts in by_atom.items():
        if len(shifts) == 2 and shifts[0] != shifts[1] and mol.GetAtomWithIdx(a).GetSymbol() == 'C':
            cosy.add((shifts[0], shifts[1], 1.0, float(pt[a]), float(pt[a]), 0.0, jhh.get((a, a), -1.0)))
    return sorted(hmbc), sorted(cosy), None


# ----------------------------------------------------------------------------- shard worker
def process_shard(args):
    zip_path, name, keys, work = args
    out_path = os.path.join(work, name + '.pkl')
    if os.path.exists(out_path):
        return name, None
    res, reasons = {}, collections.Counter()
    p = subprocess.Popen(['unzip', '-p', zip_path, name], stdout=subprocess.PIPE)
    for line in p.stdout:
        rec = json.loads(line)
        if rec['status'] != 'SUCCESS':
            reasons['status_failed'] += 1
            continue
        try:
            key = canonicalize_smiles(rec['smiles'])
        except Exception:  # noqa: BLE001
            reasons['canon_fail'] += 1
            continue
        if key not in keys or key in res:
            reasons['not_in_index' if key not in keys else 'dup'] += 1
            continue
        hm, co, why = ceiling(rec)
        if why:
            reasons[why] += 1
            continue
        res[key] = (np.asarray(hm, dtype=np.float32).reshape(-1, 6), np.asarray(co, dtype=np.float32).reshape(-1, 7))
        reasons['ok'] += 1
    p.wait()
    pickle.dump((res, reasons), open(out_path + '.tmp', 'wb'))
    os.replace(out_path + '.tmp', out_path)
    return name, dict(reasons)


# ----------------------------------------------------------------------------- assembly
def link_tree(src: Path, dst: Path, skip=()):
    """Hard-link every file of src into dst (same filesystem), skipping relative paths in `skip`."""
    for root, dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for f in files:
            r = str(rel / f)
            if r in skip:
                continue
            target = dst / rel / f
            if target.exists():
                target.unlink()
            os.link(Path(root) / f, target)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--zip', default='Snapshots/RawData/mnova_predictions.zip')
    ap.add_argument('--src', default='Datasets/MARINA-DB-OPEN')
    ap.add_argument('--out', default='Datasets/MARINA-DB-OPEN-2D')
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--limit', type=int, default=0, help='only the first N shards (smoke test)')
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    work = out.parent / (out.name + '.work')
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    index = pickle.load(open(src / 'index.pkl', 'rb'))
    keys = {e['smiles'] for e in index.values()}
    print(f'index: {len(index)} molecules, {len(keys)} unique SMILES', flush=True)

    shards = sorted(subprocess.run(['unzip', '-Z1', a.zip], capture_output=True, text=True).stdout.split())
    if a.limit:
        shards = shards[:a.limit]
    print(f'{len(shards)} shards, {a.workers} workers', flush=True)
    reasons = collections.Counter()
    with Pool(a.workers) as pool:
        for i, (name, r) in enumerate(pool.imap_unordered(process_shard, [(a.zip, s, keys, str(work)) for s in shards])):
            if r:
                reasons.update(r)
            if i % 20 == 0:
                print(f'  {i + 1}/{len(shards)} shards  {time.time() - t0:.0f}s  {dict(reasons)}', flush=True)
    print(f'shard pass done in {time.time() - t0:.0f}s: {dict(reasons)}', flush=True)

    # gather
    data = {}
    for s in shards:
        res, _ = pickle.load(open(work / (s + '.pkl'), 'rb'))
        for k, v in res.items():
            data.setdefault(k, v)
    print(f'ceilings for {len(data)} of {len(keys)} index SMILES', flush=True)

    # assemble parquet shards per split + flags
    out.mkdir(parents=True, exist_ok=True)
    rows = {sp: {m: [] for m in MOD_DIRS} for sp in SPLITS}
    stats = {sp: collections.Counter() for sp in SPLITS}
    npeaks = {m: [] for m in MOD_DIRS}
    for e in index.values():
        v = data.get(e['smiles'])
        hm = v[0] if v is not None else np.zeros((0, 6), np.float32)
        co = v[1] if v is not None else np.zeros((0, 7), np.float32)
        e['has_hmbc'] = bool(len(hm))
        e['has_cosy'] = bool(len(co))
        sp = e['split']
        stats[sp]['n'] += 1
        for m, arr in (('hmbc', hm), ('cosy', co)):
            if len(arr):
                rows[sp][m].append((int(e['idx']), arr.reshape(-1).tolist(), [int(arr.shape[0]), int(arr.shape[1])]))
                stats[sp][m] += 1
                npeaks[m].append(len(arr))
    link_tree(src, out, skip=('index.pkl',))
    pickle.dump(index, open(out / 'index.pkl', 'wb'))
    for sp in SPLITS:
        d = out / 'arrow' / sp
        d.mkdir(parents=True, exist_ok=True)
        for m, mod_dir in MOD_DIRS.items():
            rs = rows[sp][m]
            table = {'idx': [r[0] for r in rs], 'data': [r[1] for r in rs], 'shape': [r[2] for r in rs]}
            pq.write_table(pa.table(table), d / f'{mod_dir}.parquet')
            # new shard: make sure a stale packed dir never shadows it
            pk = out / 'packed' / sp / mod_dir
            if pk.exists():
                shutil.rmtree(pk)
    pack_root(str(out), force=False)     # only the two new shards lack packed dirs

    summary = {
        'src': str(src), 'out': str(out), 'zip': a.zip, 'shards': len(shards), 'reasons': dict(reasons),
        'coverage': {sp: dict(stats[sp]) for sp in SPLITS},
        'peaks_per_molecule': {m: dict(mean=float(np.mean(v)), median=float(np.median(v)), p90=float(np.percentile(v, 90)),
                                       max=int(np.max(v))) for m, v in npeaks.items() if v},
        'columns': {'HMBC_NMR': ['dC', 'dH', 'ptype', 'ctype', 'n_bonds', 'absJ_or_-1'],
                    'COSY_NMR': ['dHa', 'dHb', 'cls(0=vicinal,1=geminal)', 'ptype_a', 'ptype_b', 'exch', 'absJ_or_-1']},
        'ptype': PTYPE, 'ctype': CTYPE, 'seconds': int(time.time() - t0),
    }
    json.dump(summary, open(out / '2d_build_summary.json', 'w'), indent=1)
    print(json.dumps(summary, indent=1), flush=True)
    if not a.limit:
        shutil.rmtree(work)


if __name__ == '__main__':
    main()
