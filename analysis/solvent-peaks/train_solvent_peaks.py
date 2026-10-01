"""Are solvent residual peaks present in MARINA training data? Per source, the fraction of molecules with a peak at a
solvent line vs at control positions nearby (no excess at the line => no solvent peaks)."""
import sys, numpy as np, pyarrow.parquet as pq, pyarrow.compute as pc, os
root, label = sys.argv[1], sys.argv[2]
LINES_C = {'CDCl3 77.16': 77.16, 'DMSO 39.52': 39.52, 'CD3OD 49.00': 49.00, 'C5D5N 123.87': 123.87, 'C5D5N 135.91': 135.91, 'C5D5N 150.35': 150.35}
LINES_H = {'CHCl3 7.26': 7.26, 'DMSO 2.50': 2.50, 'H2O/DMSO 3.33': 3.33, 'CD3OD 3.31': 3.31, 'HOD 4.87': 4.87, 'C5D5N 8.74': 8.74}
WC, WH = 0.10, 0.01
CTRL_C, CTRL_H = (-2.0, -1.0, 1.0, 2.0), (-0.2, -0.1, 0.1, 0.2)
src = None
if os.path.isfile(f'{root}/nmr_sources.parquet'):
    t = pq.read_table(f'{root}/nmr_sources.parquet').to_pandas().set_index('idx'); src = t
def load(mod):
    ids, vals = [], []
    for sp in ('train', 'val', 'test'):
        t = pq.read_table(f'{root}/arrow/{sp}/{mod}.parquet', columns=['idx', 'data'])
        lens = pc.list_value_length(t['data']).to_numpy(zero_copy_only=False)
        ids.append(np.repeat(t['idx'].to_numpy(), lens)); vals.append(pc.list_flatten(t['data']).to_numpy(zero_copy_only=False))
    return np.concatenate(ids), np.concatenate(vals)
def frac(ids, v, center, w, subset):
    hit = np.unique(ids[np.abs(v - center) <= w]); return np.isin(hit, subset).sum() / len(subset)
def report(mod, lines, w, ctrl, col=None, stride=1):
    ids, v = load(mod)
    if stride > 1: v = v.reshape(-1, stride)[:, col]; ids = ids[::stride]
    groups = {'all': np.unique(ids)}
    if src is not None:
        key = {'C_NMR': 'c_nmr', 'H_NMR': 'h_nmr', 'HSQC_NMR': 'hsqc'}[mod]
        for s in ('mnova', 'chnmr'):
            groups[s] = np.intersect1d(np.unique(ids), src.index[src[key] == s].to_numpy())
    for g, sub in groups.items():
        print(f'  [{mod}{"" if col is None else f" dim{col}"} | {g}, n={len(sub):,}]')
        for name, L in lines.items():
            at = frac(ids, v, L, w, sub); c = np.mean([frac(ids, v, L + d, w, sub) for d in ctrl])
            print(f'    {name:14s} at line {100*at:6.3f}%   controls {100*c:6.3f}%   ratio {at/c if c else float("nan"):5.2f}')
print(f'== {label} ({root})')
report('C_NMR', LINES_C, WC, CTRL_C)
report('H_NMR', LINES_H, WH, CTRL_H)
