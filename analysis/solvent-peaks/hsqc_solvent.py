import sys, numpy as np, pyarrow.parquet as pq, pyarrow.compute as pc, os
root, label = sys.argv[1], sys.argv[2]
CROSS = {'CHCl3 (77.16, 7.26)': (77.16, 7.26), 'DMSO-d5 (39.52, 2.50)': (39.52, 2.50), 'CHD2OD (49.00, 3.31)': (49.00, 3.31),
         'C5D4HN (123.87,7.22)': (123.87, 7.22), 'C5D4HN (135.91,7.58)': (135.91, 7.58), 'C5D4HN (150.35,8.74)': (150.35, 8.74)}
WC, WH = 0.3, 0.03
ids, rows = [], []
for sp in ('train', 'val', 'test'):
    t = pq.read_table(f'{root}/arrow/{sp}/HSQC_NMR.parquet', columns=['idx', 'data'])
    lens = pc.list_value_length(t['data']).to_numpy(zero_copy_only=False) // 3
    ids.append(np.repeat(t['idx'].to_numpy(), lens)); rows.append(pc.list_flatten(t['data']).to_numpy(zero_copy_only=False).reshape(-1, 3))
ids, rows = np.concatenate(ids), np.concatenate(rows)
# source per molecule: MARINA-DB-OPEN from nmr_sources; MARINA-DB by row signature (ACD = intensity column, else ±1)
if os.path.isfile(f'{root}/nmr_sources.parquet'):
    s = pq.read_table(f'{root}/nmr_sources.parquet', columns=['idx', 'hsqc']).to_pandas().set_index('idx')['hsqc']
    src = s.reindex(ids).to_numpy()
else:
    acd = np.abs(np.abs(rows[:, 2]) - 1) > 1e-6
    mol_acd = np.zeros(ids.max() + 1, bool); mol_acd[ids[acd]] = True
    src = np.where(mol_acd[ids], 'ACD/Labs', 'Mnova or JEOL(CH-NMR-NP)')
print(f'== {label}: HSQC rows within ±{WC} ppm 13C and ±{WH} ppm 1H of a residual-solvent cross-peak')
for g in sorted(set(src)):
    m = src == g; nm = len(np.unique(ids[m]))
    hits = {k: int(len(np.unique(ids[m][(np.abs(rows[m, 0] - c) <= WC) & (np.abs(rows[m, 1] - h) <= WH)]))) for k, (c, h) in CROSS.items()}
    print(f'  {g:28s} molecules {nm:>8,}: ' + ', '.join(f'{k} {v}' for k, v in hits.items()))
