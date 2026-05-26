"""
Build an empirical peak distribution from the training split Arrow shards.
Saves {DATASET_ROOT}/peak_distributions.npz with keys: hsqc, c_nmr, h_nmr.

Run from the MARINA root:
    pixi run python3 scripts/dataset/build_peak_distribution.py
"""

import os
import pickle
import sys

import numpy as np
import pyarrow.parquet as pq

DATASET_ROOT = os.environ.get('DATASET_ROOT', 'data/dataset')
ARROW_DIR = os.path.join(DATASET_ROOT, 'arrow', 'train')
INDEX_PATH = os.path.join(DATASET_ROOT, 'index.pkl')
OUT_PATH = os.path.join(DATASET_ROOT, 'peak_distributions.npz')

MODALITY_CONFIGS = [
    ('hsqc',  'HSQC_NMR', 'has_hsqc',  None),     # raw shape (N, 3)
    ('c_nmr', 'C_NMR',    'has_c_nmr', (-1, 1)),   # reshape to (N, 1)
    ('h_nmr', 'H_NMR',    'has_h_nmr', (-1, 1)),   # reshape to (N, 1)
]


def load_parquet_by_idx(path):
    """Return dict idx -> (flat data list, shape list)."""
    table = pq.read_table(path, columns=['idx', 'data', 'shape'])
    return {
        int(idx): (data, shape)
        for idx, data, shape in zip(
            table['idx'].to_pylist(),
            table['data'].to_pylist(),
            table['shape'].to_pylist(),
        )
    }


def main():
    print(f'Loading index from {INDEX_PATH}')
    with open(INDEX_PATH, 'rb') as f:
        index = pickle.load(f)

    train_entries = {idx: e for idx, e in index.items() if e['split'] == 'train'}
    print(f'Training molecules: {len(train_entries)}')

    distributions = {}
    for key, arrow_name, has_field, reshape in MODALITY_CONFIGS:
        path = os.path.join(ARROW_DIR, f'{arrow_name}.parquet')
        if not os.path.isfile(path):
            print(f'  WARNING: {path} not found, skipping {key}')
            continue

        print(f'  Building distribution for {key} from {arrow_name}.parquet ...')
        by_idx = load_parquet_by_idx(path)

        chunks = []
        for idx, entry in train_entries.items():
            if not entry.get(has_field, False):
                continue
            if idx not in by_idx:
                continue
            data, shape = by_idx[idx]
            arr = np.asarray(data, dtype=np.float32)
            if shape:
                arr = arr.reshape([int(v) for v in shape])
            if reshape is not None:
                arr = arr.reshape(reshape)
            chunks.append(arr)

        if not chunks:
            print(f'  WARNING: no data found for {key}')
            continue

        combined = np.concatenate(chunks, axis=0)
        distributions[key] = combined
        print(f'    {key}: {combined.shape} rows collected')

    np.savez(OUT_PATH, **distributions)
    print(f'\nSaved distributions to {OUT_PATH}')
    for k, v in distributions.items():
        print(f'  {k}: {v.shape}')


if __name__ == '__main__':
    main()
