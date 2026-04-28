# gets dataset info for a specified path, assuming dataset is valid (index.pkl is updated)

import pickle
import os
from collections import defaultdict

DATASET_ROOT = "/data/nas-gpu/wang/atong/Datasets/MARINAFullDataset_jsonl"

index = pickle.load(open(os.path.join(DATASET_ROOT, "index.pkl"), "rb"))

for split in ["train", "val", "test"]:
    counts = defaultdict(lambda: 0)
    for item in index.values():
        if item['split'] != split:
            continue
        counts['total'] += 1
        counts['hsqc'] += int(item['has_hsqc'])
        counts['c_nmr'] += int(item['has_c_nmr'])
        counts['h_nmr'] += int(item['has_h_nmr'])
        counts['mass_spec'] += int(item['has_mass_spec'])
        counts['mw'] += int(item['has_mw'])
    print(f"Split: {split}")
    print(f"Number of items: {counts['total']}")
    print(f"Number of items with HSQC: {counts['hsqc']}")
    print(f"Number of items with C_NMR: {counts['c_nmr']}")
    print(f"Number of items with H_NMR: {counts['h_nmr']}")
    print(f"Number of items with MassSpec: {counts['mass_spec']}")
    print(f"Number of items with MW: {counts['mw']}")