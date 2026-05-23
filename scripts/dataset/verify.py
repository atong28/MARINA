# verifies dataset is valid (index.pkl is updated). only works for jsonl datasets

import pickle
import os
import json
from tqdm import tqdm

strict_mode = False

def raise_error(message: str):
    if strict_mode:
        raise ValueError(message)
    else:
        print(message)

DATASET_ROOT = "data/cleaned"

index = pickle.load(open(os.path.join(DATASET_ROOT, "index.pkl"), "rb"))
train = {json.loads(line)['idx']: json.loads(line) for line in tqdm(open(os.path.join(DATASET_ROOT, "train.jsonl")).readlines(), desc="Loading train")}
val = {json.loads(line)['idx']: json.loads(line) for line in tqdm(open(os.path.join(DATASET_ROOT, "val.jsonl")).readlines(), desc="Loading val")}
test = {json.loads(line)['idx']: json.loads(line) for line in tqdm(open(os.path.join(DATASET_ROOT, "test.jsonl")).readlines(), desc="Loading test")}

# verify idx matches
for split in ["train", "val", "test"]:
    split_data = train if split == "train" else val if split == "val" else test
    for idx, item in tqdm(index.items(), desc=f"Verifying {split}"):
        if item['split'] != split:
            continue
        if idx not in split_data:
            raise_error(f"Item {idx} in split {split} not found in {split}.jsonl")
            continue
        if split_data[idx]['smiles'] != item['smiles']:
            raise_error(f"Item {idx} in split {split} smiles mismatch")
            continue

    for idx, item in tqdm(split_data.items(), desc=f"Verifying {split} part 2"):
        if idx not in index:
            raise_error(f"Item {idx} in {split}.jsonl not found in index.pkl")
            continue

# verify every training molecule is present in the retrieval set
retrieval = pickle.load(open(os.path.join(DATASET_ROOT, "retrieval.pkl"), "rb"))
retrieval_smiles = {entry["smiles"] for entry in retrieval.values()}
missing_from_retrieval = [
    (idx, item["smiles"])
    for idx, item in index.items()
    if item["smiles"] not in retrieval_smiles
]
if missing_from_retrieval:
    for idx, smi in missing_from_retrieval[:10]:
        raise_error(f"Item {idx} (smiles={smi}) is in index but missing from retrieval.pkl")
    if len(missing_from_retrieval) > 10:
        raise_error(f"... and {len(missing_from_retrieval) - 10} more missing from retrieval set")
else:
    print(f"Retrieval superset check: PASSED ({len(retrieval_smiles)} retrieval, {len(index)} index)")

print("Passed!")