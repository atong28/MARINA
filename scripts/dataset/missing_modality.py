#!/usr/bin/env python3
"""List 2D SMILES from index.pkl that are missing a given modality."""

import argparse
import os
import pickle
import sys

DATASET_ROOT = "data/dataset"
VALID_MODALITIES = ["hsqc", "c_nmr", "h_nmr", "mass_spec", "mw"]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--modality", required=True, choices=VALID_MODALITIES)
parser.add_argument("--split", default="all", choices=["train", "val", "test", "all"])
parser.add_argument("--output", default=None, help="Write SMILES to file instead of stdout")
args = parser.parse_args()

index = pickle.load(open(os.path.join(DATASET_ROOT, "index.pkl"), "rb"))

missing = [
    item["smiles"]
    for item in index.values()
    if (args.split == "all" or item["split"] == args.split)
    and not item.get(f"has_{args.modality}", True)
]

header = f"# {len(missing)} molecules missing {args.modality}" + (
    f" (split={args.split})" if args.split != "all" else ""
)

if args.output:
    with open(args.output, "w") as f:
        f.write("\n".join(missing) + "\n")
    print(header)
    print(f"Written to {args.output}")
else:
    print(header)
    sys.stdout.write("\n".join(missing) + ("\n" if missing else ""))
