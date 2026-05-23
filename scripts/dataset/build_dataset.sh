#!/bin/bash

# usage: bash scripts/dataset/build_dataset.sh

bash download_data.sh
# optional: filter smiles_dict.json before building the dataset
# PYTHONPATH=. pixi run python scripts/dataset/filter_molecules.py
PYTHONPATH=. pixi run python scripts/dataset/generate_dataset.py
PYTHONPATH=. pixi run python scripts/dataset/generate_metadata.py
PYTHONPATH=. pixi run python scripts/dataset/convert.py --to-arrow data/cleaned data/dataset
PYTHONPATH=. pixi run python scripts/dataset/generate_fragidx.py
cp data/raw/benchmark.pkl data/dataset/benchmark.pkl