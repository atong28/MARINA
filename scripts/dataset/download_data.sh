#!/bin/bash
set -euo pipefail

# downloads the data from NP-MRD, COCONUT, and LOTUS and processes them.
mkdir -p data/raw
# download the data from NP-MRD
echo "Downloading data from NP-MRD..."
wget https://np-mrd.org/system/downloads/current/smiles_NP0000001_NP0050000.csv.gz -O data/raw/NP0000001_NP0050000.csv.gz
wget https://np-mrd.org/system/downloads/current/smiles_NP0050001_NP0100000.csv.gz -O data/raw/NP0050001_NP0100000.csv.gz
wget https://np-mrd.org/system/downloads/current/smiles_NP0100001_NP0150000.csv.gz -O data/raw/NP0100001_NP0150000.csv.gz
wget https://np-mrd.org/system/downloads/current/smiles_NP0150001_NP0200000.csv.gz -O data/raw/NP0150001_NP0200000.csv.gz
wget https://np-mrd.org/system/downloads/current/smiles_NP0200001_NP0250000.csv.gz -O data/raw/NP0200001_NP0250000.csv.gz
wget https://np-mrd.org/system/downloads/current/smiles_NP0250001_NP0300000.csv.gz -O data/raw/NP0250001_NP0300000.csv.gz
wget https://np-mrd.org/system/downloads/current/smiles_NP0300001_NP0350000.csv.gz -O data/raw/NP0300001_NP0350000.csv.gz

# unzip the data
gunzip data/raw/*.csv.gz

# concatenate the data, keep only first header line
head -n 1 data/raw/*.csv > data/raw/npmrd.csv
tail -n +2 data/raw/*.csv >> data/raw/npmrd.csv

rm data/raw/NP*.csv

# download the data from COCONUT
echo "Downloading data from COCONUT..."
wget https://coconut.s3.uni-jena.de/prod/downloads/2026-04/coconut_csv-04-2026.zip -O data/raw/coconut_csv.zip
unzip data/raw/coconut_csv.zip -d data/raw/
mv data/raw/coconut_csv-04-2026.csv data/raw/coconut.csv
rm data/raw/coconut_csv.zip

# download the data from LOTUS
echo "Downloading data from LOTUS..."
wget https://lotus.naturalproducts.net/download/smiles -O data/raw/lotus.txt

# process the data
echo "Processing SMILES data..."
pixi run python scripts/dataset/process_smiles.py

# download spectral data
gdown 1FzjNuhQCNmVWzgsotrpZF2s4ZOn6UMQW -o data/raw/spectre_data.zip
unzip data/raw/spectre_data.zip -d data/raw/
rm data/raw/spectre_data.zip

# download mnova nmr prediction data
gdown 1a6IkkpYVx5mO2HWffi5PO0gUVgXvOkAR -o data/raw/mnova_predictions.zip
unzip data/raw/mnova_predictions.zip -d data/raw/mnova_predictions/
rm data/raw/mnova_predictions.zip

# download iceberg ms prediction data
gdown 1MiMcAI5j08ti-npKvc-ynkff17Mns1wC -o data/raw/ms_predictions.zip
unzip data/raw/ms_predictions.zip -d data/raw/ms_predictions/
rm data/raw/ms_predictions.zip