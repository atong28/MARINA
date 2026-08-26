#!/bin/bash
set -euo pipefail

# Pull the MARINA-DB build inputs into data/raw/.
#
# Structure dumps (NP-MRD / COCONUT / LOTUS) come one of two ways:
#   - PINNED (used when STRUCTURE_DUMPS_ID is set): gdown the condensed
#     structure_dumps.tar snapshot, untar, gunzip. Reproducible -- two of the three
#     upstream endpoints are undated and move without notice. This is the reliable path.
#   - LIVE (fallback when STRUCTURE_DUMPS_ID is unset): wget the upstream endpoints.
#     Only COCONUT is date-pinned; NP-MRD 'current/' and LOTUS are undated.
#
# Spectral archives + MS/MS predictions always come from Drive via gdown.
#
# Drive IDs are re-synced after each Snapshots upload; provenance for every archive is in
# Snapshots/RawData/PROVENANCE.md. IDs left empty below are pending the next upload.
# cd to repo root so every path below is repo-root-relative and consistent.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."

mkdir -p data/raw

# ---- Drive IDs (see Snapshots/RawData/PROVENANCE.md) ----
STRUCTURE_DUMPS_ID="${STRUCTURE_DUMPS_ID:-1lkHAp47usjvwtnNkUyD__4wlmMTKW5_k}"       # condensed structure_dumps.tar (NP-MRD+COCONUT+LOTUS); empty -> live fallback
SPECTRE_DATA_ID="${SPECTRE_DATA_ID:-1artiYvqQLGCcP_vg5gpkOU3TBbd0d3uh}"             # verified 2026-08-25 (byte-identical to local); supersedes re-zipped data.zip 1FzjNuhQCNmVWzgsotrpZF2s4ZOn6UMQW
SPECTRE_RETRIEVAL_ID="${SPECTRE_RETRIEVAL_ID:-}"                                    # TODO: spectre_retrieval.pkl (526,316 SMILES; extracted from SPECTRE inference metadata inference_metadata_name_updated.pkl)
MNOVA_PREDICTIONS_ID="${MNOVA_PREDICTIONS_ID:-1XyiHjrWBugBl9OOEQOYkQdmEt6e4V6Mx}"   # 489-shard, verified 2026-08-25; supersedes 488-shard 1a6IkkpYVx5mO2HWffi5PO0gUVgXvOkAR
MS_PREDICTIONS_POSITIVE_ID="${MS_PREDICTIONS_POSITIVE_ID:-}"                        # TODO: re-predicted positive MS/MS; supersedes 1MiMcAI5j08ti-npKvc-ynkff17Mns1wC
MS_PREDICTIONS_NEGATIVE_ID="${MS_PREDICTIONS_NEGATIVE_ID:-1oRUUkb2NAtimGlp2H4h5GfGdGJEUi5vM}"  # negative MS/MS, verified 2026-08-25 (not yet consumed by the build -- see below)

# ---- structure dumps: NP-MRD / COCONUT / LOTUS ----
if [ -n "$STRUCTURE_DUMPS_ID" ]; then
    echo "Fetching pinned structure_dumps.tar..."
    gdown "$STRUCTURE_DUMPS_ID" -O data/raw/structure_dumps.tar
    tar -xf data/raw/structure_dumps.tar -C data/raw/
    gunzip -f data/raw/coconut.csv.gz data/raw/npmrd.csv.gz data/raw/lotus.txt.gz
    rm data/raw/structure_dumps.tar
else
    echo "STRUCTURE_DUMPS_ID unset -> downloading structure dumps live from upstream..."
    # NP-MRD (undated 'current/' endpoint)
    echo "Downloading data from NP-MRD..."
    for r in NP0000001_NP0050000 NP0050001_NP0100000 NP0100001_NP0150000 \
             NP0150001_NP0200000 NP0200001_NP0250000 NP0250001_NP0300000 NP0300001_NP0350000; do
        wget "https://np-mrd.org/system/downloads/current/smiles_${r}.csv.gz" -O "data/raw/${r}.csv.gz"
    done
    gunzip data/raw/*.csv.gz
    # concatenate the shards, keep only the first header line
    head -n 1 data/raw/*.csv > data/raw/npmrd.csv
    tail -n +2 data/raw/*.csv >> data/raw/npmrd.csv
    rm data/raw/NP*.csv

    # COCONUT: release is pinned via COCONUT_RELEASE (coconut_csv-MM-YYYY); URL date is YYYY-MM.
    COCONUT_RELEASE="${COCONUT_RELEASE:-coconut_csv-08-2026}"
    COCONUT_MMYYYY="${COCONUT_RELEASE#coconut_csv-}"          # MM-YYYY
    COCONUT_DATE="${COCONUT_MMYYYY#*-}-${COCONUT_MMYYYY%-*}"  # YYYY-MM
    echo "Downloading data from COCONUT ($COCONUT_RELEASE)..."
    wget "https://coconut.s3.uni-jena.de/prod/downloads/${COCONUT_DATE}/${COCONUT_RELEASE}.zip" -O data/raw/coconut_csv.zip
    unzip data/raw/coconut_csv.zip -d data/raw/
    mv "data/raw/${COCONUT_RELEASE}.csv" data/raw/coconut.csv
    rm data/raw/coconut_csv.zip

    # LOTUS (undated endpoint)
    echo "Downloading data from LOTUS..."
    wget https://lotus.naturalproducts.net/download/smiles -O data/raw/lotus.txt
fi

# ---- SPECTRE corpus (index.pkl, train/val/test.jsonl) ----
gdown "$SPECTRE_DATA_ID" -O data/raw/spectre_data.zip
unzip data/raw/spectre_data.zip -d data/raw/
rm data/raw/spectre_data.zip

# ---- SPECTRE retrieval bank (structure candidates folded into retrieval; 3_build_retrieval.py) ----
# List of 526,316 SMILES extracted from SPECTRE's inference metadata. Makes MARINA-DB
# retrieval a strict superset of SPECTRE's candidate pool.
if [ -n "$SPECTRE_RETRIEVAL_ID" ]; then
    echo "Downloading SPECTRE retrieval bank..."
    gdown "$SPECTRE_RETRIEVAL_ID" -O data/raw/spectre_retrieval.pkl
else
    echo "SPECTRE_RETRIEVAL_ID unset -- expecting data/raw/spectre_retrieval.pkl staged manually"
fi

# ---- Mnova NMR predictions (489-shard) ----
gdown "$MNOVA_PREDICTIONS_ID" -O data/raw/mnova_predictions.zip
unzip data/raw/mnova_predictions.zip -d data/raw/mnova_predictions/
rm data/raw/mnova_predictions.zip

# ---- ICEBERG MS/MS predictions ----
# The build's process_ms_predictions() consumes data/raw/ms_predictions/*.json only.
# Positive mode goes there. Negative-mode integration is a pending pipeline change (D5):
# it is fetched to a separate dir and NOT yet read by the build, to avoid pos/neg key collisions.
if [ -n "$MS_PREDICTIONS_POSITIVE_ID" ]; then
    echo "Downloading positive-mode MS/MS predictions..."
    gdown "$MS_PREDICTIONS_POSITIVE_ID" -O data/raw/ms_predictions_positive.zip
    unzip data/raw/ms_predictions_positive.zip -d data/raw/ms_predictions/
    rm data/raw/ms_predictions_positive.zip
else
    echo "MS_PREDICTIONS_POSITIVE_ID unset (re-prediction in flight) -- skipping positive MS/MS"
fi
if [ -n "$MS_PREDICTIONS_NEGATIVE_ID" ]; then
    echo "Downloading negative-mode MS/MS predictions (staged, not yet consumed by the build)..."
    gdown "$MS_PREDICTIONS_NEGATIVE_ID" -O data/raw/ms_predictions_negative.zip
    unzip data/raw/ms_predictions_negative.zip -d data/raw/   # zip wraps files in ms_predictions_negative/
    rm data/raw/ms_predictions_negative.zip
else
    echo "MS_PREDICTIONS_NEGATIVE_ID unset -- skipping negative MS/MS"
fi
