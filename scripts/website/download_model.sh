#!/usr/bin/env bash
# Downloads the website model and writes the models.json the backend serves.
#
# Usage (from anywhere):
#   bash scripts/website/download_model.sh           # skip a model already on disk
#   bash scripts/website/download_model.sh --force   # re-download
#
# Each id is a publicly shared Google Drive file: a zip whose entries sit at the
# model-directory root (params.json, best.ckpt, calibration.json, the
# RankingEntropy* vocabulary, ...), so it unpacks straight into checkpoints/<root>/.
# Needs only curl + unzip; gdown is used instead when it is installed.
set -euo pipefail

# marina_db_s1 = marina-db-open-chnmr-uniqmult-formula-s1 ep738 (MARINA-DB, CH-NMR-NP-first;
# RankingEntropyUniqueMultiplicity, radius 10, formula-capable), the served default.
# Override with the env var.
MARINA_DB_S1_ID="${MARINA_DB_S1_ID:-1PtxGL84MJppAtn8Sqn-sQia3nA5Kdn7S}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
MODEL_DATA_DIR="${MODEL_DATA_DIR:-./checkpoints}"

FORCE=false
[[ "${1:-}" == "--force" ]] && FORCE=true

info()  { printf '\033[1;34m[marina]\033[0m %s\n' "$*"; }
error() { printf '\033[1;31m[marina]\033[0m %s\n' "$*" >&2; }

for tool in curl unzip; do
    command -v "$tool" >/dev/null || { error "$tool is required"; exit 1; }
done

mkdir -p "$MODEL_DATA_DIR"

fetch() {
    local root=$1 id=$2
    local dest="$MODEL_DATA_DIR/$root" zip="$MODEL_DATA_DIR/$root.zip"
    if [[ -f "$dest/best.ckpt" ]] && ! $FORCE; then
        info "$root already present, skipping (--force to re-download)"
        return
    fi
    if [[ -z "$id" ]]; then
        error "no Google Drive id set for $root (edit this script or export the id variable)"
        exit 1
    fi
    info "downloading $root ..."
    if command -v gdown >/dev/null; then
        gdown "$id" --output "$zip"
    else
        # confirm=t skips Drive's "can't scan this file for viruses" page on large files.
        curl -fL --retry 3 --progress-bar -o "$zip" \
            "https://drive.usercontent.google.com/download?id=${id}&export=download&confirm=t"
    fi
    if ! unzip -tq "$zip" >/dev/null; then
        error "$zip is not a valid zip -- is the Drive file shared as 'Anyone with the link'?"
        exit 1
    fi
    rm -rf "$dest"
    unzip -q -o "$zip" -d "$dest"
    rm "$zip"
    chmod -R a+rX "$dest"
    info "$root -> $dest"
}

fetch marina_db_s1 "$MARINA_DB_S1_ID"

# The previous default model is no longer served; it is left on disk for rollback.
if [[ -d "$MODEL_DATA_DIR/marina_uniqmult_s1" ]]; then
    info "marina_uniqmult_s1 (previous model) is unused; remove it to free ~2.3 GB: rm -rf $MODEL_DATA_DIR/marina_uniqmult_s1"
fi

# The backend container runs as an unprivileged user, so the weights must be
# world-readable (checkpoints copied in by hand are often owner-only).
chmod -R a+rX "$MODEL_DATA_DIR" 2>/dev/null \
    || error "could not make $MODEL_DATA_DIR world-readable; the backend may fail to load it"

cat << EOF > "$MODEL_DATA_DIR/models.json"
{
    "models": [
        {
            "id": "marina_db_s1",
            "root": "marina_db_s1",
            "type": "marina",
            "default": true,
            "display_name": "MARINA-DB (unique-multiplicity, formula-capable)"
        }
    ]
}
EOF
info "wrote $MODEL_DATA_DIR/models.json"
