#!/usr/bin/env bash
# Downloads the website models and writes the models.json the backend serves.
#
# Usage (from anywhere):
#   bash scripts/website/download_model.sh           # skip models already on disk
#   bash scripts/website/download_model.sh --force   # re-download
#
# The bundle is one zip that unpacks into checkpoints/:
#   marina_flag_s0/   marina-db-flagship-uniqmult-s0 ep616 -- the served model (release MARINA-DB;
#                     RankingEntropyUniqueMultiplicity, radius 10, formula-capable)
#   marina_shifts_s0/ marina-db-flagship-hsqcshifts-s0 ep638 -- shifts-only HSQC fallback: answers only
#                     an HSQC with no sign/intensity column ("role": "hsqc_unsigned"); its retrieval bank,
#                     metadata and indexes are symlinks to marina_flag_s0/
#   models.json
# Bundle: marina_db_release_s0.zip, sha256 fd0e86289480492986980133b32c4c25e321be89e20f793923c92e756c81c9a2.
# Needs only curl + unzip; gdown is used for Google Drive when it is installed.
set -euo pipefail

# Where the bundle comes from, first set wins (both can be exported instead of edited):
#   MARINA_BUNDLE_URL  a direct download URL, e.g. a Zenodo record file
#                      (https://zenodo.org/records/<id>/files/marina_db_release_s0.zip?download=1)
#   MARINA_RELEASE_ID  a Google Drive file id shared as "Anyone with the link"
MARINA_BUNDLE_URL="${MARINA_BUNDLE_URL:-}"
MARINA_RELEASE_ID="${MARINA_RELEASE_ID:-}"     # not uploaded yet
BUNDLE_SHA256="fd0e86289480492986980133b32c4c25e321be89e20f793923c92e756c81c9a2"

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
ROOTS=(marina_flag_s0 marina_shifts_s0)

present=true
for root in "${ROOTS[@]}"; do
    [[ -f "$MODEL_DATA_DIR/$root/best.ckpt" ]] || present=false
done

if $present && ! $FORCE; then
    info "${ROOTS[*]} already present, skipping (--force to re-download)"
else
    zip="$MODEL_DATA_DIR/marina_db_release_s0.zip"
    if [[ -n "$MARINA_BUNDLE_URL" ]]; then
        info "downloading the model bundle from $MARINA_BUNDLE_URL ..."
        curl -fL --retry 3 --progress-bar -o "$zip" "$MARINA_BUNDLE_URL"
    elif [[ -n "$MARINA_RELEASE_ID" ]]; then
        info "downloading the model bundle from Google Drive ..."
        if command -v gdown >/dev/null; then
            gdown "$MARINA_RELEASE_ID" --output "$zip"
        else
            # confirm=t skips Drive's "can't scan this file for viruses" page on large files.
            curl -fL --retry 3 --progress-bar -o "$zip" \
                "https://drive.usercontent.google.com/download?id=${MARINA_RELEASE_ID}&export=download&confirm=t"
        fi
    else
        error "no download source for the model bundle: set MARINA_BUNDLE_URL or MARINA_RELEASE_ID"
        error "(or place marina_flag_s0/ and marina_shifts_s0/ in $MODEL_DATA_DIR yourself)"
        exit 1
    fi
    if command -v sha256sum >/dev/null; then
        got=$(sha256sum "$zip" | cut -d' ' -f1)
        if [[ "$got" != "$BUNDLE_SHA256" ]]; then
            error "$zip has sha256 $got, expected $BUNDLE_SHA256 -- incomplete download or wrong file"
            exit 1
        fi
    fi
    if ! unzip -tq "$zip" >/dev/null; then
        error "$zip is not a valid zip -- is the file shared publicly?"
        exit 1
    fi
    for root in "${ROOTS[@]}"; do rm -rf "${MODEL_DATA_DIR:?}/$root"; done
    unzip -q -o "$zip" -d "$MODEL_DATA_DIR"
    rm "$zip"
    info "${ROOTS[*]} -> $MODEL_DATA_DIR"
fi

# Earlier default models are no longer served; they are left on disk for rollback.
for old in marina_db_s1 marina_uniqmult_s1; do
    if [[ -d "$MODEL_DATA_DIR/$old" ]]; then
        info "$old (previous model) is unused; remove it to free ~2.3 GB: rm -rf $MODEL_DATA_DIR/$old"
    fi
done

# The backend container runs as an unprivileged user, so the weights must be
# world-readable (checkpoints copied in by hand are often owner-only).
chmod -R a+rX "$MODEL_DATA_DIR" 2>/dev/null \
    || error "could not make $MODEL_DATA_DIR world-readable; the backend may fail to load it"

cat << EOF > "$MODEL_DATA_DIR/models.json"
{
    "models": [
        {
            "id": "marina_flag_s0",
            "root": "marina_flag_s0",
            "type": "marina",
            "default": true,
            "display_name": "MARINA-DB flagship (unique-multiplicity, formula-capable)"
        },
        {
            "id": "marina_shifts_s0",
            "root": "marina_shifts_s0",
            "type": "marina",
            "role": "hsqc_unsigned",
            "display_name": "MARINA-DB HSQC shifts-only (no multiplicity sign)"
        }
    ]
}
EOF
info "wrote $MODEL_DATA_DIR/models.json"
