#!/usr/bin/env bash
# RECOVERY TOOL. Delta compute nodes have direct egress (confirmed 2026-08-25), so
# delta-train.sbatch logs W&B ONLINE and normally no sync is needed. Use this only if a run was
# forced offline (e.g. a node without egress, or WANDB_MODE=offline set manually): it pushes any
# offline run directories under /projects to wandb.ai. Because the chain pins the W&B run id
# across chunks, syncing all of them folds the whole chain into a single W&B run.
#
# Safe to re-run: wandb skips run directories it has already synced.
#
#   Usage:  bash singularity/delta-sync-wandb.sh <ncsa-username>
set -euo pipefail

SOCK="${NCSA_SOCK:-$HOME/.ssh/ncsa.sock}"
HOST="${NCSA_HOST:-dt-login01.delta.ncsa.illinois.edu}"
PROJECT_ROOT="${PROJECT_ROOT:-/projects/bibx/atong1}"

USER_NAME="${1:-}"
if [ -z "$USER_NAME" ]; then
    echo "usage: $0 <ncsa-username>" >&2
    exit 2
fi
TARGET="${USER_NAME}@${HOST}"

if ! ssh -S "$SOCK" -O check "$TARGET" 2>/dev/null; then
    echo "No live control socket at $SOCK. Run singularity/delta-connect.sh first." >&2
    exit 1
fi

ssh -S "$SOCK" "$TARGET" bash -s <<EOF
set -euo pipefail
PROJECT_ROOT='$PROJECT_ROOT'
# wandb appends its own wandb/ subdir to WANDB_DIR, so offline runs usually land in
# runs/wandb/wandb/, NOT runs/wandb/. A glob at one fixed depth reports "No offline runs" and
# exits 0 -- silently never reaching wandb.ai -- if the real depth differs. Search both levels
# so the sync is correct regardless of how wandb nested them here (verify actual depth once).
mapfile -t runs < <(find "\$PROJECT_ROOT/runs/wandb" -maxdepth 2 -type d -name 'offline-run-*' 2>/dev/null)
if [ \${#runs[@]} -eq 0 ]; then
    echo "No offline runs under \$PROJECT_ROOT/runs/wandb (searched 2 levels)."
    exit 0
fi
echo "Found \${#runs[@]} offline run(s)."

module load apptainer >/dev/null 2>&1 || true
export WANDB_API_KEY=\$(python3 -c "import json;print(json.load(open('\$PROJECT_ROOT/wandb_api_key.json'))['key'])")

# OPENBLAS_NUM_THREADS: defensive on a login node -- see delta-stage.sh. Harmless if Delta's
# login nodes do not cap per-user memory the way Expanse's do (verify).
apptainer exec --cleanenv \
    --env WANDB_API_KEY="\$WANDB_API_KEY" \
    --env OPENBLAS_NUM_THREADS=1 --env OMP_NUM_THREADS=1 \
    -B "\$PROJECT_ROOT":"\$PROJECT_ROOT" \
    "\$PROJECT_ROOT/images/marina.sif" \
    wandb sync "\${runs[@]}"
EOF
