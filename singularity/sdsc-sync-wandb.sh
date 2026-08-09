#!/usr/bin/env bash
# Push the chain's offline W&B runs to wandb.ai.
#
# Expanse compute nodes have no outbound network, so sdsc-train.sbatch sets
# WANDB_MODE=offline and every chunk writes an offline run directory to Lustre. Login
# nodes do have network, so the sync runs there. Because the chain pins the W&B run id
# across chunks, syncing all of them folds the whole chain into a single W&B run.
#
# Safe to re-run: wandb skips run directories it has already synced.
#
#   Usage:  bash singularity/sdsc-sync-wandb.sh <sdsc-username>
set -euo pipefail

SOCK="${SDSC_SOCK:-$HOME/.ssh/sdsc.sock}"
HOST="${SDSC_HOST:-login.expanse.sdsc.edu}"
PROJECT_ROOT="${PROJECT_ROOT:-/expanse/lustre/projects/sdp158/atong1}"

USER_NAME="${1:-}"
if [ -z "$USER_NAME" ]; then
    echo "usage: $0 <sdsc-username>" >&2
    exit 2
fi
TARGET="${USER_NAME}@${HOST}"

if ! ssh -S "$SOCK" -O check "$TARGET" 2>/dev/null; then
    echo "No live control socket at $SOCK. Run singularity/sdsc-connect.sh first." >&2
    exit 1
fi

ssh -S "$SOCK" "$TARGET" bash -s <<EOF
set -euo pipefail
PROJECT_ROOT='$PROJECT_ROOT'
WANDB_DIR="\$PROJECT_ROOT/runs/wandb"

shopt -s nullglob
runs=("\$WANDB_DIR"/offline-run-*)
if [ \${#runs[@]} -eq 0 ]; then
    echo "No offline runs under \$WANDB_DIR."
    exit 0
fi
echo "Found \${#runs[@]} offline run(s)."

module load singularitypro >/dev/null 2>&1 || true
export WANDB_API_KEY=\$(python3 -c "import json;print(json.load(open('\$PROJECT_ROOT/wandb_api_key.json'))['key'])")

# OPENBLAS_NUM_THREADS: this also runs on a login node, where OpenBLAS sizing its buffers
# from the core count overruns the per-user memory cap during import. See sdsc-stage.sh.
singularity exec --cleanenv \
    --env WANDB_API_KEY="\$WANDB_API_KEY" \
    --env OPENBLAS_NUM_THREADS=1 --env OMP_NUM_THREADS=1 \
    -B "\$PROJECT_ROOT":"\$PROJECT_ROOT" \
    "\$PROJECT_ROOT/images/marina.sif" \
    wandb sync "\${runs[@]}"
EOF
