#!/usr/bin/env bash
# Push the chain's offline W&B runs to wandb.ai.
#
# Expanse compute nodes have no outbound network, so sdsc-train.sbatch sets
# WANDB_MODE=offline and every chunk writes an offline run directory to Lustre. Login
# nodes do have network, so the sync runs there. Because the chain pins the W&B run id
# across chunks, syncing all of them folds the whole chain into a single W&B run.
#
# Incremental: each chunk (SLURM job) writes its own offline-run-* dir. A finished chunk's
# dir is static forever; only the currently-training chunk keeps growing. So we track each
# dir's newest-file mtime in a state file ($WANDB_DIR/.sync_state) and only re-sync a dir
# whose mtime has advanced since the last sync -- unchanged dirs (finished, or just idle for
# a moment) are skipped without invoking wandb, which is where the speedup comes from.
#
# We never mark a dir .synced: syncing always uses --no-mark-synced, so a chunk that goes
# quiet is merely skipped this pass and picked back up if it grows again. There is no
# finalize step and thus no premature-finalize hazard (nothing to tune like the old
# QUIET_SECS): a dir being marked done can never strand its later data.
#
# Safe to re-run.
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
WANDB_DIR="\$PROJECT_ROOT/runs/wandb/wandb"  # sdsc-train.sbatch sets WANDB_DIR=/runs/wandb; wandb nests offline-run-* under a further wandb/
STATE="\$WANDB_DIR/.sync_state"              # "<newest-mtime> <dir>" per dir; kept OUTSIDE the run dirs so writing it can't bump the mtime we watch

shopt -s nullglob
runs=("\$WANDB_DIR"/offline-run-*)
if [ \${#runs[@]} -eq 0 ]; then
    echo "No offline runs under \$WANDB_DIR."
    exit 0
fi

declare -A seen=()
if [ -f "\$STATE" ]; then
    while read -r m d; do seen["\$d"]=\$m; done < "\$STATE"
fi

declare -A mtime=()
tosync=()
for r in "\${runs[@]}"; do
    last=\$(find "\$r" -type f -printf '%T@\n' 2>/dev/null | sort -rn | head -1)
    last=\${last%.*}; last=\${last:-0}
    mtime["\$r"]=\$last
    if [ "\${seen[\$r]:-}" != "\$last" ]; then
        tosync+=("\$r")                    # new dir, or grew since the last sync
    fi
done

echo "\${#runs[@]} run(s): \$(( \${#runs[@]} - \${#tosync[@]} )) unchanged, \${#tosync[@]} to sync."
if [ \${#tosync[@]} -eq 0 ]; then
    exit 0
fi

module load singularitypro >/dev/null 2>&1 || true
export WANDB_API_KEY=\$(python3 -c "import json;print(json.load(open('\$PROJECT_ROOT/wandb_api_key.json'))['key'])")

# OPENBLAS_NUM_THREADS: this also runs on a login node, where OpenBLAS sizing its buffers
# from the core count overruns the per-user memory cap during import. See sdsc-stage.sh.
# --no-mark-synced: never finalize, so any dir stays re-syncable if it grows later.
singularity exec --cleanenv \
    --env WANDB_API_KEY="\$WANDB_API_KEY" \
    --env OPENBLAS_NUM_THREADS=1 --env OMP_NUM_THREADS=1 \
    -B "\$PROJECT_ROOT":"\$PROJECT_ROOT" \
    "\$PROJECT_ROOT/images/marina.sif" \
    wandb sync --no-mark-synced "\${tosync[@]}"

# Record mtimes only after a successful sync (set -e aborts above on failure, leaving the old
# state intact so the dir re-syncs next pass). Write every run dir's current mtime, atomically.
tmp=\$(mktemp)
for r in "\${runs[@]}"; do printf '%s %s\n' "\${mtime[\$r]}" "\$r" >> "\$tmp"; done
mv "\$tmp" "\$STATE"
EOF
