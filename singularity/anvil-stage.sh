#!/usr/bin/env bash
# Stage code + verify data for the Purdue Anvil (AnvilAI) MARINA training chain.
#
# Code staging is GIT-CLONE at a pinned COMMIT on the Anvil login node (Anvil's ~/.ssh/id_rsa is
# registered on GitHub, 2026-09-16), the same convention as DeltaAI; $CODE_DIR/COMMIT records the SHA
# so the pin is auditable on-box. Data (dataset zip, benchmark journal, W&B key) lives on
# /anvil/projects/x-bio260190/atong1 and is rsync'd only when missing/changed.
#
#   Usage:  COMMIT=<full-40-char-SHA> bash singularity/anvil-stage.sh [DATASET ...]
#
# SSH alias `anvil-ai` (pubkey, no ControlMaster) must be in ~/.ssh/config.
set -euo pipefail

HOST="${ANVIL_HOST:-anvil-ai}"
PROJECT_ROOT="${PROJECT_ROOT:-/anvil/projects/x-bio260190/atong1}"
LOCAL_ROOT="${LOCAL_ROOT:-$HOME/Workspace}"
COMMIT="${COMMIT:?set COMMIT=<full 40-char SHA>: experiments are pinned to an exact commit}"
CODE_NAME="${CODE_NAME:-MARINA-2d}"
REPO="${REPO:-git@github.com:atong28/MARINA.git}"
DATASETS=("$@")

SSH=(ssh -o BatchMode=yes "$HOST")

echo "==> layout under $PROJECT_ROOT"
"${SSH[@]}" "mkdir -p '$PROJECT_ROOT'/{code,images,datasets,benchmark,runs/wandb,slurm-logs,fp-artifacts}"

echo "==> code: git clone/checkout $COMMIT -> $PROJECT_ROOT/code/$CODE_NAME"
"${SSH[@]}" bash -s <<EOF
set -euo pipefail
export GIT_SSH_COMMAND="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new"
CODE="$PROJECT_ROOT/code/$CODE_NAME"
if [ ! -d "\$CODE/.git" ]; then
    git clone --quiet "$REPO" "\$CODE"
fi
cd "\$CODE"
git fetch --all --quiet
git checkout --quiet --detach "$COMMIT"
head=\$(git rev-parse HEAD)
[ "\$head" = "$COMMIT" ] || { echo "  checkout landed on \$head, not $COMMIT" >&2; exit 1; }
dirty=\$(git status --porcelain | grep -v "^?? COMMIT\$" || true)
[ -z "\$dirty" ] || { echo "  clone tree is dirty after checkout:" >&2; echo "\$dirty" >&2; exit 1; }
echo "$COMMIT" > COMMIT
echo "  code pinned at \$head (clean detached checkout)"
EOF

echo "==> datasets"
for ds in "${DATASETS[@]}"; do
    src="$LOCAL_ROOT/Datasets/${ds}.zip"
    [ -f "$src" ] || { echo "  MISSING $src" >&2; exit 1; }
    rsync -a --info=progress2 "$src" "$HOST:$PROJECT_ROOT/datasets/"
    [ -f "$src.sha256" ] && rsync -a "$src.sha256" "$HOST:$PROJECT_ROOT/datasets/" && \
        "${SSH[@]}" "cd '$PROJECT_ROOT/datasets' && sha256sum -c '${ds}.zip.sha256'"
done

echo "==> benchmark journal + W&B key"
rsync -a "$LOCAL_ROOT/Benchmark/benchmark-journal.pkl" "$HOST:$PROJECT_ROOT/benchmark/"
rsync -a "$LOCAL_ROOT/wandb_api_key.json" "$HOST:$PROJECT_ROOT/"

echo "==> image"
if "${SSH[@]}" "test -f '$PROJECT_ROOT/images/marina.sif'"; then
    "${SSH[@]}" "apptainer exec '$PROJECT_ROOT/images/marina.sif' python -c \
        'import torch, rdkit, pytorch_lightning as pl; print(\"  image OK | torch\", torch.__version__, \"| pl\", pl.__version__, \"| rdkit\", rdkit.__version__)'" || \
        echo "  image present but smoke import failed" >&2
else
    echo "  NO image at $PROJECT_ROOT/images/marina.sif -- rsync $LOCAL_ROOT/images/marina.sif there first" >&2
fi

echo "==> code smoke import in the image (CPU, login node)"
"${SSH[@]}" "cd '$PROJECT_ROOT/code/$CODE_NAME' && DATASET_ROOT=/tmp apptainer exec --cleanenv --pwd \$PWD -B \$PWD:\$PWD --env DATASET_ROOT=/tmp '$PROJECT_ROOT/images/marina.sif' python -c 'import src.modules; from src.modules.core.const import INPUTS_CANONICAL_ORDER; print(\"  import OK\", INPUTS_CANONICAL_ORDER)'" 2>&1 | grep -v "INFO" || true

cat <<EOF

Staged. Launch one seed with:

  ssh $HOST "cd $PROJECT_ROOT/code/$CODE_NAME && \\
     EXPERIMENT_NAME=marina-db-open-2d-uniqmult-formula-s0 DATASET=${DATASETS[0]:-MARINA-DB-OPEN-2D} \\
     FP_TYPE=RankingEntropyUniqueMultiplicity SEED=0 \\
     sbatch --job-name=mdbopen2d-s0-\$(date +%Y%m%d) singularity/anvil-train.sbatch"
EOF
