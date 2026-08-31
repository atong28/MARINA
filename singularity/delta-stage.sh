#!/usr/bin/env bash
# Push everything the Delta training chain needs onto NCSA Delta, over the ControlMaster
# socket opened by delta-connect.sh. Re-runnable: rsync only sends what changed, so this
# doubles as "ship my latest code" between chunks.
#
#   Usage:  bash singularity/delta-stage.sh <ncsa-username> [dataset ...]
#   e.g.    bash singularity/delta-stage.sh atong1 MARINA1 MARINA3
#
# Datasets default to MARINA1. The .sif itself is NOT built here -- Delta does not grant the
# root/fakeroot needed for `apptainer build`. Build it on a machine you own
# (`sudo apptainer build marina.sif singularity/marina.def`, or reuse the Expanse marina.sif
# -- it is arch-agnostic PyTorch) and upload once:
#   scp -o ControlPath=~/.ssh/ncsa.sock marina.sif \
#       <user>@dt-login01.delta.ncsa.illinois.edu:/projects/bibx/atong1/images/
set -euo pipefail

SOCK="${NCSA_SOCK:-$HOME/.ssh/ncsa.sock}"
HOST="${NCSA_HOST:-dt-login01.delta.ncsa.illinois.edu}"
PROJECT_ROOT="${PROJECT_ROOT:-/projects/bibx/atong1}"
LOCAL_ROOT="${LOCAL_ROOT:-$HOME/Workspace}"

USER_NAME="${1:-}"
if [ -z "$USER_NAME" ]; then
    echo "usage: $0 <ncsa-username> [dataset ...]" >&2
    exit 2
fi
shift
DATASETS=("$@")
[ ${#DATASETS[@]} -eq 0 ] && DATASETS=(MARINA1)

TARGET="${USER_NAME}@${HOST}"
SSH=(ssh -S "$SOCK" "$TARGET")
RSYNC_E="ssh -S $SOCK"

if ! ssh -S "$SOCK" -O check "$TARGET" 2>/dev/null; then
    echo "No live control socket at $SOCK." >&2
    echo "Run: bash singularity/delta-connect.sh $USER_NAME $HOST" >&2
    exit 1
fi

echo "==> creating layout under $PROJECT_ROOT"
"${SSH[@]}" "mkdir -p '$PROJECT_ROOT'/{code,images,datasets,benchmark,runs/wandb,slurm-logs}"

echo "==> syncing repo"
# .pixi (8.5G) and analysis (14G) are local-only working state; frontend is not used by
# training. Excluding them takes the transfer from 25G to ~20M.
rsync -az --delete -e "$RSYNC_E" \
    --exclude '.pixi/' \
    --exclude 'analysis/' \
    --exclude 'frontend/' \
    --exclude 'data/' \
    --exclude '.git/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    "$LOCAL_ROOT/MARINA/" "$TARGET:$PROJECT_ROOT/code/MARINA/"

echo "==> syncing datasets: ${DATASETS[*]}"
for ds in "${DATASETS[@]}"; do
    src="$LOCAL_ROOT/Datasets/${ds}.zip"
    if [ ! -f "$src" ]; then
        echo "  MISSING $src -- skipping" >&2
        continue
    fi
    echo "  $ds ($(du -h "$src" | cut -f1))"
    rsync -a --info=progress2 -e "$RSYNC_E" "$src" "$TARGET:$PROJECT_ROOT/datasets/"
done

echo "==> syncing benchmark data"
# BenchmarkCosineCallback reads these two from BENCHMARK_ROOT every validation epoch.
# benchmark.pkl also ships inside the dataset zips, but BENCHMARK_ROOT is a separate,
# writable location (post-training results are written under it), so it gets its own copy.
# benchmark-journal.pkl is NOT in the zips and must come from the hand-built Benchmark dir.
for f in "$LOCAL_ROOT/Datasets/MARINA1/benchmark.pkl" \
         "$LOCAL_ROOT/Benchmark/benchmark-journal.pkl"; do
    if [ ! -f "$f" ]; then
        echo "  MISSING $f -- val/benchmark*_cos will be empty" >&2
        continue
    fi
    rsync -a -e "$RSYNC_E" "$f" "$TARGET:$PROJECT_ROOT/benchmark/"
done

echo "==> syncing W&B key (used by the login-node sync, not by compute nodes)"
rsync -a -e "$RSYNC_E" "$LOCAL_ROOT/wandb_api_key.json" "$TARGET:$PROJECT_ROOT/"

echo "==> verifying image"
if "${SSH[@]}" "test -f '$PROJECT_ROOT/images/marina.sif'"; then
    ssh -S "$SOCK" "$TARGET" bash -s <<EOF
set -eu
module load apptainer >/dev/null 2>&1 || true

# OPENBLAS_NUM_THREADS is defensive here. On Expanse, login nodes cap per-user memory while
# still reporting every core, and OpenBLAS sizes its thread-local buffers from the core count
# -- enough to fail numpy's import ("Memory allocation still failed after 10 retries"). Whether
# Delta's login nodes have the same cap is (verify); the flag is harmless if they do not.
apptainer exec \
    --env OPENBLAS_NUM_THREADS=1 --env OMP_NUM_THREADS=1 \
    '$PROJECT_ROOT/images/marina.sif' \
    python -c "
# Must match what marina.def actually installs. lightning and torch_geometric were dropped
# from the image (nothing under src/ imports them) -- importing them here would fail a
# perfectly good image.
import torch, numpy, rdkit, pydantic, pytorch_lightning, torchmetrics, tyro, pyarrow, tqdm, wandb
from pydantic.dataclasses import dataclass
print('image OK | torch', torch.__version__, '| pl', pytorch_lightning.__version__)
# get_arch_list() is empty without a visible GPU and login nodes have none.
print('arch flags:', torch._C._cuda_getArchFlags())
"
EOF
else
    echo "  NO IMAGE at $PROJECT_ROOT/images/marina.sif -- build and upload it (see header)" >&2
fi

cat <<EOF

Staged. Submit a chain with (A100 supports bf16, unlike Expanse's V100):

  ssh -S $SOCK $TARGET \\
    "cd $PROJECT_ROOT/code/MARINA && \\
     EXPERIMENT_NAME=marina-delta-s0 DATASET=${DATASETS[0]} \\
     TRAIN_ARGS='--input_types hsqc c_nmr h_nmr mw mass_spec --seed 0 --precision bf16-mixed' \\
     sbatch singularity/delta-train.sbatch"

EOF
