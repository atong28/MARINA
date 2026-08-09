#!/usr/bin/env bash
# Push everything the SDSC training chain needs onto Expanse, over the ControlMaster
# socket opened by sdsc-connect.sh. Re-runnable: rsync only sends what changed, so this
# doubles as "ship my latest code" between chunks.
#
#   Usage:  bash singularity/sdsc-stage.sh <sdsc-username> [dataset ...]
#   e.g.    bash singularity/sdsc-stage.sh atong1 MARINA1 MARINA3
#
# Datasets default to MARINA1. The .sif itself is NOT built here -- Expanse does not
# grant the root/fakeroot needed for `singularity build`. Build it on a machine you own
# (`sudo singularity build marina.sif singularity/marina.def`) and upload once:
#   scp -o ControlPath=~/.ssh/sdsc.sock marina.sif \
#       <user>@login.expanse.sdsc.edu:/expanse/lustre/projects/sdp158/atong1/images/
set -euo pipefail

SOCK="${SDSC_SOCK:-$HOME/.ssh/sdsc.sock}"
HOST="${SDSC_HOST:-login.expanse.sdsc.edu}"
PROJECT_ROOT="${PROJECT_ROOT:-/expanse/lustre/projects/sdp158/atong1}"
LOCAL_ROOT="${LOCAL_ROOT:-$HOME/atong}"

USER_NAME="${1:-}"
if [ -z "$USER_NAME" ]; then
    echo "usage: $0 <sdsc-username> [dataset ...]" >&2
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
    echo "Run: bash singularity/sdsc-connect.sh $USER_NAME $HOST" >&2
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
    "${SSH[@]}" "module load singularitypro >/dev/null 2>&1; \
        singularity exec '$PROJECT_ROOT/images/marina.sif' \
        python -c \"import torch, rdkit, pydantic, lightning, torchmetrics, tyro, pyarrow, torch_geometric; \
print('image OK, torch', torch.__version__); print('arch_list:', torch.cuda.get_arch_list())\""
else
    echo "  NO IMAGE at $PROJECT_ROOT/images/marina.sif -- build and upload it (see header)" >&2
fi

cat <<EOF

Staged. Submit a chain with:

  ssh -S $SOCK $TARGET \\
    "cd $PROJECT_ROOT/code/MARINA && \\
     EXPERIMENT_NAME=marina-sdsc-s0 DATASET=${DATASETS[0]} \\
     TRAIN_ARGS='--input_types {hsqc,c_nmr,h_nmr,mw,mass_spec} --seed 0' \\
     sbatch singularity/sdsc-train.sbatch"

EOF
