#!/usr/bin/env bash
# Cache test-split fingerprint predictions for bit-confidence calibration, on Expanse.
#
# The isotonic curve a deployment ships as `calibration.json` is fitted per checkpoint,
# and a curve fitted on one fingerprint vocabulary is NOT valid for another -- the raw
# sigmoid means something different when the target space has a different density. So a
# new vocabulary needs its own pass before its confidences can be shown to a user.
#
# This is the expensive half (a forward pass over the whole test split). The fit itself
# is `analysis/bit-confidence/scripts/02_calibrate.py`, which needs scikit-learn and so
# does NOT run in marina.sif -- pull the .npz and fit it in the analysis pixi env.
#
#   Submit:  CKPT_DIR=/expanse/lustre/projects/sdp158/atong1/calibration/marina_multiplicity \
#            TAG=multiplicity \
#            sbatch singularity/sdsc-calibrate.sbatch
#
# CKPT_DIR must contain params.json and best.ckpt. A run directory names its checkpoint
# `epoch_epoch=<N>.ckpt`, so link it into place rather than renaming the original.
#
#SBATCH --job-name=marina-calib
#SBATCH --account=sdp158
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=1
#SBATCH --mem=92G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=/expanse/lustre/projects/sdp158/atong1/slurm-logs/%x-%j.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/expanse/lustre/projects/sdp158/atong1}"
CKPT_DIR="${CKPT_DIR:?set CKPT_DIR (must hold params.json and best.ckpt)}"
TAG="${TAG:?set TAG, e.g. multiplicity}"
DATASET="${DATASET:-MARINA1}"
IMAGE="${IMAGE:-$PROJECT_ROOT/images/marina.sif}"
# Deliberately a separate checkout from the training one: this must be a known commit
# with the vocabulary registered in FP_LOADERS, not whatever a training tree happens to
# have uncommitted.
CODE_DIR="${CODE_DIR:-$PROJECT_ROOT/code/MARINA-calib}"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/calibration/results}"
LIMIT="${LIMIT:-0}"          # 0 = the whole test split
BATCH_SIZE="${BATCH_SIZE:-16}"

for f in params.json best.ckpt; do
    [[ -e "$CKPT_DIR/$f" ]] || { echo "[calib] missing $CKPT_DIR/$f" >&2; exit 1; }
done
mkdir -p "$OUT_DIR"

echo "[calib] job=$SLURM_JOB_ID tag=$TAG"
echo "[calib] ckpt: $CKPT_DIR"
python3 -c "import json;p=json.load(open('$CKPT_DIR/params.json'));print('[calib] fp_type:',p.get('fp_type'),'out_dim:',p.get('out_dim'),'seed:',p.get('seed'))"

module purge
module load singularitypro

# Same staging as training: the arrow shards are many small random reads, which Lustre
# handles badly and node-local NVMe handles well.
LOCAL_DATA="/scratch/$USER/job_$SLURM_JOB_ID/data"
mkdir -p "$LOCAL_DATA"
echo "[stage] unpacking $DATASET.zip to $LOCAL_DATA"
time unzip -q "$PROJECT_ROOT/datasets/${DATASET}.zip" -d "$LOCAL_DATA"

# Non-Morgan vocabularies live outside the dataset zip. Without this the loader would
# find no feature map and rebuild the whole vocabulary inside a GPU job.
FP_OVERLAY="$PROJECT_ROOT/fp-artifacts/$DATASET"
if [ -d "$FP_OVERLAY" ] && [ -n "$(ls -A "$FP_OVERLAY" 2>/dev/null)" ]; then
    echo "[stage] overlaying prebuilt fingerprint artifacts from $FP_OVERLAY"
    rsync -a "$FP_OVERLAY/" "$LOCAL_DATA/"
fi

nvidia-smi --query-gpu=index,name,memory.total --format=csv

singularity exec --nv --cleanenv \
    --pwd /code \
    -B "$CODE_DIR":/code \
    -B "$LOCAL_DATA":/workspace \
    -B "$CKPT_DIR":/ckpt:ro \
    -B "$OUT_DIR":/out \
    -B "$PROJECT_ROOT/benchmark":/benchmark \
    --env DATASET_ROOT=/workspace \
    --env BENCHMARK_ROOT=/benchmark \
    --env PVC_ROOT=/out \
    "$IMAGE" \
    python3 analysis/bit-confidence/scripts/01_predict.py \
        --ckpt_dir /ckpt \
        --out "/out/preds_${TAG}.npz" \
        --limit "$LIMIT" \
        --batch_size "$BATCH_SIZE"

echo "[calib] wrote $OUT_DIR/preds_${TAG}.npz"
ls -la "$OUT_DIR/preds_${TAG}.npz"
echo "[calib] next: fit the curve off-cluster --"
echo "        scp this .npz down, then in analysis/bit-confidence/:"
echo "        pixi run python scripts/02_calibrate.py --preds <npz> --tag $TAG"
