#!/usr/bin/env bash
# One chunk of a self-resubmitting MARINA training chain on SDSC Expanse.
#
# Expanse caps the gpu partition at 48h, which is far short of a full MARINA run
# (750 epochs). Each chunk queues its own successor with --dependency=afterany
# BEFORE training starts, so a hard walltime kill -- which gives the script no
# chance to run cleanup -- still leaves the next chunk queued. --resume then picks
# up last.ckpt with optimizer, LR schedule and EarlyStopping patience intact.
#
# The chain stops when a chunk exits 0, which writes a COMPLETE sentinel; the
# already-queued successor sees it and exits immediately without burning SUs.
# MAX_CHUNKS is the backstop against a chunk that fails instantly every time.
#
#   Submit:  EXPERIMENT_NAME=marina-sdsc-s0 DATASET=MARINA1 \
#            TRAIN_ARGS='--input_types hsqc c_nmr h_nmr mw mass_spec --seed 0' \
#            sbatch singularity/sdsc-train.sbatch
#
# Write --input_types space-separated, NOT as {hsqc,c_nmr,...}. The braced form in the
# Nautilus job specs is bash brace expansion, which happens before variable expansion and
# is NOT reapplied to the contents of $TRAIN_ARGS -- tyro would receive one literal token
# "{hsqc,...}" and reject it as an invalid choice.
#   Watch:   ssh -S ~/.ssh/sdsc.sock <user>@login.expanse.sdsc.edu 'squeue -u <user>'
#
#SBATCH --job-name=marina
#SBATCH --account=BIO260190
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --gpus=4
#SBATCH --mem=360G
#SBATCH --time=48:00:00
#SBATCH --no-requeue
#SBATCH --output=/expanse/lustre/projects/sdp158/atong1/slurm-logs/%x-%j.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/expanse/lustre/projects/sdp158/atong1}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:?set EXPERIMENT_NAME when submitting}"
DATASET="${DATASET:-MARINA1}"
TRAIN_ARGS="${TRAIN_ARGS:-}"
IMAGE="${IMAGE:-$PROJECT_ROOT/images/marina.sif}"
CODE_DIR="${CODE_DIR:-$PROJECT_ROOT/code/MARINA}"
CHAIN_SCRIPT="${CHAIN_SCRIPT:-$CODE_DIR/singularity/sdsc-train.sbatch}"
MAX_CHUNKS="${MAX_CHUNKS:-12}"
CHAIN_INDEX="${CHAIN_INDEX:-1}"

# Pinning the run id is what makes the chain a single run rather than N runs: it fixes
# both the results directory (main.py reads SMART_RUN_ID) and the W&B run id, so every
# chunk resumes the same last.ckpt and logs to the same W&B run.
export SMART_RUN_ID="${SMART_RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"

RUN_DIR="$PROJECT_ROOT/runs/results/$EXPERIMENT_NAME/$SMART_RUN_ID"
mkdir -p "$RUN_DIR" "$PROJECT_ROOT/runs/wandb"

echo "[chain] chunk $CHAIN_INDEX/$MAX_CHUNKS  job=$SLURM_JOB_ID  run_id=$SMART_RUN_ID"
echo "[chain] run dir: $RUN_DIR"

if [[ -f "$RUN_DIR/COMPLETE" ]]; then
    echo "[chain] COMPLETE sentinel present; training already finished. Exiting."
    exit 0
fi

# Queue the successor first. A walltime kill sends SIGKILL after a short grace period,
# so anything after the training call is not guaranteed to run -- but a dependency
# registered now survives regardless of how this chunk dies.
if (( CHAIN_INDEX < MAX_CHUNKS )); then
    next_id=$(sbatch --parsable \
        --dependency=afterany:"$SLURM_JOB_ID" \
        --job-name="${SLURM_JOB_NAME:-marina}" \
        --export=ALL,SMART_RUN_ID="$SMART_RUN_ID",CHAIN_INDEX="$((CHAIN_INDEX + 1))" \
        "$CHAIN_SCRIPT")
    echo "[chain] successor queued as $next_id (afterany:$SLURM_JOB_ID)"
else
    echo "[chain] MAX_CHUNKS reached; this is the final chunk, not resubmitting."
fi

module purge
module load singularitypro

# The dataset goes on node-local NVMe, not Lustre. MARINA's arrow shards are read with
# many small random reads per batch, which is the access pattern Lustre is worst at and
# local NVMe is best at. It is only ~2GB unpacked, so the copy costs a couple of minutes
# once per chunk. Checkpoints stay on Lustre because /scratch dies with the job.
LOCAL_DATA="/scratch/$USER/job_$SLURM_JOB_ID/data"
mkdir -p "$LOCAL_DATA"
echo "[stage] unpacking $DATASET.zip to $LOCAL_DATA"
time unzip -q "$PROJECT_ROOT/datasets/${DATASET}.zip" -d "$LOCAL_DATA"

nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Derived from the allocation rather than hardcoded, so `sbatch --gpus=2` (or a
# gpu-shared submission) cannot silently leave torchrun spawning 4 ranks for 2 devices.
# NOTE this changes the effective batch size: batch_size=32 is per DataLoader process, so
# global batch is 32 * NPROC * accumulate_grad_batches_num. 4 GPUs -> 512, matching the
# Nautilus runs; 2 GPUs -> 256, which is NOT comparable unless you double
# --accumulate_grad_batches_num to compensate.
NPROC=$(nvidia-smi -L | wc -l)
if (( NPROC < 1 )); then
    echo "[chain] no GPUs visible; aborting before wasting the allocation." >&2
    exit 1
fi
echo "[chain] launching torchrun with $NPROC rank(s)"

# Compute nodes have no outbound network, so W&B runs offline and is synced from a
# login node afterwards (see sdsc-sync-wandb.sh).
set +e
singularity exec --nv --cleanenv \
    --pwd /code \
    -B "$CODE_DIR":/code \
    -B "$LOCAL_DATA":/workspace \
    -B "$PROJECT_ROOT/runs":/runs \
    -B "$PROJECT_ROOT/benchmark":/benchmark \
    --env DATASET_ROOT=/workspace \
    --env PVC_ROOT=/runs \
    --env BENCHMARK_ROOT=/benchmark \
    --env WANDB_MODE=offline \
    --env WANDB_DIR=/runs/wandb \
    --env SMART_RUN_ID="$SMART_RUN_ID" \
    "$IMAGE" \
    torchrun --standalone --nproc_per_node "$NPROC" --module src.main marina \
        --experiment_name "$EXPERIMENT_NAME" \
        --resume \
        $TRAIN_ARGS
rc=$?
set -e

if (( rc == 0 )); then
    touch "$RUN_DIR/COMPLETE"
    echo "[chain] training returned 0 (early stopping or max epochs). Chain complete."
else
    echo "[chain] chunk exited $rc -- successor resumes from $RUN_DIR/last.ckpt"
fi
exit "$rc"
