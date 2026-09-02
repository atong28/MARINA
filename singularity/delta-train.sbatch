#!/usr/bin/env bash
# One chunk of a self-resubmitting MARINA training chain on NCSA Delta.
#
# Delta caps the gpuA100x4 partition at 48h, which is far short of a full MARINA run
# (750 epochs). Each chunk queues its own successor with --dependency=afterany BEFORE
# training starts, so a hard walltime kill -- which gives the script no chance to run
# cleanup -- still leaves the next chunk queued. --resume then picks up last.ckpt with
# optimizer, LR schedule and EarlyStopping patience intact.
#
# The chain stops when a chunk exits 0, which writes a COMPLETE sentinel; the already-queued
# successor sees it and exits immediately without burning SUs. MAX_CHUNKS is the backstop
# against a chunk that fails instantly every time.
#
#   Submit (full node, 4 GPU):
#            EXPERIMENT_NAME=marina-delta-s0 DATASET=MARINA1 \
#            TRAIN_ARGS='--input_types hsqc c_nmr h_nmr mw mass_spec --seed 0 --precision bf16-mixed' \
#            sbatch singularity/delta-train.sbatch
#
#   Submit (2-GPU shared): the memmap packing step below moves the dataset out of per-worker RAM,
#   so a 2-GPU job fits in ~64G instead of 120G. GPUS_PER_NODE/CPUS_PER_TASK/MEM propagate down the
#   chain, so set them once on chunk 1:
#            EXPERIMENT_NAME=marina-delta-s0 DATASET=MARINA1 \
#            GPUS_PER_NODE=2 CPUS_PER_TASK=32 MEM=64G \
#            TRAIN_ARGS='--input_types hsqc c_nmr h_nmr mw mass_spec --seed 0 --precision bf16-mixed' \
#            sbatch --gpus-per-node=2 --cpus-per-task=32 --mem=64G singularity/delta-train.sbatch
#
# --precision bf16-mixed: A100 is sm_80 with real bf16 tensor cores. This differs from Expanse,
# where V100 (sm_70) has no bf16 hardware and must use 16-mixed.
#
# Write --input_types space-separated, NOT as {hsqc,c_nmr,...}. The braced form in the Nautilus
# job specs is bash brace expansion, which happens before variable expansion and is NOT reapplied
# to the contents of $TRAIN_ARGS -- tyro would receive one literal token and reject it.
#   Watch:   ssh -S ~/.ssh/ncsa.sock <user>@dt-login01.delta.ncsa.illinois.edu 'squeue -u <user>'
#
#SBATCH --job-name=marina
#SBATCH --account=bibx-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --mem=240G
#SBATCH --time=48:00:00
#SBATCH --no-requeue
#SBATCH --output=/projects/bibx/atong1/slurm-logs/%x-%j.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/projects/bibx/atong1}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:?set EXPERIMENT_NAME when submitting}"
DATASET="${DATASET:-MARINA1}"
TRAIN_ARGS="${TRAIN_ARGS:-}"
IMAGE="${IMAGE:-$PROJECT_ROOT/images/marina.sif}"
CODE_DIR="${CODE_DIR:-$PROJECT_ROOT/code/MARINA}"
CHAIN_SCRIPT="${CHAIN_SCRIPT:-$CODE_DIR/singularity/delta-train.sbatch}"
MAX_CHUNKS="${MAX_CHUNKS:-12}"
CHAIN_INDEX="${CHAIN_INDEX:-1}"

# Partition/account propagate down the chain: chunk 1 takes them from the #SBATCH defaults
# (or a CLI --partition/--account override on the initial sbatch, surfaced as SLURM_JOB_*),
# and every successor is resubmitted onto the SAME partition. This is what lets a run submitted
# with --partition=gpuA40x4 stay on A40 across all chunks instead of bouncing back to A100.
PARTITION="${PARTITION:-${SLURM_JOB_PARTITION:-gpuA100x4}}"
ACCOUNT="${ACCOUNT:-${SLURM_JOB_ACCOUNT:-bibx-delta-gpu}}"
# GPU/CPU/mem also propagate down the chain (same reason as partition): a run started as a
# 2-GPU shared job stays 2-GPU across all chunks instead of the successor reverting to the
# #SBATCH full-node (4-GPU) request. NPROC (torchrun ranks) is derived separately from
# nvidia-smi -L below, so it tracks whatever is actually allocated.
GPUS_PER_NODE="${GPUS_PER_NODE:-${SLURM_GPUS_PER_NODE:-4}}"
CPUS_PER_TASK="${CPUS_PER_TASK:-${SLURM_CPUS_PER_TASK:-64}}"
MEM="${MEM:-${SLURM_MEM_PER_NODE:-240G}}"

# Pinning the run id is what makes the chain a single run rather than N runs: it fixes both the
# results directory (main.py reads SMART_RUN_ID) and the W&B run id, so every chunk resumes the
# same last.ckpt and logs to the same W&B run.
export SMART_RUN_ID="${SMART_RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"

RUN_DIR="$PROJECT_ROOT/runs/results/$EXPERIMENT_NAME/$SMART_RUN_ID"
mkdir -p "$RUN_DIR" "$PROJECT_ROOT/runs/wandb"

echo "[chain] chunk $CHAIN_INDEX/$MAX_CHUNKS  job=$SLURM_JOB_ID  run_id=$SMART_RUN_ID"
echo "[chain] run dir: $RUN_DIR"

if [[ -f "$RUN_DIR/COMPLETE" ]]; then
    echo "[chain] COMPLETE sentinel present; training already finished. Exiting."
    exit 0
fi

# Queue the successor first. A walltime kill sends SIGKILL after a short grace period, so
# anything after the training call is not guaranteed to run -- but a dependency registered now
# survives regardless of how this chunk dies.
if (( CHAIN_INDEX < MAX_CHUNKS )); then
    next_id=$(sbatch --parsable \
        --dependency=afterany:"$SLURM_JOB_ID" \
        --partition="$PARTITION" \
        --account="$ACCOUNT" \
        --gpus-per-node="$GPUS_PER_NODE" \
        --cpus-per-task="$CPUS_PER_TASK" \
        --mem="$MEM" \
        --job-name="${SLURM_JOB_NAME:-marina}" \
        --export=ALL,SMART_RUN_ID="$SMART_RUN_ID",CHAIN_INDEX="$((CHAIN_INDEX + 1))" \
        "$CHAIN_SCRIPT")
    echo "[chain] successor queued as $next_id (afterany:$SLURM_JOB_ID)"
else
    echo "[chain] MAX_CHUNKS reached; this is the final chunk, not resubmitting."
fi

module purge 2>/dev/null || true
module load apptainer 2>/dev/null || true

# The dataset goes on node-local /tmp (1.5TB on GPU nodes, purged at job end), NOT /projects.
# MARINA's arrow shards are read with many small random reads per batch; /projects is Taiga
# (center-wide spinning disk), the worst pattern for that, while /tmp is local NVMe/SSD. It is
# only ~2GB unpacked, so the copy costs a couple of minutes once per chunk. Checkpoints stay on
# /projects because /tmp dies with the job.
LOCAL_DATA="/tmp/$USER/job_$SLURM_JOB_ID/data"
mkdir -p "$LOCAL_DATA"
echo "[stage] unpacking $DATASET.zip to $LOCAL_DATA"
time unzip -q "$PROJECT_ROOT/datasets/${DATASET}.zip" -d "$LOCAL_DATA"

# Pack the Arrow shards into flat memmap-ready arrays so the dataset is paged in shared (via the
# OS page cache) instead of a full private copy per DataLoader worker -- the dominant host-RAM term
# (see scripts/dataset/pack_arrow.py; the Memmap* stores auto-engage when packed/ is present). This
# is what lets a 2-GPU shared job drop --mem from 120G to ~64G. Skip-if-exists; CPU-only, runs in
# the sif for numpy/pyarrow. If the zip already ships a packed/ dir this is a fast no-op.
echo "[stage] packing arrow -> memmap arrays"
time apptainer exec --cleanenv --pwd /code \
    -B "$CODE_DIR":/code \
    -B "$LOCAL_DATA":/workspace \
    "$IMAGE" \
    python scripts/dataset/pack_arrow.py --root /workspace

nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Derived from the allocation rather than hardcoded, so a job that lands on fewer GPUs cannot
# silently leave torchrun spawning 4 ranks for 2 devices. NOTE this changes the effective batch
# size: batch_size=32 is per DataLoader process, so global batch is 32 * NPROC *
# accumulate_grad_batches_num. 4 GPUs -> 512, matching the Nautilus/Expanse runs.
NPROC=$(nvidia-smi -L | wc -l)
if (( NPROC < 1 )); then
    echo "[chain] no GPUs visible; aborting before wasting the allocation." >&2
    exit 1
fi
echo "[chain] launching torchrun with $NPROC rank(s)"

# Delta compute nodes HAVE direct outbound network (confirmed 2026-08-25 on gpua017: no proxy,
# pypi/github 200), so W&B logs online straight from the node -- no offline-then-sync step, same
# as Nautilus. delta-sync-wandb.sh remains only as a recovery tool if a node ever lacks egress.
#
# configure_wandb reads the key from a FILE at WANDB_API_KEY_FILE (it ignores WANDB_API_KEY). The
# image auto-detects a "nautilus setup" and defaults that to /root/gurusmart/Moonshot/..., which
# does not exist here -- so bind the staged key in and override the path via env (const.py honours
# $WANDB_API_KEY_FILE, the same override used for DATASET_ROOT/PVC_ROOT/BENCHMARK_ROOT).
set +e
apptainer exec --nv --cleanenv \
    --pwd /code \
    -B "$CODE_DIR":/code \
    -B "$LOCAL_DATA":/workspace \
    -B "$PROJECT_ROOT/runs":/runs \
    -B "$PROJECT_ROOT/benchmark":/benchmark \
    -B "$PROJECT_ROOT/wandb_api_key.json":/wandb_api_key.json:ro \
    --env DATASET_ROOT=/workspace \
    --env PVC_ROOT=/runs \
    --env BENCHMARK_ROOT=/benchmark \
    --env WANDB_MODE=online \
    --env WANDB_API_KEY_FILE=/wandb_api_key.json \
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
