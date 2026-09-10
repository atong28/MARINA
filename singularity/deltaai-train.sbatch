#!/usr/bin/env bash
# One chunk of a self-resubmitting MARINA training chain on NCSA DeltaAI (Grace-Hopper).
#
# DeltaAI differences vs the A100 Delta (delta-train.sbatch):
#   * partition ghx4 (GH200, 4 GPU/node, node-sharing) / account bibx-dtai-gh (2000 GPU-h).
#   * aarch64 image marina-arm64.sif (see deltaai-marina.def) -- the amd64 marina.sif will NOT run.
#   * precision bf16-mixed (GH200 = Hopper sm_90, real bf16 tensor cores).
#   * CODE IS GIT-CLONED at a pinned COMMIT, not rsync'd. The clone is done ON THE LOGIN NODE by
#     deltaai-stage.sh into $CODE_DIR (persistent, SHA-verified); this chunk binds it read-only,
#     so no compute-node git egress is required. Data (dataset/benchmark/key/fp-overlay) lives on
#     the shared /projects and is rsync'd once by the stage script.
#
# The 48h ghx4 walltime cap is beaten the same way as on Delta: queue the successor FIRST with
# --dependency=afterany:$SLURM_JOB_ID, pin SMART_RUN_ID so every chunk --resume's the same
# last.ckpt and logs to one W&B run, and stop cleanly on a COMPLETE sentinel.
#
#   Submit (1 GPU, global batch 512 = batch_size 512 * 1 rank * accum 1):
#            EXPERIMENT_NAME=marina-deltaai-substructure-s0 DATASET=MARINA-DB \
#            FP_TYPE=RankingEntropySubstructure SEED=0 \
#            sbatch singularity/deltaai-train.sbatch
#
#   Watch:  ssh -S ~/.ssh/ncsa-ai.sock atong1@gh-login01.delta.ncsa.illinois.edu 'squeue -u atong1'
#
#SBATCH --job-name=marina-ai
#SBATCH --account=bibx-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=110G
#SBATCH --time=48:00:00
#SBATCH --no-requeue
#SBATCH --output=/projects/bibx/atong1/slurm-logs/%x-%j.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/projects/bibx/atong1}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:?set EXPERIMENT_NAME when submitting}"
DATASET="${DATASET:-MARINA-DB}"
FP_TYPE="${FP_TYPE:-RankingEntropySubstructure}"
SEED="${SEED:-0}"
# Global batch = BATCH_SIZE * NPROC(=1 GPU) * ACCUM. 512*1*1 = 512, matching the 4xA100 runs
# (32 per-worker * 4 GPU * 4 accum). GH200 has 120GB, so 512 on one GPU is the plan; if the
# first job OOMs, drop to BATCH_SIZE=256 ACCUM=2 (still global 512).
BATCH_SIZE="${BATCH_SIZE:-512}"
ACCUM="${ACCUM:-1}"
INPUT_TYPES="${INPUT_TYPES:-hsqc c_nmr h_nmr mass_spec mass_spec_neg mw formula}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

IMAGE="${IMAGE:-$PROJECT_ROOT/images/marina-arm64.sif}"
CODE_DIR="${CODE_DIR:-$PROJECT_ROOT/code/MARINA-deltaai}"
CHAIN_SCRIPT="${CHAIN_SCRIPT:-$CODE_DIR/singularity/deltaai-train.sbatch}"
MAX_CHUNKS="${MAX_CHUNKS:-12}"
CHAIN_INDEX="${CHAIN_INDEX:-1}"

PARTITION="${PARTITION:-${SLURM_JOB_PARTITION:-ghx4}}"
ACCOUNT="${ACCOUNT:-${SLURM_JOB_ACCOUNT:-bibx-dtai-gh}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${SLURM_GPUS_PER_NODE:-1}}"
CPUS_PER_TASK="${CPUS_PER_TASK:-${SLURM_CPUS_PER_TASK:-16}}"
MEM="${MEM:-${SLURM_MEM_PER_NODE:-110G}}"

export SMART_RUN_ID="${SMART_RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"

RUN_DIR="$PROJECT_ROOT/runs/results/$EXPERIMENT_NAME/$SMART_RUN_ID"
mkdir -p "$RUN_DIR" "$PROJECT_ROOT/runs/wandb"

echo "[chain] chunk $CHAIN_INDEX/$MAX_CHUNKS  job=$SLURM_JOB_ID  run_id=$SMART_RUN_ID"
echo "[chain] exp=$EXPERIMENT_NAME fp=$FP_TYPE seed=$SEED batch=$BATCH_SIZE accum=$ACCUM"
echo "[chain] run dir: $RUN_DIR"

if [[ -f "$RUN_DIR/COMPLETE" ]]; then
    echo "[chain] COMPLETE sentinel present; training already finished. Exiting."
    exit 0
fi

# Queue the successor first (survives a walltime SIGKILL). GPU/CPU/mem propagate down the chain.
if (( CHAIN_INDEX < MAX_CHUNKS )); then
    next_id=$(sbatch --parsable \
        --dependency=afterany:"$SLURM_JOB_ID" \
        --partition="$PARTITION" \
        --account="$ACCOUNT" \
        --gpus-per-node="$GPUS_PER_NODE" \
        --cpus-per-task="$CPUS_PER_TASK" \
        --mem="$MEM" \
        --job-name="${SLURM_JOB_NAME:-marina-ai}" \
        --export=ALL,SMART_RUN_ID="$SMART_RUN_ID",CHAIN_INDEX="$((CHAIN_INDEX + 1))" \
        "$CHAIN_SCRIPT")
    echo "[chain] successor queued as $next_id (afterany:$SLURM_JOB_ID)"
else
    echo "[chain] MAX_CHUNKS reached; this is the final chunk, not resubmitting."
fi

module purge 2>/dev/null || true
module load apptainer 2>/dev/null || true

# Dataset -> node-local /tmp (purged at job end); /projects is Taiga (bad for many small reads).
LOCAL_DATA="/tmp/$USER/job_$SLURM_JOB_ID/data"
mkdir -p "$LOCAL_DATA"
echo "[stage] unpacking $DATASET.zip to $LOCAL_DATA"
time unzip -q "$PROJECT_ROOT/datasets/${DATASET}.zip" -d "$LOCAL_DATA"

# Overlay prebuilt fingerprint artifacts not shipped inside the dataset zip (e.g. the
# RankingEntropySubstructure vocabulary: rankingset.pt + bitinfo_to_idx.pkl). FragIdx and packed/
# for substructure ARE in the zip; only the vocab dir needs overlaying. Staged by deltaai-stage.sh.
FP_OVERLAY="$PROJECT_ROOT/fp-artifacts/$DATASET"
if [ -d "$FP_OVERLAY" ] && [ -n "$(ls -A "$FP_OVERLAY" 2>/dev/null)" ]; then
    echo "[stage] overlaying prebuilt fingerprint artifacts from $FP_OVERLAY"
    rsync -a "$FP_OVERLAY/" "$LOCAL_DATA/"
fi

# Re-pack any FragIdx that only exists as arrow (skip-if-exists; the zip already ships packed/).
echo "[stage] packing arrow -> memmap arrays (skip-if-exists)"
time apptainer exec --cleanenv --pwd /code \
    -B "$CODE_DIR":/code \
    -B "$LOCAL_DATA":/workspace \
    "$IMAGE" \
    python scripts/dataset/pack_arrow.py --root /workspace

nvidia-smi --query-gpu=index,name,memory.total --format=csv

NPROC=$(nvidia-smi -L | wc -l)
if (( NPROC < 1 )); then
    echo "[chain] no GPUs visible; aborting before wasting the allocation." >&2
    exit 1
fi
echo "[chain] launching torchrun with $NPROC rank(s); global batch = $BATCH_SIZE * $NPROC * $ACCUM"

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
        --fp_type "$FP_TYPE" \
        --seed "$SEED" \
        --batch_size "$BATCH_SIZE" \
        --accumulate_grad_batches_num "$ACCUM" \
        --precision bf16-mixed \
        --input_types $INPUT_TYPES \
        --resume \
        $EXTRA_ARGS
rc=$?
set -e

if (( rc == 0 )); then
    touch "$RUN_DIR/COMPLETE"
    echo "[chain] training returned 0 (early stopping or max epochs). Chain complete."
else
    echo "[chain] chunk exited $rc -- successor resumes from $RUN_DIR/last.ckpt"
fi
exit "$rc"
