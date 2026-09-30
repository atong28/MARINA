#!/usr/bin/env bash
# One chunk of a self-resubmitting MARINA training chain on Purdue Anvil (AnvilAI, H100).
#
# Anvil vs the NCSA targets (delta-train.sbatch / deltaai-train.sbatch):
#   * partition `ai` (21 nodes x 4x H100 80GB, node-sharing, QOS part-ai: 2-day MaxWall, 12 GPU/user)
#     / account bio260190-ai (1000 SU; SU factor 1 => 1 SU per GPU-hour).
#   * x86-64, so the amd64 marina.sif (the Delta/Expanse image) runs as-is.
#   * apptainer 1.4 at /usr/bin/apptainer, no module needed.
#   * CODE is git-cloned at a pinned COMMIT by anvil-stage.sh on the login node (Anvil's id_rsa is
#     on GitHub); this chunk binds it read-only. $CODE_DIR/COMMIT records the SHA.
#   * Slurm reports TmpDisk=0 for the ai nodes: node-local /tmp size is unknown up front, so the
#     dataset is unpacked to /tmp only if it has >= TMP_MIN_GB free, else to $SCRATCH_ROOT.
#   * W&B mode decided per chunk by an egress probe from the compute node (online if api.wandb.ai
#     answers, else offline + sync later).
#
# Chain mechanics are identical to the other targets: queue the successor FIRST with
# --dependency=afterany, pin SMART_RUN_ID so every chunk --resume's one last.ckpt and logs to one
# W&B run, stop on a COMPLETE sentinel, MAX_CHUNKS backstop.
#
#   Submit (1 GPU, global batch 512 = 512 * 1 rank * 1 accum; H100 80GB):
#       EXPERIMENT_NAME=marina-db-open-solvjit-uniqmult-formula-s0 DATASET=MARINA-DB-OPEN \
#       FP_TYPE=RankingEntropyUniqueMultiplicity SEED=0 \
#       EXTRA_ARGS="--solvent_jitter_p 1.0" \
#       sbatch --job-name=mdbopen-solvjit-s0-$(date +%Y%m%d-%H%M) singularity/anvil-train.sbatch
#   Watch:  ssh anvil-ai 'squeue -u x-atong1'
#
#SBATCH --job-name=marina-anvil
#SBATCH --account=bio260190-ai
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=110G
#SBATCH --time=48:00:00
#SBATCH --no-requeue
#SBATCH --output=/anvil/projects/x-bio260190/atong1/slurm-logs/%x-%j.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/anvil/projects/x-bio260190/atong1}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/anvil/scratch/x-atong1}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:?set EXPERIMENT_NAME when submitting}"
DATASET="${DATASET:-MARINA-DB-OPEN}"
FP_TYPE="${FP_TYPE:-RankingEntropyUniqueMultiplicity}"
SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-512}"
ACCUM="${ACCUM:-1}"
# MARINA2.0: the flagship 7 inputs plus the ceiling HMBC/COSY modalities (bernoulli sampler = default).
INPUT_TYPES="${INPUT_TYPES:-hsqc c_nmr h_nmr mass_spec mass_spec_neg mw formula}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
TMP_MIN_GB="${TMP_MIN_GB:-40}"

IMAGE="${IMAGE:-$PROJECT_ROOT/images/marina.sif}"
CODE_DIR="${CODE_DIR:-$PROJECT_ROOT/code/MARINA-open-solvjit}"
CHAIN_SCRIPT="${CHAIN_SCRIPT:-$CODE_DIR/singularity/anvil-train.sbatch}"
MAX_CHUNKS="${MAX_CHUNKS:-12}"
CHAIN_INDEX="${CHAIN_INDEX:-1}"

PARTITION="${PARTITION:-${SLURM_JOB_PARTITION:-ai}}"
ACCOUNT="${ACCOUNT:-${SLURM_JOB_ACCOUNT:-bio260190-ai}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${SLURM_GPUS_PER_NODE:-1}}"
CPUS_PER_TASK="${CPUS_PER_TASK:-${SLURM_CPUS_PER_TASK:-16}}"
MEM="${MEM:-${SLURM_MEM_PER_NODE:-110G}}"

export SMART_RUN_ID="${SMART_RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"

RUN_DIR="$PROJECT_ROOT/runs/results/$EXPERIMENT_NAME/$SMART_RUN_ID"
mkdir -p "$RUN_DIR" "$PROJECT_ROOT/runs/wandb"

echo "[chain] chunk $CHAIN_INDEX/$MAX_CHUNKS  job=$SLURM_JOB_ID  run_id=$SMART_RUN_ID  node=$(hostname)"
echo "[chain] exp=$EXPERIMENT_NAME fp=$FP_TYPE seed=$SEED batch=$BATCH_SIZE accum=$ACCUM inputs=[$INPUT_TYPES]"
echo "[chain] code=$CODE_DIR commit=$(cat "$CODE_DIR/COMMIT" 2>/dev/null || echo unknown)"
echo "[chain] run dir: $RUN_DIR"

if [[ -f "$RUN_DIR/COMPLETE" ]]; then
    echo "[chain] COMPLETE sentinel present; training already finished. Exiting."
    exit 0
fi

if (( CHAIN_INDEX < MAX_CHUNKS )); then
    next_id=$(sbatch --parsable \
        --dependency=afterany:"$SLURM_JOB_ID" \
        --partition="$PARTITION" \
        --account="$ACCOUNT" \
        --gpus-per-node="$GPUS_PER_NODE" \
        --cpus-per-task="$CPUS_PER_TASK" \
        --mem="$MEM" \
        --job-name="${SLURM_JOB_NAME:-marina-anvil}" \
        --export=ALL,SMART_RUN_ID="$SMART_RUN_ID",CHAIN_INDEX="$((CHAIN_INDEX + 1))" \
        "$CHAIN_SCRIPT")
    echo "[chain] successor queued as $next_id (afterany:$SLURM_JOB_ID)"
else
    echo "[chain] MAX_CHUNKS reached; this is the final chunk, not resubmitting."
fi

# Dataset staging: node-local /tmp if it is big enough (unknown on Anvil up front), else scratch.
tmp_free_gb=$(df -BG --output=avail /tmp | tail -1 | tr -dc '0-9')
if (( tmp_free_gb >= TMP_MIN_GB )); then
    LOCAL_DATA="/tmp/$USER/job_$SLURM_JOB_ID/data"
else
    LOCAL_DATA="$SCRATCH_ROOT/job_$SLURM_JOB_ID/data"
    echo "[stage] /tmp has only ${tmp_free_gb}G free (< ${TMP_MIN_GB}G); staging to scratch instead"
fi
mkdir -p "$LOCAL_DATA"
echo "[stage] /tmp free: ${tmp_free_gb}G; unpacking $DATASET.zip to $LOCAL_DATA"
time unzip -q "$PROJECT_ROOT/datasets/${DATASET}.zip" -d "$LOCAL_DATA"
# the zips are archived with the dataset dir as top level; flatten if so
if [ -d "$LOCAL_DATA/$DATASET" ]; then
    mv "$LOCAL_DATA/$DATASET"/* "$LOCAL_DATA"/ && rmdir "$LOCAL_DATA/$DATASET"
fi
cleanup() { rm -rf "$LOCAL_DATA"; }
trap cleanup EXIT

FP_OVERLAY="$PROJECT_ROOT/fp-artifacts/$DATASET"
if [ -d "$FP_OVERLAY" ] && [ -n "$(ls -A "$FP_OVERLAY" 2>/dev/null)" ]; then
    echo "[stage] overlaying prebuilt fingerprint artifacts from $FP_OVERLAY"
    rsync -a "$FP_OVERLAY/" "$LOCAL_DATA/"
fi

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

# W&B: online if the compute node has egress, else offline (sync with `wandb sync` later).
if curl -sI --max-time 10 https://api.wandb.ai >/dev/null 2>&1; then
    WANDB_MODE=online
else
    WANDB_MODE=offline
fi
echo "[chain] W&B mode: $WANDB_MODE"
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
    --env WANDB_MODE="$WANDB_MODE" \
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
