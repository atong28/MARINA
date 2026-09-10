#!/usr/bin/env bash
# Stage code + verify data for the DeltaAI (Grace-Hopper) MARINA training chain.
#
# Code staging is GIT-CLONE at a pinned COMMIT (NOT rsync), done here on the login node over the
# ControlMaster socket. The compute nodes then bind the SHA-pinned clone read-only, so no
# compute-node git egress is needed. Data (dataset zip, benchmark files, W&B key, fp-artifact
# overlays) lives on the shared /projects Taiga filesystem and is rsync'd separately (usually
# already present, since DeltaAI shares /projects/bibx with the A100 Delta).
#
#   Usage:  COMMIT=<full-40-char-SHA> bash singularity/deltaai-stage.sh atong1
#
# COMMIT is REQUIRED: experiments are pinned to an exact commit (wiki/active/experiment-registry.md).
set -euo pipefail

SOCK="${NCSA_AI_SOCK:-$HOME/.ssh/ncsa-ai.sock}"
HOST="${NCSA_AI_HOST:-gh-login01.delta.ncsa.illinois.edu}"
PROJECT_ROOT="${PROJECT_ROOT:-/projects/bibx/atong1}"
REPO="${REPO:-git@github.com:atong28/MARINA.git}"
COMMIT="${COMMIT:?set COMMIT=<full 40-char SHA>: experiments are pinned to an exact commit}"

USER_NAME="${1:-atong1}"
TARGET="${USER_NAME}@${HOST}"
SSH=(ssh -S "$SOCK" "$TARGET")

if ! ssh -S "$SOCK" -O check "$TARGET" 2>/dev/null; then
    echo "No live control socket at $SOCK. Open one first (ssh ncsa-ai and complete Duo)." >&2
    exit 1
fi

echo "==> creating layout under $PROJECT_ROOT"
"${SSH[@]}" "mkdir -p '$PROJECT_ROOT'/{code,images,datasets,benchmark,runs/wandb,slurm-logs,fp-artifacts}"

echo "==> git-clone/checkout MARINA at $COMMIT (using DeltaAI's github ssh key)"
"${SSH[@]}" bash -s <<EOF
set -euo pipefail
export GIT_SSH_COMMAND="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new"
CODE="$PROJECT_ROOT/code/MARINA-deltaai"
if [ ! -d "\$CODE/.git" ]; then
    git clone "$REPO" "\$CODE"
fi
cd "\$CODE"
git fetch --all --quiet
git checkout --quiet --detach "$COMMIT"
head=\$(git rev-parse HEAD)
if [ "\$head" != "$COMMIT" ]; then
    echo "  checkout landed on \$head, not $COMMIT" >&2; exit 1
fi
if [ -n "\$(git status --porcelain)" ]; then
    echo "  clone tree is dirty after checkout (unexpected):" >&2
    git status --porcelain >&2; exit 1
fi
echo "  code pinned at \$head (clean detached checkout)"
EOF

echo "==> verifying data on shared /projects (rsync only what is missing)"
"${SSH[@]}" bash -s <<EOF
set -eu
ok=1
for f in datasets/MARINA-DB.zip benchmark/benchmark-journal.pkl benchmark/benchmark.pkl wandb_api_key.json; do
    if [ -e "$PROJECT_ROOT/\$f" ]; then
        echo "  present: \$f (\$(du -h "$PROJECT_ROOT/\$f" | cut -f1))"
    else
        echo "  MISSING: \$f" >&2; ok=0
    fi
done
sub="$PROJECT_ROOT/fp-artifacts/MARINA-DB/RankingEntropySubstructure"
if [ -f "\$sub/rankingset.pt" ] && [ -f "\$sub/bitinfo_to_idx.pkl" ]; then
    echo "  present: fp-artifacts/MARINA-DB/RankingEntropySubstructure/ (vocab overlay)"
else
    echo "  MISSING: substructure vocab overlay under fp-artifacts/MARINA-DB/" >&2; ok=0
fi
[ "\$ok" = 1 ] || { echo "  --> stage the missing data before submitting." >&2; exit 1; }
EOF

echo "==> verifying image"
if "${SSH[@]}" "test -f '$PROJECT_ROOT/images/marina-arm64.sif'"; then
    "${SSH[@]}" "apptainer exec '$PROJECT_ROOT/images/marina-arm64.sif' python -c \
        'import torch, rdkit, pytorch_lightning as pl; print(\"image OK | torch\", torch.__version__, \"| pl\", pl.__version__, \"| rdkit\", rdkit.__version__)'" || \
        echo "  image present but smoke import failed -- check the build" >&2
else
    echo "  NO aarch64 image at $PROJECT_ROOT/images/marina-arm64.sif" >&2
    echo "  build it on an aarch64 host from singularity/deltaai-marina.def and upload." >&2
fi

cat <<EOF

Staged. Launch the substructure smoke job (1 GPU, global batch 512) with:

  ssh -S $SOCK $TARGET \\
    "cd $PROJECT_ROOT/code/MARINA && \\
     EXPERIMENT_NAME=marina-deltaai-substructure-s0 DATASET=MARINA-DB \\
     FP_TYPE=RankingEntropySubstructure SEED=0 \\
     sbatch singularity/deltaai-train.sbatch"

Once s0 is confirmed training, launch s1 and s2 (SEED=1,2; EXPERIMENT_NAME ...-s1/-s2).
EOF
