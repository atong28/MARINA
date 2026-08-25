#!/bin/bash
# MARINA training-pod bootstrap: clone the code at a frozen ref, stage datasets onto the
# node, install the pixi env. Fully configured from the Job spec via env vars; the old
# positional form `startup.sh <DATASET>.zip [<BRANCH>]` is still honored.
#
# Env vars (all optional):
#   REPO          git remote to clone                 (default git@github.com:atong28/MARINA.git)
#   BRANCH        branch OR tag to clone              (default main; set exp/<name> to train frozen code)
#   COMMIT        exact SHA to check out after clone   (default: the BRANCH tip)
#   DATASETS      space-separated zip names to unzip   (default MARINA1.zip)
#   CODE_DIR      clone destination                    (default /code)
#   WORKSPACE     dataset unzip destination            (default /workspace)
#   DATASET_DIR   where the zips live                  (default /root/datasets)
#   PIXI_INSTALL  1 = run pixi install, 0 = skip       (default 1)
set -euo pipefail

REPO="${REPO:-git@github.com:atong28/MARINA.git}"
BRANCH="${BRANCH:-${2:-main}}"
COMMIT="${COMMIT:-}"
DATASETS="${DATASETS:-${1:-MARINA1.zip}}"
CODE_DIR="${CODE_DIR:-/code}"
WORKSPACE="${WORKSPACE:-/workspace}"
DATASET_DIR="${DATASET_DIR:-/root/datasets}"
PIXI_INSTALL="${PIXI_INSTALL:-1}"

export GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no"

# --- ssh key for the private clone ---
mkdir -p ~/.ssh
cp /root/gurusmart/.ssh/id_rsa ~/.ssh/id_rsa
cp /root/gurusmart/.ssh/id_rsa.pub ~/.ssh/id_rsa.pub
chmod 600 ~/.ssh/id_rsa

# --- clone code at the frozen ref ---
echo "[startup] cloning $REPO @ ${COMMIT:-$BRANCH} -> $CODE_DIR"
git clone --branch "$BRANCH" "$REPO" "$CODE_DIR"
if [ -n "$COMMIT" ]; then
    git -C "$CODE_DIR" checkout --quiet "$COMMIT"
fi
echo "[startup] code frozen at $(git -C "$CODE_DIR" rev-parse --short HEAD) (${COMMIT:-$BRANCH})"

# --- stage datasets onto the node ---
mkdir -p "$WORKSPACE"
for ds in $DATASETS; do
    src="$DATASET_DIR/$ds"
    if [ ! -f "$src" ]; then
        echo "[startup] MISSING dataset $src" >&2
        exit 1
    fi
    echo "[startup] unzipping $ds -> $WORKSPACE"
    unzip -q "$src" -d "$WORKSPACE"
done

# --- pixi env ---
if [ "$PIXI_INSTALL" = "1" ]; then
    echo "[startup] pixi install"
    pixi install --manifest-path "$CODE_DIR/pixi.toml"
fi
echo "[startup] ready"
