#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# MARINA – full-stack start / stop / restart script
#
# Usage:
#   ./start.sh [command] [options]
#
# Commands:
#   start     Start containers (no rebuild)
#   stop      Stop and remove containers
#   restart   Rebuild images and recreate containers  [default]
#   build     Build images only (no start)
#   logs      Tail all container logs (Ctrl-C to exit)
#   status    Show container status
#   help      Show this message
#
# Options:
#   --gpu     Build with CUDA support (uses TORCH_INDEX_URL for cu128 by default)
#   --cpu     Build with CPU-only PyTorch (overrides --gpu and any .env value)
#
# Build-time (.env at repo root, before build):
#   LEGACY_NUMPY=true   Older NumPy/pandas wheels for CPUs without x86-64-v2 (see .env.example)
#
# Examples:
#   ./start.sh                     # rebuild + restart (CPU)
#   ./start.sh restart --gpu       # rebuild + restart with CUDA 12.8
#   ./start.sh restart --gpu TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
#   ./start.sh logs
#   ./start.sh stop
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Load .env if present ──────────────────────────────────────────────────────
if [[ -f ".env" ]]; then
    set -a
    # shellcheck source=/dev/null
    source ".env"
    set +a
fi

# ── Parse arguments ───────────────────────────────────────────────────────────
CMD="${1:-restart}"
GPU=false
CPU=false

shift || true   # shift past CMD (safe even if no args)
for arg in "$@"; do
    case "$arg" in
        --gpu) GPU=true ;;
        --cpu) CPU=true ;;
    esac
done

# ── Resolve PyTorch wheel index ───────────────────────────────────────────────
if $CPU; then
    export TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
    export DEVICE="${DEVICE:-cpu}"
elif $GPU; then
    export TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
    export DEVICE="${DEVICE:-cuda}"
else
    # Honour whatever is already set in the environment / .env; fall back to cpu.
    export TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
    export DEVICE="${DEVICE:-cpu}"
fi

# ── Helpers ───────────────────────────────────────────────────────────────────
info()  { printf '\033[1;34m[marina]\033[0m %s\n' "$*"; }
error() { printf '\033[1;31m[marina]\033[0m %s\n' "$*" >&2; }

require_model_data() {
    MODEL_DATA_DIR="${MODEL_DATA_DIR:-./checkpoints}"
    if [[ ! -d "$MODEL_DATA_DIR" ]]; then
        error "MODEL_DATA_DIR='$MODEL_DATA_DIR' does not exist or is not a directory."
        error "Set MODEL_DATA_DIR in .env or place your model data under ./checkpoints."
        exit 1
    fi
}

# ── Commands ──────────────────────────────────────────────────────────────────
case "$CMD" in

    start)
        require_model_data
        info "Starting containers (DEVICE=${DEVICE}, TORCH_INDEX_URL=${TORCH_INDEX_URL})"
        docker compose up -d
        info "App → http://localhost:${NGINX_PORT:-80}"
        ;;

    stop)
        info "Stopping containers…"
        docker compose down
        info "Done."
        ;;

    restart)
        require_model_data
        info "Building images and restarting containers…"
        info "  DEVICE            = ${DEVICE}"
        info "  TORCH_INDEX_URL   = ${TORCH_INDEX_URL}"
        info "  LEGACY_NUMPY      = ${LEGACY_NUMPY:-false}"
        info "  MODEL_DATA_DIR    = ${MODEL_DATA_DIR}"
        docker compose build
        docker compose up -d --force-recreate
        info "App → http://localhost:${NGINX_PORT:-80}"
        ;;

    build)
        info "Building images…"
        info "  TORCH_INDEX_URL = ${TORCH_INDEX_URL}"
        info "  LEGACY_NUMPY    = ${LEGACY_NUMPY:-false}"
        docker compose build
        info "Build complete."
        ;;

    logs)
        exec docker compose logs -f
        ;;

    status)
        docker compose ps
        ;;

    help|--help|-h)
        sed -n '2,/^# ─\+$/p' "$0" | grep '^#' | sed 's/^# \?//'
        ;;

    *)
        error "Unknown command: '$CMD'"
        error "Run './start.sh help' for usage."
        exit 1
        ;;

esac
