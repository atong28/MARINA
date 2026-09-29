#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# MARINA – one-shot deploy on a fresh Docker host
#
# Usage:
#   bash scripts/website/deploy.sh [PORT]      # default PORT=8643
#
# Downloads the model weights (skipped if already present), builds the images
# and starts the stack with the site on http://<host>:PORT. Re-running is safe:
# it keeps the weights and rebuilds/recreates the containers, so it is also the
# update path after a `git pull`.
#
# The port is persisted as NGINX_PORT in .env; every other setting keeps its
# default (CPU inference, see .env.example). Needs docker with the compose
# plugin (or docker-compose), curl and unzip.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PORT="${1:-8643}"
if ! [[ "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
    echo "usage: $0 [PORT]   (PORT must be 1-65535, got '$PORT')" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

info()  { printf '\033[1;34m[marina]\033[0m %s\n' "$*"; }
error() { printf '\033[1;31m[marina]\033[0m %s\n' "$*" >&2; }

if docker compose version >/dev/null 2>&1; then
    COMPOSE=(docker compose)
elif command -v docker-compose >/dev/null; then
    COMPOSE=(docker-compose)
else
    error "docker with the compose plugin (or docker-compose) is required"
    exit 1
fi
# Pin the base file so a COMPOSE_FILE overlay in .env (e.g. the Cloudflare
# tunnel, which removes the host port) can't override the exposed port.
COMPOSE+=(-f docker-compose.yml)

# ── 1. Weights ────────────────────────────────────────────────────────────────
bash scripts/website/download_model.sh

# ── 2. Port ───────────────────────────────────────────────────────────────────
touch .env
if grep -q '^NGINX_PORT=' .env; then
    sed -i "s/^NGINX_PORT=.*/NGINX_PORT=${PORT}/" .env
else
    echo "NGINX_PORT=${PORT}" >> .env
fi

# ── 3. Build + start ──────────────────────────────────────────────────────────
info "building images (first build takes several minutes) ..."
"${COMPOSE[@]}" build
"${COMPOSE[@]}" up -d --force-recreate --remove-orphans

# ── 4. Wait for the backend to load the model ─────────────────────────────────
# Polled end-to-end through nginx, so a pass also proves the port and proxy chain.
info "waiting for the backend to load the model ..."
for _ in $(seq 1 120); do
    if curl -fsS "http://localhost:${PORT}/api/health" 2>/dev/null | grep -q '"model_loaded":true'; then
        info "MARINA is up → http://$(hostname -f 2>/dev/null || hostname):${PORT}"
        exit 0
    fi
    sleep 5
done
error "model not loaded after 10 min; check: ${COMPOSE[*]} logs backend nginx"
exit 1
