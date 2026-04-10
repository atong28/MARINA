# ─────────────────────────────────────────────────────────────────────────────
# MARINA Backend – common development and deployment tasks
# ─────────────────────────────────────────────────────────────────────────────

# Load .env if it exists (for docker-compose targets).
-include .env
export

# Dev env file for local (no-Docker) runs.
DEV_ENV_FILE ?= .env.dev

.PHONY: help dev install build up down logs restart clean

# ── Default target ────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  MARINA Backend"
	@echo ""
	@echo "  Local development (no Docker):"
	@echo "    make install        Install Python dependencies into current env"
	@echo "    make dev            Run with hot-reload (edit $(DEV_ENV_FILE) first)"
	@echo ""
	@echo "  Docker:"
	@echo "    make build          Build the Docker image"
	@echo "    make up             Start the container (detached)"
	@echo "    make down           Stop and remove the container"
	@echo "    make logs           Tail container logs"
	@echo "    make restart        Rebuild image and restart container"
	@echo ""
	@echo "  GPU build:"
	@echo "    TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 make build"
	@echo "    DEVICE=cuda make up"
	@echo ""
	@echo "  Older CPU (NumPy x86-64-v2 error): set in .env then build"
	@echo "    LEGACY_NUMPY=true"
	@echo ""

# ── Local dev (no Docker) ─────────────────────────────────────────────────────

install:
	pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
	pip install -r requirements.txt

# Run uvicorn with hot-reload, loading variables from .env.dev.
# Uses bash `source` + `set -a` so all assignments are exported (unlike
# `env $(grep … | xargs)`, which breaks on spaces, CRLF, and does not
# affect shell-side expansions like ${BACKEND_PORT}).
dev:
	@if [ ! -f "$(DEV_ENV_FILE)" ]; then \
		echo "ERROR: $(DEV_ENV_FILE) not found."; \
		echo "       Copy .env.example to $(DEV_ENV_FILE) and set your paths."; \
		exit 1; \
	fi
	@echo "Starting dev server (sourcing $(CURDIR)/$(DEV_ENV_FILE))…"
	bash -eu -o pipefail -c '\
		cd "$(CURDIR)" && \
		set -a && \
		source "$(CURDIR)/$(DEV_ENV_FILE)" && \
		set +a && \
		export PYTHONPATH="$(CURDIR)" && \
		exec uvicorn app.main:app \
			--host 0.0.0.0 \
			--port "$${BACKEND_PORT:-5000}" \
			--reload \
			--reload-dir "$(CURDIR)/app" \
			--log-level debug'

# ── Docker targets ────────────────────────────────────────────────────────────

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f backend

restart: build
	docker compose up -d --force-recreate

clean:
	docker compose down --rmi local --volumes
