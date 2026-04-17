# ── Stage 1: dependency installation ─────────────────────────────────────────
# Python 3.12: stable wheels across hosts; pixi.toml may use 3.13 for local dev only.
FROM python:3.12-slim AS deps

# System libraries required by RDKit and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-base.txt .

# Install PyTorch.
# Default index is CPU-only for widest compatibility.
# Override at build time for CUDA wheels:
#   docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 ...
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch==2.8.0 --index-url "$TORCH_INDEX_URL"

# NumPy / pandas: modern stack vs legacy CPU-safe wheels (no x86-64-v2 baseline).
# Set LEGACY_NUMPY=true in the root .env when building for older CPUs / VMs without v2 features.
ARG LEGACY_NUMPY=false
RUN set -eux; \
    LEGACY="$(echo "${LEGACY_NUMPY}" | tr '[:upper:]' '[:lower:]')"; \
    if [ "$LEGACY" = "true" ] || [ "$LEGACY" = "1" ] || [ "$LEGACY" = "yes" ]; then \
      pip install --no-cache-dir \
        "numpy==1.26.4" \
        "pandas>=2.2.3,<3"; \
    else \
      pip install --no-cache-dir \
        "numpy>=2.4.2,<3" \
        "pandas>=3.0.1,<4"; \
    fi

RUN pip install --no-cache-dir -r requirements-base.txt


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from the build stage
COPY --from=deps /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=deps /usr/local/bin            /usr/local/bin

# Copy application source
COPY app/ ./app/

# MARINA source is mounted at runtime via a bind mount (see docker-compose.yml).
ENV MARINA_ROOT=/marina \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    KMP_DUPLICATE_LIB_OK=TRUE

EXPOSE 5000

ENV UVICORN_WORKERS=2

CMD uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 5000 \
    --workers ${UVICORN_WORKERS} \
    --log-level info
