# ── Stage 1: dependency installation ─────────────────────────────────────────
FROM python:3.11-slim AS deps

# System libraries needed by RDKit and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install CPU-only PyTorch first (avoids pulling CUDA binaries by default).
# Override TORCH_INDEX_URL at build time to install a GPU-capable wheel:
#   docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 ...
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch torchvision --index-url "$TORCH_INDEX_URL" \
 && pip install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from the deps stage
COPY --from=deps /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=deps /usr/local/bin            /usr/local/bin

# Copy application source
COPY app/ ./app/

# MARINA source is mounted at runtime via a bind mount (see docker-compose.yml).
# MARINA_ROOT=/marina is set as default; override via environment variable.
ENV MARINA_ROOT=/marina \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    KMP_DUPLICATE_LIB_OK=TRUE

EXPOSE 5000

# Use multiple uvicorn workers to handle concurrent requests.
# UVICORN_WORKERS can be overridden in docker-compose or at runtime.
ENV UVICORN_WORKERS=2

CMD uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 5000 \
    --workers ${UVICORN_WORKERS} \
    --log-level info
