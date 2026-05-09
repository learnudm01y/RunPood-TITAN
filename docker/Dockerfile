# =============================================================================
# Dockerfile
# RunPood-histo-TITAN – Feature Extraction Container
#
# Base image: PyTorch 2.1 + CUDA 12.1 (Ubuntu 22.04)
# The /workspace directory is expected to be a RunPod persistent volume.
# =============================================================================

FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# ── System packages ────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        unzip \
        git \
        ca-certificates \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

# ── rclone ────────────────────────────────────────────────────────────────────
RUN curl -fsSL https://rclone.org/install.sh | bash

# ── Python dependencies ───────────────────────────────────────────────────────
WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r /tmp/requirements.txt

# ── Copy project source ───────────────────────────────────────────────────────
# Build context must be the project root (RunPood-histo-TITAN/):
#   docker build -f docker/Dockerfile -t runpood-histo-titan .
COPY app/          /workspace/app/
COPY scripts/      /workspace/scripts/
COPY rclone.conf   /workspace/rclone.conf

RUN chmod +x /workspace/scripts/*.sh

# ── Create directory structure ────────────────────────────────────────────────
# These directories will be bind-mounted from the persistent volume at runtime.
# Creating them here ensures the container starts correctly even without a volume.
RUN mkdir -p \
    /workspace/input/patches \
    /workspace/input/metadata \
    /workspace/output/features \
    /workspace/output/logs \
    /workspace/output/manifests \
    /workspace/models/titan \
    /workspace/models/conch \
    /workspace/models/cache

# ── Environment defaults ──────────────────────────────────────────────────────
ENV WORKSPACE_DIR=/workspace \
    HF_HOME=/workspace/models/cache \
    HF_HUB_CACHE=/workspace/models/cache/hub \
    TRANSFORMERS_CACHE=/workspace/models/cache \
    PYTHONPATH=/workspace \
    PYTHONUNBUFFERED=1

# ── Entrypoint ────────────────────────────────────────────────────────────────
# By default we run the HTTP server which receives jobs from Laravel.
# Override CMD at runtime to do a one-shot CLI extraction instead:
#   docker run ... runpood-histo-titan bash scripts/run_extract.sh
#
# Required env at runtime:
#   RUNPOD_API_KEY  – shared secret (must match servers_names.api_key in Laravel)
#   RCLONE_REMOTE   – e.g. "gdrive"  (must match rclone remote name in rclone.conf)
#   HUGGING_FACE_HUB_TOKEN – for first-time model download

EXPOSE 8000

ENTRYPOINT ["bash", "scripts/start_server.sh"]
