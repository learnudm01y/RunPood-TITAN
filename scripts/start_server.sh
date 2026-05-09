#!/usr/bin/env bash
# =============================================================================
# start_server.sh
# Boot the RunPod feature-extraction HTTP server.
#
# This is the recommended entry-point on a RunPod pod with a public URL
# (e.g. https://histopathology.cloud/).  It listens for POST /jobs/start from
# the Laravel management server and processes them in a background worker.
#
# Required environment variables:
#   RUNPOD_API_KEY   – Shared secret matching servers_names.api_key in Laravel
#   RCLONE_REMOTE    – rclone remote name (e.g. "gdrive") matching rclone.conf
#
# Optional:
#   PORT             – default 8000
#   HOST             – default 0.0.0.0
#   HUGGING_FACE_HUB_TOKEN – for downloading gated weights on first run
# =============================================================================

set -euo pipefail

# Determine project directory (the folder containing scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

WORKSPACE="${WORKSPACE_DIR:-/workspace}"
export WORKSPACE_DIR="${WORKSPACE}"
export PYTHONPATH="${PROJECT_DIR}"
export HF_HOME="${WORKSPACE}/models/cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

if [ -z "${RUNPOD_API_KEY:-}" ]; then
    echo "ERROR: RUNPOD_API_KEY is not set."
    echo "       Set it to the same value as servers_names.api_key in Laravel."
    exit 1
fi

if [ -z "${RCLONE_REMOTE:-}" ]; then
    echo "WARN: RCLONE_REMOTE is not set; defaulting to 'gdrive'."
fi

# Restore rclone config from persistent volume if present
if [ -f "${WORKSPACE}/rclone.conf" ]; then
    mkdir -p "${HOME:-/root}/.config/rclone"
    cp "${WORKSPACE}/rclone.conf" "${HOME:-/root}/.config/rclone/rclone.conf"
    echo "[start_server] Restored rclone config from ${WORKSPACE}/rclone.conf"
fi

# Pre-warm models so the first job doesn't pay the download cost
bash "${PROJECT_DIR}/scripts/download_models.sh" || true

cd "${PROJECT_DIR}"

echo "================================================================"
echo "  RunPood-histo-TITAN  |  HTTP Server"
echo "  Host : ${HOST}:${PORT}"
echo "  RClone remote : ${RCLONE_REMOTE:-gdrive}"
echo "================================================================"

exec uvicorn app.server:app --host "${HOST}" --port "${PORT}" --workers 1
