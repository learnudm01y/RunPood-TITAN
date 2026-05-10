#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
WORKSPACE="${WORKSPACE_DIR:-/workspace}"
SETUP_SCRIPT="${WORKSPACE}/setup_env.sh"
VENV_PYTHON="${WORKSPACE}/venv/bin/python"
LOG_DIR="${WORKSPACE}/logs"
SERVER_LOG="${LOG_DIR}/server.log"

mkdir -p "${LOG_DIR}"

echo "[start] workspace : ${WORKSPACE}"
echo "[start] project   : ${PROJECT_DIR}"
echo "[start] log       : ${SERVER_LOG}"

# ── Fix DNS (RunPod resets resolv.conf on every start) ────────────────────────
echo "nameserver 8.8.8.8" > /etc/resolv.conf
echo "nameserver 8.8.4.4" >> /etc/resolv.conf
echo "[start] DNS fixed"

if [ ! -f "${SETUP_SCRIPT}" ]; then
    echo "ERROR: Missing ${SETUP_SCRIPT} — create it on the Network Volume first."
    exit 1
fi

if [ ! -x "${VENV_PYTHON}" ]; then
    echo "ERROR: Missing ${VENV_PYTHON} — recreate /workspace/venv first."
    exit 1
fi

# ── Restore environment variables and rclone config ───────────────────────────
source "${SETUP_SCRIPT}"
echo "[start] Environment ready"

# ── Pull latest code from git (non-fatal if no network yet) ───────────────────
cd "${PROJECT_DIR}"
git pull --quiet 2>/dev/null && echo "[start] Code updated" || echo "[start] git pull skipped (non-fatal)"

export PYTHONPATH="${PROJECT_DIR}"

# ── Ensure required packages are installed ────────────────────────────────────
if ! "${VENV_PYTHON}" -c "import fastapi, uvicorn, httpx, torch" >/dev/null 2>&1; then
    echo "[start] Installing missing packages..."
    "${VENV_PYTHON}" -m pip install -q -r "${PROJECT_DIR}/requirements.txt"
fi

echo "[start] Starting uvicorn on port 8000..."
exec env \
    PYTHONPATH="${PROJECT_DIR}" \
    API_BASE_URL="${API_BASE_URL:-}" \
    API_KEY="${API_KEY:-}" \
    HF_TOKEN="${HF_TOKEN:-}" \
    LARAVEL_BASE_URL="${LARAVEL_BASE_URL:-}" \
    LARAVEL_SERVER_ID="${LARAVEL_SERVER_ID:-}" \
    RUNPOD_API_KEY="${RUNPOD_API_KEY:-}" \
    "${VENV_PYTHON}" -m uvicorn app.server:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    2>&1 | tee -a "${SERVER_LOG}"
