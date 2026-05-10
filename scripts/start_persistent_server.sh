#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
WORKSPACE="${WORKSPACE_DIR:-/workspace}"
SETUP_SCRIPT="${WORKSPACE}/setup_env.sh"
VENV_PYTHON="${WORKSPACE}/venv/bin/python"
SERVER_LOG="${WORKSPACE}/server.log"

echo "[start_persistent_server] workspace      : ${WORKSPACE}"
echo "[start_persistent_server] project        : ${PROJECT_DIR}"

if [ ! -f "${SETUP_SCRIPT}" ]; then
    echo "ERROR: Missing ${SETUP_SCRIPT}"
    echo "       Create it inside the Network Volume before using this script."
    exit 1
fi

if [ ! -x "${VENV_PYTHON}" ]; then
    echo "ERROR: Missing virtual environment python at ${VENV_PYTHON}"
    echo "       Recreate /workspace/venv inside the Network Volume first."
    exit 1
fi

# Restore environment variables, rclone config, and anything else the pod needs.
source "${SETUP_SCRIPT}"

cd "${PROJECT_DIR}"

export PYTHONPATH="${PROJECT_DIR}"

ensure_packages() {
    if ! "${VENV_PYTHON}" - <<'PY' >/dev/null 2>&1
import fastapi, uvicorn, httpx, torch
PY
    then
        echo "[start_persistent_server] Installing missing Python packages into /workspace/venv ..."
        "${VENV_PYTHON}" -m pip install -q -r "${PROJECT_DIR}/requirements.txt"
    fi
}

ensure_packages

if [ -f "${SERVER_LOG}" ]; then
    echo "[start_persistent_server] Existing server log: ${SERVER_LOG}"
fi

echo "[start_persistent_server] Starting server..."
exec env \
    PYTHONPATH="${PROJECT_DIR}" \
    API_BASE_URL="${API_BASE_URL:-}" \
    API_KEY="${API_KEY:-}" \
    HF_TOKEN="${HF_TOKEN:-}" \
    "${VENV_PYTHON}" -m uvicorn app.server:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1
