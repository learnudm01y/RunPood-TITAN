#!/usr/bin/env bash
# =============================================================================
# setup_env.sh  –  TITAN feature-extraction server
# Source this before starting the server:
#   source /workspace/RunPood-histo-TITAN/scripts/setup_env.sh
# =============================================================================

# ─── Activate venv if it exists ───────────────────────────────────────────────
VENV="/workspace/venv"
if [ -f "${VENV}/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
    echo "[setup_env] venv activated: ${VENV}"
fi

# ─── Paths ────────────────────────────────────────────────────────────────────
export WORKSPACE_DIR="/workspace"
export HF_HOME="${WORKSPACE_DIR}/models/cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}"
export PYTHONPATH="/workspace/RunPood-histo-TITAN"

# ─── Server port (TITAN runs on 8001 when Virchow2 is on 8000) ───────────────
export PORT="${PORT:-8001}"

# ─── Laravel connectivity ─────────────────────────────────────────────────────
export API_BASE_URL="${API_BASE_URL:-https://ai.histopathology.cloud}"
export LARAVEL_BASE_URL="${API_BASE_URL}"   # alias used by server.py self-registration

# ─── Auth ─────────────────────────────────────────────────────────────────────
# Set API_KEY as RunPod template env var (servers_names.api_key for this server)
export API_KEY="${API_KEY:-}"
export RUNPOD_API_KEY="${API_KEY}"

# ─── rclone ───────────────────────────────────────────────────────────────────
export RCLONE_REMOTE="${RCLONE_REMOTE:-gdrive}"

echo "[setup_env] TITAN environment ready."
echo "  WORKSPACE_DIR   = ${WORKSPACE_DIR}"
echo "  PORT            = ${PORT}"
echo "  API_BASE_URL    = ${API_BASE_URL}"
echo "  RCLONE_REMOTE   = ${RCLONE_REMOTE}"
