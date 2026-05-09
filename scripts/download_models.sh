#!/usr/bin/env bash
# =============================================================================
# download_models.sh
# Download TITAN and CONCH weights from Hugging Face into the persistent
# /workspace/models directory.
#
# This script is idempotent — it skips any model that is already present.
# Safe to run on every pod startup.
#
# Requirements:
#   - Python 3.10+
#   - huggingface_hub installed (pip install huggingface_hub)
#   - Optionally: HUGGING_FACE_HUB_TOKEN env var for gated repos
#
# Usage:
#   bash scripts/download_models.sh
# =============================================================================

set -euo pipefail

WORKSPACE="${WORKSPACE_DIR:-/workspace}"
MODELS_DIR="${WORKSPACE}/models"
CACHE_DIR="${MODELS_DIR}/cache"
TITAN_DIR="${MODELS_DIR}/titan"
CONCH_DIR="${MODELS_DIR}/conch"

export HF_HOME="${CACHE_DIR}"
export HF_HUB_CACHE="${CACHE_DIR}/hub"
export TRANSFORMERS_CACHE="${CACHE_DIR}"

TITAN_REPO="${TITAN_HF_REPO:-MahmoodLab/TITAN}"
CONCH_REPO="${CONCH_HF_REPO:-MahmoodLab/conch}"

mkdir -p "${TITAN_DIR}" "${CONCH_DIR}" "${CACHE_DIR}"

echo "================================================================"
echo "  Model download script"
echo "  TITAN repo : ${TITAN_REPO}  →  ${TITAN_DIR}"
echo "  CONCH repo : ${CONCH_REPO}  →  ${CONCH_DIR}"
echo "================================================================"

# ── TITAN ─────────────────────────────────────────────────────────────────────
if [ -f "${TITAN_DIR}/config.json" ]; then
    echo "[TITAN] Weights already present – skipping."
else
    echo "[TITAN] Downloading from Hugging Face …"
    python3 - <<'PYEOF'
import os, sys
from huggingface_hub import snapshot_download
from pathlib import Path

titan_dir = os.environ.get("TITAN_DIR", "/workspace/models/titan")
titan_repo = os.environ.get("TITAN_HF_REPO", "MahmoodLab/TITAN")

print(f"  repo_id   : {titan_repo}")
print(f"  local_dir : {titan_dir}")

snapshot_download(
    repo_id=titan_repo,
    local_dir=titan_dir,
    local_dir_use_symlinks=False,
    ignore_patterns=["*.ot", "*.msgpack"],
)
print("[TITAN] Download complete.")
PYEOF
fi

# ── CONCH ─────────────────────────────────────────────────────────────────────
# CONCH v1.5 is bundled inside the TITAN repo (conch_v1_5_pytorch_model.bin).
# No separate download needed — TITAN's return_conch() extracts it at runtime.
if [ -f "${TITAN_DIR}/conch_v1_5_pytorch_model.bin" ]; then
    echo "[CONCH] Found bundled inside TITAN weights – no separate download needed."
else
    echo "[CONCH] WARNING: conch_v1_5_pytorch_model.bin not found in TITAN dir."
    echo "        Re-run TITAN download or check ${TITAN_DIR} contents."
fi

echo "================================================================"
echo "  All models ready."
echo "================================================================"
