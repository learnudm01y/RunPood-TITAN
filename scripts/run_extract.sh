#!/usr/bin/env bash
# =============================================================================
# run_extract.sh
# Convenience wrapper to run the full TITAN feature-extraction pipeline.
#
# All required values are read from environment variables OR passed as
# positional arguments (for quick manual testing).
#
# Usage (environment-variable driven — recommended for RunPod):
#   SAMPLE_ID=42 \
#   SLIDE_ID=TCGA-A1-A0SD-01Z-00-DX1 \
#   PATCH_SIZE=256 \
#   MAGNIFICATION=20x \
#   GDRIVE_INPUT_PATH="histo-pipeline/input/patches/TCGA-A1-A0SD-01Z-00-DX1" \
#   GDRIVE_OUTPUT_PATH="histo-pipeline/output/features" \
#   API_BASE_URL="https://your-management-server.com" \
#   API_KEY="your-secret-api-key" \
#   bash scripts/run_extract.sh
#
# Usage (positional):
#   bash scripts/run_extract.sh <sample_id> <slide_id> [patch_size] [magnification]
# =============================================================================

set -euo pipefail

WORKSPACE="${WORKSPACE_DIR:-/workspace}"
export WORKSPACE_DIR="${WORKSPACE}"

# ── Resolve arguments (positional > env) ──────────────────────────────────────
SAMPLE_ID="${1:-${SAMPLE_ID:?'SAMPLE_ID not set'}}"
SLIDE_ID="${2:-${SLIDE_ID:?'SLIDE_ID not set'}}"
PATCH_SIZE="${3:-${PATCH_SIZE:-256}}"
MAGNIFICATION="${4:-${MAGNIFICATION:-20x}}"
BATCH_SIZE="${BATCH_SIZE:-32}"

GDRIVE_INPUT_PATH="${GDRIVE_INPUT_PATH:-histo-pipeline/input/patches/${SLIDE_ID}}"
GDRIVE_OUTPUT_PATH="${GDRIVE_OUTPUT_PATH:-histo-pipeline/output/features}"
RCLONE_REMOTE="${RCLONE_REMOTE:-gdrive}"
API_BASE_URL="${API_BASE_URL:-}"
API_KEY="${API_KEY:-}"

# ── Set HuggingFace cache dirs ─────────────────────────────────────────────────
export HF_HOME="${WORKSPACE}/models/cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}"

echo "================================================================"
echo "  RunPood-histo-TITAN  |  Feature Extraction"
echo "  sample_id   : ${SAMPLE_ID}"
echo "  slide_id    : ${SLIDE_ID}"
echo "  patch_size  : ${PATCH_SIZE}"
echo "  magnification: ${MAGNIFICATION}"
echo "  gdrive_in   : ${GDRIVE_INPUT_PATH}"
echo "  gdrive_out  : ${GDRIVE_OUTPUT_PATH}"
echo "================================================================"

cd "${WORKSPACE}"

python3 -m app.main \
    --sample-id        "${SAMPLE_ID}" \
    --slide-id         "${SLIDE_ID}" \
    --patch-size       "${PATCH_SIZE}" \
    --magnification    "${MAGNIFICATION}" \
    --batch-size       "${BATCH_SIZE}" \
    --gdrive-input-path  "${GDRIVE_INPUT_PATH}" \
    --gdrive-output-path "${GDRIVE_OUTPUT_PATH}" \
    --rclone-remote    "${RCLONE_REMOTE}" \
    --api-base-url     "${API_BASE_URL}" \
    --api-key          "${API_KEY}"

echo "================================================================"
echo "  Extraction finished successfully."
echo "================================================================"
