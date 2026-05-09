#!/usr/bin/env bash
# =============================================================================
# sync_to_gdrive.sh
# Sync output features from the local RunPod workspace to Google Drive.
#
# Usage:
#   bash scripts/sync_to_gdrive.sh <remote> <local_dir> <gdrive_path>
#
# Example:
#   bash scripts/sync_to_gdrive.sh \
#       gdrive \
#       /workspace/output/features/TCGA-A1-A0SD-01Z-00-DX1 \
#       "histo-pipeline/output/features/TCGA-A1-A0SD-01Z-00-DX1"
# =============================================================================

set -euo pipefail

REMOTE="${1:?Usage: $0 <remote> <local_dir> <gdrive_path>}"
LOCAL_DIR="${2:?Missing local_dir argument}"
GDRIVE_PATH="${3:?Missing gdrive_path argument}"

echo "[sync_to_gdrive] Local   : ${LOCAL_DIR}"
echo "[sync_to_gdrive] Remote  : ${REMOTE}:${GDRIVE_PATH}"

if [ ! -d "${LOCAL_DIR}" ]; then
    echo "[sync_to_gdrive] ERROR: local directory not found: ${LOCAL_DIR}"
    exit 1
fi

rclone sync \
    "${LOCAL_DIR}" \
    "${REMOTE}:${GDRIVE_PATH}" \
    --progress \
    --transfers 8 \
    --checkers 16 \
    --contimeout 60s \
    --timeout 300s \
    --retries 3 \
    --low-level-retries 10 \
    --log-level INFO

echo "[sync_to_gdrive] Done."
