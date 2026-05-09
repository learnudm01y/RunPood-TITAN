#!/usr/bin/env bash
# =============================================================================
# sync_from_gdrive.sh
# Sync input patches from Google Drive to the local RunPod workspace.
#
# Usage:
#   bash scripts/sync_from_gdrive.sh <remote> <gdrive_path> <local_dir>
#
# Example:
#   bash scripts/sync_from_gdrive.sh \
#       gdrive \
#       "histo-pipeline/input/patches/TCGA-A1-A0SD-01Z-00-DX1" \
#       /workspace/input/patches/TCGA-A1-A0SD-01Z-00-DX1
# =============================================================================

set -euo pipefail

REMOTE="${1:?Usage: $0 <remote> <gdrive_path> <local_dir>}"
GDRIVE_PATH="${2:?Missing gdrive_path argument}"
LOCAL_DIR="${3:?Missing local_dir argument}"

echo "[sync_from_gdrive] Remote  : ${REMOTE}:${GDRIVE_PATH}"
echo "[sync_from_gdrive] Local   : ${LOCAL_DIR}"

mkdir -p "${LOCAL_DIR}"

rclone sync \
    "${REMOTE}:${GDRIVE_PATH}" \
    "${LOCAL_DIR}" \
    --progress \
    --transfers 8 \
    --checkers 16 \
    --contimeout 60s \
    --timeout 300s \
    --retries 3 \
    --low-level-retries 10 \
    --log-level INFO

echo "[sync_from_gdrive] Done."
