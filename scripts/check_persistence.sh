#!/usr/bin/env bash

set -euo pipefail

WORKSPACE="${WORKSPACE_DIR:-/workspace}"
PROJECT_DIR="${PROJECT_DIR:-${WORKSPACE}/RunPood-TITAN}"
RCLONE_CONF="${HOME:-/root}/.config/rclone/rclone.conf"
VENV_DIR="${WORKSPACE}/venv"
MODEL_DIR="${WORKSPACE}/models/titan"

section() {
    printf '\n============================================================\n'
    printf '%s\n' "$1"
    printf '============================================================\n'
}

count_items() {
    local path="$1"
    if [ ! -e "$path" ]; then
        echo "MISSING"
        return
    fi

    if [ -d "$path" ]; then
        find "$path" -mindepth 1 -maxdepth 1 2>/dev/null | wc -l | awk '{print $1}'
    else
        echo "FILE"
    fi
}

section "RunPod Persistence Check"
echo "Date               : $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo "Workspace          : ${WORKSPACE}"
echo "Project directory   : ${PROJECT_DIR}"
echo "Hostname            : $(hostname)"
echo "Pod hostname        : ${RUNPOD_POD_HOSTNAME:-unset}"
echo "Public IP           : ${RUNPOD_PUBLIC_IP:-unset}"
echo "TCP SSH port        : ${RUNPOD_TCP_PORT_22:-unset}"

section "Mount Status"
if mountpoint -q "$WORKSPACE" 2>/dev/null; then
    echo "Workspace mount     : mounted"
else
    echo "Workspace mount     : not a mountpoint (still checking path contents)"
fi
df -h "$WORKSPACE" 2>/dev/null || true

section "Workspace Top-Level"
if [ -d "$WORKSPACE" ]; then
    ls -lah "$WORKSPACE"
else
    echo "Workspace directory not found: $WORKSPACE"
fi

section "Critical Paths"
for path in \
    "$WORKSPACE/setup_env.sh" \
    "$WORKSPACE/PERSISTENCE_TEST.txt" \
    "$PROJECT_DIR" \
    "$PROJECT_DIR/app" \
    "$PROJECT_DIR/scripts" \
    "$MODEL_DIR" \
    "$VENV_DIR" \
    "$RCLONE_CONF"
do
    if [ -e "$path" ]; then
        if [ -d "$path" ]; then
            echo "OK   DIR  $path  (items: $(count_items "$path"))"
        else
            echo "OK   FILE $path  (size: $(du -h "$path" 2>/dev/null | awk '{print $1}') )"
        fi
    else
        echo "MISS      $path"
    fi
done

section "Model Inventory"
if [ -d "$MODEL_DIR" ]; then
    find "$MODEL_DIR" -maxdepth 2 -type f 2>/dev/null | sed "s#^${WORKSPACE}/##" | sort
else
    echo "No model directory found at $MODEL_DIR"
fi

section "Project Inventory"
if [ -d "$PROJECT_DIR" ]; then
    find "$PROJECT_DIR" -maxdepth 2 -mindepth 1 2>/dev/null | sed "s#^${WORKSPACE}/##" | sort
else
    echo "No project directory found at $PROJECT_DIR"
fi

section "Python Environment"
if [ -x "$VENV_DIR/bin/python" ]; then
    "$VENV_DIR/bin/python" - <<'PY'
import sys
print(f"venv python      : {sys.executable}")
try:
    import torch
    print(f"torch version    : {torch.__version__}")
    print(f"torch cuda       : {torch.cuda.is_available()}")
except Exception as exc:
    print(f"torch import err : {exc}")

for module_name in ["fastapi", "uvicorn", "httpx", "einops", "einops_exts", "transformers"]:
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", "unknown")
        print(f"{module_name:<16}: {version}")
    except Exception as exc:
        print(f"{module_name:<16}: MISSING ({exc})")
PY
else
    echo "No venv python found at $VENV_DIR/bin/python"
fi

section "Rclone Config"
if [ -f "$RCLONE_CONF" ]; then
    echo "rclone config path : $RCLONE_CONF"
    grep -E '^\[' "$RCLONE_CONF" || true
else
    echo "No rclone config found at $RCLONE_CONF"
fi

section "Environment Vars"
for name in API_KEY API_BASE_URL HF_TOKEN RCLONE_REMOTE RUNPOD_POD_HOSTNAME RUNPOD_PUBLIC_IP RUNPOD_TCP_PORT_22; do
    value="${!name:-}"
    if [ -n "$value" ]; then
        case "$name" in
            HF_TOKEN|API_KEY)
                echo "$name=SET (len ${#value})"
                ;;
            *)
                echo "$name=$value"
                ;;
        esac
    else
        echo "$name=unset"
    fi
done

section "Quick Readiness"
missing=0
for path in "$WORKSPACE/setup_env.sh" "$PROJECT_DIR" "$MODEL_DIR" "$VENV_DIR/bin/python"; do
    [ -e "$path" ] || missing=$((missing + 1))
done

if [ "$missing" -eq 0 ]; then
    echo "READY: workspace, project, models, and venv are present."
else
    echo "NOT READY: $missing critical path(s) missing."
fi
