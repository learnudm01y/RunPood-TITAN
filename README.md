# RunPood-histo-TITAN

Feature extraction pipeline for histopathology whole-slide image patches using TITAN / CONCH.  
Designed to run on **RunPod GPU pods** with Google Drive as input/output storage and the **histopathology_laravel102** management system as the controlling API.

---

## Purpose

This project does **one thing only**: extract patch-level feature vectors from already-prepared image patches (PNG / JPG) using the TITAN foundation model, write them to HDF5 files, and report the results back to the management server.

It does **not**:
- Cut patches from WSI files (that is a separate pipeline stage)
- Train any model
- Run inference / classification

---

## Directory layout

```
/workspace/
├── app/
│   ├── __init__.py
│   ├── config.py           ← All configuration, env-variable driven
│   ├── main.py             ← Pipeline entry-point (argparse CLI)
│   ├── extractor.py        ← TITAN model loader + batch extractor
│   ├── io_utils.py         ← HDF5 writer, metadata, manifest helpers
│   ├── sync_google_drive.py← rclone wrappers (sync in / sync out / cleanup)
│   └── api_client.py       ← HTTP client for management API reporting
├── scripts/
│   ├── download_models.sh  ← Download TITAN + CONCH weights (idempotent)
│   ├── sync_from_gdrive.sh ← Sync patches from Google Drive
│   ├── sync_to_gdrive.sh   ← Sync output features to Google Drive
│   └── run_extract.sh      ← Full pipeline wrapper (recommended entry-point)
├── docker/
│   └── Dockerfile
├── input/
│   ├── patches/            ← Downloaded patches land here (per-slide sub-folder)
│   └── metadata/
├── output/
│   ├── features/           ← HDF5 + metadata JSON (per-slide sub-folder)
│   ├── logs/               ← Per-run log files (kept after cleanup)
│   └── manifests/          ← Job manifests + summary JSON (kept after cleanup)
├── models/
│   ├── titan/              ← TITAN weights (downloaded once, persistent)
│   ├── conch/              ← CONCH weights (downloaded once, persistent)
│   └── cache/              ← Hugging Face hub cache
├── requirements.txt
└── README.md
```

---

## Prerequisites

| Requirement | Notes |
|---|---|
| RunPod GPU pod | A100 / RTX 4090 recommended. Minimum 16 GB VRAM. |
| Persistent volume | Mount at `/workspace`. Models, logs, and manifests survive across restarts. |
| rclone | Pre-installed in the Docker image. Configure a remote named `gdrive` (see below). |
| Hugging Face token | Required for gated models (TITAN / CONCH). Set `HUGGING_FACE_HUB_TOKEN`. |
| Management API | The Laravel server running `histopathology_laravel102`. |

---

## Environment variables

All configuration is driven by environment variables. Set them in the RunPod pod template or pass them via `docker run -e`.

| Variable | Required | Default | Description |
|---|---|---|---|
| `SAMPLE_ID` | **yes** | — | `samples.id` in the management database |
| `SLIDE_ID` | **yes** | — | Human-readable slide identifier (used as folder name) |
| `PATCH_SIZE` | no | `256` | Patch size in pixels |
| `MAGNIFICATION` | no | `20x` | Magnification label |
| `BATCH_SIZE` | no | `32` | GPU batch size |
| `GDRIVE_INPUT_PATH` | no | `histo-pipeline/input/patches/<SLIDE_ID>` | rclone path to input patches |
| `GDRIVE_OUTPUT_PATH` | no | `histo-pipeline/output/features` | rclone path root for output |
| `RCLONE_REMOTE` | no | `gdrive` | rclone remote name |
| `API_BASE_URL` | **yes** | — | Base URL of management server, e.g. `http://127.0.0.1:8000` |
| `API_KEY` | **yes** | — | Bearer token for the management API |
| `HUGGING_FACE_HUB_TOKEN` | yes (first run) | — | HF token for gated model download |
| `TITAN_HF_REPO` | no | `MahmoodLab/TITAN` | Hugging Face repo for TITAN |
| `CONCH_HF_REPO` | no | `MahmoodLab/conch` | Hugging Face repo for CONCH |
| `WORKSPACE_DIR` | no | `/workspace` | Root of the persistent workspace |

---

## Quick start on RunPod

### Step 1 – Create a RunPod persistent volume

In the RunPod console, create a **Network Volume** of ≥ 50 GB and mount it at `/workspace`.

### Step 2 – Configure rclone for Google Drive

SSH into the pod and run:

```bash
rclone config
# → n (new remote)
# → name: gdrive
# → type: drive (Google Drive)
# → Follow the OAuth flow
```

The rclone config file is saved to `~/.config/rclone/rclone.conf`.  
**Copy it to the persistent volume so it survives pod deletion:**

```bash
cp ~/.config/rclone/rclone.conf /workspace/rclone.conf
```

On every new pod, restore it:

```bash
mkdir -p ~/.config/rclone
cp /workspace/rclone.conf ~/.config/rclone/rclone.conf
```

### Step 3 – First-time model download

Run once. Models are saved to `/workspace/models/` and never re-downloaded:

```bash
cd /workspace
bash scripts/download_models.sh
```

### Step 4 – Run feature extraction

```bash
export SAMPLE_ID=42
export SLIDE_ID=TCGA-A1-A0SD-01Z-00-DX1
export PATCH_SIZE=256
export MAGNIFICATION=20x
export GDRIVE_INPUT_PATH="histo-pipeline/input/patches/TCGA-A1-A0SD-01Z-00-DX1"
export GDRIVE_OUTPUT_PATH="histo-pipeline/output/features"
export API_BASE_URL="http://127.0.0.1:8000"
export API_KEY="your-server-api-key"
export HUGGING_FACE_HUB_TOKEN="hf_..."

bash scripts/run_extract.sh
```

Or directly via Python:

```bash
python3 -m app.main \
    --sample-id 42 \
    --slide-id TCGA-A1-A0SD-01Z-00-DX1 \
    --patch-size 256 \
    --magnification 20x \
    --gdrive-input-path "histo-pipeline/input/patches/TCGA-A1-A0SD-01Z-00-DX1" \
    --gdrive-output-path "histo-pipeline/output/features" \
    --api-base-url "http://127.0.0.1:8000" \
    --api-key "your-key"
```

### Step 5 – Verify outputs

```bash
ls /workspace/output/features/TCGA-A1-A0SD-01Z-00-DX1/
# → TCGA-A1-A0SD-01Z-00-DX1.h5
# → TCGA-A1-A0SD-01Z-00-DX1_metadata.json

ls /workspace/output/manifests/
# → TCGA-A1-A0SD-01Z-00-DX1_manifest.json
# → TCGA-A1-A0SD-01Z-00-DX1_summary.json
```

---

## Docker usage

Build the image:

```bash
docker build -t runpood-histo-titan -f docker/Dockerfile .
```

Run on RunPod (mount persistent volume):

```bash
docker run --gpus all --rm \
    -e SAMPLE_ID=42 \
    -e SLIDE_ID=TCGA-A1-A0SD-01Z-00-DX1 \
    -e API_BASE_URL=https://your-server.com \
    -e API_KEY=secret \
    -e HUGGING_FACE_HUB_TOKEN=hf_... \
    -v /mnt/persistent:/workspace \
    runpood-histo-titan
```

---

## Pipeline behavior

```
Startup
  └─ create dirs
  └─ notify API: status = processing

Stage 1 – rclone sync (Google Drive → /workspace/input/patches/<slide_id>/)
Stage 2 – check / download TITAN + CONCH weights (idempotent)
Stage 3 – load TITAN once
Stage 4 – resume check (skip already-processed patches if HDF5 exists)
Stage 5 – batch feature extraction → /workspace/output/features/<slide_id>/<slide_id>.h5
Stage 6 – write per-slide metadata JSON
Stage 7 – write local job manifest (fallback if API is down)
Stage 8 – notify API: status = completed + runpod_output_path + features_gdrive_path
Stage 9 – rclone sync (/workspace/output/features/<slide_id>/ → Google Drive)
Stage 10 – delete local output files (HDF5 + metadata) from RunPod
Stage 11 – delete local input patches from RunPod
Stage 12 – write job summary JSON
```

Logs and manifests are **never deleted** from RunPod so you always have a local trace.

---

## Output files

| File | Location | Description |
|---|---|---|
| `<slide_id>.h5` | `output/features/<slide_id>/` | HDF5: features (N×D), patch names, coordinates |
| `<slide_id>_metadata.json` | `output/features/<slide_id>/` | Per-slide metadata (model, patch size, counts) |
| `<slide_id>_manifest.json` | `output/manifests/` | Job status, paths, timestamps (local fallback) |
| `<slide_id>_summary.json` | `output/manifests/` | High-level job summary |
| `<slide_id>_<timestamp>.log` | `output/logs/` | Full pipeline log |

---

## HDF5 file structure

```
/features        float32  (N, D)   – feature vectors
/patch_names     str      (N,)     – original file names
/coordinates     float32  (N, 2)   – (row, col) pixel coordinates
attrs:
  slide_id       str
  patch_size_px  int
  magnification  str
  model_name     str
  created_at     ISO-8601
  updated_at     ISO-8601
```

---

## API contract (what must be added to histopathology_laravel102)

See [API requirements below](#missing-api-endpoints-in-the-management-system) and `app/api_client.py`.

---

## Troubleshooting

| Issue | Solution |
|---|---|
| `rclone: command not found` | Install rclone: `curl https://rclone.org/install.sh \| bash` |
| `No GPU detected` | Check `nvidia-smi`. Ensure `--gpus all` in docker run. |
| `TITAN load error: trust_remote_code` | Add `--trust-remote-code` or use `trust_remote_code=True` (already set). |
| `API_BASE_URL not configured` | Set the env var. The job still completes; only API reporting is skipped. |
| `No patch images found` | Verify rclone sync succeeded and patches are PNG/JPG. |
| HF download fails | Set `HUGGING_FACE_HUB_TOKEN` and accept model terms on huggingface.co. |

---

## Resume behavior

If the job is interrupted mid-extraction, simply rerun the same command.  
The pipeline detects the existing HDF5 file, reads the list of already-processed patch names, and skips them. Only new patches are processed.
