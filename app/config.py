"""
config.py
---------
Central configuration for the TITAN feature-extraction pipeline.

All values can be overridden via environment variables so the Docker image
needs zero modification between runs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# ─── Workspace root (persistent volume on RunPod) ─────────────────────────────
WORKSPACE = Path(os.environ.get("WORKSPACE_DIR", "/workspace"))

# ─── Sub-directories ──────────────────────────────────────────────────────────
INPUT_PATCHES_DIR: Path = WORKSPACE / "input" / "patches"
INPUT_METADATA_DIR: Path = WORKSPACE / "input" / "metadata"
OUTPUT_FEATURES_DIR: Path = WORKSPACE / "output" / "features"
OUTPUT_LOGS_DIR: Path = WORKSPACE / "output" / "logs"
OUTPUT_MANIFESTS_DIR: Path = WORKSPACE / "output" / "manifests"
MODELS_TITAN_DIR: Path = WORKSPACE / "models" / "titan"
MODELS_CONCH_DIR: Path = WORKSPACE / "models" / "conch"
MODELS_CACHE_DIR: Path = WORKSPACE / "models" / "cache"

# ─── Model identifiers on Hugging Face ────────────────────────────────────────
TITAN_HF_REPO: str = os.environ.get(
    "TITAN_HF_REPO", "MahmoodLab/TITAN"
)
CONCH_HF_REPO: str = os.environ.get(
    "CONCH_HF_REPO", "MahmoodLab/conch"
)

# ─── Google Drive (rclone remote name) ────────────────────────────────────────
RCLONE_REMOTE: str = os.environ.get("RCLONE_REMOTE", "gdrive")
GDRIVE_INPUT_PATH: str = os.environ.get(
    "GDRIVE_INPUT_PATH", "histo-pipeline/input/patches"
)
GDRIVE_OUTPUT_PATH: str = os.environ.get(
    "GDRIVE_OUTPUT_PATH", "histo-pipeline/output/features"
)

# ─── API server (Laravel management system) ───────────────────────────────────
API_BASE_URL: str = os.environ.get("API_BASE_URL", "").rstrip("/")
API_KEY: str = os.environ.get("API_KEY", "")
API_TIMEOUT: int = int(os.environ.get("API_TIMEOUT", "30"))

# ─── Extraction parameters (can be overridden by CLI or API response) ─────────
BATCH_SIZE: int = int(os.environ.get("BATCH_SIZE", "32"))
NUM_WORKERS: int = int(os.environ.get("NUM_WORKERS", "4"))
PATCH_SIZE_PX: int = int(os.environ.get("PATCH_SIZE_PX", "256"))
MAGNIFICATION: str = os.environ.get("MAGNIFICATION", "20x")

# ─── Hugging Face cache override ──────────────────────────────────────────────
HF_HOME: str = str(MODELS_CACHE_DIR)

# ─── Supported image extensions ───────────────────────────────────────────────
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg")


@dataclass
class JobConfig:
    """
    Runtime configuration assembled from environment + CLI arguments.
    Passed around as a single object so callers never read raw globals.
    """

    sample_id: int
    slide_id: str
    patch_size_px: int = PATCH_SIZE_PX
    magnification: str = MAGNIFICATION
    batch_size: int = BATCH_SIZE
    num_workers: int = NUM_WORKERS

    # Paths (set after config is resolved)
    input_patches_dir: Path = field(default_factory=lambda: INPUT_PATCHES_DIR)
    input_metadata_dir: Path = field(default_factory=lambda: INPUT_METADATA_DIR)
    output_features_dir: Path = field(default_factory=lambda: OUTPUT_FEATURES_DIR)
    output_logs_dir: Path = field(default_factory=lambda: OUTPUT_LOGS_DIR)
    output_manifests_dir: Path = field(default_factory=lambda: OUTPUT_MANIFESTS_DIR)

    # Google Drive paths (resolved per job)
    gdrive_input_path: str = GDRIVE_INPUT_PATH
    gdrive_output_path: str = GDRIVE_OUTPUT_PATH

    # API reporting
    api_base_url: str = API_BASE_URL
    api_key: str = API_KEY

    def ensure_dirs(self) -> None:
        """Create all required directories if they do not exist."""
        for d in (
            self.input_patches_dir,
            self.input_metadata_dir,
            self.output_features_dir,
            self.output_logs_dir,
            self.output_manifests_dir,
            MODELS_TITAN_DIR,
            MODELS_CONCH_DIR,
            MODELS_CACHE_DIR,
        ):
            d.mkdir(parents=True, exist_ok=True)
