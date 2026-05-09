"""
io_utils.py
-----------
Utilities for reading patch images, writing HDF5 feature files,
writing per-slide JSON metadata, and writing job-level manifests.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Optional

import h5py
import numpy as np
from PIL import Image, UnidentifiedImageError

from app.config import SUPPORTED_EXTENSIONS, JobConfig

logger = logging.getLogger(__name__)


# ─── Image loading ────────────────────────────────────────────────────────────

def iter_patch_paths(patches_dir: Path) -> Generator[Path, None, None]:
    """
    Yield all valid patch image paths from *patches_dir* (non-recursive).
    Only files with a supported extension are yielded.
    Files whose stem starts with 'overview' (case-insensitive) are skipped —
    these are whole-slide thumbnail images, not tissue patches.
    """
    for path in sorted(patches_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if path.stem.lower().startswith("overview"):
            logger.debug("Skipping overview image: %s", path.name)
            continue
        yield path


def load_patch_image(path: Path) -> Optional[np.ndarray]:
    """
    Load a single patch image as a (H, W, 3) uint8 numpy array.

    Returns None and logs a warning when the file is missing or corrupted.
    """
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            return np.asarray(img, dtype=np.uint8)
    except FileNotFoundError:
        logger.warning("Patch not found: %s", path)
    except UnidentifiedImageError:
        logger.warning("Corrupted/unreadable image: %s", path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load %s – %s", path, exc)
    return None


# ─── HDF5 feature writer ──────────────────────────────────────────────────────

class FeatureWriter:
    """
    Incremental writer that appends feature vectors and patch coordinates
    to a single HDF5 file.  The file is opened once and closed explicitly
    so the job can be resumed safely without data loss.

    Layout inside the HDF5 file
    ───────────────────────────
    /features        float32  (N, D)   – feature vectors
    /patch_names     bytes    (N,)     – original file names (UTF-8)
    /coordinates     float32  (N, 2)   – (row, col) extracted from filename
    /metadata        attrs            – slide_id, patch_size, magnification, model
    """

    def __init__(
        self,
        h5_path: Path,
        slide_id: str,
        patch_size_px: int,
        magnification: str,
        model_name: str,
        feature_dim: int,
    ) -> None:
        self._path = h5_path
        self._slide_id = slide_id
        self._patch_size_px = patch_size_px
        self._magnification = magnification
        self._model_name = model_name
        self._feature_dim = feature_dim
        self._file: Optional[h5py.File] = None

        # Check whether the file already exists (resume scenario)
        self._resume = h5_path.exists()

    def open(self) -> None:
        mode = "a" if self._resume else "w"
        self._file = h5py.File(self._path, mode)

        if not self._resume:
            # Create resizable datasets
            self._file.create_dataset(
                "features",
                shape=(0, self._feature_dim),
                maxshape=(None, self._feature_dim),
                dtype="float32",
                compression="gzip",
                compression_opts=4,
            )
            self._file.create_dataset(
                "patch_names",
                shape=(0,),
                maxshape=(None,),
                dtype=h5py.special_dtype(vlen=str),
            )
            self._file.create_dataset(
                "coordinates",
                shape=(0, 2),
                maxshape=(None, 2),
                dtype="float32",
            )
            # Metadata stored as HDF5 attributes
            attrs = self._file.attrs
            attrs["slide_id"] = self._slide_id
            attrs["patch_size_px"] = self._patch_size_px
            attrs["magnification"] = self._magnification
            attrs["model_name"] = self._model_name
            attrs["created_at"] = datetime.now(timezone.utc).isoformat()

    def already_written(self) -> set[str]:
        """Return the set of patch names already stored (for resume)."""
        if self._file is None:
            raise RuntimeError("FeatureWriter not opened – call open() first.")
        if "patch_names" in self._file:
            return set(self._file["patch_names"][:].tolist())
        return set()

    def write_batch(
        self,
        features: np.ndarray,
        patch_names: list[str],
        coordinates: np.ndarray,
    ) -> None:
        """Append a batch of features to the open HDF5 file."""
        if self._file is None:
            raise RuntimeError("FeatureWriter not opened – call open() first.")

        n = features.shape[0]
        for dataset_name, data in (
            ("features", features),
            ("patch_names", np.array(patch_names, dtype=object)),
            ("coordinates", coordinates),
        ):
            ds = self._file[dataset_name]
            old_size = ds.shape[0]
            ds.resize(old_size + n, axis=0)
            ds[old_size: old_size + n] = data

        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.attrs["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._file.close()
            self._file = None

    def __enter__(self) -> "FeatureWriter":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# ─── Per-slide JSON metadata ──────────────────────────────────────────────────

def write_slide_metadata(
    meta_path: Path,
    slide_id: str,
    sample_id: int,
    patch_count: int,
    failed_count: int,
    patch_size_px: int,
    magnification: str,
    model_name: str,
    model_version: str,
    h5_path: Path,
    runpod_storage_path: str,
) -> None:
    """Write a JSON sidecar file next to the HDF5 output."""
    payload = {
        "slide_id": slide_id,
        "sample_id": sample_id,
        "patch_count": patch_count,
        "failed_patch_count": failed_count,
        "patch_size_px": patch_size_px,
        "magnification": magnification,
        "model_name": model_name,
        "model_version": model_version,
        "h5_file": str(h5_path),
        "runpod_storage_path": runpod_storage_path,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Slide metadata written → %s", meta_path)


# ─── Job manifest ─────────────────────────────────────────────────────────────

def write_job_manifest(
    manifest_path: Path,
    sample_id: int,
    slide_id: str,
    status: str,
    input_gdrive_path: str,
    output_gdrive_path: str,
    runpod_output_path: str,
    patch_count: int,
    failed_count: int,
    error_message: str = "",
) -> None:
    """
    Write (or overwrite) the structured JSON manifest for a single job.
    This is the source of truth when the API server is temporarily unavailable.
    """
    payload = {
        "sample_id": sample_id,
        "slide_id": slide_id,
        "status": status,                           # pending / processing / completed / failed
        "input_gdrive_path": input_gdrive_path,
        "output_gdrive_path": output_gdrive_path,
        "runpod_output_path": runpod_output_path,
        "patch_count": patch_count,
        "failed_patch_count": failed_count,
        "error_message": error_message,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Manifest written → %s", manifest_path)


# ─── Job summary ──────────────────────────────────────────────────────────────

def write_job_summary(
    summary_path: Path,
    cfg: JobConfig,
    slides_processed: int,
    total_patches: int,
    total_failed: int,
    output_paths: list[str],
    errors: list[str],
) -> None:
    """Write a high-level summary JSON for the entire extraction run."""
    payload = {
        "sample_id": cfg.sample_id,
        "slide_id": cfg.slide_id,
        "patch_size_px": cfg.patch_size_px,
        "magnification": cfg.magnification,
        "slides_processed": slides_processed,
        "total_patches": total_patches,
        "total_failed": total_failed,
        "output_paths": output_paths,
        "errors": errors,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Job summary written → %s", summary_path)
