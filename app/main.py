"""
main.py
-------
Entry-point for the TITAN feature-extraction pipeline.

Typical invocation on RunPod
----------------------------
python -m app.main \
    --sample-id 42 \
    --slide-id TCGA-A1-A0SD-01Z-00-DX1 \
    --patch-size 256 \
    --magnification 20x \
    --batch-size 32 \
    --gdrive-input-path "histo-pipeline/input/patches/TCGA-A1-A0SD-01Z-00-DX1" \
    --gdrive-output-path "histo-pipeline/output/features"

Pipeline stages
---------------
1. Validate environment and create all required directories.
2. Sync input patches from Google Drive  → /workspace/input/patches/<slide_id>/.
3. Check / download TITAN and CONCH model weights.
4. Load TITAN.
5. Extract features from all patches.
6. Write per-slide HDF5 + metadata JSON.
7. Write job manifest (local fallback if API is down).
8. Report status to external API server.
9. Sync output features to Google Drive.
10. Delete local output files from RunPod (keep logs + manifests).
11. Delete local input patches from RunPod.
12. Write final job summary.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

# ─── Bootstrap logging before any local imports ───────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("titan_pipeline")

from app.api_client import APIClient
from app.config import (
    API_BASE_URL,
    API_KEY,
    API_TIMEOUT,
    BATCH_SIZE,
    GDRIVE_INPUT_PATH,
    GDRIVE_OUTPUT_PATH,
    HF_HOME,
    MAGNIFICATION,
    MODELS_CACHE_DIR,
    NUM_WORKERS,
    OUTPUT_FEATURES_DIR,
    OUTPUT_LOGS_DIR,
    OUTPUT_MANIFESTS_DIR,
    PATCH_SIZE_PX,
    RCLONE_REMOTE,
    JobConfig,
)
from app.extractor import TITANExtractor, download_conch_weights, download_titan_weights, get_extractor
from app.io_utils import (
    FeatureWriter,
    iter_patch_paths,
    write_job_manifest,
    write_job_summary,
    write_slide_metadata,
)
from app.sync_google_drive import (
    delete_local_input,
    delete_local_output,
    extract_tar_archive,
    sync_input_from_gdrive,
    sync_output_to_gdrive,
)

# Expose HF_HOME to transformers / datasets early
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(MODELS_CACHE_DIR / "hub"))


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TITAN feature extraction pipeline for RunPod",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required identifiers
    parser.add_argument("--sample-id", type=int, required=True,
                        help="samples.id in the management database.")
    parser.add_argument("--slide-id", type=str, required=True,
                        help="Human-readable slide identifier (used as folder name).")

    # Extraction parameters (overridable per-job)
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE_PX,
                        help="Patch width/height in pixels.")
    parser.add_argument("--magnification", type=str, default=MAGNIFICATION,
                        help="Magnification label, e.g. 20x.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of patches per GPU batch.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="DataLoader worker count (unused here, kept for compatibility).")

    # Google Drive paths
    parser.add_argument("--gdrive-input-path", type=str, default=GDRIVE_INPUT_PATH,
                        help="Remote rclone path for input patches.")
    parser.add_argument("--gdrive-output-path", type=str, default=GDRIVE_OUTPUT_PATH,
                        help="Remote rclone path root for output features.")

    # API
    parser.add_argument("--api-base-url", type=str, default=API_BASE_URL,
                        help="Base URL of the management API server.")
    parser.add_argument("--api-key", type=str, default=API_KEY,
                        help="Bearer token for the management API.")

    # Rclone
    parser.add_argument("--rclone-remote", type=str, default=RCLONE_REMOTE,
                        help="rclone remote name for Google Drive.")

    # Flags
    parser.add_argument("--skip-gdrive-sync", action="store_true",
                        help="Skip Google Drive sync (useful for local testing).")
    parser.add_argument("--skip-cleanup", action="store_true",
                        help="Do not delete local files after sync (useful for debugging).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Do everything except GPU extraction (for pipeline testing).")

    return parser.parse_args()


# ─── File log handler ─────────────────────────────────────────────────────────

def _setup_file_logging(log_dir: Path, slide_id: str) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    log_path = log_dir / f"{slide_id}_{ts}.log"
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    ))
    logging.getLogger().addHandler(fh)
    return log_path


# ─── Main pipeline ────────────────────────────────────────────────────────────

def run(cfg: JobConfig, args: argparse.Namespace) -> int:
    """
    Execute the full extraction pipeline.

    Returns
    -------
    int
        Exit code: 0 = success, 1 = failure.
    """
    api = APIClient(cfg.api_base_url, cfg.api_key, timeout=API_TIMEOUT)

    # ── 1. Ensure directories ─────────────────────────────────────────────────
    cfg.ensure_dirs()
    log_path = _setup_file_logging(cfg.output_logs_dir, cfg.slide_id)
    logger.info("=" * 60)
    logger.info("TITAN Feature Extraction Pipeline")
    logger.info("  sample_id   : %d", cfg.sample_id)
    logger.info("  slide_id    : %s", cfg.slide_id)
    logger.info("  patch_size  : %d px", cfg.patch_size_px)
    logger.info("  magnification: %s", cfg.magnification)
    logger.info("  batch_size  : %d", cfg.batch_size)
    logger.info("  log_file    : %s", log_path)
    logger.info("=" * 60)

    # ── 2. Notify API: starting ───────────────────────────────────────────────
    api.report_started(cfg.sample_id, cfg.slide_id)

    # Per-slide input and output directories
    slide_input_dir = cfg.input_patches_dir / cfg.slide_id
    slide_output_dir = cfg.output_features_dir / cfg.slide_id
    slide_output_dir.mkdir(parents=True, exist_ok=True)

    h5_path = slide_output_dir / f"{cfg.slide_id}.h5"
    meta_path = slide_output_dir / f"{cfg.slide_id}_metadata.json"
    manifest_path = cfg.output_manifests_dir / f"{cfg.slide_id}_manifest.json"
    summary_path = cfg.output_manifests_dir / f"{cfg.slide_id}_summary.json"

    # When dispatched by the FastAPI server (Laravel push), gdrive_output_path
    # is already the absolute per-sample folder. When invoked from the CLI
    # against a root, append the slide_id so each slide gets its own subdir.
    if getattr(cfg, "ai_model_name", None):
        gdrive_output_slide_path = cfg.gdrive_output_path
    else:
        gdrive_output_slide_path = f"{cfg.gdrive_output_path}/{cfg.slide_id}"

    # ── 3. Sync input patches from Google Drive ───────────────────────────────
    if not args.skip_gdrive_sync:
        logger.info("Stage 3: Syncing input patches from Google Drive …")
        try:
            sync_input_from_gdrive(
                rclone_remote=args.rclone_remote,
                gdrive_path=cfg.gdrive_input_path,
                local_dir=slide_input_dir,
            )
        except Exception as exc:
            msg = f"Google Drive sync failed: {exc}"
            logger.error(msg)
            write_job_manifest(
                manifest_path,
                sample_id=cfg.sample_id,
                slide_id=cfg.slide_id,
                status="failed",
                input_gdrive_path=cfg.gdrive_input_path,
                output_gdrive_path=gdrive_output_slide_path,
                runpod_output_path=str(slide_output_dir),
                patch_count=0,
                failed_count=0,
                error_message=msg,
            )
            api.report_failed(cfg.sample_id, cfg.slide_id, error_message=msg)
            return 1

        # If the synced folder contains a tar.gz archive (produced by Laravel's
        # PatchExtractionJob), extract it before proceeding.
        archive_name = getattr(cfg, "gdrive_input_archive", "patches.tar.gz")
        archive_path = slide_input_dir / archive_name
        if archive_path.exists():
            logger.info("Stage 3b: Extracting %s …", archive_path)
            try:
                extract_tar_archive(archive_path, slide_input_dir)
                # Remove archive to avoid the iterator picking it up as an image
                archive_path.unlink(missing_ok=True)
            except Exception as exc:
                msg = f"Archive extraction failed: {exc}"
                logger.error(msg)
                api.report_failed(cfg.sample_id, cfg.slide_id, error_message=msg)
                return 1
    else:
        logger.info("Stage 3: Skipping Google Drive sync (--skip-gdrive-sync).")
        slide_input_dir = cfg.input_patches_dir  # fall back to root patches dir

    # Verify input patches exist
    patch_paths = list(iter_patch_paths(slide_input_dir))
    logger.info("Found %d patch files in %s", len(patch_paths), slide_input_dir)
    if not patch_paths:
        msg = f"No patch images found in {slide_input_dir}"
        logger.error(msg)
        write_job_manifest(
            manifest_path,
            sample_id=cfg.sample_id,
            slide_id=cfg.slide_id,
            status="failed",
            input_gdrive_path=cfg.gdrive_input_path,
            output_gdrive_path=gdrive_output_slide_path,
            runpod_output_path=str(slide_output_dir),
            patch_count=0,
            failed_count=0,
            error_message=msg,
        )
        api.report_failed(cfg.sample_id, cfg.slide_id, error_message=msg)
        return 1

    # ── 4. Download model weights ─────────────────────────────────────────────
    logger.info("Stage 4: Checking / downloading model weights …")
    try:
        download_titan_weights()
        download_conch_weights()
    except Exception as exc:
        msg = f"Model download failed: {exc}"
        logger.error(msg)
        api.report_failed(cfg.sample_id, cfg.slide_id, error_message=msg)
        return 1

    # ── 5. Dry-run short-circuit ───────────────────────────────────────────────
    if args.dry_run:
        logger.info("DRY RUN – skipping actual extraction. Pipeline test passed.")
        return 0

    # ── 6. Load TITAN (singleton – loaded once at startup, reused across all jobs) ────
    logger.info("Stage 5: Loading TITAN (singleton) …")
    try:
        extractor = get_extractor(
            patch_size_px=cfg.patch_size_px,
            batch_size=cfg.batch_size,
        )
    except Exception as exc:
        msg = f"TITAN load error: {exc}"
        logger.error("%s\n%s", msg, traceback.format_exc())
        api.report_failed(cfg.sample_id, cfg.slide_id, error_message=msg)
        return 1

    # ── 7. Resume check ───────────────────────────────────────────────────────
    already_written: set[str] = set()
    if h5_path.exists():
        logger.info("Resuming: existing HDF5 found at %s", h5_path)
        try:
            writer_probe = FeatureWriter(
                h5_path, cfg.slide_id, cfg.patch_size_px,
                cfg.magnification, "TITAN", extractor.feature_dim,
            )
            writer_probe.open()
            already_written = writer_probe.already_written()
            writer_probe.close()
            logger.info("  Already processed: %d patches", len(already_written))
        except Exception as exc:
            logger.warning("Could not read existing HDF5 for resume check: %s", exc)

    remaining_paths = [p for p in patch_paths if p.name not in already_written]
    logger.info(
        "Patches to process: %d  (already done: %d)",
        len(remaining_paths),
        len(already_written),
    )

    # ── 8. Extract features ───────────────────────────────────────────────────
    logger.info("Stage 6: Extracting features …")
    total_patches = 0
    total_failed = 0

    try:
        with FeatureWriter(
            h5_path,
            slide_id=cfg.slide_id,
            patch_size_px=cfg.patch_size_px,
            magnification=cfg.magnification,
            model_name="TITAN",
            feature_dim=extractor.feature_dim,
        ) as writer:
            if remaining_paths:
                features, names, failed_names = extractor.extract_patches(
                    remaining_paths,
                    desc=f"Extracting [{cfg.slide_id}]",
                )

                total_patches = len(already_written) + len(names)
                total_failed = len(failed_names)

                if features.shape[0] > 0:
                    import numpy as np
                    # Build dummy coordinates from file names
                    # Patches are expected to be named as: row_col.png or row-col.png
                    coords = _parse_coordinates(names)
                    writer.write_batch(features, names, coords)

                if failed_names:
                    logger.warning(
                        "%d patches failed extraction: %s",
                        len(failed_names),
                        failed_names[:10],
                    )
            else:
                logger.info("All patches already extracted – nothing to do.")
                total_patches = len(already_written)
    except Exception as exc:
        msg = f"Feature extraction error: {exc}"
        logger.error("%s\n%s", msg, traceback.format_exc())
        write_job_manifest(
            manifest_path,
            sample_id=cfg.sample_id,
            slide_id=cfg.slide_id,
            status="failed",
            input_gdrive_path=cfg.gdrive_input_path,
            output_gdrive_path=gdrive_output_slide_path,
            runpod_output_path=str(slide_output_dir),
            patch_count=total_patches,
            failed_count=total_failed,
            error_message=msg,
        )
        api.report_failed(cfg.sample_id, cfg.slide_id, error_message=msg,
                          patch_count=total_patches, failed_patch_count=total_failed)
        return 1

    # ── 9. Write slide metadata ───────────────────────────────────────────────
    logger.info("Stage 7: Writing slide metadata …")
    write_slide_metadata(
        meta_path=meta_path,
        slide_id=cfg.slide_id,
        sample_id=cfg.sample_id,
        patch_count=total_patches,
        failed_count=total_failed,
        patch_size_px=cfg.patch_size_px,
        magnification=cfg.magnification,
        model_name="TITAN",
        model_version=extractor.model_version,
        h5_path=h5_path,
        runpod_storage_path=str(slide_output_dir),
    )

    # ── 10. Write local manifest ───────────────────────────────────────────────
    write_job_manifest(
        manifest_path,
        sample_id=cfg.sample_id,
        slide_id=cfg.slide_id,
        status="completed",
        input_gdrive_path=cfg.gdrive_input_path,
        output_gdrive_path=gdrive_output_slide_path,
        runpod_output_path=str(slide_output_dir),
        patch_count=total_patches,
        failed_count=total_failed,
    )

    # ── 11. Sync output → Google Drive ────────────────────────────────────────
    if not args.skip_gdrive_sync:
        logger.info("Stage 8: Syncing output to Google Drive …")
        try:
            sync_output_to_gdrive(
                rclone_remote=args.rclone_remote,
                local_dir=slide_output_dir,
                gdrive_path=gdrive_output_slide_path,
            )
        except Exception as exc:
            logger.error("Output sync to Google Drive failed: %s", exc)
            # Don't fail the job – the data is still local; update manifest
            write_job_manifest(
                manifest_path,
                sample_id=cfg.sample_id,
                slide_id=cfg.slide_id,
                status="completed",
                input_gdrive_path=cfg.gdrive_input_path,
                output_gdrive_path="",
                runpod_output_path=str(slide_output_dir),
                patch_count=total_patches,
                failed_count=total_failed,
                error_message=f"Output GDrive sync failed: {exc}",
            )
    else:
        logger.info("Stage 8: Skipping Google Drive output sync (--skip-gdrive-sync).")

    # ── 12. Report to API ─────────────────────────────────────────────────────
    logger.info("Stage 9: Reporting to management API …")
    model_name = getattr(cfg, "ai_model_name", None) or "TITAN"
    api.report_completed(
        sample_id=cfg.sample_id,
        slide_id=cfg.slide_id,
        runpod_output_path=str(slide_output_dir),
        features_gdrive_path=gdrive_output_slide_path,
        features_gdrive_folder_id="",   # rclone does not return folder IDs
        patch_count=total_patches,
        failed_patch_count=total_failed,
        model_name=model_name,
        model_version=extractor.model_version,
    )

    # ── 13. Clean up local files from RunPod ─────────────────────────────────
    if not args.skip_cleanup and not args.skip_gdrive_sync:
        logger.info("Stage 10: Removing local output files …")
        try:
            delete_local_output(slide_output_dir)
        except Exception as exc:
            logger.warning("Cleanup of output dir failed: %s", exc)

        logger.info("Stage 11: Removing local input patches …")
        try:
            delete_local_input(slide_input_dir)
        except Exception as exc:
            logger.warning("Cleanup of input dir failed: %s", exc)
    else:
        logger.info("Skipping cleanup (--skip-cleanup or --skip-gdrive-sync is set).")

    # ── 14. Write job summary ─────────────────────────────────────────────────
    write_job_summary(
        summary_path=summary_path,
        cfg=cfg,
        slides_processed=1,
        total_patches=total_patches,
        total_failed=total_failed,
        output_paths=[str(slide_output_dir)],
        errors=[],
    )

    logger.info("=" * 60)
    logger.info("Pipeline completed successfully.")
    logger.info("  patches processed : %d", total_patches)
    logger.info("  patches failed    : %d", total_failed)
    logger.info("  HDF5 output       : %s", h5_path)
    logger.info("  GDrive output     : %s", gdrive_output_slide_path)
    logger.info("=" * 60)
    return 0


# ─── Coordinate parser ────────────────────────────────────────────────────────

def _parse_coordinates(names: list[str]) -> "np.ndarray":
    """
    Parse (row, col) pixel coordinates from patch file names.

    Supported naming conventions:
      - row_col.png        →  e.g.  1024_2048.png
      - row-col.png        →  e.g.  1024-2048.png
      - Any other format   →  coordinates set to (-1, -1)
    """
    import re
    import numpy as np

    coords = []
    pattern = re.compile(r"(\d+)[_\-](\d+)")
    for name in names:
        m = pattern.search(Path(name).stem)
        if m:
            coords.append((float(m.group(1)), float(m.group(2))))
        else:
            coords.append((-1.0, -1.0))
    return np.array(coords, dtype=np.float32)


# ─── Entrypoint ───────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    cfg = JobConfig(
        sample_id=args.sample_id,
        slide_id=args.slide_id,
        patch_size_px=args.patch_size,
        magnification=args.magnification,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        gdrive_input_path=args.gdrive_input_path,
        gdrive_output_path=args.gdrive_output_path,
        api_base_url=args.api_base_url,
        api_key=args.api_key,
    )

    exit_code = run(cfg, args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
