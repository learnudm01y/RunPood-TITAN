"""
sync_google_drive.py
--------------------
Google Drive ↔ RunPod synchronisation using rclone.

All rclone calls are wrapped in Python so errors are captured,
logged, and surfaced as exceptions rather than silent failures.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

_RCLONE_BIN = "rclone"


def _check_rclone() -> None:
    """Raise RuntimeError if rclone is not found on PATH."""
    if shutil.which(_RCLONE_BIN) is None:
        raise RuntimeError(
            "rclone not found on PATH. "
            "Install rclone and configure a remote named matching RCLONE_REMOTE."
        )


def _run(cmd: list[str], step_name: str) -> None:
    """
    Run a subprocess command.  Raises RuntimeError on non-zero exit.
    stdout and stderr are logged at DEBUG / ERROR level respectively.
    """
    logger.debug("rclone command: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=3600,  # 1 hour max per sync operation
    )
    if result.stdout:
        for line in result.stdout.splitlines():
            logger.debug("[rclone stdout] %s", line)
    if result.returncode != 0:
        logger.error("[rclone stderr] %s", result.stderr)
        raise RuntimeError(
            f"{step_name} failed (exit {result.returncode}): {result.stderr[:500]}"
        )
    logger.info("%s completed successfully.", step_name)


def sync_input_from_gdrive(
    rclone_remote: str,
    gdrive_path: str,
    local_dir: Path,
) -> None:
    """
    Sync input patches from Google Drive to *local_dir*.

    Uses ``rclone sync`` so the local directory mirrors the remote exactly.
    Only new / changed files are transferred; existing identical files are skipped.

    Parameters
    ----------
    rclone_remote : str
        The rclone remote name (e.g. "gdrive").
    gdrive_path : str
        Path inside the remote (e.g. "histo-pipeline/input/patches/TCGA-slide-001").
    local_dir : Path
        Destination directory on the RunPod filesystem.
    """
    _check_rclone()
    local_dir.mkdir(parents=True, exist_ok=True)
    remote_src = f"{rclone_remote}:{gdrive_path}"

    logger.info("Syncing patches from %s → %s", remote_src, local_dir)
    _run(
        [
            _RCLONE_BIN, "sync",
            remote_src,
            str(local_dir),
            "--progress",
            "--transfers", "8",
            "--checkers", "16",
            "--contimeout", "60s",
            "--timeout", "300s",
            "--retries", "3",
            "--low-level-retries", "10",
        ],
        step_name="sync_input_from_gdrive",
    )


def sync_output_to_gdrive(
    rclone_remote: str,
    local_dir: Path,
    gdrive_path: str,
) -> None:
    """
    Sync output features from *local_dir* back to Google Drive.

    Parameters
    ----------
    rclone_remote : str
        The rclone remote name.
    local_dir : Path
        Source directory containing HDF5 files and metadata.
    gdrive_path : str
        Destination path inside the remote
        (e.g. "histo-pipeline/output/features/TCGA-slide-001").
    """
    _check_rclone()
    remote_dst = f"{rclone_remote}:{gdrive_path}"

    logger.info("Syncing features from %s → %s", local_dir, remote_dst)
    _run(
        [
            _RCLONE_BIN, "sync",
            str(local_dir),
            remote_dst,
            "--progress",
            "--transfers", "8",
            "--checkers", "16",
            "--contimeout", "60s",
            "--timeout", "300s",
            "--retries", "3",
            "--low-level-retries", "10",
        ],
        step_name="sync_output_to_gdrive",
    )


def delete_local_output(local_dir: Path) -> None:
    """
    Remove all files in *local_dir* after a successful sync to Google Drive.
    The directory itself is kept so the next job has a clean landing zone.

    Raises RuntimeError if any file deletion fails.
    """
    errors: list[str] = []
    for item in local_dir.iterdir():
        if item.is_file():
            try:
                item.unlink()
                logger.debug("Deleted local file: %s", item)
            except OSError as exc:
                errors.append(f"{item}: {exc}")
        elif item.is_dir():
            import shutil as _shutil
            try:
                _shutil.rmtree(item)
                logger.debug("Deleted local dir: %s", item)
            except OSError as exc:
                errors.append(f"{item}: {exc}")

    if errors:
        msg = "Some files could not be deleted after sync:\n" + "\n".join(errors)
        logger.error(msg)
        raise RuntimeError(msg)

    logger.info("Local output directory cleaned: %s", local_dir)


def delete_local_input(local_dir: Path) -> None:
    """
    Remove all downloaded input patches after a successful extraction.
    Frees disk space on the RunPod pod.
    """
    import shutil as _shutil
    errors: list[str] = []
    for item in local_dir.iterdir():
        if item.name == ".gitkeep":
            continue
        try:
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                _shutil.rmtree(item)
            logger.debug("Deleted input file/dir: %s", item)
        except OSError as exc:
            errors.append(f"{item}: {exc}")

    if errors:
        logger.warning("Some input files could not be deleted: %s", errors)


def extract_tar_archive(archive_path: Path, dest_dir: Path) -> int:
    """
    Extract a .tar / .tar.gz archive into *dest_dir*.

    Returns the number of regular files extracted.
    Files are flattened into *dest_dir* (no nested subdirectories) so the
    extractor can iterate them with a single glob.
    """
    import tarfile

    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0

    with tarfile.open(archive_path, "r:*") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            # Flatten: keep only the basename
            name = Path(member.name).name
            if not name:
                continue
            # Security: refuse absolute or parent-traversal paths
            if name.startswith("/") or ".." in Path(name).parts:
                logger.warning("Skipping unsafe archive entry: %s", member.name)
                continue
            target = dest_dir / name
            with tar.extractfile(member) as src, open(target, "wb") as dst:
                if src is None:
                    continue
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
            extracted += 1

    logger.info("Extracted %d files from %s → %s", extracted, archive_path, dest_dir)
    return extracted
