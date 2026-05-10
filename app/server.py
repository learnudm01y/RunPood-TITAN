"""
server.py
---------
Minimal FastAPI HTTP server that receives feature-extraction jobs
from the Laravel management system (histopathology_laravel102).

Endpoints
---------
GET  /health                – liveness probe (no auth)
POST /jobs/start            – accept a new job (Bearer-token auth)
GET  /jobs/{job_id}         – read a job's local status (auth)

Architecture
------------
1. Laravel POSTs the full job spec to /jobs/start.
2. The server validates the Bearer token against an env-configured shared
   secret (the same value stored in `servers_names.api_key` on Laravel).
3. The job is queued in an in-memory `asyncio.Queue` and a background worker
   runs `app.main.run()` for each one.
4. The worker reports progress back to Laravel via `app.api_client.APIClient`
   (URL is provided in the dispatched payload's `callback.url`).

This design keeps the GPU pod stateless — Laravel is the source of truth.
The server is intentionally single-process / single-worker; one GPU pod
typically handles one extraction at a time.

Usage
-----
    # Inside the RunPod pod:
    export RUNPOD_API_KEY="your-shared-secret"
    uvicorn app.server:app --host 0.0.0.0 --port 8000

The same shared secret is stored on Laravel in the `servers_names` row whose
`type=external`, `api_url=https://histopathology.cloud`, `api_key=<secret>`.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

from app.api_client import APIClient
from app.config import (
    API_TIMEOUT,
    BATCH_SIZE,
    NUM_WORKERS,
    PATCH_SIZE_PX,
    RCLONE_REMOTE,
    JobConfig,
)

logger = logging.getLogger("titan_pipeline.server")

# ─── Shared-secret authentication ─────────────────────────────────────────────
SHARED_SECRET = os.environ.get("RUNPOD_API_KEY", "")


def _verify_token(authorization: Optional[str] = Header(default=None)) -> None:
    """FastAPI dependency: rejects requests without a valid Bearer token."""
    if not SHARED_SECRET:
        # Hard-fail when the server was started without a configured secret.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="RUNPOD_API_KEY not configured on the server.",
        )
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token.")
    token = authorization[7:].strip()
    # Constant-time comparison
    import hmac
    if not hmac.compare_digest(token, SHARED_SECRET):
        raise HTTPException(status_code=403, detail="Invalid API key.")


# ─── Request / response models ────────────────────────────────────────────────

class AiModelSpec(BaseModel):
    id: int
    name: str
    slug: str
    huggingface: Optional[str] = None
    version: Optional[str] = None
    embedding_dim: Optional[str] = None
    input_resolution: Optional[str] = None


class CallbackSpec(BaseModel):
    url: str
    token: str
    method: str = "POST"


class JobStartRequest(BaseModel):
    sample_id: int
    slide_id: str
    patch_size_px: int = Field(default=PATCH_SIZE_PX)
    magnification: str = "20x"
    magnification_folder: Optional[str] = None
    gdrive_input_path: str
    gdrive_input_archive: Optional[str] = "patches.tar.gz"
    gdrive_output_path: str
    ai_model: AiModelSpec
    callback: CallbackSpec
    dispatched_at: Optional[str] = None


class JobStartResponse(BaseModel):
    success: bool
    job_id: str
    sample_id: int
    accepted_at: str


# ─── In-memory job tracker ────────────────────────────────────────────────────

class JobTracker:
    """Tracks the local status of jobs accepted by this server."""
    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def add(self, job_id: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self._jobs[job_id] = {
                "job_id":       job_id,
                "sample_id":    payload.get("sample_id"),
                "slide_id":     payload.get("slide_id"),
                "status":       "queued",
                "accepted_at":  datetime.now(timezone.utc).isoformat(),
                "started_at":   None,
                "finished_at":  None,
                "error":        None,
            }

    def update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(fields)

    def get(self, job_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            return self._jobs.get(job_id)


_tracker = JobTracker()


# ─── Background worker ────────────────────────────────────────────────────────

_job_queue: "asyncio.Queue[tuple[str, JobStartRequest]]" = asyncio.Queue()


async def _worker_loop() -> None:
    """Consumes jobs from the queue and runs them sequentially."""
    while True:
        job_id, req = await _job_queue.get()
        try:
            _tracker.update(job_id, status="running",
                            started_at=datetime.now(timezone.utc).isoformat())
            await asyncio.get_running_loop().run_in_executor(None, _run_job_sync, job_id, req)
            _tracker.update(job_id, status="completed",
                            finished_at=datetime.now(timezone.utc).isoformat())
        except Exception as exc:  # noqa: BLE001
            logger.exception("Job %s failed: %s", job_id, exc)
            _tracker.update(job_id, status="failed",
                            finished_at=datetime.now(timezone.utc).isoformat(),
                            error=str(exc))
        finally:
            _job_queue.task_done()


def _run_job_sync(job_id: str, req: JobStartRequest) -> None:
    """
    Run a single job synchronously in a worker thread.
    Imports `run` lazily to avoid loading torch at server-import time.
    """
    from argparse import Namespace
    from app.main import run

    cfg = JobConfig(
        sample_id=req.sample_id,
        slide_id=req.slide_id,
        patch_size_px=req.patch_size_px,
        magnification=req.magnification,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        gdrive_input_path=req.gdrive_input_path,
        gdrive_output_path=req.gdrive_output_path,
        api_base_url=(os.environ.get("API_BASE_URL") or req.callback.url.replace("/api/v1/feature-extraction/report", "")),
        api_key=req.callback.token,
    )

    # Pass ai_model name + slug to the JobConfig via attributes so main.run() can
    # use it to set features_runpod_path and forward model_name in the report.
    setattr(cfg, "ai_model_name", req.ai_model.name)
    setattr(cfg, "ai_model_slug", req.ai_model.slug)
    setattr(cfg, "ai_model_id",   req.ai_model.id)
    setattr(cfg, "ai_model_hf",   req.ai_model.huggingface)
    setattr(cfg, "gdrive_input_archive", req.gdrive_input_archive or "patches.tar.gz")

    args = Namespace(
        skip_gdrive_sync=False,
        skip_cleanup=False,
        dry_run=False,
        rclone_remote=RCLONE_REMOTE,
    )

    logger.info("Worker starting job %s (sample=%d, slide=%s)",
                job_id, req.sample_id, req.slide_id)
    rc = run(cfg, args)
    if rc != 0:
        raise RuntimeError(f"Pipeline returned non-zero exit code {rc}")


# ─── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="RunPood-histo-TITAN",
    description="Feature extraction worker for histopathology patches.",
    version="1.0.0",
)


async def _self_register() -> None:
    """
    Detect this pod's public proxy URL and update Laravel's servers_names row.
    Runs once at startup. Fails silently so the server still boots if Laravel
    is unreachable.
    """
    pod_hostname = os.environ.get("RUNPOD_POD_HOSTNAME", "")
    laravel_url = os.environ.get("LARAVEL_BASE_URL", "")
    server_id = os.environ.get("LARAVEL_SERVER_ID", "2")

    if not pod_hostname:
        logger.info("RUNPOD_POD_HOSTNAME not set — skipping self-registration.")
        return
    if not laravel_url or not SHARED_SECRET:
        logger.info("LARAVEL_BASE_URL or RUNPOD_API_KEY not set — skipping self-registration.")
        return

    public_url = f"https://{pod_hostname}-8000.proxy.runpod.net"
    register_endpoint = f"{laravel_url.rstrip('/')}/api/v1/servers/{server_id}/update-url"

    import httpx
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                register_endpoint,
                json={"api_url": public_url},
                headers={"Authorization": f"Bearer {SHARED_SECRET}"},
            )
        if resp.status_code == 200:
            logger.info("Self-registered api_url=%s on Laravel (server_id=%s)", public_url, server_id)
        else:
            logger.warning("Self-registration returned HTTP %d: %s", resp.status_code, resp.text[:200])
    except Exception as exc:  # noqa: BLE001
        logger.warning("Self-registration failed (non-fatal): %s", exc)


@app.on_event("startup")
async def _startup() -> None:
    asyncio.create_task(_worker_loop())
    asyncio.create_task(_self_register())
    logger.info("Worker loop started. Queue ready.")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "success": True,
        "service": "runpood-histo-titan",
        "time": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/jobs/start", response_model=JobStartResponse,
          dependencies=[Depends(_verify_token)])
async def start_job(req: JobStartRequest) -> JobStartResponse:
    """
    Accepts a new feature-extraction job.  Returns immediately after queueing.
    The actual extraction runs in a background worker thread.
    """
    job_id = uuid.uuid4().hex
    _tracker.add(job_id, req.model_dump())

    # Pre-emptively notify Laravel that the job is now processing so the
    # admin UI flips to the right state without waiting for the GPU work.
    api = APIClient(
        base_url=(os.environ.get("API_BASE_URL") or req.callback.url.replace("/api/v1/feature-extraction/report", "")),
        api_key=req.callback.token,
        timeout=API_TIMEOUT,
    )
    api.report_started(req.sample_id, req.slide_id)

    await _job_queue.put((job_id, req))
    logger.info("Queued job %s for sample %d (slide=%s, model=%s)",
                job_id, req.sample_id, req.slide_id, req.ai_model.name)

    return JobStartResponse(
        success=True,
        job_id=job_id,
        sample_id=req.sample_id,
        accepted_at=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/jobs/{job_id}", dependencies=[Depends(_verify_token)])
def get_job(job_id: str) -> dict[str, Any]:
    job = _tracker.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return {"success": True, "job": job}


# ─── Credential update endpoint ───────────────────────────────────────────────

class CredentialsUpdateRequest(BaseModel):
    rclone_refresh_token: Optional[str] = None
    hf_token: Optional[str] = None


@app.post("/admin/update-credentials", dependencies=[Depends(_verify_token)])
def update_credentials(req: CredentialsUpdateRequest) -> dict[str, Any]:
    """
    Laravel calls this to push updated rclone / HF credentials.
    Updates rclone.conf and setup_env.sh in-place without restarting.
    """
    import re
    updated: list[str] = []
    workspace = os.environ.get("WORKSPACE_DIR", "/workspace")
    setup_env_path = f"{workspace}/setup_env.sh"

    if req.rclone_refresh_token:
        rclone_conf_path = os.path.expanduser("~/.config/rclone/rclone.conf")
        os.makedirs(os.path.dirname(rclone_conf_path), exist_ok=True)
        token_json = (
            '{"access_token":"","token_type":"Bearer",'
            f'"refresh_token":"{req.rclone_refresh_token}",'
            '"expiry":"2020-01-01T00:00:00Z"}'
        )
        with open(rclone_conf_path, "w") as fh:
            fh.write(f"[gdrive]\ntype = drive\nscope = drive\ntoken = {token_json}\nteam_drive =\n")
        if os.path.exists(setup_env_path):
            content = open(setup_env_path).read()
            content = re.sub(
                r'token = \{"access_token":.*?"expiry":"[^"]*"\}',
                f'token = {token_json}',
                content,
            )
            open(setup_env_path, "w").write(content)
        updated.append("rclone_refresh_token")
        logger.info("Credentials updated: rclone_refresh_token")

    if req.hf_token:
        os.environ["HF_TOKEN"] = req.hf_token
        if os.path.exists(setup_env_path):
            content = open(setup_env_path).read()
            content = re.sub(
                r'export HF_TOKEN="[^"]*"',
                f'export HF_TOKEN="{req.hf_token}"',
                content,
            )
            open(setup_env_path, "w").write(content)
        updated.append("hf_token")
        logger.info("Credentials updated: hf_token")

    return {"success": True, "updated": updated}


# ─── CLI entry-point (for local dev) ──────────────────────────────────────────

def _main() -> None:
    parser = argparse.ArgumentParser(description="RunPod feature-extraction server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    import uvicorn
    uvicorn.run("app.server:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    _main()
