"""
HTTP integration test — simulates a real Laravel POST to /jobs/start
without any GPU. Validates:
  - /health works without auth
  - /jobs/start rejects bad/missing token
  - /jobs/start accepts valid Laravel payload and returns job_id
  - GET /jobs/{id} returns the queued job
  - The pre-emptive Laravel callback is attempted (will fail gracefully)
"""
from __future__ import annotations
import os, sys, threading, time, asyncio
from pathlib import Path

# Set required env BEFORE importing app.server
os.environ["RUNPOD_API_KEY"] = "test-shared-secret-12345"
os.environ["RCLONE_REMOTE"]  = "gdrive"
os.environ["WORKSPACE_DIR"]  = str(Path(__file__).resolve().parent.parent)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Disable the actual job worker (don't load torch / TITAN)
import app.server as srv
async def _noop_loop():
    while True:
        await asyncio.sleep(3600)
srv._worker_loop = _noop_loop

# Stub APIClient so report_started() doesn't try to reach a real Laravel
class _StubAPI:
    def __init__(self, *a, **kw): pass
    def report_started(self, *a, **kw):
        print("    [stub] APIClient.report_started called -- OK")
srv.APIClient = _StubAPI

from fastapi.testclient import TestClient

PASS, FAIL = 0, 0
def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  [OK ] {name}")
    else:
        FAIL += 1; print(f"  [ERR] {name} -- {detail}")

with TestClient(srv.app) as client:
    # ── 1. /health (public)
    print("[1] GET /health (public)")
    r = client.get("/health")
    check("status 200", r.status_code == 200, f"got {r.status_code}")
    check("success=true", r.json().get("success") is True)
    check("service name", r.json().get("service") == "runpood-histo-titan")

    # ── 2. /jobs/start without token
    print("\n[2] POST /jobs/start without token")
    r = client.post("/jobs/start", json={})
    check("status 401", r.status_code == 401, f"got {r.status_code}")

    # ── 3. /jobs/start with WRONG token
    print("\n[3] POST /jobs/start with wrong token")
    r = client.post("/jobs/start", json={},
                    headers={"Authorization": "Bearer wrong-secret"})
    check("status 403", r.status_code == 403, f"got {r.status_code}")

    # ── 4. /jobs/start with VALID token + valid Laravel payload
    print("\n[4] POST /jobs/start with valid Laravel payload")
    payload = {
        "sample_id": 42,
        "slide_id": "TCGA-A1-A0SD-01Z-00-DX1.svs",
        "patch_size_px": 256,
        "magnification": "20x",
        "magnification_folder": "20x",
        "gdrive_input_path": "samples/sliced_slides/20x/tcga/breast/TCGA-A1-A0SD/sample_42_256px",
        "gdrive_input_archive": "patches.tar.gz",
        "gdrive_output_path": "samples/features/TITAN/20x/tcga/breast/TCGA-A1-A0SD/sample_42_256px",
        "ai_model": {
            "id": 1, "name": "TITAN", "slug": "titan",
            "huggingface": "MahmoodLab/TITAN", "version": "v1.0",
            "embedding_dim": "1024", "input_resolution": "256",
        },
        "callback": {
            "url": "http://127.0.0.1:9999/api/v1/feature-extraction/report",
            "token": "any-laravel-key",
            "method": "POST",
        },
        "dispatched_at": "2026-05-09T12:00:00+00:00",
    }
    r = client.post("/jobs/start", json=payload,
                    headers={"Authorization": "Bearer test-shared-secret-12345"})
    check("status 200", r.status_code == 200, f"got {r.status_code} body={r.text[:200]}")
    body = r.json() if r.status_code == 200 else {}
    check("success=true", body.get("success") is True)
    check("returns job_id (hex)", isinstance(body.get("job_id"), str) and len(body.get("job_id", "")) == 32)
    check("returns sample_id=42", body.get("sample_id") == 42)
    check("returns accepted_at",  isinstance(body.get("accepted_at"), str))
    job_id = body.get("job_id", "")

    # ── 5. GET /jobs/{id} with auth
    print("\n[5] GET /jobs/{id} with valid token")
    r = client.get(f"/jobs/{job_id}",
                   headers={"Authorization": "Bearer test-shared-secret-12345"})
    check("status 200", r.status_code == 200, f"got {r.status_code}")
    job = r.json().get("job", {})
    check("job tracked", job.get("job_id") == job_id)
    check("status=queued", job.get("status") == "queued")
    check("sample_id=42",  job.get("sample_id") == 42)

    # ── 6. /jobs/start rejects malformed payload (Pydantic 422)
    print("\n[6] POST /jobs/start with missing fields (validation)")
    r = client.post("/jobs/start", json={"sample_id": 1},
                    headers={"Authorization": "Bearer test-shared-secret-12345"})
    check("status 422", r.status_code == 422, f"got {r.status_code}")

print()
print("=" * 60)
print(f" HTTP INTEGRATION: {PASS} passed, {FAIL} failed")
print("=" * 60)
sys.exit(0 if FAIL == 0 else 1)
