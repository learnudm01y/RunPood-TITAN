"""
Contract validation: Laravel ↔ RunPod end-to-end.

Simulates real payloads in BOTH directions and verifies schema parity:
  1. Laravel -> RunPod  (FeatureExtractionJob -> /jobs/start)
  2. RunPod  -> Laravel (APIClient -> /api/v1/feature-extraction/report)

No network calls, no DB. Pure schema + path verification.
"""
from __future__ import annotations
import sys
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Import RunPod Pydantic models
from app.server import JobStartRequest, JobStartResponse  # noqa: E402

PASSED, FAILED = 0, 0

def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [OK ] {name}")
    else:
        FAILED += 1
        print(f"  [ERR] {name}  -- {detail}")


# ─────────────────────────────────────────────────────────────────────────────
# 1) Laravel -> RunPod: build payload exactly as FeatureExtractionJob does
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print(" [1/4] Laravel -> RunPod  (POST /jobs/start)")
print("=" * 70)

# Mirror buildPayload() in FeatureExtractionJob.php
fake_sample = {
    "id": 42,
    "file_name": "TCGA-A1-A0SD-01Z-00-DX1.svs",
    "tiles_gdrive_path": "samples/sliced_slides/20x/tcga/breast/TCGA-A1-A0SD/sample_42_256px",
    "data_source": "TCGA",
    "category_label_en": "Breast",
    "case_id": "TCGA-A1-A0SD",
}
fake_patch_size_px = 256
fake_mag = "20x"
fake_model = {
    "id": 1, "name": "TITAN", "huggingface": "MahmoodLab/TITAN",
    "version": "v1.0", "embedding_dim": "1024", "input_resolution": "256",
}

def php_str_slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")

model_slug = php_str_slug(fake_model["name"])
source_slug = php_str_slug(fake_sample["data_source"])
category_slug = php_str_slug(fake_sample["category_label_en"])
gdrive_root = "samples"

output_path = "/".join([
    gdrive_root, "features", fake_model["name"], fake_mag,
    source_slug, category_slug, fake_sample["case_id"],
    f"sample_{fake_sample['id']}_{fake_patch_size_px}px",
])

laravel_payload = {
    "sample_id": fake_sample["id"],
    "slide_id": fake_sample["file_name"],
    "patch_size_px": fake_patch_size_px,
    "magnification": fake_mag,
    "magnification_folder": fake_mag,
    "gdrive_input_path": fake_sample["tiles_gdrive_path"],
    "gdrive_input_archive": "patches.tar.gz",
    "gdrive_output_path": output_path,
    "ai_model": {
        "id": fake_model["id"],
        "name": fake_model["name"],
        "slug": model_slug,
        "huggingface": fake_model["huggingface"],
        "version": fake_model["version"],
        "embedding_dim": fake_model["embedding_dim"],
        "input_resolution": fake_model["input_resolution"],
    },
    "callback": {
        "url": "https://management.histopathology.cloud/api/v1/feature-extraction/report",
        "token": "53025e0a-3da8-446a-a1b3-304139ac8d87-bkru42r0yZN2vsAg",
        "method": "POST",
    },
    "dispatched_at": "2026-05-09T12:00:00+00:00",
}

# Validate against RunPod's Pydantic schema
try:
    parsed = JobStartRequest.model_validate(laravel_payload)
    check("Pydantic JobStartRequest accepts Laravel payload", True)
except Exception as e:
    check("Pydantic JobStartRequest accepts Laravel payload", False, str(e))

# Field-by-field
check("sample_id is int",        isinstance(parsed.sample_id, int))
check("slide_id is str",         isinstance(parsed.slide_id, str))
check("patch_size_px=256",       parsed.patch_size_px == 256)
check("magnification='20x'",     parsed.magnification == "20x")
check("ai_model.slug present",   parsed.ai_model.slug == "titan")
check("callback.url ends with /report",
      parsed.callback.url.endswith("/api/v1/feature-extraction/report"))
check("callback.method=POST",    parsed.callback.method == "POST")
check("gdrive_output_path under samples/features/",
      parsed.gdrive_output_path.startswith("samples/features/"))


# ─────────────────────────────────────────────────────────────────────────────
# 2) Path correctness — Laravel hierarchy
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print(" [2/4] GDrive output path layout (matches Laravel sliced_slides tree)")
print("=" * 70)
parts = output_path.split("/")
check("8 path segments",         len(parts) == 8, f"got {len(parts)}: {parts}")
check("seg[0] = root 'samples'", parts[0] == "samples")
check("seg[1] = 'features'",     parts[1] == "features")
check("seg[2] = AI model name",  parts[2] == "TITAN")
check("seg[3] = magnification",  parts[3] == "20x")
check("seg[4] = source slug",    parts[4] == "tcga")
check("seg[5] = category slug",  parts[5] == "breast")
check("seg[6] = case_id",        parts[6] == "TCGA-A1-A0SD")
check("seg[7] = sample folder",  parts[7] == "sample_42_256px")
print(f"  -> {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3) RunPod base_url derivation from callback.url (server.py logic)
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print(" [3/4] RunPod base_url derivation (round-trip)")
print("=" * 70)
api_endpoint = "/api/v1/feature-extraction/report"
derived_base = parsed.callback.url.replace(api_endpoint, "")
reconstructed = derived_base.rstrip("/") + api_endpoint
check("base_url has no /api suffix", not derived_base.endswith("/api/v1/feature-extraction/report"))
check("Round-trip URL == original",  reconstructed == parsed.callback.url,
      f"{reconstructed} != {parsed.callback.url}")
print(f"  -> base_url = {derived_base}")


# ─────────────────────────────────────────────────────────────────────────────
# 4) RunPod -> Laravel: payloads exactly as APIClient sends them
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print(" [4/4] RunPod -> Laravel  (POST /api/v1/feature-extraction/report)")
print("=" * 70)

# Laravel validation rules (mirrored from FeatureExtractionApiController::report)
LARAVEL_RULES = {
    "sample_id":                 {"required": True,  "type": int,  "max_len": None},
    "slide_id":                  {"required": False, "type": str,  "max_len": 255},
    "status":                    {"required": True,  "type": str,  "in": ["processing", "completed", "failed"]},
    "runpod_output_path":        {"required": False, "type": str,  "max_len": 500},
    "features_gdrive_path":      {"required": False, "type": str,  "max_len": 500},
    "features_gdrive_folder_id": {"required": False, "type": str,  "max_len": 100},
    "patch_count":               {"required": False, "type": int,  "min": 0},
    "failed_patch_count":        {"required": False, "type": int,  "min": 0},
    "model_name":                {"required": False, "type": str,  "max_len": 100},
    "model_version":             {"required": False, "type": str,  "max_len": 100},
    "error_message":             {"required": False, "type": str,  "max_len": 2000},
}

def validate(payload: dict, label: str) -> None:
    print(f"\n  -- {label}")
    for field, rule in LARAVEL_RULES.items():
        present = field in payload
        if rule["required"]:
            check(f"required field '{field}' present", present)
            if not present:
                continue
        if not present:
            continue
        val = payload[field]
        # Allow empty string for non-required, since Laravel 'nullable' allows null/empty
        if val is None or val == "":
            continue
        # Type
        check(f"'{field}' is {rule['type'].__name__}", isinstance(val, rule["type"]))
        # max_len
        if rule.get("max_len") and isinstance(val, str):
            check(f"'{field}' length <= {rule['max_len']}",
                  len(val) <= rule["max_len"], f"len={len(val)}")
        # in
        if "in" in rule:
            check(f"'{field}' in {rule['in']}", val in rule["in"])
        # min
        if "min" in rule and isinstance(val, int):
            check(f"'{field}' >= {rule['min']}", val >= rule["min"])

# Build payloads exactly as APIClient methods do
report_started_payload = {
    "sample_id": 42, "slide_id": "TCGA-A1-A0SD-01Z-00-DX1.svs",
    "status": "processing",
    "runpod_output_path": "", "features_gdrive_path": "", "features_gdrive_folder_id": "",
    "patch_count": 0, "failed_patch_count": 0,
    "model_name": "TITAN", "model_version": "", "error_message": "",
}
report_completed_payload = {
    "sample_id": 42, "slide_id": "TCGA-A1-A0SD-01Z-00-DX1.svs",
    "status": "completed",
    "runpod_output_path": "/workspace/output/features/TCGA-A1-A0SD-01Z-00-DX1.svs",
    "features_gdrive_path": output_path,
    "features_gdrive_folder_id": "",
    "patch_count": 1024, "failed_patch_count": 0,
    "model_name": "TITAN", "model_version": "TITAN-v1", "error_message": "",
}
report_failed_payload = {
    "sample_id": 42, "slide_id": "TCGA-A1-A0SD-01Z-00-DX1.svs",
    "status": "failed",
    "runpod_output_path": "", "features_gdrive_path": "", "features_gdrive_folder_id": "",
    "patch_count": 0, "failed_patch_count": 0,
    "model_name": "TITAN", "model_version": "", "error_message": "Test failure mode.",
}

validate(report_started_payload,   "report_started -> Laravel rules")
validate(report_completed_payload, "report_completed -> Laravel rules")
validate(report_failed_payload,    "report_failed -> Laravel rules")


# ─────────────────────────────────────────────────────────────────────────────
# Final
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print(f" RESULT: {PASSED} passed, {FAILED} failed")
print("=" * 70)
sys.exit(0 if FAILED == 0 else 1)
