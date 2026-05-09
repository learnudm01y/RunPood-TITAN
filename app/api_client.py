"""
api_client.py
-------------
HTTP client for reporting job status and storage paths to the
Laravel management API (histopathology_laravel102).

Expected API contract (must be implemented in Laravel routes/api.php):
─────────────────────────────────────────────────────────────────────

POST  /api/v1/feature-extraction/report
    Authorization: Bearer <api_key>
    Content-Type:  application/json

    {
      "sample_id":              <int>     – samples.id in the DB,
      "slide_id":               <str>     – human-readable slide identifier,
      "status":                 <str>     – "processing" | "completed" | "failed",
      "runpod_output_path":     <str>     – absolute path on the RunPod server,
      "features_gdrive_path":   <str>     – Google Drive path where HDF5 was synced,
      "features_gdrive_folder_id": <str>  – Google Drive folder ID (optional),
      "patch_count":            <int>     – number of patches processed,
      "failed_patch_count":     <int>     – number of patches that failed,
      "model_name":             <str>     – e.g. "TITAN",
      "model_version":          <str>     – e.g. "TITAN-v1",
      "error_message":          <str>     – empty string if no error
    }

    Expected response:
    {
      "success": true,
      "message": "Status updated."
    }

All errors are caught and logged; they NEVER crash the extraction pipeline.
A local manifest is always written as a fallback (see io_utils.write_job_manifest).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

_ENDPOINT = "/api/v1/feature-extraction/report"


class APIClient:
    """
    Thin wrapper around httpx for reporting feature-extraction job status.
    """

    def __init__(self, base_url: str, api_key: str, timeout: int = 30) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _post(self, endpoint: str, payload: dict[str, Any]) -> Optional[dict[str, Any]]:
        """
        POST JSON payload to *endpoint*.

        Returns the parsed JSON response or None on failure.
        Never raises – all exceptions are swallowed so extraction continues.
        """
        if not self._base_url:
            logger.warning(
                "API_BASE_URL is not configured – skipping API report. "
                "Set the API_BASE_URL environment variable."
            )
            return None

        url = f"{self._base_url}{endpoint}"
        try:
            response = httpx.post(
                url,
                json=payload,
                headers=self._headers,
                timeout=self._timeout,
                verify=False,
            )
            if response.is_success:
                logger.info("API report accepted (HTTP %d): %s", response.status_code, url)
                return response.json()
            else:
                logger.error(
                    "API returned HTTP %d for %s – body: %s",
                    response.status_code,
                    url,
                    response.text[:500],
                )
        except httpx.ConnectError as exc:
            logger.error("Cannot reach API server at %s – %s", url, exc)
        except httpx.TimeoutException:
            logger.error("API request timed out after %ds: %s", self._timeout, url)
        except Exception as exc:  # noqa: BLE001
            logger.error("Unexpected API error for %s – %s", url, exc)

        return None

    # ── Public reporting methods ───────────────────────────────────────────────

    def report_started(
        self,
        sample_id: int,
        slide_id: str,
    ) -> None:
        """Notify the management server that feature extraction has started."""
        self._post(
            _ENDPOINT,
            {
                "sample_id": sample_id,
                "slide_id": slide_id,
                "status": "processing",
                "runpod_output_path": "",
                "features_gdrive_path": "",
                "features_gdrive_folder_id": "",
                "patch_count": 0,
                "failed_patch_count": 0,
                "model_name": "TITAN",
                "model_version": "",
                "error_message": "",
            },
        )

    def report_completed(
        self,
        sample_id: int,
        slide_id: str,
        runpod_output_path: str,
        features_gdrive_path: str,
        features_gdrive_folder_id: str,
        patch_count: int,
        failed_patch_count: int,
        model_name: str,
        model_version: str,
    ) -> None:
        """Notify the management server that feature extraction finished successfully."""
        self._post(
            _ENDPOINT,
            {
                "sample_id": sample_id,
                "slide_id": slide_id,
                "status": "completed",
                "runpod_output_path": runpod_output_path,
                "features_gdrive_path": features_gdrive_path,
                "features_gdrive_folder_id": features_gdrive_folder_id,
                "patch_count": patch_count,
                "failed_patch_count": failed_patch_count,
                "model_name": model_name,
                "model_version": model_version,
                "error_message": "",
            },
        )

    def report_failed(
        self,
        sample_id: int,
        slide_id: str,
        error_message: str,
        patch_count: int = 0,
        failed_patch_count: int = 0,
    ) -> None:
        """Notify the management server that feature extraction failed."""
        self._post(
            _ENDPOINT,
            {
                "sample_id": sample_id,
                "slide_id": slide_id,
                "status": "failed",
                "runpod_output_path": "",
                "features_gdrive_path": "",
                "features_gdrive_folder_id": "",
                "patch_count": patch_count,
                "failed_patch_count": failed_patch_count,
                "model_name": "TITAN",
                "model_version": "",
                "error_message": error_message[:1000],
            },
        )
