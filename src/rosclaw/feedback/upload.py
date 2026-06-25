"""Manual rich feedback bundle upload with mandatory redaction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests

from rosclaw import __version__ as rosclaw_version

from .config import FeedbackConfig
from .export import FeedbackExporter
from .installation import InstallationManager
from .store import directory_size_mb


class FeedbackUploader:
    """Build and upload a redacted feedback bundle."""

    def __init__(self, home: Path) -> None:
        self.home = Path(home)
        self.config = FeedbackConfig.load(home)
        self.installation = InstallationManager(home)

    def upload(
        self,
        *,
        redact: bool,
        days: int = 30,
        dry_run: bool = False,
        include_media: bool = False,
    ) -> dict[str, Any]:
        """Upload a feedback bundle; redaction is mandatory."""
        if not redact:
            return {
                "ok": False,
                "error": "redaction_required",
                "message": "Feedback upload requires --redact. Use `rosclaw feedback upload --redact`.",
            }

        anonymous_id = self.installation.get_anonymous_installation_id()
        if anonymous_id is None:
            return {"ok": False, "error": "no_installation"}

        media_dir = self.home / "feedback" / "media" / "local_only"
        media_files = list(media_dir.glob("*")) if include_media and media_dir.exists() else []

        exporter = FeedbackExporter(self.home)
        bundle_path = exporter.export(days=days, redact=True)
        size_mb = directory_size_mb(self.home / "feedback" / "bundles")
        max_mb = self.config.upload.get("max_bundle_mb", 25)
        if size_mb > max_mb:
            return {"ok": False, "error": "bundle_too_large", "size_mb": size_mb, "max_mb": max_mb}

        if dry_run:
            return {
                "ok": True,
                "dry_run": True,
                "bundle_path": str(bundle_path),
                "anonymous_installation_id": anonymous_id,
                "media_files": [str(p.name) for p in media_files],
            }

        endpoint = self.config.upload.get("endpoint")
        if not endpoint:
            return {"ok": False, "error": "no_upload_endpoint"}

        return self._post_bundle(
            endpoint,
            bundle_path,
            anonymous_id,
            media_files,
            days,
        )

    def _post_bundle(
        self,
        endpoint: str,
        bundle_path: Path,
        anonymous_id: str,
        media_files: list[Path],
        days: int,
    ) -> dict[str, Any]:
        timeout = self.config.upload.get("timeout_seconds", 10)
        max_retries = self.config.upload.get("max_retries", 1)

        metadata = {
            "schema_version": "rosclaw.feedback.upload.v1",
            "anonymous_installation_id": anonymous_id,
            "client_version": rosclaw_version,
            "redacted": True,
            "media_count": len(media_files),
            "days": days,
        }

        headers = {"User-Agent": f"rosclaw/{rosclaw_version}"}

        last_error: dict[str, Any] | None = None
        for attempt in range(max_retries + 1):
            try:
                with bundle_path.open("rb") as bundle_file:
                    files = {
                        "bundle": (bundle_path.name, bundle_file, "application/gzip"),
                    }
                    response = requests.post(
                        endpoint,
                        data=metadata,
                        files=files,
                        headers=headers,
                        timeout=timeout,
                    )
                    response.raise_for_status()
                    return response.json() if response.text else {"ok": True}
            except requests.exceptions.Timeout:
                last_error = {"ok": False, "error": "timeout"}
            except requests.exceptions.HTTPError as exc:
                last_error = {
                    "ok": False,
                    "error": "http_error",
                    "status": exc.response.status_code if exc.response else None,
                }
            except requests.exceptions.RequestException as exc:
                last_error = {"ok": False, "error": "request_error", "reason": str(exc)}
            except json.JSONDecodeError:
                last_error = {"ok": False, "error": "invalid_json_response"}

        return last_error or {"ok": False, "error": "upload_failed"}
