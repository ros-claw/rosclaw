"""Manual rich feedback bundle upload with mandatory redaction."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

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

        payload = {
            "schema_version": "rosclaw.feedback.upload.v1",
            "anonymous_installation_id": anonymous_id,
            "client_version": rosclaw_version,
            "redacted": True,
            "bundle_path": str(bundle_path),
            "media_count": len(media_files),
            "days": days,
        }
        response = self._post(endpoint, payload)
        return response or {"ok": False, "error": "upload_failed"}

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        timeout = self.config.upload.get("timeout_seconds", 10)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=body,
            headers={
                "Content-Type": "application/json",
                "User-Agent": f"rosclaw/{rosclaw_version}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
                if raw:
                    return json.loads(raw)
                return {"ok": True}
        except urllib.error.HTTPError as exc:
            return {"ok": False, "error": "http_error", "status": exc.code}
        except urllib.error.URLError as exc:
            return {"ok": False, "error": "url_error", "reason": str(exc.reason)}
        except TimeoutError:
            return {"ok": False, "error": "timeout"}
        except Exception:
            return {"ok": False, "error": "unknown"}
