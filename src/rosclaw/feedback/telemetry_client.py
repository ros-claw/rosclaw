"""Telemetry client: local spooling, bucketing, and fire-and-forget upload."""

from __future__ import annotations

import contextlib
import json
import platform
import sys
import threading
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rosclaw import __version__ as rosclaw_version

from .config import TelemetryConfig
from .installation import InstallationManager
from .store import append_event, event_file_for_date

ALLOWED_EVENT_TYPES = frozenset([
    "install_started",
    "install_completed",
    "firstboot_started",
    "firstboot_completed",
    "doctor_started",
    "doctor_completed",
    "command_completed",
    "module_enabled",
    "provider_installed",
    "provider_served",
    "hub_asset_installed",
    "dashboard_opened",
    "practice_started",
    "heartbeat",
    "telemetry_ping",
])

FORBIDDEN_FIELDS = frozenset([
    "hostname",
    "username",
    "ip",
    "local_path",
    "cwd",
    "full_command",
    "full_args",
    "prompt",
    "system_prompt",
    "tool_arguments",
    "provider_response",
    "stacktrace",
    "log",
    "video",
    "image",
    "audio",
    "mcap",
    "trace",
    "api_key",
    "secret",
    "robot_serial",
])

ERROR_CLASS_BUCKETS = frozenset([
    "ImportError",
    "ConfigError",
    "DockerUnavailable",
    "ROSNotFound",
    "ProviderTimeout",
    "PermissionDenied",
    "NetworkError",
    "ValidationError",
    "RuntimeError",
])


class TelemetryClient:
    """Record, bucket, and optionally upload product telemetry events."""

    def __init__(self, home: Path) -> None:
        self.home = Path(home)
        self.config = TelemetryConfig.load(home)
        self.installation = InstallationManager(home)

    def record_event(
        self,
        event_type: str,
        command_name: str | None = None,
        command_status: str | None = None,
        module_name: str | None = None,
        payload: dict[str, Any] | None = None,
        error: BaseException | None = None,
    ) -> dict[str, Any] | None:
        """Record a telemetry event locally and upload if allowed."""
        if not self._is_enabled_for_event_type(event_type):
            return None

        event = self._build_event(
            event_type=event_type,
            command_name=command_name,
            command_status=command_status,
            module_name=module_name,
            payload=payload,
            error=error,
        )
        if event is None:
            return None

        path = event_file_for_date(self.home, "telemetry")
        append_event(path, event)

        if self._should_upload():
            self._upload_async(event)
        return event

    def record_command(
        self,
        args: Any,
        status: str,
        duration_ms: int,
        error: BaseException | None = None,
    ) -> dict[str, Any] | None:
        """Record a command_completed event from CLI dispatch."""
        command_name = self._command_name(args)
        module_name = getattr(args, "command", None)
        return self.record_event(
            event_type="command_completed",
            command_name=command_name,
            command_status=status,
            module_name=module_name,
            payload={"duration_ms": duration_ms},
            error=error,
        )

    def heartbeat_if_due(self) -> dict[str, Any] | None:
        """Send a heartbeat event if the configured interval has elapsed."""
        if not self._should_upload():
            return None

        last_path = self.home / "telemetry" / "heartbeat" / "last_heartbeat.json"
        interval_hours = self.config.product_telemetry.get("heartbeat_interval_hours", 24)
        now = datetime.now(UTC)
        if last_path.exists():
            try:
                last = json.loads(last_path.read_text(encoding="utf-8"))
                last_ts = last.get("timestamp", "")
                if last_ts:
                    last_dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
                    if (now - last_dt) < timedelta(hours=interval_hours):
                        return None
            except (json.JSONDecodeError, ValueError, OSError):
                pass

        event = self._build_event("heartbeat")
        if event is None:
            return None

        path = event_file_for_date(self.home, "telemetry")
        append_event(path, event)

        response = self._upload_sync(
            self.config.upload.get("heartbeat_endpoint") or self.config.upload.get("endpoint"),
            event,
        )
        try:
            last_path.parent.mkdir(parents=True, exist_ok=True)
            last_path.write_text(
                json.dumps({"timestamp": event["created_at"]}, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            pass
        return response

    def ping(self) -> dict[str, Any]:
        """Send a telemetry_ping event synchronously and return server response."""
        event = self._build_event("telemetry_ping")
        if event is None:
            return {"ok": False, "error": "no_installation"}
        response = self._upload_sync(self.config.upload.get("endpoint"), event)
        return response or {"ok": False, "error": "upload_failed"}

    def _build_event(
        self,
        event_type: str,
        command_name: str | None = None,
        command_status: str | None = None,
        module_name: str | None = None,
        payload: dict[str, Any] | None = None,
        error: BaseException | None = None,
    ) -> dict[str, Any] | None:
        anonymous_id = self.installation.get_anonymous_installation_id()
        if anonymous_id is None and event_type != "install_started":
            # Events before installation exists are best-effort only.
            pass

        install = self.installation.get_installation()
        install_channel = install.install_channel if install else "unknown"

        event: dict[str, Any] = {
            "schema_version": "rosclaw.telemetry.event.v1",
            "event_type": event_type,
            "anonymous_installation_id": anonymous_id,
            "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "rosclaw_version": rosclaw_version,
            "cli_version": rosclaw_version,
            "os_family": platform.system().lower() or "unknown",
            "arch": platform.machine().lower() or "unknown",
            "python": f"{sys.version_info.major}.{sys.version_info.minor}",
            "install_channel": install_channel,
            "deployment_mode": "local",
        }

        if anonymous_id is None:
            event["anonymous_installation_id"] = None

        if command_name is not None:
            event["command_name"] = command_name
        if command_status is not None:
            event["command_status"] = command_status
        if module_name is not None:
            event["module_name"] = module_name

        payload = dict(payload or {})
        if error is not None:
            event["error_class_bucket"] = error_class_bucket(error)
            payload["error_class"] = error_class_bucket(error)

        if "duration_ms" in payload:
            event["duration_bucket"] = duration_bucket(payload.pop("duration_ms"))

        if payload:
            event["payload"] = payload

        event = self._scrub(event)
        return event

    def _should_upload(self) -> bool:
        mode = self.config.mode
        return bool(mode.get("enabled", True) and mode.get("product_telemetry", True))

    def _is_enabled_for_event_type(self, event_type: str) -> bool:
        if event_type not in ALLOWED_EVENT_TYPES:
            return False
        if event_type in ("telemetry_ping", "heartbeat"):
            return True
        return self._should_upload()

    def _upload_async(self, event: dict[str, Any]) -> None:
        """Fire-and-forget upload in a background thread."""
        def _upload() -> None:
            with contextlib.suppress(Exception):
                self._upload_sync(self.config.upload.get("endpoint"), event)

        thread = threading.Thread(target=_upload, daemon=True)
        thread.start()

    def _upload_sync(self, endpoint: str | None, event: dict[str, Any]) -> dict[str, Any] | None:
        if not endpoint:
            return None
        timeout = self.config.upload.get("timeout_seconds", 3)
        body = json.dumps(event, ensure_ascii=False).encode("utf-8")
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

    def _scrub(self, event: dict[str, Any]) -> dict[str, Any]:
        """Remove forbidden fields from the event and payload recursively."""
        def scrub_value(value: Any) -> Any:
            if isinstance(value, dict):
                return {k: scrub_value(v) for k, v in value.items() if k not in FORBIDDEN_FIELDS}
            if isinstance(value, list):
                return [scrub_value(v) for v in value]
            return value

        return scrub_value(event)

    @staticmethod
    def _command_name(args: Any) -> str | None:
        command = getattr(args, "command", None)
        if not command:
            return None
        return str(command)


from datetime import timedelta  # noqa: E402


def duration_bucket(duration_ms: int) -> str:
    """Map a duration in milliseconds to a privacy-safe bucket."""
    if duration_ms < 100:
        return "<100ms"
    if duration_ms < 1000:
        return "100ms-1s"
    if duration_ms < 5000:
        return "1s-5s"
    if duration_ms < 30000:
        return "5s-30s"
    if duration_ms < 300000:
        return "30s-5m"
    return ">5m"


def error_class_bucket(error: BaseException | None) -> str | None:
    """Classify an exception into a privacy-safe error bucket."""
    if error is None:
        return None
    name = type(error).__name__
    if name == "ValueError":
        return "ValidationError"
    if name in ERROR_CLASS_BUCKETS:
        return name
    return "Unknown"
