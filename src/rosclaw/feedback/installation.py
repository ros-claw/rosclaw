"""Installation identity: UUID, anonymous ID derivation, consent persistence."""

from __future__ import annotations

import hashlib
import hmac
import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .store import append_event


@dataclass
class Installation:
    """In-memory representation of installation.json."""

    installation_id: str
    created_at: str
    install_channel: str
    telemetry_enabled: bool
    diagnostics_enabled: bool
    rich_feedback_enabled: bool
    schema_version: str = "rosclaw.installation.v1"


class InstallationManager:
    """Manage the local installation identity and consent flags."""

    def __init__(self, home: Path) -> None:
        self.home = Path(home)
        self.path = self.home / "config" / "installation.json"
        self.salt_path = self.home / "telemetry" / "local_salt"
        self.audit_path = self.home / "feedback" / "consent" / "audit.jsonl"

    def ensure_installation(
        self,
        install_channel: str = "stable",
        telemetry_enabled: bool = True,
        diagnostics_enabled: bool = False,
        rich_feedback_enabled: bool = False,
    ) -> Installation:
        """Create or update installation.json and salt."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.salt_path.parent.mkdir(parents=True, exist_ok=True)
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)

        data = self._load_raw()
        if not data.get("installation_id"):
            data["installation_id"] = str(uuid.uuid4())
            data["created_at"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        if not self.salt_path.exists():
            self.salt_path.write_text(
                uuid.uuid4().hex + uuid.uuid4().hex,
                encoding="utf-8",
            )

        data["schema_version"] = "rosclaw.installation.v1"
        data["install_channel"] = install_channel
        data["telemetry_enabled"] = telemetry_enabled
        data["diagnostics_enabled"] = diagnostics_enabled
        data["rich_feedback_enabled"] = rich_feedback_enabled
        data["updated_at"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")

        self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        self._audit("ensure_installation", {
            "telemetry_enabled": telemetry_enabled,
            "diagnostics_enabled": diagnostics_enabled,
            "rich_feedback_enabled": rich_feedback_enabled,
        })
        return self._to_installation(data)

    def get_installation(self) -> Installation | None:
        data = self._load_raw()
        if not data.get("installation_id"):
            return None
        return self._to_installation(data)

    def get_anonymous_installation_id(self) -> str | None:
        """Return a stable anonymous ID: rci_ + HMAC_SHA256(installation_id, salt)[0:32]."""
        data = self._load_raw()
        install_id = data.get("installation_id")
        if not install_id or not self.salt_path.exists():
            return None
        salt = self.salt_path.read_text(encoding="utf-8").strip()
        digest = hmac.new(
            salt.encode("utf-8"),
            install_id.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return f"rci_{digest[:32]}"

    def set_telemetry_enabled(self, enabled: bool) -> None:
        data = self._load_raw()
        if not data.get("installation_id"):
            self.ensure_installation(telemetry_enabled=enabled)
            return
        data["telemetry_enabled"] = enabled
        data["updated_at"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        self._audit("set_telemetry_enabled", {"enabled": enabled})

    def set_diagnostics_enabled(self, enabled: bool) -> None:
        data = self._load_raw()
        if not data.get("installation_id"):
            self.ensure_installation(diagnostics_enabled=enabled)
            return
        data["diagnostics_enabled"] = enabled
        data["updated_at"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        self._audit("set_diagnostics_enabled", {"enabled": enabled})

    def set_rich_feedback_enabled(self, enabled: bool) -> None:
        data = self._load_raw()
        if not data.get("installation_id"):
            self.ensure_installation(rich_feedback_enabled=enabled)
            return
        data["rich_feedback_enabled"] = enabled
        data["updated_at"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        self._audit("set_rich_feedback_enabled", {"enabled": enabled})

    def reset_installation_id(self) -> Installation:
        """Rotate installation ID and salt; old telemetry is no longer associable."""
        new_id = str(uuid.uuid4())
        new_salt = uuid.uuid4().hex + uuid.uuid4().hex
        self.salt_path.write_text(new_salt, encoding="utf-8")
        data = {
            "schema_version": "rosclaw.installation.v1",
            "installation_id": new_id,
            "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "updated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "install_channel": self._load_raw().get("install_channel", "stable"),
            "telemetry_enabled": self._load_raw().get("telemetry_enabled", True),
            "diagnostics_enabled": self._load_raw().get("diagnostics_enabled", False),
            "rich_feedback_enabled": self._load_raw().get("rich_feedback_enabled", False),
        }
        self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        self._audit("reset_installation_id", {"new_installation_id_prefix": new_id[:8]})
        return self._to_installation(data)

    def _load_raw(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _to_installation(self, data: dict[str, Any]) -> Installation:
        return Installation(
            installation_id=data["installation_id"],
            created_at=data.get("created_at", ""),
            install_channel=data.get("install_channel", "stable"),
            telemetry_enabled=data.get("telemetry_enabled", True),
            diagnostics_enabled=data.get("diagnostics_enabled", False),
            rich_feedback_enabled=data.get("rich_feedback_enabled", False),
            schema_version=data.get("schema_version", "rosclaw.installation.v1"),
        )

    def _audit(self, action: str, details: dict[str, Any]) -> None:
        append_event(
            self.audit_path,
            {
                "action": action,
                "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "details": details,
            },
        )
