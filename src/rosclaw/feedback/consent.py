"""User consent management for diagnostics and rich feedback."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .installation import InstallationManager
from .store import append_event


@dataclass
class ConsentState:
    """Aggregated consent state."""

    product_telemetry: bool
    diagnostics: bool
    rich_feedback: bool
    updated_at: str


class ConsentManager:
    """Read and write user consent for diagnostic/rich feedback upload."""

    def __init__(self, home: Path) -> None:
        self.home = Path(home)
        self.path = self.home / "feedback" / "consent" / "consent.json"
        self.audit_path = self.home / "feedback" / "consent" / "audit.jsonl"
        self.installation = InstallationManager(home)

    def ensure(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def show(self) -> ConsentState:
        install = self.installation.get_installation()
        if install is None:
            install = self.installation.ensure_installation()
        consent = self._load_consent()
        return ConsentState(
            product_telemetry=install.telemetry_enabled,
            diagnostics=consent.get("diagnostics", install.diagnostics_enabled),
            rich_feedback=consent.get("rich_feedback", install.rich_feedback_enabled),
            updated_at=consent.get("updated_at", install.created_at),
        )

    def set_diagnostics(self, enabled: bool) -> ConsentState:
        self.ensure()
        consent = self._load_consent()
        consent["diagnostics"] = enabled
        consent["updated_at"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        self.path.write_text(json.dumps(consent, indent=2, ensure_ascii=False), encoding="utf-8")
        self.installation.set_diagnostics_enabled(enabled)
        self._audit("set_diagnostics", {"enabled": enabled})
        return self.show()

    def set_rich_feedback(self, enabled: bool) -> ConsentState:
        self.ensure()
        consent = self._load_consent()
        consent["rich_feedback"] = enabled
        consent["updated_at"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        self.path.write_text(json.dumps(consent, indent=2, ensure_ascii=False), encoding="utf-8")
        self.installation.set_rich_feedback_enabled(enabled)
        self._audit("set_rich_feedback", {"enabled": enabled})
        return self.show()

    def revoke_all(self) -> ConsentState:
        self.ensure()
        consent = {
            "diagnostics": False,
            "rich_feedback": False,
            "updated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        }
        self.path.write_text(json.dumps(consent, indent=2, ensure_ascii=False), encoding="utf-8")
        self.installation.set_diagnostics_enabled(False)
        self.installation.set_rich_feedback_enabled(False)
        self._audit("revoke_all", {})
        return self.show()

    def _load_consent(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _audit(self, action: str, details: dict[str, Any]) -> None:
        append_event(
            self.audit_path,
            {
                "action": action,
                "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "details": details,
            },
        )
