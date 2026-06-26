"""Maintenance log — structured JSONL notes and events."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rosclaw.body.schema import MaintenanceEvent


class MaintenanceLog:
    """Append/read structured maintenance events as JSONL."""

    def __init__(self, log_path: Path):
        self.log_path = log_path

    def append(self, event: MaintenanceEvent) -> None:
        """Append a single event to the log."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_dict(), default=str) + "\n")

    def read_events(
        self,
        limit: int = 1000,
        type_filter: str | None = None,
    ) -> list[MaintenanceEvent]:
        """Read recent events, optionally filtered by type."""
        if not self.log_path.exists():
            return []
        events: list[MaintenanceEvent] = []
        with open(self.log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if type_filter and data.get("type") != type_filter:
                    continue
                events.append(MaintenanceEvent.from_dict(data))
        if limit > 0:
            events = events[-limit:]
        return events

    def events_affecting_skills(self) -> list[MaintenanceEvent]:
        """Return events that should trigger a skill recheck."""
        return [e for e in self.read_events() if e.requires_skill_recheck]

    def write_init_event(
        self,
        body_instance_id: str,
        eurdf_uri: str,
        author: str = "rosclaw",
    ) -> MaintenanceEvent:
        """Write the initial link-eurdf event."""
        event = MaintenanceEvent(
            ts=_utc_now(),
            type="maintenance",
            severity="info",
            author=author,
            body_instance_id=body_instance_id,
            message=f"Initial e-URDF linked: {eurdf_uri}",
            affects=["body"],
            tags=["init"],
            requires_skill_recheck=True,
        )
        self.append(event)
        return event

    def write_update_event(
        self,
        body_instance_id: str,
        change_summary: str,
        affects: list[str],
        author: str = "human",
        reason: str = "",
        event_type: str = "maintenance",
    ) -> MaintenanceEvent:
        """Write a body state update event."""
        event = MaintenanceEvent(
            ts=_utc_now(),
            type=event_type,
            severity="info",
            author=author,
            body_instance_id=body_instance_id,
            message=change_summary + (f" — reason: {reason}" if reason else ""),
            component=affects[0] if affects else "",
            affects=affects,
            tags=["update-state"],
            requires_skill_recheck=_should_recheck(affects),
        )
        self.append(event)
        return event


    def write_fault_event(
        self,
        body_instance_id: str,
        component: str,
        severity: str,
        summary: str,
        fault_id: str,
        author: str = "human",
    ) -> MaintenanceEvent:
        """Write a new fault event."""
        event = MaintenanceEvent(
            ts=_utc_now(),
            type="fault",
            severity=severity,
            author=author,
            body_instance_id=body_instance_id,
            message=f"Fault opened: {summary}",
            summary=summary,
            component=component,
            affects=[component],
            tags=["fault", severity],
            requires_skill_recheck=True,
            before={"status": "ok"},
            after={"status": "fault", "fault_id": fault_id, "severity": severity},
            result={"fault_id": fault_id, "status": "open"},
        )
        self.append(event)
        return event

    def write_resolution_event(
        self,
        body_instance_id: str,
        fault_id: str,
        component: str,
        summary: str,
        author: str = "human",
    ) -> MaintenanceEvent:
        """Write a fault resolution event."""
        event = MaintenanceEvent(
            ts=_utc_now(),
            type="repair",
            severity="info",
            author=author,
            body_instance_id=body_instance_id,
            message=f"Fault resolved: {summary}",
            summary=summary,
            component=component,
            affects=[component],
            tags=["fault", "resolution"],
            requires_skill_recheck=True,
            before={"status": "fault", "fault_id": fault_id},
            after={"status": "ok"},
            result={"fault_id": fault_id, "status": "resolved"},
        )
        self.append(event)
        return event

    def write_render_event(
        self,
        body_instance_id: str,
        reason: str = "",
        author: str = "rosclaw",
    ) -> MaintenanceEvent:
        """Write an EMBODIMENT.md render event."""
        event = MaintenanceEvent(
            ts=_utc_now(),
            type="render",
            severity="info",
            author=author,
            body_instance_id=body_instance_id,
            message="EMBODIMENT.md rendered" + (f": {reason}" if reason else ""),
            summary=reason or "EMBODIMENT.md rendered",
            component="embodiment",
            affects=["embodiment"],
            tags=["render"],
            requires_skill_recheck=False,
            requires_render=True,
            before={},
            after={},
            result={"status": "rendered"},
        )
        self.append(event)
        return event

    def write_calibration_event(
        self,
        body_instance_id: str,
        summary: str,
        before: dict[str, Any] | None = None,
        after: dict[str, Any] | None = None,
        author: str = "human",
    ) -> MaintenanceEvent:
        """Write a calibration update event."""
        event = MaintenanceEvent(
            ts=_utc_now(),
            type="calibration",
            severity="info",
            author=author,
            body_instance_id=body_instance_id,
            message=f"Calibration updated: {summary}",
            summary=summary,
            component="calibration",
            affects=["calibration"],
            tags=["calibration"],
            requires_skill_recheck=True,
            before=before or {},
            after=after or {},
            result={"status": "updated"},
        )
        self.append(event)
        return event

    def write_retrofit_event(
        self,
        body_instance_id: str,
        component: str,
        retrofit_type: str,
        summary: str,
        author: str = "human",
    ) -> MaintenanceEvent:
        """Write a retrofit (hardware modification) event."""
        event = MaintenanceEvent(
            ts=_utc_now(),
            type="maintenance",
            severity="info",
            author=author,
            body_instance_id=body_instance_id,
            message=f"Retrofit ({retrofit_type}): {summary}",
            summary=summary,
            component=component,
            affects=[component],
            tags=["retrofit", retrofit_type],
            requires_skill_recheck=True,
            before={"installed": False},
            after={"installed": True, "retrofit_type": retrofit_type},
            result={"status": "installed"},
        )
        self.append(event)
        return event

    def write_capability_event(
        self,
        body_instance_id: str,
        capability: str,
        action: str,
        reason: str,
        author: str = "human",
    ) -> MaintenanceEvent:
        """Write a capability disable/degrade/enable event."""
        event = MaintenanceEvent(
            ts=_utc_now(),
            type="capability_update",
            severity="warning" if action in ("disable", "degrade") else "info",
            author=author,
            body_instance_id=body_instance_id,
            message=f"Capability {action}d: {capability} — {reason}",
            summary=f"Capability {action}d: {capability}",
            component=capability,
            affects=[capability],
            tags=["capability", action],
            requires_skill_recheck=True,
            before={"capability_state": "unknown"},
            after={"capability_state": action},
            result={"capability": capability, "action": action, "reason": reason},
        )
        self.append(event)
        return event


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _should_recheck(affects: list[str]) -> bool:
    """Heuristic: does this affect list warrant a skill recheck?"""
    if not affects:
        return False
    recheck_keywords = {"sensor", "camera", "actuator", "arm", "leg", "motor", "gripper", "capability", "safety", "calibration"}
    for affect in affects:
        low = affect.lower()
        if any(kw in low for kw in recheck_keywords):
            return True
    return False
