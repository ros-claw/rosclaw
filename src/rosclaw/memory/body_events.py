"""Memory body events — write body-change events to the memory layer."""

from __future__ import annotations

import logging
import time
from typing import Any

from rosclaw.body.diff import BodyDiff

logger = logging.getLogger("rosclaw.memory.body_events")


class BodyMemoryEventWriter:
    """Write body-related events to memory without blocking body updates.

    The writer receives structured body change information (including the diff
    and affected skills) and records it.  In P0/P0.5 it may be a mock/no-op;
    in P1 it must be wired to the active memory store.
    """

    EVENT_TYPES = {
        "body_initialized",
        "body_changed",
        "body_fault_added",
        "body_fault_resolved",
        "body_calibration_updated",
        "body_capability_changed",
        "skill_compatibility_changed",
        "active_body_switched",
    }

    def __init__(self, memory_client: Any | None = None):
        self._client = memory_client

    def write_body_change(
        self,
        body_instance_id: str,
        old_hash: str,
        new_hash: str,
        diff: BodyDiff,
        affected_skills: list[str],
    ) -> dict[str, Any]:
        """Record a body change event.

        Failures are logged and returned in the result but never raised, so that
        a memory outage cannot block body updates.
        """
        event = {
            "event_type": "body_changed",
            "body_instance_id": body_instance_id,
            "old_effective_body_hash": old_hash,
            "new_effective_body_hash": new_hash,
            "diff": diff.to_dict() if diff else {},
            "affected_skills": affected_skills,
            "timestamp": time.time(),
        }

        if self._client is None:
            return {"recorded": False, "event": event, "reason": "memory_client_not_configured"}

        try:
            record_id = self._client.insert("body_events", event)
            return {"recorded": True, "event": event, "record_id": record_id}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to write body change event to memory: %s", exc)
            return {"recorded": False, "event": event, "reason": str(exc)}

    def write_event(
        self,
        event_type: str,
        body_instance_id: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record an arbitrary body event type."""
        if event_type not in self.EVENT_TYPES:
            raise ValueError(f"Unknown body event type: {event_type}")

        event = {
            "event_type": event_type,
            "body_instance_id": body_instance_id,
            "payload": payload or {},
            "timestamp": time.time(),
        }

        if self._client is None:
            return {"recorded": False, "event": event, "reason": "memory_client_not_configured"}

        try:
            record_id = self._client.insert("body_events", event)
            return {"recorded": True, "event": event, "record_id": record_id}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to write body event to memory: %s", exc)
            return {"recorded": False, "event": event, "reason": str(exc)}
