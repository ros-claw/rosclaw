"""Event factory helpers for rosclaw.sense."""

from __future__ import annotations

import time
import uuid
from typing import Any

from rosclaw.core.event_bus import Event, EventPriority
from rosclaw.core.event_topics import EventTopics
from rosclaw.sense.schemas import BodyEvent, BodyReadiness, BodySense, BodyState


def _event(
    topic: str,
    payload: dict[str, Any],
    priority: EventPriority = EventPriority.NORMAL,
) -> Event:
    return Event(
        topic=topic,
        payload=payload,
        source="sense_runtime",
        priority=priority,
    )


def state_updated_event(state: BodyState) -> Event:
    return _event(
        EventTopics.SENSE_STATE_UPDATED,
        {"robot_id": state.robot_id, "timestamp": state.timestamp, "state": state.to_dict()},
    )


def body_updated_event(sense: BodySense) -> Event:
    return _event(
        EventTopics.SENSE_BODY_UPDATED,
        {"robot_id": sense.robot_id, "timestamp": sense.timestamp, "body_sense": sense.to_dict()},
    )


def event_detected_event(event: BodyEvent) -> Event:
    return _event(
        EventTopics.SENSE_EVENT_DETECTED,
        {"robot_id": event.robot_id, "timestamp": event.timestamp, "event": event.to_dict()},
        priority=_event_priority(event.severity),
    )


def readiness_updated_event(readiness: BodyReadiness) -> Event:
    return _event(
        EventTopics.SENSE_READINESS_UPDATED,
        {
            "robot_id": readiness.robot_id,
            "timestamp": readiness.timestamp,
            "readiness": readiness.to_dict(),
        },
    )


def capability_blocked_event(capability: str, reason: str, evidence: dict[str, Any]) -> Event:
    return _event(
        EventTopics.SENSE_CAPABILITY_BLOCKED,
        {
            "capability": capability,
            "reason": reason,
            "evidence": evidence,
            "timestamp": time.time(),
        },
        priority=EventPriority.HIGH,
    )


def capability_degraded_event(capability: str, reason: str, evidence: dict[str, Any]) -> Event:
    return _event(
        EventTopics.SENSE_CAPABILITY_DEGRADED,
        {
            "capability": capability,
            "reason": reason,
            "evidence": evidence,
            "timestamp": time.time(),
        },
        priority=EventPriority.NORMAL,
    )


def _event_priority(severity: str) -> EventPriority:
    if severity in ("critical", "high"):
        return EventPriority.HIGH
    if severity == "medium":
        return EventPriority.NORMAL
    return EventPriority.LOW


def make_event_id() -> str:
    return f"evt_{uuid.uuid4().hex[:12]}"
