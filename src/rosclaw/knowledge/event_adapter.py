"""Redacted EventBus projection for knowledge lifecycle observability."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rosclaw.core.event_bus import Event, EventBus, EventPriority

KNOWLEDGE_EVENT_TOPICS = frozenset(
    {
        "know.research.requested",
        "know.research.started",
        "know.research.completed",
        "know.research.failed",
        "know.source.discovered",
        "know.source.snapshotted",
        "know.source.updated",
        "know.source.superseded",
        "know.project.indexed",
        "know.wiki.updated",
        "know.reference_pack.created",
        "know.reference_pack.stale",
        "how.advice.created",
        "how.advice.abstained",
        "how.feedback.recorded",
    }
)

_FORBIDDEN_KEYS = {
    "api_key",
    "authorization",
    "password",
    "token",
    "document_content",
    "memory_content",
    "trajectory",
    "sensor_data",
    "permit",
    "action_authorization",
}
_ALLOWED_KEYS = {
    "request_id",
    "job_id",
    "source_id",
    "snapshot_id",
    "project_id",
    "reference_pack_id",
    "advice_id",
    "feedback_id",
    "knowledge_unit_id",
    "index_version",
    "status",
    "mode",
    "verdict",
    "count",
    "stale",
    "error_type",
}


def safe_event_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if _FORBIDDEN_KEYS.intersection(key.casefold() for key in payload):
        raise ValueError("knowledge event payload contains a forbidden field")
    result: dict[str, Any] = {}
    for key, value in payload.items():
        if key not in _ALLOWED_KEYS:
            continue
        if isinstance(value, str):
            result[key] = value[:240]
        elif isinstance(value, (bool, int, float)) or value is None:
            result[key] = value
    return result


class KnowledgeEventAdapter:
    def __init__(self, event_bus: EventBus | None) -> None:
        self.event_bus = event_bus

    def publish(self, topic: str, payload: Mapping[str, Any]) -> None:
        if self.event_bus is None:
            return
        if topic not in KNOWLEDGE_EVENT_TOPICS:
            raise ValueError(f"unsupported knowledge event topic: {topic}")
        self.event_bus.publish(
            Event(
                topic=topic,
                payload=safe_event_payload(payload),
                source="knowledge_adapter",
                priority=EventPriority.LOW,
            )
        )
