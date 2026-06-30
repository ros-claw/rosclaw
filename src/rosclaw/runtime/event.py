"""Unified RuntimeEvent schema for ROSClaw Runtime Kernel v2.

All producers/consumers in the runtime communicate through this envelope.
It is intentionally flat and serialization-friendly so it can be stored in
JSONL, replayed, and consumed by any runtime component without coupling to a
particular module's dataclass.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RuntimeEvent(BaseModel):
    """Canonical event envelope for the Runtime Kernel.

    Attributes:
        id: Unique event identifier.
        timestamp: UTC datetime when the event was published.
        source: Producer name (e.g. "realsense_camera", "cosmos_reasoner").
        robot: Robot profile/model id (e.g. "realsense-d405").
        body_id: Body instance id (e.g. "d405_lab_01").
        type: Event type in dotted form (e.g. "camera.rgbd_frame").
        payload: Event-specific payload. Must be JSON-serializable.
        metadata: Optional tracing, quality, or policy context.
    """

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source: str = "unknown"
    robot: str | None = None
    body_id: str | None = None
    type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def topic(self) -> str:
        """Return the canonical bus topic for this event type."""
        return f"rosclaw.{self.type}"

    def to_event_bus_payload(self) -> dict[str, Any]:
        """Serialize to the payload dict used by the legacy EventBus."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "robot": self.robot,
            "body_id": self.body_id,
            "type": self.type,
            "payload": self.payload,
            "metadata": self.metadata,
        }

    @classmethod
    def from_event_bus_payload(
        cls, payload: dict[str, Any], topic: str = ""
    ) -> "RuntimeEvent":
        """Reconstruct a RuntimeEvent from a legacy EventBus payload dict."""
        data = dict(payload)
        ts = data.get("timestamp")
        if isinstance(ts, datetime):
            pass
        elif isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        elif isinstance(ts, (int, float)):
            ts = datetime.fromtimestamp(ts, tz=UTC)
        else:
            ts = datetime.now(UTC)
        data["timestamp"] = ts
        if topic and ("type" not in data or not data.get("type")):
            # Derive type from canonical topic rosclaw.<type>
            prefix = "rosclaw."
            if topic.startswith(prefix):
                data["type"] = topic[len(prefix) :]
            else:
                data["type"] = topic
        return cls.model_validate(data)
