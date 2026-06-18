"""ROS Connector - Practice capture adapter.

Bridges ROS capability execution events into the ROSClaw practice / memory
ecosystem. It listens to the events emitted by ``RosCapabilityProvider`` and
forwards them as ``praxis.recorded`` events that ``EpisodeRecorder`` and
``MemoryInterface`` already ingest.

No ROS Python imports.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from rosclaw.core.event_bus import Event, EventBus
from rosclaw.core.lifecycle import LifecycleMixin

logger = logging.getLogger("rosclaw.connectors.ros.practice.ros_practice_adapter")


class RosPracticeAdapter(LifecycleMixin):
    """Adapter that turns ROS provider events into practice episodes."""

    _SUBSCRIBED_TOPICS = [
        "rosclaw.practice.event.created",
        "rosclaw.sandbox.episode.failed",
        "rosclaw.how.recovery_hint.generated",
    ]

    def __init__(self, event_bus: EventBus):
        super().__init__()
        self._event_bus = event_bus
        self._handlers: dict[str, Callable[[Event], None]] = {}

    def _do_initialize(self) -> None:
        self._handlers = {
            "rosclaw.practice.event.created": self._on_practice_event_created,
            "rosclaw.sandbox.episode.failed": self._on_sandbox_episode_failed,
            "rosclaw.how.recovery_hint.generated": self._on_recovery_hint_generated,
        }
        for topic, handler in self._handlers.items():
            self._event_bus.subscribe(topic, handler)
        logger.info("RosPracticeAdapter initialized")

    def _do_stop(self) -> None:
        for topic, handler in self._handlers.items():
            self._event_bus.unsubscribe(topic, handler)
        self._handlers.clear()

    def _on_practice_event_created(self, event: Event) -> None:
        """Convert a successful ROS execution into a praxis.recorded event."""
        payload = event.payload if isinstance(event.payload, dict) else {}
        self._publish_praxis_recorded(payload, outcome="success")

    def _on_sandbox_episode_failed(self, event: Event) -> None:
        """Convert a failed/blocked ROS execution into a praxis.recorded event."""
        payload = event.payload if isinstance(event.payload, dict) else {}
        outcome = "failure" if not payload.get("sandbox_blocked") else "blocked"
        self._publish_praxis_recorded(payload, outcome=outcome)

    def _on_recovery_hint_generated(self, event: Event) -> None:
        """Forward recovery hints so Memory can attach them to failures."""
        payload = event.payload if isinstance(event.payload, dict) else {}
        failure_id = payload.get("episode_id") or payload.get("request_id")
        if failure_id is None:
            return
        self._event_bus.publish(Event(
            topic="rosclaw.memory.failure.update",
            payload={
                "failure_id": failure_id,
                "recovery_hint": payload.get("hint", ""),
                "rule_id": payload.get("rule_id", ""),
                "source": "ros_practice_adapter",
            },
            source="ros_practice_adapter",
        ))

    def _publish_praxis_recorded(self, payload: dict[str, Any], outcome: str) -> None:
        """Publish a normalized praxis.recorded event."""
        capability_id = payload.get("capability_id", "unknown")
        trace_id = payload.get("trace_id", "")
        error = payload.get("error")
        sandbox_decision = payload.get("sandbox_decision") or {}
        instruction = f"ROS {capability_id}"

        self._event_bus.publish(Event(
            topic="praxis.recorded",
            payload={
                "event_id": trace_id or f"ros_{capability_id}",
                "event_type": outcome,
                "robot_id": payload.get("robot_id", "unknown"),
                "instruction": instruction,
                "duration_sec": 0.0,
                "outcome": outcome,
                "error_details": error,
                "capability_id": capability_id,
                "ros_name": payload.get("ros_name"),
                "ros_kind": payload.get("ros_kind"),
                "sandbox_decision": sandbox_decision,
                "raw": payload.get("result"),
            },
            source="ros_practice_adapter",
        ))
        logger.debug(
            "Forwarded ROS %s as praxis.recorded: %s", outcome, capability_id
        )
