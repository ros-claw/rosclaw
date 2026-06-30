"""MemoryConsumer — RuntimeBus consumer for the ROSClaw Knowledge Plane.

The MemoryConsumer subscribes to runtime events and writes them directly into
SeekDB as they arrive, removing the need for a separate ``memory ingest`` step.

Tables written:
- ``experience_graph`` — episode summaries and skill execution outcomes.
- ``praxis_events`` — camera, provider, sandbox, and skill events.
- ``artifacts`` — frame references and other artifact URIs.
- ``failures`` — blocked sandbox decisions and skill failures.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any

from rosclaw.memory.interface import MemoryInterface
from rosclaw.memory.seekdb_client import SeekDBClient
from rosclaw.memory.types import ArtifactRef, FailureMemory, PraxisEvent
from rosclaw.runtime.component import RuntimeConsumer
from rosclaw.runtime.event import RuntimeEvent

logger = logging.getLogger("rosclaw.memory.consumer")


class MemoryConsumer(RuntimeConsumer):
    """Real-time memory writer driven by RuntimeBus events.

    Usage:
        bus = RuntimeBus()
        consumer = MemoryConsumer(bus, robot_id="realsense-d405")
        consumer.initialize()
        consumer.start()

        # ... runtime events flow through bus ...

        consumer.stop()
    """

    def __init__(
        self,
        runtime_bus,
        robot_id: str = "default_robot",
        seekdb_client: SeekDBClient | None = None,
        event_bus=None,
    ) -> None:
        super().__init__("memory_consumer", runtime_bus)
        self._robot_id = robot_id
        # event_bus is only used for MemoryInterface's own publications; the
        # consumer itself subscribes through RuntimeBus.
        self._memory = MemoryInterface(
            robot_id=robot_id,
            event_bus=event_bus,
            seekdb_client=seekdb_client,
        )

    def _do_initialize(self) -> None:
        self._memory.initialize()
        super()._do_initialize()

    def _do_start(self) -> None:
        super()._do_start()
        self._memory.start()

    def _do_stop(self) -> None:
        super()._do_stop()
        self._memory.stop()

    def on_event(self, event: RuntimeEvent) -> None:
        """Route runtime events to the appropriate SeekDB tables."""
        handlers = {
            "camera.rgbd_frame": self._handle_camera_frame,
            "provider.result": self._handle_provider_result,
            "provider.request": self._handle_provider_request,
            "skill.invoke": self._handle_skill_invoke,
            "skill.complete": self._handle_skill_complete,
            "sandbox.decision": self._handle_sandbox_decision,
            "practice.stop": self._handle_practice_stop,
        }
        handler = handlers.get(event.type)
        if handler is not None:
            try:
                handler(event)
            except Exception as exc:
                logger.warning("MemoryConsumer failed to store %s: %s", event.type, exc)
        elif event.type.startswith("runtime."):
            self._write_praxis_event(event)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _handle_camera_frame(self, event: RuntimeEvent) -> None:
        payload = event.payload or {}
        episode_id = self._episode_id(event)
        refs = {
            "rgb_frame": payload.get("rgb_ref"),
            "depth_frame": payload.get("depth_ref"),
            "camera_info": payload.get("camera_info_ref"),
            "extrinsics": payload.get("extrinsics_ref"),
        }
        for artifact_type, uri in refs.items():
            if not isinstance(uri, str) or not uri:
                continue
            artifact_id = f"{episode_id or 'no_ep'}_{event.id}_{artifact_type}"
            self._memory.store_artifact(
                ArtifactRef(
                    artifact_id=artifact_id,
                    artifact_type=artifact_type,
                    uri=uri,
                    episode_id=episode_id,
                    metadata={"camera_id": payload.get("camera_id"), "event_type": "camera.rgbd_frame"},
                )
            )
        self._write_praxis_event(event, episode_id=episode_id)

    def _handle_provider_result(self, event: RuntimeEvent) -> None:
        self._write_praxis_event(event)

    def _handle_provider_request(self, event: RuntimeEvent) -> None:
        self._write_praxis_event(event)

    def _handle_skill_invoke(self, event: RuntimeEvent) -> None:
        self._write_praxis_event(event)

    def _handle_skill_complete(self, event: RuntimeEvent) -> None:
        self._write_praxis_event(event)
        payload = event.payload or {}
        outcome = self._normalize_outcome(payload.get("status") or payload.get("outcome"))
        skill_id = payload.get("skill_id") or event.metadata.get("skill_id") or "unknown"
        self._memory.store_experience(
            event_id=event.id,
            event_type="skill_execution",
            instruction=f"Skill execution: {skill_id}",
            outcome=outcome,
            duration_sec=payload.get("duration_sec", 0.0),
            tags=["skill", skill_id],
            metadata={"skill_id": skill_id, **payload},
        )
        if outcome == "failure":
            self._memory.write_failure_memory(
                FailureMemory(
                    failure_id=event.id,
                    robot_id=event.robot or self._robot_id,
                    episode_id=self._episode_id(event),
                    failure_type="skill_failure",
                    root_cause=payload.get("error", ""),
                    recovery_hint=payload.get("recovery_hint", ""),
                    metadata=payload,
                )
            )

    def _handle_sandbox_decision(self, event: RuntimeEvent) -> None:
        self._write_praxis_event(event)
        payload = event.payload or {}
        decision = payload.get("decision")
        if decision == "BLOCK":
            self._memory.write_failure_memory(
                FailureMemory(
                    failure_id=event.id,
                    robot_id=event.robot or self._robot_id,
                    episode_id=self._episode_id(event),
                    failure_type="sandbox_blocked",
                    root_cause=payload.get("reason", "sandbox blocked action"),
                    recovery_hint=payload.get("reason", ""),
                    sandbox_intervened=True,
                    category="safety",
                    metadata=payload,
                )
            )

    def _handle_practice_stop(self, event: RuntimeEvent) -> None:
        payload = event.payload or {}
        episode_id = payload.get("practice_id") or self._episode_id(event)
        task = payload.get("task", {})
        task_label = (
            task.get("skill_id")
            or task.get("task_name")
            or task.get("task_id")
            or "unknown"
        )
        outcome = self._normalize_outcome(payload.get("outcome"))
        duration_ms = payload.get("duration_ms", 0)
        duration_sec = duration_ms / 1000.0 if isinstance(duration_ms, (int, float)) else 0.0
        robot_id = payload.get("robot_id") or event.robot or self._robot_id
        robot_type = payload.get("robot_type") or "unknown"
        tags = ["practice", robot_type, robot_id]
        if task.get("skill_id"):
            tags.append(str(task["skill_id"]))

        self._memory.store_experience(
            event_id=episode_id,
            event_type="practice_episode",
            instruction=f"Practice episode {episode_id}: {task_label}",
            outcome=outcome,
            duration_sec=duration_sec,
            tags=tags,
            metadata={"episode_payload": payload},
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _write_praxis_event(self, event: RuntimeEvent, episode_id: str | None = None) -> None:
        if episode_id is None:
            episode_id = self._episode_id(event)
        self._memory.write_praxis_event(
            PraxisEvent(
                event_id=event.id,
                robot_id=event.robot or self._robot_id,
                event_type=event.type,
                timestamp=_datetime_to_epoch(event.timestamp),
                episode_id=episode_id,
                task_id=event.metadata.get("task_id"),
                payload=event.payload or {},
                metadata=event.metadata,
            )
        )

    def _episode_id(self, event: RuntimeEvent) -> str | None:
        return event.metadata.get("trace_id") or event.metadata.get("practice_id") or event.metadata.get("episode_id")

    @staticmethod
    def _normalize_outcome(value: Any) -> str:
        if value is None:
            return "unknown"
        text = str(value).lower()
        if text in {"success", "succeeded", "ok", "pass"}:
            return "success"
        if text in {"failure", "failed", "fail", "error"}:
            return "failure"
        return text


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _datetime_to_epoch(dt: datetime) -> float:
    return dt.timestamp()
