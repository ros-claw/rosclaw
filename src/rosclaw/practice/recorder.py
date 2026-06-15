"""Practice Recorder - Timeline Grounding

Records robot execution traces using the DataFlywheel.
All communication goes through EventBus — no direct module calls.

v1.0 EventBus Integration:
- Publishes praxis.completed / praxis.failed events
- Subscribes to knowledge.ingest_complete
- Publishes heuristic.recovery_executed outcomes
"""

import logging
import time
from pathlib import Path
from typing import Any

from rosclaw.core.event_bus import Event, EventBus, EventPriority
from rosclaw.core.lifecycle import LifecycleMixin
from rosclaw.data.flywheel import DataFlywheel, EventType

logger = logging.getLogger("rosclaw.practice.recorder")


class PracticeRecorder(LifecycleMixin):
    """
    Records robot practice sessions and execution traces.

    Uses the DataFlywheel for high-frequency capture
    and event-triggered persistence.

    Subscribes to agent.command via EventBus to auto-record
    all robot actions.

    EventBus Integration (v1.0):
    - Publishes praxis.completed after successful execution
    - Publishes praxis.failed on failure (with error context)
    - Subscribes to knowledge.ingest_complete for KNOW tracking
    - Publishes heuristic.recovery_executed after recovery attempts
    """

    def __init__(self, robot_id: str, joint_dof: int = 6, event_bus: EventBus | None = None):
        super().__init__()
        self.robot_id = robot_id
        self.joint_dof = joint_dof
        self.event_bus = event_bus
        self._flywheel: DataFlywheel | None = None
        self._recording = False

        # Failure context tracking for praxis.failed events
        self._failure_context: dict[str, Any] = {
            "previous_scores": [],
            "current_iteration": 0,
            "last_error": "",
        }

        # Knowledge ingest tracking
        self._knowledge_ingest_log: list[dict] = []

    def _do_initialize(self) -> None:
        """Initialize the practice recorder (flywheel + EventBus subscription)."""
        self._flywheel = DataFlywheel(
            robot_id=self.robot_id,
            joint_dof=self.joint_dof,
        )
        if self.event_bus is not None:
            self.event_bus.subscribe("skill.execution.complete", self._on_skill_complete)
            self.event_bus.subscribe("knowledge.ingest_complete", self._on_knowledge_ingest_complete)
        logger.info("Recorder initialized for %s (flywheel mode)", self.robot_id)

    def _do_stop(self) -> None:
        """Stop the recorder and unsubscribe from EventBus."""
        if self.event_bus is not None:
            self.event_bus.unsubscribe("skill.execution.complete", self._on_skill_complete)
            self.event_bus.unsubscribe("knowledge.ingest_complete", self._on_knowledge_ingest_complete)

    def _on_skill_complete(self, event) -> None:
        """Handle skill.execution.complete and publish praxis.completed/failed."""
        if self.event_bus is None:
            return
        payload = event.payload if hasattr(event, "payload") else {}
        if not isinstance(payload, dict):
            return
        result = payload.get("result", {})
        status = result.get("status")
        correlation_id = payload.get("correlation_id", "")
        skill_name = payload.get("skill_name", "")
        # Update failure context tracking
        self._failure_context["current_iteration"] += 1
        iteration = self._failure_context["current_iteration"]
        if status == "success":
            self._failure_context["previous_scores"].append(result.get("reward", 1.0))
            self.event_bus.publish(Event(
                topic="praxis.completed",
                payload={
                    "practice_id": correlation_id,
                    "event_type": "praxis.completed",
                    "robot_id": self.robot_id,
                    "outcome": {
                        "status": "success",
                        "reward": result.get("reward", 1.0),
                        "skill_name": skill_name,
                        "details": {"details": result.get("details", {})},
                    },
                    "current_iteration": iteration,
                    "previous_scores": list(self._failure_context["previous_scores"]),
                    "timestamp": time.time(),
                },
                source="practice_recorder",
                priority=EventPriority.NORMAL,
            ))
        elif status == "failure":
            error = result.get("error", "")
            self._failure_context["last_error"] = error
            self._failure_context["previous_scores"].append(result.get("reward", -1.0))
            self.event_bus.publish(Event(
                topic="praxis.failed",
                payload={
                    "practice_id": correlation_id,
                    "event_type": "praxis.failed",
                    "robot_id": self.robot_id,
                    "outcome": {
                        "status": "failure",
                        "reward": result.get("reward", -1.0),
                        "skill_name": skill_name,
                        "details": {"details": result.get("details", {})},
                    },
                    "error_log": error,
                    "current_iteration": iteration,
                    "previous_scores": list(self._failure_context["previous_scores"]),
                    "timestamp": time.time(),
                },
                source="practice_recorder",
                priority=EventPriority.HIGH,
            ))

    def record_recovery_outcome(self, rule_id: str, success: bool, duration: float, correlation_id: str = "") -> None:
        """Record heuristic recovery outcome and publish event. Subscribers: HOW.

        Args:
            rule_id: Identifier of the heuristic rule that was applied
            success: Whether the recovery was successful
            duration: Time taken for recovery in seconds
            correlation_id: Optional correlation ID linking to the original praxis
        """
        if self.event_bus is None:
            return

        self.event_bus.publish(Event(
            topic="heuristic.recovery_executed",
            payload={
                "rule_id": rule_id,
                "success": success,
                "duration": duration,
                "robot_id": self.robot_id,
                "timestamp": time.time(),
                "correlation_id": correlation_id,
            },
            source="practice_recorder",
            priority=EventPriority.NORMAL,
        ))
        logger.info("Published heuristic.recovery_executed for rule %s", rule_id)

    def _on_knowledge_ingest_complete(self, event) -> None:
        """Handle knowledge.ingest_complete and log to knowledge_ingest_log."""
        payload = event.payload if hasattr(event, "payload") else {}
        if not isinstance(payload, dict):
            return
        self._knowledge_ingest_log.append({
            "practice_id": payload.get("practice_id", "unknown"),
            "knowledge_version": payload.get("knowledge_version", "unknown"),
            "status": payload.get("status", "unknown"),
            "ingest_timestamp": payload.get("timestamp", time.time()),
        })

    def start_recording(self) -> None:
        """Start a recording session."""
        self._recording = True
        logger.info("Recording started")

    def stop_recording(self) -> None:
        """Stop recording."""
        self._recording = False
        logger.info("Recording stopped")

    def log_state(self, joint_positions: list[float], timestamp: float) -> None:
        """Log a robot state sample."""
        if not self._recording or self._flywheel is None:
            return
        import numpy as np

        from rosclaw.data.flywheel import RobotState as FlywheelRobotState
        state = FlywheelRobotState(
            timestamp=timestamp,
            joint_positions=np.array(joint_positions),
            joint_velocities=np.zeros(self.joint_dof),
            joint_torques=np.zeros(self.joint_dof),
        )
        self._flywheel.on_control_cycle(state)

    def mark_event(self, event_type: EventType, metadata: dict | None = None) -> str:
        """Mark an event in the recording."""
        if self._flywheel is None:
            return ""
        return self._flywheel.trigger_event(event_type, metadata)

    def export_session(self, output_path: Path) -> Path:
        """Export recorded session to LeRobot format."""
        if self._flywheel is None:
            raise RuntimeError("Flywheel not initialized")
        return self._flywheel.export_to_lerobot(output_path)

    def record_praxis_event(self, event=None, event_id: str = "", event_type: str = "", instruction: str = "", metadata: dict | None = None) -> str:
        """Record a praxis event on the timeline.

        Args:
            event: A PraxisEvent dataclass instance (alternative to individual args)
            event_id: Unique event identifier (ignored if event is provided)
            event_type: Event classification (success, failure, milestone)
            instruction: Natural language instruction that triggered this event
            metadata: Additional context data

        Returns:
            Event marker ID
        """
        if not self._recording or self._flywheel is None:
            return ""

        from rosclaw.core.types import PraxisEvent
        from rosclaw.data.flywheel import EventType as FlywheelEventType

        if isinstance(event, PraxisEvent):
            data = {
                "event_id": event.event_id,
                "event_type": event.event_type,
                "instruction": event.agent_instruction,
                "duration_sec": event.duration_sec,
                "cot_trace": event.cot_trace,
                "trajectory": event.trajectory,
            }
            try:
                fw_type = FlywheelEventType[event.event_type.upper()]
            except KeyError:
                fw_type = FlywheelEventType.MILESTONE
            return self._flywheel.trigger_event(fw_type, data)

        try:
            fw_type = FlywheelEventType[event_type.upper()]
        except KeyError:
            fw_type = FlywheelEventType.MILESTONE
        return self._flywheel.trigger_event(
            fw_type,
            {
                "event_id": event_id,
                "instruction": instruction,
                **(metadata or {}),
            },
        )

    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._recording

    @property
    def failure_context(self) -> dict:
        """Get current failure context (for testing/inspection)."""
        return self._failure_context.copy()

    @property
    def knowledge_ingest_log(self) -> list[dict]:
        """Get knowledge ingest completion log (for testing/inspection)."""
        return self._knowledge_ingest_log.copy()
