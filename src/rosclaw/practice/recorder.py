"""Practice Recorder - Timeline Grounding

Records robot execution traces using the DataFlywheel.
All communication goes through EventBus — no direct module calls.

v1.0 EventBus Integration:
- Publishes praxis.completed / praxis.failed events
- Subscribes to knowledge.ingest_complete
- Publishes heuristic.recovery_executed outcomes
"""

import time
from pathlib import Path
from typing import Any, Optional

from rosclaw.core.event_bus import EventBus, Event, EventPriority
from rosclaw.core.lifecycle import LifecycleMixin
from rosclaw.data.flywheel import DataFlywheel, EventType


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

    def __init__(self, robot_id: str, joint_dof: int = 6, event_bus: Optional[EventBus] = None):
        super().__init__()
        self.robot_id = robot_id
        self.joint_dof = joint_dof
        self.event_bus = event_bus
        self._flywheel: Optional[DataFlywheel] = None
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
        """Initialize the practice recorder."""
        self._flywheel = DataFlywheel(
            robot_id=self.robot_id,
            joint_dof=self.joint_dof,
        )
        if self.event_bus is not None:
            self.event_bus.subscribe("agent.command", self._on_agent_command)
            self.event_bus.subscribe("skill.execution.start", self._on_skill_start)
            self.event_bus.subscribe("skill.execution.complete", self._on_skill_complete)
            self.event_bus.subscribe("knowledge.ingest_complete", self._on_knowledge_ingest_complete)
        print(f"[Practice] Recorder initialized for {self.robot_id}")

    def _do_stop(self) -> None:
        if self.event_bus is not None:
            self.event_bus.unsubscribe("agent.command", self._on_agent_command)
            self.event_bus.unsubscribe("skill.execution.start", self._on_skill_start)
            self.event_bus.unsubscribe("skill.execution.complete", self._on_skill_complete)
            self.event_bus.unsubscribe("knowledge.ingest_complete", self._on_knowledge_ingest_complete)

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
        print(f"[Practice] Published heuristic.recovery_executed for rule {rule_id}")

    def start_recording(self) -> None:
        """Start a recording session."""
        self._recording = True
        print("[Practice] Recording started")

    def stop_recording(self) -> None:
        """Stop recording."""
        self._recording = False
        print("[Practice] Recording stopped")

    def log_state(self, joint_positions: list[float], timestamp: float) -> None:
        """Log a robot state sample."""
        if not self._recording or self._flywheel is None:
            return
        from rosclaw.data.flywheel import RobotState as FlywheelRobotState
        import numpy as np
        state = FlywheelRobotState(
            timestamp=timestamp,
            joint_positions=np.array(joint_positions),
            joint_velocities=np.zeros(self.joint_dof),
            joint_torques=np.zeros(self.joint_dof),
        )
        self._flywheel.on_control_cycle(state)

    def mark_event(self, event_type: EventType, metadata: Optional[dict] = None) -> str:
        """Mark an event in the recording."""
        if self._flywheel is None:
            return ""
        return self._flywheel.trigger_event(event_type, metadata)

    def export_session(self, output_path: Path) -> Path:
        """Export recorded session to LeRobot format."""
        if self._flywheel is None:
            raise RuntimeError("Flywheel not initialized")
        return self._flywheel.export_to_lerobot(output_path)

    def record_praxis_event(self, event=None, event_id: str = "", event_type: str = "", instruction: str = "", metadata: Optional[dict] = None) -> str:
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

        from rosclaw.data.flywheel import EventType as FlywheelEventType
        from rosclaw.core.types import PraxisEvent

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

    def _on_agent_command(self, event: Event) -> None:
        """Auto-start recording when agent commands are issued."""
        if not self._recording:
            return
        payload = event.payload if isinstance(event.payload, dict) else {}
        action = payload.get("action", "unknown")
        self.mark_event(EventType.MILESTONE, {"action": action, "source": event.source})

    def _on_skill_start(self, event: Event) -> None:
        """Mark skill execution start."""
        if self._recording and self._flywheel is not None:
            payload = event.payload if isinstance(event.payload, dict) else {}
            self._flywheel.trigger_event(
                EventType.MILESTONE,
                {"skill_name": payload.get("skill_name"), "phase": "start"},
            )

    def _on_skill_complete(self, event: Event) -> None:
        """Mark skill execution completion and publish praxis event."""
        payload = event.payload if isinstance(event.payload, dict) else {}
        result = payload.get("result", {})
        status = result.get("status", "unknown")
        skill_name = payload.get("skill_name", "unknown")
        correlation_id = payload.get("correlation_id", "")

        # Record on flywheel
        if self._recording and self._flywheel is not None:
            event_type = EventType.SUCCESS if status == "success" else EventType.FAILURE
            self._flywheel.trigger_event(
                event_type,
                {"skill_name": skill_name, "phase": "complete", "result": result},
            )

        # Publish praxis event on EventBus
        if self.event_bus is not None:
            if status == "success":
                self._publish_praxis_completed(correlation_id, skill_name, result)
            else:
                self._publish_praxis_failed(correlation_id, skill_name, result)

    def _publish_praxis_completed(self, correlation_id: str, skill_name: str, result: dict) -> None:
        """Publish praxis.completed event. Subscribers: KNOW, DASHBOARD."""
        if self.event_bus is None:
            return
        self.event_bus.publish(Event(
            topic="praxis.completed",
            payload={
                "practice_id": correlation_id,
                "event_type": "praxis.completed",
                "timestamp": time.time(),
                "robot_id": self.robot_id,
                "outcome": {
                    "status": "success",
                    "skill_name": skill_name,
                    "reward": result.get("reward", 1.0),
                    "details": result,
                },
            },
            source="practice_recorder",
            priority=EventPriority.NORMAL,
        ))

    def _publish_praxis_failed(self, correlation_id: str, skill_name: str, result: dict) -> None:
        """Publish praxis.failed event with full context. Subscribers: HOW, MEMORY."""
        if self.event_bus is None:
            return
        self._failure_context["current_iteration"] += 1
        self._failure_context["previous_scores"].append(result.get("reward", -1.0))
        self._failure_context["last_error"] = result.get("error", "unknown error")
        self.event_bus.publish(Event(
            topic="praxis.failed",
            payload={
                "practice_id": correlation_id,
                "event_type": "praxis.failed",
                "timestamp": time.time(),
                "robot_id": self.robot_id,
                "outcome": {
                    "status": "failure",
                    "skill_name": skill_name,
                    "reward": result.get("reward", -1.0),
                },
                "error_log": self._failure_context["last_error"],
                "previous_scores": self._failure_context["previous_scores"].copy(),
                "current_iteration": self._failure_context["current_iteration"],
            },
            source="practice_recorder",
            priority=EventPriority.HIGH,
        ))

    def _on_knowledge_ingest_complete(self, event: Event) -> None:
        """Handle knowledge.ingest_complete from KNOW module."""
        payload = event.payload if isinstance(event.payload, dict) else {}
        log_entry = {
            "practice_id": payload.get("practice_id", "unknown"),
            "ingest_timestamp": event.timestamp,
            "knowledge_version": payload.get("knowledge_version", "unknown"),
            "status": payload.get("status", "unknown"),
        }
        self._knowledge_ingest_log.append(log_entry)

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
