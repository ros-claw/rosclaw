"""Practice Recorder - Timeline Grounding

Records robot execution traces using the DataFlywheel.
All communication goes through EventBus — no direct module calls.
"""

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
    """

    def __init__(self, robot_id: str, joint_dof: int = 6, event_bus: Optional[EventBus] = None):
        super().__init__()
        self.robot_id = robot_id
        self.joint_dof = joint_dof
        self.event_bus = event_bus
        self._flywheel: Optional[DataFlywheel] = None
        self._recording = False

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
        print(f"[Practice] Recorder initialized for {self.robot_id}")

    def _do_stop(self) -> None:
        if self.event_bus is not None:
            self.event_bus.unsubscribe("agent.command", self._on_agent_command)
            self.event_bus.unsubscribe("skill.execution.start", self._on_skill_start)
            self.event_bus.unsubscribe("skill.execution.complete", self._on_skill_complete)

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
        """Mark skill execution completion."""
        if self._recording and self._flywheel is not None:
            payload = event.payload if isinstance(event.payload, dict) else {}
            result = payload.get("result", {})
            status = result.get("status", "unknown")
            event_type = EventType.SUCCESS if status == "success" else EventType.FAILURE
            self._flywheel.trigger_event(
                event_type,
                {"skill_name": payload.get("skill_name"), "phase": "complete", "result": result},
            )

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

    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._recording
