"""UnifiedTimeline - Multi-channel timeline binding LLM CoT with sensorimotor data.

Records all events on a single timeline with nanosecond precision.
Assembles PraxisEvent from timeline entries on praxis.completed.

Sprint 4 of DESIGN_SPRINT3_5.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Any
import json
import time

import numpy as np

from rosclaw.core.event_bus import EventBus, Event, EventPriority
from rosclaw.core.lifecycle import LifecycleMixin
from rosclaw.core.types import RobotState, PraxisEvent


class TimelineChannel(Enum):
    """Channels on the unified timeline."""
    LLM_REASONING = "llm_reasoning"
    AGENT_COMMAND = "agent_command"
    FIREWALL_RESULT = "firewall_result"
    DRIVER_STATE = "driver_state"
    SENSORIMOTOR = "sensorimotor"
    PRAXIS_EVENT = "praxis_event"
    SKILL_EXECUTION = "skill_execution"
    SWARM_MESSAGE = "swarm_message"


@dataclass
class TimelineEntry:
    """Single entry on the unified timeline."""
    timestamp_ns: int
    channel: TimelineChannel
    sequence: int
    data: dict
    correlation_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "timestamp_ns": self.timestamp_ns,
            "channel": self.channel.value,
            "sequence": self.sequence,
            "data": self.data,
            "correlation_id": self.correlation_id,
        }


class UnifiedTimeline(LifecycleMixin):
    """
    Multi-channel timeline for practice recording.

    Subscribes to EventBus channels and records entries on a
    single timeline. Assembles PraxisEvent when practice session ends.

    Architecture:
        Layer 1: LLM Reasoning   (~1 Hz)
        Layer 2: Agent Commands   (~50 Hz)
        Layer 3: Sensorimotor    (~1000 Hz, direct recording bypasses EventBus)
        Layer 4: Events          (async)
    """

    def __init__(
        self,
        robot_id: str,
        event_bus: EventBus,
        output_dir: str = "./practice_data",
        enable_mcap: bool = False,
        buffer_size: int = 100_000,
    ):
        super().__init__()
        self._robot_id = robot_id
        self._event_bus = event_bus
        self._output_dir = Path(output_dir)
        self._enable_mcap = enable_mcap
        self._buffer_size = buffer_size

        self._entries: list[TimelineEntry] = []
        self._sequence_counters: dict[TimelineChannel, int] = {ch: 0 for ch in TimelineChannel}
        self._pending_praxis: dict[str, dict] = {}
        self._mcap_writer: Optional[Any] = None
        self._sensorimotor_buffer: list[TimelineEntry] = []
        self._sensorimotor_max = 10_000

    def _do_initialize(self) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._event_bus.subscribe("agent.command", self._on_agent_command)
        self._event_bus.subscribe("praxis.completed", self._on_praxis_completed)
        self._event_bus.subscribe("praxis.failed", self._on_praxis_failed)
        self._event_bus.subscribe("skill.execution.start", self._on_skill_event)
        self._event_bus.subscribe("skill.execution.complete", self._on_skill_event)
        self._event_bus.subscribe("swarm.message", self._on_swarm_message)

        if self._enable_mcap:
            try:
                from mcap.writer import Writer
                self._mcap_writer = True
            except ImportError:
                print("[UnifiedTimeline] mcap not available, using JSONL only")
                self._enable_mcap = False

        print(f"[UnifiedTimeline] Initialized for {self._robot_id}, "
              f"MCAP={'enabled' if self._enable_mcap else 'disabled'}, "
              f"buffer_size={self._buffer_size}")

    def _do_start(self) -> None:
        self._event_bus.publish(Event(
            topic="timeline.status",
            payload={"state": "running", "robot_id": self._robot_id},
            source="unified_timeline",
        ))

    def _do_stop(self) -> None:
        self._flush_pending_praxis()
        if self._mcap_writer:
            self._mcap_writer = None

    def _on_agent_command(self, event: Event) -> None:
        self._record(TimelineChannel.AGENT_COMMAND, event.payload,
                     correlation_id=event.metadata.get("request_id"))

    def _on_praxis_completed(self, event: Event) -> None:
        payload = event.payload
        correlation_id = payload.get("correlation_id", "unknown")

        related_entries = [e for e in self._entries if e.correlation_id == correlation_id]
        llm_entries = [e for e in related_entries if e.channel == TimelineChannel.LLM_REASONING]
        cmd_entries = [e for e in related_entries if e.channel == TimelineChannel.AGENT_COMMAND]
        sensor_entries = [e for e in self._sensorimotor_buffer
                         if e.correlation_id == correlation_id]

        cot_trace = []
        for e in llm_entries:
            steps = e.data.get("reasoning_steps", [])
            cot_trace.extend(steps)
        trajectory = [e.data.get("waypoint", []) for e in cmd_entries]

        praxis_event = PraxisEvent(
            event_id=payload.get("event_id", correlation_id),
            event_type="success",
            timestamp=time.time(),
            robot_id=self._robot_id,
            agent_instruction=payload.get("instruction", ""),
            cot_trace=cot_trace,
            initial_state=payload.get("initial_state"),
            final_state=payload.get("final_state"),
            trajectory=trajectory,
            mcap_path=payload.get("mcap_path"),
            error_details=None,
            duration_sec=payload.get("duration_sec", 0.0),
            metadata={
                "firewall_results": [e.data for e in related_entries
                                     if e.channel == TimelineChannel.FIREWALL_RESULT],
                "sensorimotor_count": len(sensor_entries),
                "timeline_entries": len(related_entries),
            },
        )

        self._record(TimelineChannel.PRAXIS_EVENT, {
            "event_id": praxis_event.event_id,
            "event_type": praxis_event.event_type,
            "trajectory_waypoints": len(praxis_event.trajectory),
            "cot_steps": len(praxis_event.cot_trace),
        }, correlation_id=correlation_id)

        self._event_bus.publish(Event(
            topic="praxis.recorded",
            payload={
                "event_id": praxis_event.event_id,
                "event_type": praxis_event.event_type,
                "robot_id": praxis_event.robot_id,
                "duration_sec": praxis_event.duration_sec,
                "trajectory_waypoints": len(praxis_event.trajectory),
                "cot_steps": len(praxis_event.cot_trace),
                "sensorimotor_samples": len(sensor_entries),
            },
            source="unified_timeline",
            priority=EventPriority.NORMAL,
        ))

        self._export_timeline(correlation_id, related_entries, sensor_entries)

    def _on_praxis_failed(self, event: Event) -> None:
        correlation_id = event.payload.get("correlation_id", "unknown")
        self._record(TimelineChannel.PRAXIS_EVENT, {
            "event_type": "failure",
            "error": event.payload.get("error", "unknown"),
        }, correlation_id=correlation_id)

    def _on_skill_event(self, event: Event) -> None:
        self._record(TimelineChannel.SKILL_EXECUTION, event.payload,
                     correlation_id=event.payload.get("skill_name"))

    def _on_swarm_message(self, event: Event) -> None:
        self._record(TimelineChannel.SWARM_MESSAGE, event.payload)

    def record_sensorimotor(
        self,
        joint_positions: list[float],
        joint_velocities: list[float],
        joint_torques: list[float],
        correlation_id: Optional[str] = None,
    ) -> None:
        """Record sensorimotor data directly (1kHz, bypasses EventBus)."""
        entry = TimelineEntry(
            timestamp_ns=time.time_ns(),
            channel=TimelineChannel.SENSORIMOTOR,
            sequence=self._sequence_counters[TimelineChannel.SENSORIMOTOR],
            data={
                "positions": joint_positions,
                "velocities": joint_velocities,
                "torques": joint_torques,
            },
            correlation_id=correlation_id,
        )
        self._sequence_counters[TimelineChannel.SENSORIMOTOR] += 1
        self._sensorimotor_buffer.append(entry)

        if len(self._sensorimotor_buffer) > self._sensorimotor_max:
            self._sensorimotor_buffer = self._sensorimotor_buffer[-self._sensorimotor_max:]

    def record_llm_reasoning(
        self,
        instruction: str,
        reasoning_steps: list[str],
        correlation_id: str,
    ) -> None:
        """Record LLM Chain-of-Thought on the timeline."""
        self._record(TimelineChannel.LLM_REASONING, {
            "instruction": instruction,
            "reasoning_steps": reasoning_steps,
        }, correlation_id=correlation_id)

    def record_agent_command(
        self,
        action: str,
        joint_positions: list[float],
        correlation_id: Optional[str] = None,
    ) -> None:
        """Record an agent command on the timeline.

        Args:
            action: Command action (e.g., "move_joints", "grasp")
            joint_positions: Joint position values for the command
            correlation_id: Optional correlation ID for grouping related entries
        """
        self._record(TimelineChannel.AGENT_COMMAND, {
            "action": action,
            "joint_positions": joint_positions,
        }, correlation_id=correlation_id)

    def export_session(self, correlation_id: str) -> Path:
        """Manually export a session's timeline and sensorimotor data.

        Args:
            correlation_id: Session identifier to export

        Returns:
            Path to the exported session directory

        Note:
            Sessions are normally auto-exported when praxis.completed fires.
            This method allows manual export for incomplete or in-progress sessions.
        """
        entries = [e for e in self._entries if e.correlation_id == correlation_id]
        sensor_entries = [e for e in self._sensorimotor_buffer if e.correlation_id == correlation_id]

        if not entries and not sensor_entries:
            raise ValueError(f"No timeline entries found for session '{correlation_id}'")

        self._export_timeline(correlation_id, entries, sensor_entries)
        return self._output_dir / f"session_{correlation_id}"

    def _record(
        self,
        channel: TimelineChannel,
        data: dict,
        correlation_id: Optional[str] = None,
    ) -> None:
        entry = TimelineEntry(
            timestamp_ns=time.time_ns(),
            channel=channel,
            sequence=self._sequence_counters[channel],
            data=data,
            correlation_id=correlation_id,
        )
        self._sequence_counters[channel] += 1
        self._entries.append(entry)

        if len(self._entries) > self._buffer_size:
            self._entries = self._entries[-self._buffer_size:]

    def _export_timeline(
        self,
        correlation_id: str,
        entries: list[TimelineEntry],
        sensor_entries: list[TimelineEntry],
    ) -> None:
        session_dir = self._output_dir / f"session_{correlation_id}"
        session_dir.mkdir(parents=True, exist_ok=True)

        jsonl_path = session_dir / "timeline.jsonl"
        with open(jsonl_path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry.to_dict(), default=str) + "\n")

        if sensor_entries:
            positions = np.array([e.data["positions"] for e in sensor_entries])
            velocities = np.array([e.data["velocities"] for e in sensor_entries])
            torques = np.array([e.data["torques"] for e in sensor_entries])
            timestamps = np.array([e.timestamp_ns for e in sensor_entries])

            np.savez_compressed(
                session_dir / "sensorimotor.npz",
                positions=positions,
                velocities=velocities,
                torques=torques,
                timestamps=timestamps,
            )

        print(f"[UnifiedTimeline] Exported {len(entries)} events + "
              f"{len(sensor_entries)} sensorimotor samples to {session_dir}")

    def record(self, event_type: str = "praxis", **kwargs: Any) -> None:
        """Convenience method to record a generic event on the timeline.

        Publishes a ``praxis.completed`` event so that Memory auto-ingests
        the experience and the timeline exports the session.
        """
        self._event_bus.publish(Event(
            topic="praxis.completed",
            payload={
                "event_id": kwargs.get("event_id", f"evt_{time.time_ns()}"),
                "event_type": event_type,
                "correlation_id": kwargs.get("correlation_id"),
                "instruction": kwargs.get("instruction", ""),
                "initial_state": kwargs.get("initial_state"),
                "final_state": kwargs.get("final_state"),
                "duration_sec": kwargs.get("duration_sec", 0.0),
            },
            source="unified_timeline",
            priority=EventPriority.NORMAL,
        ))

    def _flush_pending_praxis(self) -> None:
        for cid, state in self._pending_praxis.items():
            print(f"[UnifiedTimeline] Flushing incomplete praxis: {cid}")
        self._pending_praxis.clear()

    def get_entries(
        self,
        channel: Optional[TimelineChannel] = None,
        correlation_id: Optional[str] = None,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None,
    ) -> list[TimelineEntry]:
        result = self._entries
        if channel:
            result = [e for e in result if e.channel == channel]
        if correlation_id:
            result = [e for e in result if e.correlation_id == correlation_id]
        if start_ns:
            result = [e for e in result if e.timestamp_ns >= start_ns]
        if end_ns:
            result = [e for e in result if e.timestamp_ns <= end_ns]
        return result

    def get_summary(self) -> dict:
        channel_counts = {}
        for ch in TimelineChannel:
            count = sum(1 for e in self._entries if e.channel == ch)
            if count > 0:
                channel_counts[ch.value] = count
        return {
            "total_entries": len(self._entries),
            "sensorimotor_samples": len(self._sensorimotor_buffer),
            "channels": channel_counts,
            "buffer_size": self._buffer_size,
        }
