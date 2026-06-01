"""Tests for UnifiedTimeline (Sprint 4)."""

import numpy as np
from pathlib import Path

from rosclaw.core.event_bus import EventBus, Event
from rosclaw.practice.timeline import UnifiedTimeline, TimelineChannel


def test_multi_channel_recording():
    """Record entries on all channels, verify ordering."""
    bus = EventBus()
    timeline = UnifiedTimeline("test_robot", bus, output_dir="/tmp/test_timeline")
    timeline.initialize()

    timeline.record_llm_reasoning("pick up block", ["step1", "step2"], "corr1")
    timeline._record(TimelineChannel.AGENT_COMMAND, {"action": "move"}, "corr1")
    timeline._record(TimelineChannel.SKILL_EXECUTION, {"skill": "grasp"}, "corr1")

    entries = timeline.get_entries()
    assert len(entries) == 3
    assert entries[0].channel == TimelineChannel.LLM_REASONING
    assert entries[1].channel == TimelineChannel.AGENT_COMMAND
    assert entries[2].channel == TimelineChannel.SKILL_EXECUTION
    timeline.stop()


def test_sensorimotor_direct_recording():
    """record_sensorimotor() bypasses EventBus, stores at 1kHz."""
    bus = EventBus()
    timeline = UnifiedTimeline("test_robot", bus, output_dir="/tmp/test_timeline")
    timeline.initialize()

    for i in range(100):
        timeline.record_sensorimotor(
            joint_positions=[0.1] * 6,
            joint_velocities=[0.0] * 6,
            joint_torques=[0.5] * 6,
            correlation_id="test_session",
        )

    summary = timeline.get_summary()
    assert summary["sensorimotor_samples"] == 100
    assert summary["total_entries"] == 0  # Not counted in main entries
    timeline.stop()


def test_sensorimotor_ring_buffer():
    """Sensorimotor buffer evicts old entries."""
    bus = EventBus()
    timeline = UnifiedTimeline("test_robot", bus, output_dir="/tmp/test_timeline")
    timeline.initialize()

    for i in range(15_000):
        timeline.record_sensorimotor(
            joint_positions=[i * 0.001] * 6,
            joint_velocities=[0.0] * 6,
            joint_torques=[0.5] * 6,
            correlation_id="overflow_test",
        )

    assert len(timeline._sensorimotor_buffer) == 10_000
    timeline.stop()


def test_praxis_event_assembly():
    """praxis.completed -> assembled PraxisEvent with all channels."""
    bus = EventBus()
    timeline = UnifiedTimeline("test_robot", bus, output_dir="/tmp/test_timeline")
    timeline.initialize()

    recorded_events = []
    bus.subscribe("praxis.recorded", lambda e: recorded_events.append(e.payload))

    timeline.record_llm_reasoning("pick up red block", ["identify", "reach"], "sess1")
    timeline._record(TimelineChannel.AGENT_COMMAND, {"waypoint": [0.1] * 6}, "sess1")
    timeline.record_sensorimotor([0.1] * 6, [0.0] * 6, [0.5] * 6, "sess1")

    bus.publish(Event(
        topic="praxis.completed",
        payload={
            "correlation_id": "sess1",
            "event_id": "evt1",
            "instruction": "pick up red block",
            "duration_sec": 2.5,
        },
        source="test",
    ))

    assert len(recorded_events) == 1
    assert recorded_events[0]["event_id"] == "evt1"
    assert recorded_events[0]["cot_steps"] == 2
    assert recorded_events[0]["trajectory_waypoints"] == 1
    assert recorded_events[0]["sensorimotor_samples"] == 1
    timeline.stop()


def test_timeline_export():
    """Verify exported files contain correct data."""
    bus = EventBus()
    output_dir = "/tmp/test_timeline_export"
    timeline = UnifiedTimeline("test_robot", bus, output_dir=output_dir)
    timeline.initialize()

    timeline.record_llm_reasoning("test instruction", ["reason"], "export_test")
    timeline._record(TimelineChannel.AGENT_COMMAND, {"waypoint": [0.2] * 6}, "export_test")
    timeline.record_sensorimotor([0.2] * 6, [0.0] * 6, [0.1] * 6, "export_test")

    bus.publish(Event(
        topic="praxis.completed",
        payload={
            "correlation_id": "export_test",
            "event_id": "evt_export",
            "instruction": "test instruction",
            "duration_sec": 1.0,
        },
        source="test",
    ))

    session_dir = Path(output_dir) / "session_export_test"
    assert (session_dir / "timeline.jsonl").exists()
    assert (session_dir / "sensorimotor.npz").exists()

    npz = np.load(session_dir / "sensorimotor.npz")
    assert "positions" in npz
    assert "velocities" in npz
    assert "torques" in npz
    assert len(npz["positions"]) == 1
    timeline.stop()


def test_entry_filtering():
    """Query timeline entries with filters."""
    bus = EventBus()
    timeline = UnifiedTimeline("test_robot", bus, output_dir="/tmp/test_timeline")
    timeline.initialize()

    timeline._record(TimelineChannel.AGENT_COMMAND, {"a": 1}, "c1")
    timeline._record(TimelineChannel.AGENT_COMMAND, {"a": 2}, "c2")
    timeline._record(TimelineChannel.SKILL_EXECUTION, {"s": 1}, "c1")

    cmd_entries = timeline.get_entries(channel=TimelineChannel.AGENT_COMMAND)
    assert len(cmd_entries) == 2

    c1_entries = timeline.get_entries(correlation_id="c1")
    assert len(c1_entries) == 2
    timeline.stop()


def test_buffer_eviction():
    """Main buffer evicts old entries when over capacity."""
    bus = EventBus()
    timeline = UnifiedTimeline("test_robot", bus, output_dir="/tmp/test_timeline", buffer_size=50)
    timeline.initialize()

    for i in range(100):
        timeline._record(TimelineChannel.AGENT_COMMAND, {"idx": i})

    assert len(timeline._entries) == 50
    assert timeline._entries[0].data["idx"] == 50
    timeline.stop()
