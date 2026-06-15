"""Tests for praxis.failed event publishing with full context.

Task 2: Publish praxis.failed event with error_log, previous_scores, current_iteration.
"""

from rosclaw.core.event_bus import Event, EventBus, EventPriority
from rosclaw.practice.recorder import PracticeRecorder


def test_praxis_failed_event_published():
    """Failed skill execution publishes praxis.failed event."""
    bus = EventBus()
    recorder = PracticeRecorder("ur5e_01", joint_dof=6, event_bus=bus)
    recorder.initialize()
    recorder.start_recording()

    captured = []
    bus.subscribe("praxis.failed", lambda e: captured.append(e))

    bus.publish(Event(
        topic="skill.execution.complete",
        payload={
            "skill_name": "pick_red_cup",
            "result": {"status": "failure", "reward": -1.0, "error": "gripper slip"},
            "correlation_id": "prac_fail_001",
        },
    ))

    assert len(captured) == 1
    evt = captured[0]
    assert evt.topic == "praxis.failed"
    assert evt.payload["practice_id"] == "prac_fail_001"
    assert evt.payload["event_type"] == "praxis.failed"
    assert evt.payload["robot_id"] == "ur5e_01"
    assert evt.payload["outcome"]["status"] == "failure"
    assert evt.payload["error_log"] == "gripper slip"

    recorder.stop()


def test_praxis_failed_priority_is_high():
    """praxis.failed uses HIGH priority for fast HOW response."""
    bus = EventBus()
    recorder = PracticeRecorder("ur5e_01", joint_dof=6, event_bus=bus)
    recorder.initialize()
    recorder.start_recording()

    captured = []
    bus.subscribe("praxis.failed", lambda e: captured.append(e))

    bus.publish(Event(
        topic="skill.execution.complete",
        payload={
            "skill_name": "test",
            "result": {"status": "failure", "error": "motor fault"},
            "correlation_id": "prac_fail_002",
        },
    ))

    assert captured[0].priority == EventPriority.HIGH
    recorder.stop()


def test_praxis_failed_tracks_iteration():
    """Each failure increments current_iteration counter."""
    bus = EventBus()
    recorder = PracticeRecorder("ur5e_01", joint_dof=6, event_bus=bus)
    recorder.initialize()
    recorder.start_recording()

    captured = []
    bus.subscribe("praxis.failed", lambda e: captured.append(e.payload))

    for i in range(3):
        bus.publish(Event(
            topic="skill.execution.complete",
            payload={
                "skill_name": "pick",
                "result": {"status": "failure", "reward": -0.5, "error": f"fail_{i}"},
                "correlation_id": f"prac_{i}",
            },
        ))

    assert len(captured) == 3
    assert captured[0]["current_iteration"] == 1
    assert captured[1]["current_iteration"] == 2
    assert captured[2]["current_iteration"] == 3

    recorder.stop()


def test_praxis_failed_tracks_previous_scores():
    """previous_scores accumulates across failures."""
    bus = EventBus()
    recorder = PracticeRecorder("ur5e_01", joint_dof=6, event_bus=bus)
    recorder.initialize()
    recorder.start_recording()

    captured = []
    bus.subscribe("praxis.failed", lambda e: captured.append(e.payload))

    for reward in [-0.5, -0.8, -1.0]:
        bus.publish(Event(
            topic="skill.execution.complete",
            payload={
                "skill_name": "pick",
                "result": {"status": "failure", "reward": reward, "error": "slip"},
                "correlation_id": "prac_scores",
            },
        ))

    assert captured[0]["previous_scores"] == [-0.5]
    assert captured[1]["previous_scores"] == [-0.5, -0.8]
    assert captured[2]["previous_scores"] == [-0.5, -0.8, -1.0]

    recorder.stop()


def test_praxis_failed_context_exposed():
    """failure_context property exposes current state for inspection."""
    bus = EventBus()
    recorder = PracticeRecorder("ur5e_01", joint_dof=6, event_bus=bus)
    recorder.initialize()
    recorder.start_recording()

    bus.publish(Event(
        topic="skill.execution.complete",
        payload={
            "skill_name": "pick",
            "result": {"status": "failure", "reward": -0.3, "error": "collision"},
            "correlation_id": "prac_ctx",
        },
    ))

    ctx = recorder.failure_context
    assert ctx["current_iteration"] == 1
    assert ctx["previous_scores"] == [-0.3]
    assert ctx["last_error"] == "collision"

    recorder.stop()


def test_praxis_failed_payload_full_context():
    """praxis.failed includes all context required by HOW and MEMORY."""
    bus = EventBus()
    recorder = PracticeRecorder("ur5e_01", joint_dof=6, event_bus=bus)
    recorder.initialize()
    recorder.start_recording()

    captured = []
    bus.subscribe("praxis.failed", lambda e: captured.append(e.payload))

    bus.publish(Event(
        topic="skill.execution.complete",
        payload={
            "skill_name": "grasp",
            "result": {"status": "failure", "reward": -0.9, "error": "force exceeded"},
            "correlation_id": "prac_full",
        },
    ))

    p = captured[0]
    assert "practice_id" in p
    assert "event_type" in p
    assert "timestamp" in p
    assert "robot_id" in p
    assert "outcome" in p
    assert "error_log" in p
    assert "previous_scores" in p
    assert "current_iteration" in p
    assert p["error_log"] == "force exceeded"

    recorder.stop()
