"""Tests for praxis.completed and praxis.failed event publishing.

Task 1: Publish praxis.completed event after successful execution.
"""

from rosclaw.core.event_bus import Event, EventBus
from rosclaw.practice.recorder import PracticeRecorder


def test_praxis_completed_event_published():
    """Successful skill execution publishes praxis.completed event."""
    bus = EventBus()
    recorder = PracticeRecorder("ur5e_01", joint_dof=6, event_bus=bus)
    recorder.initialize()
    recorder.start_recording()

    captured = []
    bus.subscribe("praxis.completed", lambda e: captured.append(e))

    bus.publish(
        Event(
            topic="skill.execution.complete",
            payload={
                "skill_name": "pick_red_cup",
                "result": {"status": "success", "reward": 1.0},
                "correlation_id": "prac_001",
            },
        )
    )

    assert len(captured) == 1
    evt = captured[0]
    assert evt.topic == "praxis.completed"
    assert evt.payload["practice_id"] == "prac_001"
    assert evt.payload["event_type"] == "praxis.completed"
    assert evt.payload["robot_id"] == "ur5e_01"
    assert evt.payload["outcome"]["status"] == "success"
    assert evt.payload["outcome"]["reward"] == 1.0
    assert "timestamp" in evt.payload

    recorder.stop()


def test_praxis_completed_without_eventbus():
    """Recorder without EventBus does not crash on skill complete."""
    recorder = PracticeRecorder("ur5e_01", joint_dof=6, event_bus=None)
    recorder.initialize()
    recorder.start_recording()

    recorder._on_skill_complete(
        Event(
            topic="skill.execution.complete",
            payload={
                "skill_name": "test",
                "result": {"status": "success"},
            },
        )
    )

    recorder.stop()


def test_praxis_completed_event_priority():
    """praxis.completed uses NORMAL priority."""
    from rosclaw.core.event_bus import EventPriority

    bus = EventBus()
    recorder = PracticeRecorder("ur5e_01", joint_dof=6, event_bus=bus)
    recorder.initialize()
    recorder.start_recording()

    captured = []
    bus.subscribe("praxis.completed", lambda e: captured.append(e))

    bus.publish(
        Event(
            topic="skill.execution.complete",
            payload={
                "skill_name": "test",
                "result": {"status": "success"},
                "correlation_id": "prac_002",
            },
        )
    )

    assert captured[0].priority == EventPriority.NORMAL
    recorder.stop()


def test_praxis_completed_falls_back_to_episode_id():
    """Skill events without correlation_id keep their episode identity."""
    bus = EventBus()
    recorder = PracticeRecorder("ur5e_01", joint_dof=6, event_bus=bus)
    recorder.initialize()

    captured = []
    bus.subscribe("praxis.completed", lambda event: captured.append(event.payload))
    bus.publish(
        Event(
            topic="skill.execution.complete",
            payload={
                "episode_id": "ep_reach_001",
                "skill_name": "reach",
                "result": {"status": "success", "reward": 0.92},
            },
        )
    )

    assert captured[0]["practice_id"] == "ep_reach_001"
    assert captured[0]["episode_id"] == "ep_reach_001"
    assert captured[0]["correlation_id"] == "ep_reach_001"
    recorder.stop()


def test_praxis_completed_payload_structure():
    """praxis.completed payload has all required fields for KNOW/DASHBOARD."""
    bus = EventBus()
    recorder = PracticeRecorder("ur5e_01", joint_dof=6, event_bus=bus)
    recorder.initialize()
    recorder.start_recording()

    captured = []
    bus.subscribe("praxis.completed", lambda e: captured.append(e.payload))

    bus.publish(
        Event(
            topic="skill.execution.complete",
            payload={
                "skill_name": "insert_usb",
                "result": {"status": "success", "reward": 0.95, "details": {"force": 2.3}},
                "correlation_id": "prac_003",
            },
        )
    )

    payload = captured[0]
    assert "practice_id" in payload
    assert "event_type" in payload
    assert "timestamp" in payload
    assert "robot_id" in payload
    assert "outcome" in payload
    assert payload["outcome"]["skill_name"] == "insert_usb"
    assert payload["outcome"]["details"]["details"]["force"] == 2.3

    recorder.stop()
