"""Tests for knowledge.ingest_complete subscription.

Task 3: Practice subscribes to knowledge.ingest_complete and logs KNOW processing.
"""

import time

from rosclaw.core.event_bus import Event, EventBus
from rosclaw.practice.recorder import PracticeRecorder


def test_knowledge_ingest_complete_logged():
    """KNOW publishes knowledge.ingest_complete, Practice logs it."""
    bus = EventBus()
    recorder = PracticeRecorder("ur5e_01", joint_dof=6, event_bus=bus)
    recorder.initialize()

    bus.publish(Event(
        topic="knowledge.ingest_complete",
        payload={
            "practice_id": "prac_123",
            "knowledge_version": "v1.2.3",
            "status": "success",
        },
    ))

    log = recorder.knowledge_ingest_log
    assert len(log) == 1
    assert log[0]["practice_id"] == "prac_123"
    assert log[0]["knowledge_version"] == "v1.2.3"
    assert log[0]["status"] == "success"

    recorder.stop()


def test_knowledge_ingest_multiple_events():
    """Multiple ingest events are all logged."""
    bus = EventBus()
    recorder = PracticeRecorder("ur5e_01", joint_dof=6, event_bus=bus)
    recorder.initialize()

    for i in range(3):
        bus.publish(Event(
            topic="knowledge.ingest_complete",
            payload={
                "practice_id": f"prac_{i}",
                "knowledge_version": "v1.0",
                "status": "success",
            },
        ))

    log = recorder.knowledge_ingest_log
    assert len(log) == 3
    assert log[0]["practice_id"] == "prac_0"
    assert log[2]["practice_id"] == "prac_2"

    recorder.stop()


def test_knowledge_ingest_without_eventbus():
    """Recorder without EventBus initializes without error."""
    recorder = PracticeRecorder("ur5e_01", joint_dof=6, event_bus=None)
    recorder.initialize()
    assert len(recorder.knowledge_ingest_log) == 0
    recorder.stop()


def test_knowledge_ingest_default_values():
    """Payload with missing fields uses defaults."""
    bus = EventBus()
    recorder = PracticeRecorder("ur5e_01", joint_dof=6, event_bus=bus)
    recorder.initialize()

    bus.publish(Event(
        topic="knowledge.ingest_complete",
        payload={},  # Empty payload
    ))

    log = recorder.knowledge_ingest_log
    assert len(log) == 1
    assert log[0]["practice_id"] == "unknown"
    assert log[0]["knowledge_version"] == "unknown"
    assert log[0]["status"] == "unknown"

    recorder.stop()


def test_knowledge_ingest_timestamp_captured():
    """Event timestamp is captured in the log."""
    bus = EventBus()
    recorder = PracticeRecorder("ur5e_01", joint_dof=6, event_bus=bus)
    recorder.initialize()

    before = time.time()
    bus.publish(Event(
        topic="knowledge.ingest_complete",
        payload={"practice_id": "prac_ts"},
    ))
    after = time.time()

    log = recorder.knowledge_ingest_log
    assert len(log) == 1
    assert before <= log[0]["ingest_timestamp"] <= after

    recorder.stop()
