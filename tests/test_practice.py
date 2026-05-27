"""Tests for Practice module."""

from rosclaw.practice.recorder import PracticeRecorder
from rosclaw.data.flywheel import EventType


def test_practice_lifecycle():
    rec = PracticeRecorder("test_bot", joint_dof=6)
    rec.initialize()
    assert rec.is_ready
    rec.start_recording()
    assert rec.is_recording is True
    rec.stop_recording()
    assert rec.is_recording is False
    rec.stop()


def test_practice_mark_event():
    rec = PracticeRecorder("test_bot", joint_dof=6)
    rec.initialize()
    rec.start_recording()
    event_id = rec.mark_event(EventType.SUCCESS, {"task": "test"})
    assert event_id != ""
    rec.stop()


def test_practice_record_praxis_event():
    rec = PracticeRecorder("test_bot", joint_dof=6)
    rec.initialize()
    rec.start_recording()
    event_id = rec.record_praxis_event(
        event_id="evt1",
        event_type="success",
        instruction="pick up block",
        metadata={"outcome": "ok"},
    )
    assert event_id != ""
    rec.stop()


def test_practice_record_praxis_event_not_recording():
    rec = PracticeRecorder("test_bot", joint_dof=6)
    rec.initialize()
    # Not started recording
    event_id = rec.record_praxis_event(
        event_id="evt2", event_type="milestone", instruction="test"
    )
    assert event_id == ""
