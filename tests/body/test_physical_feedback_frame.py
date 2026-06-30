"""Tests for generic PhysicalFeedbackFrame."""
from __future__ import annotations

from rosclaw.body.physical_feedback_frame import PhysicalFeedbackFrame


def test_roundtrip_dict():
    frame = PhysicalFeedbackFrame(
        frame_id="f1",
        body_id="body_1",
        timestamp=1.0,
        target={"thumb": 420.0},
        actual={"thumb": 418.0},
        force_net={"thumb": 100.0},
        primary_event="desired_contact",
        secondary_tags=["stable"],
    )
    restored = PhysicalFeedbackFrame.from_dict(frame.to_dict())
    assert restored.frame_id == "f1"
    assert restored.force_net["thumb"] == 100.0
    assert restored.primary_event == "desired_contact"


def test_default_primary_event_is_unknown():
    frame = PhysicalFeedbackFrame(timestamp=0.0)
    assert frame.primary_event == "unknown"
    assert frame.secondary_tags == []


def test_dof_names_union():
    frame = PhysicalFeedbackFrame(
        target={"thumb": 0.0, "index": 0.0},
        force_net={"index": 5.0, "middle": 1.0},
    )
    assert frame.dof_names() == ["index", "middle", "thumb"]


def test_serial_timeout_flag():
    frame = PhysicalFeedbackFrame(serial_timeout=True)
    assert frame.to_dict()["serial_timeout"] is True
