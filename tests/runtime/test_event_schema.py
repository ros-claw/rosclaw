"""Tests for the RuntimeEvent schema (Milestone 1)."""

from __future__ import annotations

from datetime import UTC, datetime

from rosclaw.runtime.event import RuntimeEvent


def test_runtime_event_has_required_fields() -> None:
    event = RuntimeEvent(
        id="ev_1",
        timestamp=datetime.now(UTC),
        source="test_source",
        robot="realsense-d405",
        body_id="d405_lab_01",
        type="camera.rgbd_frame",
        payload={"width": 640, "height": 480},
        metadata={"trace_id": "trace_1"},
    )
    assert event.id == "ev_1"
    assert event.source == "test_source"
    assert event.type == "camera.rgbd_frame"
    assert event.topic == "rosclaw.camera.rgbd_frame"


def test_event_round_trip_through_event_bus_payload() -> None:
    original = RuntimeEvent(
        id="ev_2",
        timestamp=datetime(2026, 6, 30, 12, 0, 0, tzinfo=UTC),
        source="camera",
        type="camera.frame",
        payload={"foo": "bar"},
        metadata={"trace_id": "t2"},
    )
    payload = original.to_event_bus_payload()
    restored = RuntimeEvent.from_event_bus_payload(payload)
    assert restored.id == original.id
    assert restored.timestamp == original.timestamp
    assert restored.payload == original.payload
    assert restored.metadata == original.metadata


def test_event_type_derived_from_topic() -> None:
    payload = {
        "id": "ev_3",
        "timestamp": datetime.now(UTC).isoformat(),
        "source": "test",
        "type": "",
        "payload": {},
        "metadata": {},
    }
    restored = RuntimeEvent.from_event_bus_payload(payload, topic="rosclaw.skill.complete")
    assert restored.type == "skill.complete"


def test_event_accepts_string_timestamp() -> None:
    payload = {
        "id": "ev_4",
        "timestamp": "2026-06-30T12:00:00Z",
        "source": "test",
        "type": "camera.frame",
        "payload": {},
        "metadata": {},
    }
    restored = RuntimeEvent.from_event_bus_payload(payload)
    assert restored.timestamp.year == 2026
    assert restored.timestamp.hour == 12
