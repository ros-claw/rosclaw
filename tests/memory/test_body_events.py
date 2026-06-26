"""Dedicated tests for BodyMemoryEventWriter."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rosclaw.body.diff import BodyDiff
from rosclaw.body.resolver import BodyResolver
from rosclaw.body.schema import BodyChange
from rosclaw.body.service import BodyInstanceService
from rosclaw.memory.body_events import BodyMemoryEventWriter


@pytest.fixture
def linked_workspace(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    BodyInstanceService(workspace=tmp_path).create_or_init(robot="unitree-g1", name="g1-memory", mode="single")
    return tmp_path


def test_write_body_change_stores_event(linked_workspace: Path):
    resolver = BodyResolver(workspace=linked_workspace)
    body = resolver.resolve("rosclaw://body/current/effective")

    mock_client = MagicMock()
    mock_client.insert.return_value = "record-123"
    writer = BodyMemoryEventWriter(memory_client=mock_client)

    diff = BodyDiff(changed_paths=["calibration"], affected_ids=["camera"])
    result = writer.write_body_change(
        body_instance_id=body.body_instance_id,
        old_hash="hash-old",
        new_hash=body.effective_body_hash,
        diff=diff,
        affected_skills=["walk"],
    )

    assert result["recorded"] is True
    assert result["record_id"] == "record-123"
    mock_client.insert.assert_called_once()
    table, event = mock_client.insert.call_args[0]
    assert table == "body_events"
    assert event["event_type"] == "body_changed"
    assert event["old_effective_body_hash"] == "hash-old"
    assert event["new_effective_body_hash"] == body.effective_body_hash
    assert event["affected_skills"] == ["walk"]
    assert event["diff"]["changed_paths"] == ["calibration"]


def test_write_body_change_non_blocking(linked_workspace: Path):
    resolver = BodyResolver(workspace=linked_workspace)
    body = resolver.resolve("rosclaw://body/current/effective")

    mock_client = MagicMock()
    mock_client.insert.side_effect = RuntimeError("memory down")
    writer = BodyMemoryEventWriter(memory_client=mock_client)

    result = writer.write_body_change(
        body_instance_id=body.body_instance_id,
        old_hash="hash-old",
        new_hash=body.effective_body_hash,
        diff=BodyDiff(),
        affected_skills=[],
    )

    assert result["recorded"] is False
    assert "memory down" in result["reason"]


def test_write_body_change_no_client_is_no_op(linked_workspace: Path):
    resolver = BodyResolver(workspace=linked_workspace)
    body = resolver.resolve("rosclaw://body/current/effective")

    writer = BodyMemoryEventWriter(memory_client=None)
    result = writer.write_body_change(
        body_instance_id=body.body_instance_id,
        old_hash="hash-old",
        new_hash=body.effective_body_hash,
        diff=BodyDiff(),
        affected_skills=[],
    )

    assert result["recorded"] is False
    assert result["reason"] == "memory_client_not_configured"


def test_write_event_accepts_known_event_types(linked_workspace: Path):
    resolver = BodyResolver(workspace=linked_workspace)
    body = resolver.resolve("rosclaw://body/current/effective")

    mock_client = MagicMock()
    mock_client.insert.return_value = "record-456"
    writer = BodyMemoryEventWriter(memory_client=mock_client)

    for event_type in BodyMemoryEventWriter.EVENT_TYPES:
        result = writer.write_event(
            event_type=event_type,
            body_instance_id=body.body_instance_id,
            payload={"note": "test"},
        )
        assert result["recorded"] is True
        assert result["event"]["event_type"] == event_type


def test_write_event_rejects_unknown_event_type(linked_workspace: Path):
    resolver = BodyResolver(workspace=linked_workspace)
    body = resolver.resolve("rosclaw://body/current/effective")

    writer = BodyMemoryEventWriter(memory_client=MagicMock())
    with pytest.raises(ValueError, match="Unknown body event type"):
        writer.write_event(
            event_type="unknown_event",
            body_instance_id=body.body_instance_id,
        )


def test_write_body_change_with_real_changes(linked_workspace: Path):
    resolver = BodyResolver(workspace=linked_workspace)
    body = resolver.resolve("rosclaw://body/current/effective")

    mock_client = MagicMock()
    mock_client.insert.return_value = "record-789"
    writer = BodyMemoryEventWriter(memory_client=mock_client)

    diff = BodyDiff(
        changes=[
            BodyChange(
                path="installed_components.sensors.head_rgb_camera.status",
                old="available",
                new="unavailable",
                category="sensor_status",
                severity="warning",
                requires_skill_recheck=True,
            )
        ],
        affected_ids=["head_rgb_camera"],
        affected_categories=["sensor_status"],
    )
    result = writer.write_body_change(
        body_instance_id=body.body_instance_id,
        old_hash="hash-old",
        new_hash=body.effective_body_hash,
        diff=diff,
        affected_skills=["camera_nav"],
    )

    assert result["recorded"] is True
    event = result["event"]
    assert event["diff"]["changed_paths"] == ["installed_components.sensors.head_rgb_camera.status"]
    assert event["diff"]["affected_ids"] == ["head_rgb_camera"]
