"""Contract tests: memory body event writer consumes body change metadata."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rosclaw.body.resolver import BodyResolver
from rosclaw.body.schema import BodyDiff
from rosclaw.body.service import BodyInstanceService
from rosclaw.memory.body_events import BodyMemoryEventWriter


@pytest.fixture
def linked_workspace(tmp_path: Path, monkeypatch) -> Path:
    workspace = tmp_path / ".rosclaw"
    monkeypatch.setenv("HOME", str(tmp_path))
    BodyInstanceService().create_or_init(
        robot="unitree-g1", name="g1-memory", mode="registry", update_registry=True, switch_active=True
    )
    return workspace


def test_memory_writer_records_body_change(linked_workspace: Path):
    resolver = BodyResolver()
    body = resolver.resolve("rosclaw://body/current/effective")

    mock_client = MagicMock()
    mock_client.insert.return_value = "record-123"
    writer = BodyMemoryEventWriter(memory_client=mock_client)

    diff = BodyDiff(changed_paths=["calibration"])
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
    call_args = mock_client.insert.call_args[0]
    assert call_args[0] == "body_events"
    assert call_args[1]["new_effective_body_hash"] == body.effective_body_hash
    assert call_args[1]["diff"]["changed_paths"] == ["calibration"]


def test_memory_writer_does_not_block_on_failure(linked_workspace: Path):
    resolver = BodyResolver()
    body = resolver.resolve("rosclaw://body/current/effective")

    mock_client = MagicMock()
    mock_client.insert.side_effect = RuntimeError("memory store unavailable")
    writer = BodyMemoryEventWriter(memory_client=mock_client)

    diff = BodyDiff()
    result = writer.write_body_change(
        body_instance_id=body.body_instance_id,
        old_hash="hash-old",
        new_hash=body.effective_body_hash,
        diff=diff,
        affected_skills=[],
    )

    assert result["recorded"] is False
    assert "memory store unavailable" in result["reason"]


def test_memory_writer_without_client_is_no_op(linked_workspace: Path):
    resolver = BodyResolver()
    body = resolver.resolve("rosclaw://body/current/effective")

    writer = BodyMemoryEventWriter(memory_client=None)
    diff = BodyDiff()
    result = writer.write_body_change(
        body_instance_id=body.body_instance_id,
        old_hash="hash-old",
        new_hash=body.effective_body_hash,
        diff=diff,
        affected_skills=[],
    )

    assert result["recorded"] is False
    assert result["reason"] == "memory_client_not_configured"
