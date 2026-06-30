"""Tests for provider trace adapter file linkage."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rosclaw.core.event_bus import Event, EventBus
from rosclaw.practice.adapters.provider_trace_adapter import ProviderTraceAdapter


def test_provider_event_writes_files_and_references_them(tmp_path: Path):
    """Provider inference event should write request/response JSONL and emit reference."""
    bus = EventBus()
    adapter = ProviderTraceAdapter(robot_id="r1", event_bus=bus, output_root=str(tmp_path))
    session = MagicMock(practice_id="prac_test", session_dir=str(tmp_path))
    adapter.start(session)

    bus.publish(
        Event(
            topic="rosclaw.provider.inference.completed",
            payload={
                "request_id": "req_1",
                "provider_id": "openai",
                "model": "gpt-4o",
                "input_summary": {"messages": 3},
                "output_summary": {"text": "hello"},
                "latency_ms": 120.0,
                "token_usage": {"prompt": 10, "completion": 5},
            },
            source="provider",
        )
    )

    events = list(adapter.poll())
    adapter.stop()

    assert len(events) == 1
    ev = events[0]
    assert ev.source == "provider"
    assert ev.event_type == "provider.inference"
    assert "requests_ref" in ev.payload_ref
    assert "responses_ref" in ev.payload_ref

    req_path = tmp_path / ev.payload_ref["requests_ref"]
    resp_path = tmp_path / ev.payload_ref["responses_ref"]
    assert req_path.exists()
    assert resp_path.exists()

    req_lines = req_path.read_text().strip().split("\n")
    resp_lines = resp_path.read_text().strip().split("\n")
    assert len(req_lines) == 1
    assert len(resp_lines) == 1

    req = json.loads(req_lines[0])
    resp = json.loads(resp_lines[0])
    assert req["request_id"] == "req_1"
    assert resp["status"] == "success"
    assert resp["latency_ms"] == 120.0


def test_provider_failed_event_status(tmp_path: Path):
    """Failed provider event should write response with status failed."""
    bus = EventBus()
    adapter = ProviderTraceAdapter(robot_id="r1", event_bus=bus, output_root=str(tmp_path))
    session = MagicMock(practice_id="prac_test", session_dir=str(tmp_path))
    adapter.start(session)

    bus.publish(
        Event(
            topic="rosclaw.provider.inference.failed",
            payload={
                "request_id": "req_2",
                "provider_id": "openai",
                "error": "timeout",
                "latency_ms": 5000.0,
            },
            source="provider",
        )
    )

    events = list(adapter.poll())
    adapter.stop()

    assert len(events) == 1
    resp_path = tmp_path / events[0].payload_ref["responses_ref"]
    resp = json.loads(resp_path.read_text().strip().split("\n")[0])
    assert resp["status"] == "failed"
    assert resp["error"] == "timeout"
