"""Bounded persistence tests for the runtime JSONL event sink."""

from __future__ import annotations

import json

from rosclaw.core.event_sink import JsonlEventSink


def test_event_sink_summarizes_oversized_record(tmp_path) -> None:
    sink = JsonlEventSink(home=tmp_path, max_record_mb=0.001)
    try:
        sink._write(
            {
                "topic": "test.oversized",
                "metadata": {"nested": "y" * 10_000},
                "payload": {"blob": "x" * 10_000, "status": "blocked"},
            }
        )
    finally:
        sink.close()

    path = tmp_path / "events" / "live.jsonl"
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["topic"] == "test.oversized"
    assert persisted["metadata"]["persistence_truncated"] is True
    assert persisted["payload"]["keys"] == ["blob", "status"]
    assert path.stat().st_size < 1024


def test_event_sink_rotates_before_crossing_limit(tmp_path) -> None:
    sink = JsonlEventSink(home=tmp_path, rotate_mb=0.001, max_record_mb=1.0)
    try:
        sink._write({"topic": "first", "payload": {"blob": "x" * 700}})
        sink._write({"topic": "second", "payload": {"blob": "y" * 700}})
    finally:
        sink.close()

    events = tmp_path / "events"
    assert (events / "live.jsonl.001").stat().st_size < 1024
    assert (events / "live.jsonl").stat().st_size < 1024
