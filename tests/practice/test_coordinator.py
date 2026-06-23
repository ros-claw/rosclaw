"""Tests for PracticeCoordinator."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from rosclaw.core.event_bus import EventBus
from rosclaw.practice.config import PracticeConfig, SourceConfig
from rosclaw.practice.coordinator import PracticeCoordinator


def test_coordinator_mock_session_writes_manifest_and_jsonl():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = PracticeConfig(
            robot_id="test_bot",
            task_name="pick cup",
            data_root=tmp,
            sources=SourceConfig(agent=True, runtime=True),
            mock=True,
            publish_to_event_bus=False,
        )
        coord = PracticeCoordinator(cfg)
        coord.initialize()
        coord.start()
        time.sleep(0.4)
        coord.stop()

        summary = coord.summary
        assert summary is not None
        assert summary.outcome == "SUCCESS"
        assert summary.event_count >= 7

        session_dir = Path(tmp) / "sessions" / summary.practice_id
        assert (session_dir / "manifest.yaml").exists()
        assert (session_dir / "raw" / "events.jsonl").exists()

        events_jsonl = session_dir / "raw" / "events.jsonl"
        lines = events_jsonl.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == summary.event_count
        first = json.loads(lines[0])
        assert first["schema_version"] == "practice.event.v1"
        assert first["practice_id"] == summary.practice_id


def test_coordinator_publishes_session_events_on_event_bus():
    with tempfile.TemporaryDirectory() as tmp:
        bus = EventBus()
        received = []
        bus.subscribe("practice.session_started", lambda e: received.append(("started", e.payload)))
        bus.subscribe("practice.session_finished", lambda e: received.append(("finished", e.payload)))

        cfg = PracticeConfig(
            robot_id="test_bot",
            task_name="pick cup",
            data_root=tmp,
            sources=SourceConfig(agent=True),
            mock=True,
            event_bus=bus,
        )
        coord = PracticeCoordinator(cfg)
        coord.initialize()
        coord.start()
        time.sleep(0.2)
        coord.stop()

        assert any(t == "started" for t, _ in received)
        assert any(t == "finished" for t, _ in received)


def test_coordinator_catalog_indexed():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = PracticeConfig(
            robot_id="test_bot",
            task_name="pick cup",
            data_root=tmp,
            sources=SourceConfig(agent=True),
            mock=True,
            publish_to_event_bus=False,
        )
        coord = PracticeCoordinator(cfg)
        coord.initialize()
        coord.start()
        time.sleep(0.2)
        coord.stop()

        record = coord.catalog.get_practice(coord.summary.practice_id)
        assert record is not None
        assert record["outcome"] == "SUCCESS"
        # event_count is tracked in the summary, not the practices table
        assert coord.summary.event_count > 0
        assert coord.catalog.count_events(coord.summary.practice_id) == coord.summary.event_count
