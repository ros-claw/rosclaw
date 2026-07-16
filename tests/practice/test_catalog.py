"""Tests for PracticeCatalog."""

from __future__ import annotations

import tempfile
from pathlib import Path

from rosclaw.practice.storage.catalog import PracticeCatalog


def test_insert_and_get_practice():
    with tempfile.TemporaryDirectory() as tmp:
        db = PracticeCatalog(Path(tmp) / "catalog.sqlite", event_batch_size=1)
        db.insert_practice(
            {
                "practice_id": "prac_001",
                "robot_id": "r1",
                "robot_type": "humanoid",
                "task_id": "t1",
                "task_name": "walk",
                "skill_id": "s1",
                "start_time": "2024-01-01T00:00:00Z",
                "manifest_path": "/tmp/manifest.yaml",
                "events_jsonl_path": "/tmp/events.jsonl",
                "outcome": "running",
            }
        )
        record = db.get_practice("prac_001")
        assert record is not None
        assert record["robot_id"] == "r1"
        assert record["task_name"] == "walk"
        db.close()


def test_update_practice():
    with tempfile.TemporaryDirectory() as tmp:
        db = PracticeCatalog(Path(tmp) / "catalog.sqlite", event_batch_size=1)
        db.insert_practice(
            {
                "practice_id": "prac_002",
                "robot_id": "r1",
                "start_time": "2024-01-01T00:00:00Z",
                "outcome": "running",
            }
        )
        ok = db.update_practice("prac_002", {"outcome": "SUCCESS", "reward": 0.8})
        assert ok is True
        record = db.get_practice("prac_002")
        assert record["outcome"] == "SUCCESS"
        assert record["reward"] == 0.8
        db.close()


def test_list_practices_filters():
    with tempfile.TemporaryDirectory() as tmp:
        db = PracticeCatalog(Path(tmp) / "catalog.sqlite", event_batch_size=1)
        for i, robot in enumerate(["r1", "r1", "r2"]):
            db.insert_practice(
                {
                    "practice_id": f"prac_{i}",
                    "robot_id": robot,
                    "task_id": "t1",
                    "start_time": f"2024-01-0{i + 1}T00:00:00Z",
                    "outcome": "SUCCESS",
                }
            )
        results = db.list_practices(robot_id="r1")
        assert len(results) == 2
        results = db.list_practices(outcome="SUCCESS", limit=2)
        assert len(results) == 2
        db.close()


def test_insert_event():
    with tempfile.TemporaryDirectory() as tmp:
        db = PracticeCatalog(Path(tmp) / "catalog.sqlite", event_batch_size=1)
        db.insert_practice(
            {
                "practice_id": "prac_003",
                "robot_id": "r1",
                "start_time": "2024-01-01T00:00:00Z",
                "outcome": "running",
            }
        )
        db.insert_event(
            {
                "event_id": "evt_1",
                "practice_id": "prac_003",
                "source": "agent",
                "event_type": "agent.task_received",
                "timestamp_ns": 123,
                "timestamp_utc": "2024-01-01T00:00:00Z",
                "tags": "",
            }
        )
        with db._lock:
            rows = db._conn.execute(
                "SELECT * FROM events WHERE practice_id = ?", ("prac_003",)
            ).fetchall()
        assert len(rows) == 1
        db.close()
