"""Tests for PracticeCatalog v2 tables and migration behavior."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from rosclaw.practice.storage.catalog import PracticeCatalog


@pytest.fixture
def catalog(tmp_path: Path):
    return PracticeCatalog(tmp_path / "catalog.sqlite", event_batch_size=1)


def test_v2_tables_are_created(catalog: PracticeCatalog):
    with catalog._lock:
        tables = {
            row[0]
            for row in catalog._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert "practice_sessions" in tables
    assert "practice_episodes" in tables
    assert "practice_artifacts" in tables
    assert "practice_event_index" in tables
    # Legacy tables still exist
    assert "practices" in tables
    assert "events" in tables


def test_insert_and_get_session(catalog: PracticeCatalog):
    catalog.insert_session(
        {
            "session_id": "sess_1",
            "practice_id": "prac_1",
            "body_id": "body_rh56_left",
            "task_name": "rh56_ok_search",
            "started_at": "2026-07-07T00:00:00Z",
            "status": "running",
            "metadata": {"note": "initial run"},
        }
    )
    row = catalog.get_session("sess_1")
    assert row["body_id"] == "body_rh56_left"
    assert row["metadata_json"]["note"] == "initial run"


def test_list_sessions_by_body(catalog: PracticeCatalog):
    catalog.insert_session({"session_id": "s1", "body_id": "body_a", "status": "running"})
    catalog.insert_session({"session_id": "s2", "body_id": "body_b", "status": "running"})
    catalog.insert_session({"session_id": "s3", "body_id": "body_a", "status": "closed"})
    rows = catalog.list_sessions(body_id="body_a")
    assert len(rows) == 2


def test_insert_and_get_episode(catalog: PracticeCatalog):
    catalog.insert_episode(
        {
            "episode_id": "ep_1",
            "session_id": "sess_1",
            "body_id": "body_rh56_left",
            "skill_id": "skill_ok_contact",
            "started_at": "2026-07-07T00:00:00Z",
            "outcome": "success",
            "success": True,
            "failure_labels": ["no_contact"],
            "metrics": {"contact_rate": 0.9},
        }
    )
    row = catalog.get_episode("ep_1")
    assert row["outcome"] == "success"
    assert row["success"] is True
    assert row["failure_labels_json"] == ["no_contact"]
    assert row["metrics_json"]["contact_rate"] == 0.9


def test_update_episode(catalog: PracticeCatalog):
    catalog.insert_episode({"episode_id": "ep_1", "session_id": "sess_1", "outcome": "running"})
    catalog.update_episode("ep_1", {"outcome": "success", "success": True})
    row = catalog.get_episode("ep_1")
    assert row["outcome"] == "success"
    assert row["success"] is True


def test_insert_and_list_artifacts_v2(catalog: PracticeCatalog):
    catalog.insert_artifact_v2(
        {
            "artifact_id": "art_1",
            "session_id": "sess_1",
            "episode_id": "ep_1",
            "artifact_type": "events",
            "path": "/data/sessions/sess_1/episodes/ep_1/artifacts/events/art_1.jsonl",
            "sha256": "abc123",
            "size_bytes": 1024,
            "schema_name": "jsonl.event.stream",
            "created_at": "2026-07-07T00:00:00Z",
            "metadata": {"rows": 10},
        }
    )
    rows = catalog.list_artifacts_v2(session_id="sess_1", artifact_type="events")
    assert len(rows) == 1
    assert rows[0]["metadata_json"]["rows"] == 10


def test_event_index_roundtrip(catalog: PracticeCatalog):
    catalog.insert_event_index(
        {
            "event_id": "evt_1",
            "session_id": "sess_1",
            "episode_id": "ep_1",
            "timestamp_ns": 1000,
            "event_type": "physical_feedback_event",
            "artifact_id": "art_events_1",
            "byte_offset": 0,
            "summary": {"primary_event": "desired_contact"},
        }
    )
    row = catalog.get_event_index("evt_1")
    assert row["event_type"] == "physical_feedback_event"
    assert row["summary_json"]["primary_event"] == "desired_contact"

    rows = catalog.list_event_index(session_id="sess_1", event_type="physical_feedback_event")
    assert len(rows) == 1


def test_migration_preserves_existing_data(tmp_path: Path):
    db_path = tmp_path / "existing.sqlite"
    # Simulate an existing v1 catalog
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE practices (practice_id TEXT PRIMARY KEY, robot_id TEXT)")
    conn.execute("INSERT INTO practices VALUES ('prac_old', 'r1')")
    conn.commit()
    conn.close()

    catalog = PracticeCatalog(db_path, event_batch_size=1)
    row = catalog.get_practice("prac_old")
    assert row["robot_id"] == "r1"
    assert catalog.get_session("nonexistent") is None


def test_legacy_insert_event_still_works(catalog: PracticeCatalog):
    catalog.insert_event(
        {
            "event_id": "evt_legacy",
            "practice_id": "prac_1",
            "source": "runtime",
            "event_type": "physical_feedback_event",
            "timestamp_ns": 123,
            "timestamp_utc": "2026-07-07T00:00:00Z",
        }
    )
    assert catalog.count_events("prac_1") == 1
