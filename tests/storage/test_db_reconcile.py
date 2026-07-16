"""Tests for rosclaw db reconcile."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import yaml

from rosclaw.storage.cli import cmd_db_reconcile, reconcile_practice


def _write_session(
    root: Path,
    practice_id: str,
    event_ids: list[str],
    *,
    session_id: str | None = None,
    episode_id: str | None = None,
) -> Path:
    session_id = session_id or f"sess_{practice_id}"
    episode_id = episode_id or f"ep_{practice_id}"
    session_dir = root / "sessions" / practice_id
    raw = session_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "events.jsonl").open("w", encoding="utf-8") as f:
        for i, event_id in enumerate(event_ids):
            f.write(
                json.dumps(
                    {
                        "event_id": event_id,
                        "practice_id": practice_id,
                        "session_id": session_id,
                        "episode_id": episode_id,
                        "event_type": "skill.invoke",
                        "sequence_id": i + 1,
                    }
                )
                + "\n"
            )
    (session_dir / "episode.json").write_text(
        json.dumps({"event_count": len(event_ids)}), encoding="utf-8"
    )
    (session_dir / "manifest.yaml").write_text(
        yaml.safe_dump({"practice_id": practice_id, "event_count": len(event_ids)}),
        encoding="utf-8",
    )

    indexes = root / "indexes"
    indexes.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(indexes / "practice_catalog.sqlite"))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS practices (practice_id TEXT PRIMARY KEY, session_id TEXT, episode_id TEXT)"
    )
    conn.execute("CREATE TABLE IF NOT EXISTS events (event_id TEXT PRIMARY KEY, practice_id TEXT)")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS practice_event_index (event_id TEXT PRIMARY KEY, session_id TEXT, episode_id TEXT)"
    )
    conn.execute(
        "INSERT OR REPLACE INTO practices (practice_id, session_id, episode_id) VALUES (?, ?, ?)",
        (practice_id, session_id, episode_id),
    )
    for event_id in event_ids:
        conn.execute(
            "INSERT OR REPLACE INTO events (event_id, practice_id) VALUES (?, ?)",
            (event_id, practice_id),
        )
        conn.execute(
            "INSERT OR REPLACE INTO practice_event_index (event_id, session_id, episode_id) VALUES (?, ?, ?)",
            (event_id, session_id, episode_id),
        )
    conn.commit()
    conn.close()
    return session_dir


def test_reconcile_passes_for_consistent_session(tmp_path: Path) -> None:
    ids = [f"evt_{i:04d}" for i in range(50)]
    _write_session(tmp_path, "prac_ok", ids)
    report = reconcile_practice("prac_ok", str(tmp_path))
    assert report["passed"]
    assert report["raw_jsonl"] == 50
    assert report["catalog_events"] == 50
    assert report["event_index"] == 50
    assert report["manifest_event_count"] == 50
    assert report["episode_event_count"] == 50
    assert report["duplicates"] == 0
    hashes = report["hashes"]
    assert hashes["jsonl_event_ids"] == hashes["catalog_event_ids"] == hashes["event_index_ids"]


def test_reconcile_fails_when_catalog_missing_events(tmp_path: Path) -> None:
    ids = [f"evt_{i:04d}" for i in range(10)]
    _write_session(tmp_path, "prac_gap", ids)
    catalog = tmp_path / "indexes" / "practice_catalog.sqlite"
    conn = sqlite3.connect(str(catalog))
    conn.execute("DELETE FROM events WHERE event_id = 'evt_0003'")
    conn.commit()
    conn.close()
    report = reconcile_practice("prac_gap", str(tmp_path))
    assert not report["passed"]
    assert report["missing"] == ["evt_0003"]
    assert report["missing_count"] == 1


def test_reconcile_detects_duplicate_event_ids(tmp_path: Path) -> None:
    ids = [f"evt_{i:04d}" for i in range(5)]
    session_dir = _write_session(tmp_path, "prac_dup", ids)
    with (session_dir / "raw" / "events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"event_id": "evt_0001", "practice_id": "prac_dup"}) + "\n")
    report = reconcile_practice("prac_dup", str(tmp_path))
    assert not report["passed"]
    assert report["duplicates"] == 1


def test_reconcile_missing_session_reports_error(tmp_path: Path) -> None:
    report = reconcile_practice("prac_missing", str(tmp_path))
    assert not report["passed"]
    assert report["error"] == "events.jsonl not found"


def test_reconcile_remote_counts(tmp_path: Path) -> None:
    ids = [f"evt_{i:04d}" for i in range(3)]
    _write_session(tmp_path, "prac_remote", ids)

    class _FakeRemote:
        def count(self, table: str, filters: dict | None = None) -> int:
            assert filters == {"practice_id": "prac_remote"}
            return {"episodes": 1, "body_cognition": 4}.get(table, 0)

    report = reconcile_practice("prac_remote", str(tmp_path), remote_client=_FakeRemote())
    assert report["passed"]
    assert report["remote_episode"] == 1
    assert report["remote_memories"] == 4


def test_cmd_db_reconcile_all_json(tmp_path: Path, capsys) -> None:
    _write_session(tmp_path, "prac_a", [f"evt_a{i}" for i in range(4)])
    _write_session(tmp_path, "prac_b", [f"evt_b{i}" for i in range(6)])
    args = argparse.Namespace(
        practice_id=None,
        all=True,
        data_root=str(tmp_path),
        remote=False,
        json=True,
        backend="sqlite",
        url=None,
        path=str(tmp_path / "knowledge.sqlite"),
        max_missing=20,
    )
    rc = cmd_db_reconcile(args)
    assert rc == 0
    output = json.loads(capsys.readouterr().out)
    assert output["passed"]
    assert {r["practice_id"] for r in output["sessions"]} == {"prac_a", "prac_b"}
