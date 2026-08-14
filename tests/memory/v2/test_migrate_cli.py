"""PR-DF-06: rosclaw memory migrate — legacy experience_graph -> memory_items."""

import json
import sys

from rosclaw.cli import main
from rosclaw.memory.seekdb_client import SQLiteStructuredStore


def _seed_legacy(db_path, rows=3):
    store = SQLiteStructuredStore(str(db_path))
    store.connect()
    for i in range(rows):
        store.insert(
            "experience_graph",
            {
                "id": f"exp_{i}",
                "robot_id": "sim_test",
                "event_type": "praxis",
                "instruction": f"task {i}",
                "outcome": "failure" if i == 0 else "success",
                "error_details": "slip" if i == 0 else None,
                "timestamp": 1700000000.0 + i,
            },
        )
    store.disconnect()
    return store


def test_migrate_cli_and_idempotency(tmp_path, monkeypatch, capsys):
    home = tmp_path / ".rosclaw"
    monkeypatch.setenv("ROSCLAW_HOME", str(home))
    db = home / "memory" / "seekdb.sqlite"
    db.parent.mkdir(parents=True)
    _seed_legacy(db)

    monkeypatch.setattr(sys, "argv", ["rosclaw", "memory", "migrate", "--json"])
    assert main() == 0
    first = json.loads(capsys.readouterr().out)
    assert first["from"] == "experience_graph"
    assert first["scanned"] == 3
    assert first["migrated"] == 3

    # failure outcome maps to failure memory; the rest episodic
    store = SQLiteStructuredStore(str(db))
    store.connect()
    assert store.count("memory_items", {"memory_type": "failure"}) == 1
    assert store.count("memory_items", {"memory_type": "episodic"}) == 2
    store.disconnect()

    # rerun: nothing duplicates
    monkeypatch.setattr(sys, "argv", ["rosclaw", "memory", "migrate", "--json"])
    assert main() == 0
    second = json.loads(capsys.readouterr().out)
    assert second["migrated"] == 0
    assert second["deduplicated"] == 3


def test_migrate_missing_db_is_clean_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path / ".rosclaw"))
    monkeypatch.setattr(sys, "argv", ["rosclaw", "memory", "migrate"])
    assert main() == 1
    assert "not found" in capsys.readouterr().err
