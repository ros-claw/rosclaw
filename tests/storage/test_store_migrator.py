"""PR-SDB-140-2: StoreMigrator dump/restore/parity unit tests (sqlite, no engine)."""

from __future__ import annotations

import json

import pytest

from rosclaw.memory.seekdb_client import SQLiteStructuredStore
from rosclaw.storage.migrate import MANIFEST, StoreMigrator


@pytest.fixture
def pair(tmp_path):
    src = SQLiteStructuredStore(str(tmp_path / "src.sqlite"))
    dst = SQLiteStructuredStore(str(tmp_path / "dst.sqlite"))
    src.connect()
    dst.connect()
    yield src, dst
    src.disconnect()
    dst.disconnect()


def test_dump_restore_parity_roundtrip(pair, tmp_path):
    src, dst = pair
    src.insert(
        "memory_items",
        {"id": "m1", "title": "故障 一", "document": "关节越界 sandbox 阻断", "outcome": "failure", "memory_type": "failure", "robot_id": "ur5e_00"},
    )
    src.insert("memory_items", {"id": "m2", "title": "ok", "document": "recovery worked", "memory_type": "intervention", "robot_id": "ur5e_00"})
    src.insert("heuristic_rules", {"id": "r1", "condition": "c", "action": "a", "success_count": 3})

    dump_dir = tmp_path / "dump"
    manifest = StoreMigrator(src).dump(dump_dir)
    assert (dump_dir / MANIFEST).exists()
    assert manifest["tables"]["memory_items"]["rows"] == 2

    restored = StoreMigrator(dst).restore(dump_dir)
    assert restored["memory_items"] == 2

    report = StoreMigrator.parity(manifest, dst)
    assert report["ok"] is True
    assert report["tables"]["memory_items"]["checksum_match"] is True


def test_parity_detects_drift(pair, tmp_path):
    src, dst = pair
    src.insert("memory_items", {"id": "m1", "title": "x", "document": "y", "memory_type": "episode", "robot_id": "r"})
    manifest = StoreMigrator(src).dump(tmp_path / "dump")
    # never restored -> parity fails
    report = StoreMigrator.parity(manifest, dst)
    assert report["ok"] is False
    assert report["tables"]["memory_items"]["ok"] is False


def test_checksum_is_deterministic(pair, tmp_path):
    src, _ = pair
    for i in range(5):
        src.insert("memory_items", {"id": f"m{i}", "document": f"doc {i}", "memory_type": "episode", "robot_id": "r"})
    m1 = StoreMigrator(src).dump(tmp_path / "d1")
    m2 = StoreMigrator(src).dump(tmp_path / "d2")
    assert m1["tables"]["memory_items"]["checksum"] == m2["tables"]["memory_items"]["checksum"]
    # manifest timestamps differ; checksums must not
    assert json.dumps(m1["tables"], sort_keys=True) == json.dumps(m2["tables"], sort_keys=True)
