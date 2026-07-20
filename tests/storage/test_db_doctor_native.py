"""db doctor acceptance tests for native SeekDB backends (P0-4).

Uses the real embedded engine (no mocks) and, when reachable, the live
SeekDB server container.
"""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path

import pytest

pyseekdb = pytest.importorskip(  # noqa: E402
    "pyseekdb", reason="native SeekDB engine not installed"
)

from rosclaw.storage.cli import cmd_db_doctor  # noqa: E402
from rosclaw.storage.seekdb_native import SeekDBEmbeddedStore  # noqa: E402


def _args(path: str, **overrides) -> argparse.Namespace:
    base = {
        "json": True,
        "fix": False,
        "backend": "seekdb_embedded",
        "url": None,
        "path": path,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _doctor_payload(capsys) -> dict:
    return json.loads(capsys.readouterr().out)


def test_doctor_embedded_full_check_set(shared_embedded_seekdb_target, capsys) -> None:
    """P0-4: doctor accepts seekdb_embedded with the full native check set."""
    path = shared_embedded_seekdb_target["path"]
    store = SeekDBEmbeddedStore(path=path, database=shared_embedded_seekdb_target["database"])
    store.connect()
    store.delete_where("memory_items", {})
    store.insert("memory_items", {"id": "seed-1", "title": "doctor seed", "document": "x"})
    store.disconnect()

    try:
        rc = cmd_db_doctor(_args(path))
        payload = _doctor_payload(capsys)
    finally:
        cleanup = SeekDBEmbeddedStore(path=path, database=shared_embedded_seekdb_target["database"])
        cleanup.connect()
        cleanup.delete_where("memory_items", {})
        cleanup.disconnect()

    assert rc == 0, payload["issues"]
    by_name = {check["name"]: check for check in payload["checks"]}
    assert by_name["backend"]["value"] == "seekdb_embedded"
    assert by_name["engine ready"]["ok"]
    assert by_name["collections"]["ok"]
    assert by_name["vector dimension"]["ok"]
    assert by_name["embedder model"]["ok"]
    assert by_name["collection counts"]["ok"]
    # Restart persistence: reopen must see the same collections (the shared
    # engine accumulates collections across test files, so only ok matters).
    assert by_name["restart persistence"]["ok"]
    # schema_migrations must be n/a (not "missing") for native backends.
    assert by_name["schema_migrations table"]["ok"]
    assert by_name["schema_migrations table"]["value"] == "n/a"
    # Structured seekdb block with embedder + dimension per collection.
    seekdb = payload["seekdb"]
    assert seekdb["deployment"]["mode"] == "embedded"
    assert seekdb["deployment"]["path"] == path
    assert seekdb["collections"]["memory_items"]["count"] == 1
    assert seekdb["collections"]["memory_items"]["dimension"] == 384


def test_doctor_embedded_dimension_mismatch_flagged(shared_embedded_seekdb_target, capsys) -> None:
    """Consistent vector dimensions across collections must not be flagged."""
    path = shared_embedded_seekdb_target["path"]
    store = SeekDBEmbeddedStore(path=path, database=shared_embedded_seekdb_target["database"])
    store.connect()
    store.delete_where("memory_items", {})
    store.insert("memory_items", {"id": "a", "title": "t", "document": "d"})
    store.disconnect()

    try:
        rc = cmd_db_doctor(_args(path))
        payload = _doctor_payload(capsys)
    finally:
        cleanup = SeekDBEmbeddedStore(path=path, database=shared_embedded_seekdb_target["database"])
        cleanup.connect()
        cleanup.delete_where("memory_items", {})
        cleanup.disconnect()
    # Single-engine run: dimensions consistent → no dimension issue.
    assert rc == 0
    assert not any("dimension" in issue.lower() for issue in payload["issues"])


def _server_reachable(host: str = "127.0.0.1", port: int = 2881) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


@pytest.mark.skipif(not _server_reachable(), reason="SeekDB server not reachable on :2881")
def test_doctor_server_dsn_auth_and_counts(capsys) -> None:
    """P0-4: server backend reports dsn auth and per-collection counts."""
    rc = cmd_db_doctor(
        _args(
            "",
            backend="seekdb_server",
            url="seekdb://root@127.0.0.1:2881/rosclaw",
        )
    )
    payload = _doctor_payload(capsys)
    assert rc == 0, payload["issues"]
    by_name = {check["name"]: check for check in payload["checks"]}
    assert by_name["dsn auth"]["ok"]
    assert "root@127.0.0.1:2881/rosclaw" in by_name["dsn auth"]["value"]
    assert by_name["engine ready"]["ok"]
    seekdb = payload["seekdb"]
    assert seekdb["deployment"]["mode"] == "server"
    assert "password" not in json.dumps(seekdb["deployment"])
    assert seekdb["collections"], "expected at least one collection"


def test_doctor_rejects_unknown_backend_still(tmp_path: Path, capsys) -> None:
    """The pre-existing unknown-backend guard still fires."""
    rc = cmd_db_doctor(_args(str(tmp_path / "x"), backend="not_a_backend"))
    payload = _doctor_payload(capsys)
    assert rc == 1
    assert any("Unknown backend" in issue for issue in payload["issues"])
