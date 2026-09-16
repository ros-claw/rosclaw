"""PR-SDB-140-4: SeekDBLocalRuntime unit tests (no 1.4 bindings needed).

aarch64 CI/hosts lack the `seekdb` bindings wheel — every test here uses
fakes for the engine object and real filesystem/pid logic for lifecycle
and crash recovery.
"""

from __future__ import annotations

import os

import pytest

from rosclaw.storage.seekdb_runtime import (
    LocalRuntimeUnavailableError,
    SeekDBLocalRuntime,
)


class _FakeInstance:
    def __init__(self):
        self.closed = False

    def connection_options(self):
        return {"host": "127.0.0.1", "port": 2881, "pid": os.getpid()}

    def close(self):
        self.closed = True


@pytest.fixture
def fake_seekdb(monkeypatch):
    import sys
    import types

    instances: list[_FakeInstance] = []

    def _open(db_dir):
        inst = _FakeInstance()
        instances.append(inst)
        return inst

    module = types.SimpleNamespace(open=_open)
    monkeypatch.setitem(sys.modules, "seekdb", module)
    monkeypatch.setattr(SeekDBLocalRuntime, "available", staticmethod(lambda: True))
    return instances


def test_unavailable_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(SeekDBLocalRuntime, "available", staticmethod(lambda: False))
    rt = SeekDBLocalRuntime(tmp_path / "db")
    with pytest.raises(LocalRuntimeUnavailableError, match="not available"):
        rt.start()


def test_start_health_close(fake_seekdb, tmp_path):
    rt = SeekDBLocalRuntime(tmp_path / "db")
    info = rt.start()
    assert info.pid == os.getpid()
    assert info.connection_options["port"] == 2881
    assert rt.health()["healthy"] is True
    # idempotent: second start returns the same instance
    assert rt.start() is info
    assert len(fake_seekdb) == 1
    rt.close()
    assert rt.health()["healthy"] is False
    assert fake_seekdb[0].closed is True
    assert not (tmp_path / "db" / "runtime.lock.json").exists()


def test_connection_options_require_start(tmp_path):
    rt = SeekDBLocalRuntime(tmp_path / "db")
    with pytest.raises(RuntimeError, match="not started"):
        rt.connection_options()


def test_crash_recovery_clears_stale_lock(fake_seekdb, tmp_path):
    rt = SeekDBLocalRuntime(tmp_path / "db")
    rt.start()
    # simulate kill -9: lock file left behind, pid recycled to a dead one
    lock = tmp_path / "db" / "runtime.lock.json"
    import json

    payload = json.loads(lock.read_text())
    payload["pid"] = 999_999_999  # dead
    lock.write_text(json.dumps(payload))
    rt._instance = None  # process lost its handle too
    assert rt.recover_if_crashed() is True
    assert not lock.exists()
    # and a fresh start works after recovery
    info = rt.start()
    assert info.pid == os.getpid()


def test_live_owner_blocks_second_start(fake_seekdb, tmp_path):
    rt = SeekDBLocalRuntime(tmp_path / "db")
    rt.start()
    other = SeekDBLocalRuntime(tmp_path / "db")
    with pytest.raises(RuntimeError, match="already owned"):
        other.start()


class _FakeInnerStore:
    """In-memory StructuredStore stand-in for composition tests."""

    def __init__(self):
        self.rows: dict[str, dict] = {}
        self.connected = False

    def connect(self):
        self.connected = True

    def is_connected(self):
        return self.connected

    def disconnect(self):
        self.connected = False

    def insert(self, table, record):
        self.rows[record["id"]] = record
        return record["id"]

    def query(self, table, filters=None, order_by=None, limit=100):
        return [
            r for r in self.rows.values() if all(r.get(k) == v for k, v in (filters or {}).items())
        ][:limit]

    def update(self, table, record_id, updates):
        if record_id not in self.rows:
            return False
        self.rows[record_id].update(updates)
        return True

    def count(self, table, filters=None):
        return len(self.query(table, filters, limit=10**9))

    def delete(self, table, record_id):
        return self.rows.pop(record_id, None) is not None

    def delete_where(self, table, filters):
        doomed = [r["id"] for r in self.query(table, filters, limit=10**9)]
        for rid in doomed:
            self.rows.pop(rid, None)
        return len(doomed)


@pytest.fixture
def fake_inner(monkeypatch):
    """Composition tests must NOT touch a real server — stub the inner store."""
    from rosclaw.storage.seekdb_runtime import LocalRuntimeStructuredStore

    made: list[_FakeInnerStore] = []

    def _build(self, options):
        inner = _FakeInnerStore()
        made.append(inner)
        return inner

    monkeypatch.setattr(LocalRuntimeStructuredStore, "_build_inner", _build)
    return made


# ---------------------------------------------------------------------------
# PR-SDB-140-5 (P0-7/8/9): attach semantics, ownership composition, socket
# ---------------------------------------------------------------------------


def test_attach_requires_live_owner(tmp_path):
    rt = SeekDBLocalRuntime(tmp_path / "db")
    with pytest.raises(RuntimeError, match="no live seekdb local runtime"):
        rt.attach()


def test_attacher_shares_options_and_never_closes_engine(fake_seekdb, tmp_path):
    owner = SeekDBLocalRuntime(tmp_path / "db")
    info = owner.start()
    assert info.role == "owner"

    attacher = SeekDBLocalRuntime(tmp_path / "db")
    ainfo = attacher.attach()
    assert ainfo.role == "attacher"
    assert ainfo.pid == info.pid
    assert ainfo.connection_options == info.connection_options

    attacher.release()  # must NOT close the owner's engine
    assert fake_seekdb[0].closed is False
    assert owner.health()["pid_alive"] is True
    owner.close()
    assert fake_seekdb[0].closed is True


def test_start_or_attach_owner_then_attacher(fake_seekdb, tmp_path):
    first = SeekDBLocalRuntime(tmp_path / "db")
    assert first.start_or_attach().role == "owner"
    second = SeekDBLocalRuntime(tmp_path / "db")
    assert second.start_or_attach().role == "attacher"
    assert len(fake_seekdb) == 1, "attacher must not start a second engine"


def test_composed_store_owns_runtime_lifecycle(fake_seekdb, fake_inner, tmp_path):
    from rosclaw.storage.seekdb_runtime import LocalRuntimeStructuredStore

    store = LocalRuntimeStructuredStore(tmp_path / "db")
    with pytest.raises(RuntimeError, match="not connected"):
        store.count("memory_items")
    store.connect()
    assert store.is_connected()
    assert store.runtime.role == "owner"
    inner = store._inner
    store.disconnect()
    assert inner.is_connected() is False
    assert fake_seekdb[0].closed is True, "owner disconnect must close the engine"


def test_composed_store_attacher_disconnect_leaves_engine(fake_seekdb, fake_inner, tmp_path):
    from rosclaw.storage.seekdb_runtime import LocalRuntimeStructuredStore

    owner = LocalRuntimeStructuredStore(tmp_path / "db")
    owner.connect()
    attacher = LocalRuntimeStructuredStore(tmp_path / "db")
    attacher.connect()
    assert attacher.runtime.role == "attacher"
    attacher.disconnect()
    assert fake_seekdb[0].closed is False, "attacher disconnect closed the owner's engine"
    owner.disconnect()


def test_composed_store_delegates_storage(fake_seekdb, fake_inner, tmp_path):
    from rosclaw.storage.seekdb_runtime import LocalRuntimeStructuredStore

    store = LocalRuntimeStructuredStore(tmp_path / "db")
    store.connect()
    rid = store.insert("memory_items", {"id": "m1", "memory_type": "episode", "robot_id": "r"})
    assert rid == "m1"
    assert store.count("memory_items", {"robot_id": "r"}) == 1
    store.disconnect()


def test_unix_socket_options_build_socket_store(fake_seekdb, tmp_path, monkeypatch):
    """P0-7: connection_options with unix_socket must produce a store bound
    to THAT socket — never a silent fallthrough to 127.0.0.1:2881."""
    from rosclaw.storage.seekdb_runtime import LocalRuntimeStructuredStore

    for inst in fake_seekdb:
        inst.connection_options = lambda: {
            "user": "root",
            "unix_socket": "/tmp/x.sock",
            "pid": os.getpid(),
        }

    store = LocalRuntimeStructuredStore(tmp_path / "db")
    # build the inner store without connecting (no real socket exists here)
    inner = store._build_inner({"user": "root", "unix_socket": "/tmp/x.sock"})
    assert inner._unix_socket == "/tmp/x.sock"
    assert inner._database == "rosclaw"


def test_sqlstore_unix_socket_overrides_tcp(monkeypatch):
    """SeekDBSQLStore with unix_socket must pass it to pymysql and omit TCP."""
    from rosclaw.memory.seekdb_client import SeekDBSQLStore

    captured: dict = {}

    class _FakePymysql:
        class cursors:  # noqa: N801 - mirrors pymysql.cursors module
            DictCursor = object

        @staticmethod
        def connect(**kwargs):
            captured.update(kwargs)
            raise RuntimeError("stop here")

    monkeypatch.setitem(__import__("sys").modules, "pymysql", _FakePymysql)
    store = SeekDBSQLStore("mysql://root@/rosclaw", unix_socket="/tmp/rt.sock")
    with pytest.raises(RuntimeError, match="stop here"):
        store._open_connection(None)
    assert captured["unix_socket"] == "/tmp/rt.sock"
    assert "host" not in captured and "port" not in captured
