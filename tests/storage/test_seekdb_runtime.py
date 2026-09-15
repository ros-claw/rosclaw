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
