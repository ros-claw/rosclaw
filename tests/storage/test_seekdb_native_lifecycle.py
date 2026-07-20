from __future__ import annotations

from types import SimpleNamespace

import pytest

from rosclaw.storage import seekdb_native


class _FakeProxy:
    def __init__(self) -> None:
        self.enter_count = 0
        self.exit_count = 0

    def __enter__(self) -> _FakeProxy:
        self.enter_count += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.exit_count += 1

    def create_database(self, name: str) -> None:
        return None

    def list_collections(self) -> list:
        return []


def test_native_store_closes_pyseekdb_proxies(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(seekdb_native, "_EMBEDDED_PROCESS_TARGET", None)
    admin = _FakeProxy()
    client = _FakeProxy()
    pyseekdb = SimpleNamespace(
        AdminClient=lambda **kwargs: admin,
        Client=lambda **kwargs: client,
    )
    monkeypatch.setattr(seekdb_native, "_require_pyseekdb", lambda: pyseekdb)
    store = seekdb_native.SeekDBEmbeddedStore(path=str(tmp_path))

    store.connect()

    assert admin.enter_count == 1
    assert admin.exit_count == 1
    assert client.exit_count == 0

    store.disconnect()
    store.disconnect()

    assert client.exit_count == 1


def test_embedded_store_rejects_process_path_switch(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(seekdb_native, "_EMBEDDED_PROCESS_TARGET", None)
    pyseekdb = SimpleNamespace(
        AdminClient=lambda **kwargs: _FakeProxy(),
        Client=lambda **kwargs: _FakeProxy(),
    )
    monkeypatch.setattr(seekdb_native, "_require_pyseekdb", lambda: pyseekdb)
    first = seekdb_native.SeekDBEmbeddedStore(path=str(tmp_path / "first"))
    second = seekdb_native.SeekDBEmbeddedStore(path=str(tmp_path / "second"))

    first.connect()
    first.disconnect()

    with pytest.raises(RuntimeError, match="one embedded path/database target per process"):
        second.connect()


def test_embedded_store_rejects_process_database_switch(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(seekdb_native, "_EMBEDDED_PROCESS_TARGET", None)
    pyseekdb = SimpleNamespace(
        AdminClient=lambda **kwargs: _FakeProxy(),
        Client=lambda **kwargs: _FakeProxy(),
    )
    monkeypatch.setattr(seekdb_native, "_require_pyseekdb", lambda: pyseekdb)
    first = seekdb_native.SeekDBEmbeddedStore(path=str(tmp_path), database="first")
    second = seekdb_native.SeekDBEmbeddedStore(path=str(tmp_path), database="second")

    first.connect()
    first.disconnect()

    with pytest.raises(RuntimeError, match="one embedded path/database target per process"):
        second.connect()
