"""PR-SDB-140-4: ROSCLAW_SEEKDB_MODE explicit deployment-mode wiring."""

from __future__ import annotations

import pytest

from rosclaw.storage.factory import StoreFactory


def test_mode_server_maps_to_seekdb_server(monkeypatch):
    monkeypatch.setenv("ROSCLAW_SEEKDB_MODE", "server")
    monkeypatch.delenv("ROSCLAW_SEEKDB_URL", raising=False)
    from rosclaw.storage.seekdb_native import SeekDBServerRetrievalStore

    store = StoreFactory.create_structured_store(url="mysql://root@127.0.0.1:2881/rosclaw")
    assert isinstance(store, SeekDBServerRetrievalStore)


def test_mode_legacy_embedded_maps(monkeypatch, tmp_path):
    monkeypatch.setenv("ROSCLAW_SEEKDB_MODE", "legacy_embedded")
    monkeypatch.delenv("ROSCLAW_SEEKDB_URL", raising=False)
    from rosclaw.storage.seekdb_native import SeekDBEmbeddedRetrievalStore

    store = StoreFactory.create_structured_store(path=str(tmp_path / "db"))
    assert isinstance(store, SeekDBEmbeddedRetrievalStore)


def test_unknown_mode_fails_closed(monkeypatch):
    monkeypatch.setenv("ROSCLAW_SEEKDB_MODE", "magic")
    with pytest.raises(ValueError, match="not a deployment mode"):
        StoreFactory.create_structured_store()


def test_mode_conflicts_with_explicit_backend(monkeypatch):
    monkeypatch.setenv("ROSCLAW_SEEKDB_MODE", "server")
    with pytest.raises(ValueError, match="conflicts"):
        StoreFactory.create_structured_store(backend="sqlite", path="/tmp/x.sqlite")


def test_local_runtime_unavailable_fails_closed(monkeypatch, tmp_path):
    """aarch64: no bindings wheel -> explicit, immediate failure (§十六)."""
    monkeypatch.setenv("ROSCLAW_SEEKDB_MODE", "local_runtime")
    from rosclaw.storage.seekdb_runtime import SeekDBLocalRuntime

    monkeypatch.setattr(SeekDBLocalRuntime, "available", staticmethod(lambda: False))
    from rosclaw.storage.seekdb_runtime import LocalRuntimeUnavailableError

    with pytest.raises(LocalRuntimeUnavailableError, match="unavailable"):
        StoreFactory.create_structured_store(path=str(tmp_path / "db"))
