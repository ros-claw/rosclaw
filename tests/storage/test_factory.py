"""Tests for rosclaw.storage.factory."""

from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw.memory.seekdb_client import (
    InMemoryKnowledgeStore,
    SeekDBMySQLClient,
    SQLiteKnowledgeStore,
)
from rosclaw.storage.factory import StorageFactory


def test_factory_defaults_to_memory():
    client = StorageFactory.create_knowledge_store()
    assert isinstance(client, InMemoryKnowledgeStore)


def test_factory_detects_sqlite_url():
    client = StorageFactory.create_knowledge_store(url="sqlite:///tmp/test.sqlite")
    assert isinstance(client, SQLiteKnowledgeStore)


def test_factory_detects_mysql_url():
    # We cannot connect, but instantiation should succeed with a valid DSN.
    client = StorageFactory.create_knowledge_store(url="mysql://root@127.0.0.1:2881/rosclaw")
    assert isinstance(client, SeekDBMySQLClient)


def test_factory_detects_seekdb_url():
    client = StorageFactory.create_knowledge_store(url="seekdb://root@127.0.0.1:2881/rosclaw")
    assert isinstance(client, SeekDBMySQLClient)


def test_factory_sqlite_path(tmp_path: Path):
    db = tmp_path / "knowledge.sqlite"
    client = StorageFactory.create_knowledge_store(backend="sqlite", path=str(db))
    assert isinstance(client, SQLiteKnowledgeStore)
    client.connect()
    assert db.exists()


def test_factory_memory_backend_defers_to_url_scheme():
    # "memory" is the neutral default; a concrete sqlite:// URL selects the real backend.
    client = StorageFactory.create_knowledge_store(
        backend="memory", url="sqlite:///tmp/ignored.sqlite"
    )
    assert isinstance(client, SQLiteKnowledgeStore)


def test_factory_rejects_http_for_sql_backend():
    with pytest.raises(ValueError):
        StorageFactory.create_knowledge_store(backend="mysql", url="http://localhost:2882")


def test_factory_rejects_http_when_detected_with_memory_backend():
    with pytest.raises(ValueError):
        StorageFactory.create_knowledge_store(url="http://localhost:2882")


def test_factory_rejects_unknown_backend():
    with pytest.raises(ValueError):
        StorageFactory.create_knowledge_store(backend="postgres")


def test_resolve_backend():
    assert StorageFactory.resolve_backend(url="mysql://x") == "mysql"
    assert StorageFactory.resolve_backend(url="sqlite://x") == "sqlite"
    assert StorageFactory.resolve_backend(url="http://x") == "http"
    assert StorageFactory.resolve_backend(backend="sqlite", url="mysql://x") == "sqlite"


def test_capabilities():
    mem = StorageFactory.create_knowledge_store()
    assert StorageFactory.capabilities(mem)["persistent"] is False
    sqlite = StorageFactory.create_knowledge_store(backend="sqlite", path=":memory:")
    caps = StorageFactory.capabilities(sqlite)
    assert caps["persistent"] is True
    assert caps["sqlite"] is True
    assert caps["mysql"] is False


def test_ping_memory():
    mem = StorageFactory.create_knowledge_store()
    result = StorageFactory.ping(mem)
    assert result["connected"] is True
    assert result["latency_ms"] is not None
    assert result["error"] is None


def test_ping_sqlite(tmp_path: Path):
    db = tmp_path / "knowledge.sqlite"
    sqlite = StorageFactory.create_knowledge_store(backend="sqlite", path=str(db))
    result = StorageFactory.ping(sqlite)
    assert result["connected"] is True
    assert result["latency_ms"] is not None
    assert "wal_size_bytes" in result


def test_sanitize_url_redacts_password():
    from rosclaw.storage.factory import _sanitize_url

    assert "secret" not in _sanitize_url("mysql://user:secret@host/db")
    assert _sanitize_url("mysql://user:secret@host/db").count("***") == 1
