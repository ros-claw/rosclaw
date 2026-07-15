"""Consolidated integration tests for the Phase 1 storage unification layer.

These tests exercise ``StorageFactory``, ``OutboxStore``, ``VectorStore``,
and ``MigrationRunner`` together through the public ``rosclaw.storage`` API.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw.storage import StorageFactory
from rosclaw.storage.factory import _sanitize_url
from rosclaw.storage.migrations import MigrationRunner
from rosclaw.storage.outbox import OutboxStore


class TestStorageFactoryUnification:
    def test_factory_creates_sqlite_with_vector_and_outbox(self, tmp_path: Path) -> None:
        db_path = tmp_path / "knowledge.sqlite"
        client = StorageFactory.create_knowledge_store(
            backend="sqlite",
            path=str(db_path),
            vector_enabled=True,
        )
        try:
            caps = StorageFactory.capabilities(client)
            assert caps["persistent"]
            assert caps["sql"]
            assert caps["sqlite"]
            assert caps["vector"]

            ping = StorageFactory.ping(client)
            assert ping["connected"]
            assert ping["latency_ms"] is not None

            # Initialize vector extension tables.
            client._init_vector_store()
            # Force creation of a vector side table.
            assert client._vector_store is not None
            client._vector_store._ensure_table("experience_graph")

            # Vector extension tables should have been created.
            tables = {
                row[0]
                for row in client._connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            assert "schema_migrations" in tables
            assert any(name.startswith("vec_") for name in tables)
        finally:
            getattr(client, "disconnect", getattr(client, "close", lambda: None))()

    def test_factory_rejects_http_url_for_sql_backend(self) -> None:
        with pytest.raises(ValueError, match="HTTP"):
            StorageFactory.create_knowledge_store(
                backend="mysql",
                url="http://localhost:2882",
            )

    def test_sanitize_url_redacts_password(self) -> None:
        url = "mysql://root:secret@127.0.0.1:2881/rosclaw"
        assert _sanitize_url(url) == "mysql://root:***@127.0.0.1:2881/rosclaw"


class TestOutboxAndMigrationsTogether:
    def test_outbox_migrations_and_stats(self, tmp_path: Path) -> None:
        db_path = tmp_path / "outbox.sqlite"
        outbox = OutboxStore(db_path=str(db_path))
        stats = outbox.stats()
        assert stats == {"total": 0, "pending": 0, "failed": 0}

    def test_migration_runner_baseline_idempotent(self, tmp_path: Path) -> None:
        import sqlite3

        db_path = tmp_path / "mig.sqlite"
        conn = sqlite3.connect(str(db_path))
        runner = MigrationRunner()
        applied1 = runner.apply(conn, "sqlite")
        applied2 = runner.apply(conn, "sqlite")
        assert "001" in applied1
        assert applied2 == []

        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert "schema_migrations" in tables
        conn.close()


class TestVectorSearchEndToEnd:
    def test_similar_search_returns_results(self, tmp_path: Path) -> None:
        db_path = tmp_path / "vec.sqlite"
        client = StorageFactory.create_knowledge_store(
            backend="sqlite",
            path=str(db_path),
            vector_enabled=True,
        )
        try:
            # Insert two records into a table that has a vector side table.
            id1 = client.insert(
                "experience_graph",
                {
                    "event_type": "test",
                    "robot_id": "r1",
                    "timestamp": 1.0,
                    "instruction": "grasp a red cube on the table",
                    "outcome": "success",
                },
            )
            id2 = client.insert(
                "experience_graph",
                {
                    "event_type": "test",
                    "robot_id": "r1",
                    "timestamp": 2.0,
                    "instruction": "push the cube off the table",
                    "outcome": "failure",
                },
            )

            client._init_vector_store()
            embedder = client._embedder
            assert embedder is not None
            client._vector_store.upsert(
                "experience_graph", id1, "grasp a red cube on the table", embedder.encode("grasp a red cube on the table")
            )
            client._vector_store.upsert(
                "experience_graph", id2, "push the cube off the table", embedder.encode("push the cube off the table")
            )

            results = client.similar(
                "experience_graph",
                query_text="grasp cube",
                filters={"robot_id": "r1"},
                limit=5,
            )
            assert len(results) == 2
            # The grasp record should rank at least as high as the push record.
            assert results[0]["id"] == id1
        finally:
            getattr(client, "disconnect", getattr(client, "close", lambda: None))()
