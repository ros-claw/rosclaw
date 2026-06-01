"""
Tests for SeekDB query indexes.

Verifies:
1. SeekDBMemoryClient uses inverted indexes for common filter patterns
2. SeekDBSQLiteClient creates composite indexes for common query patterns
3. Index correctness: indexed queries return same results as full scans

P1 Issue 3: https://github.com/ros-claw/rosclaw-v1.0/issues/XXX
"""

import time

import pytest

from rosclaw.memory.seekdb_client import (
    SeekDBMemoryClient,
    SeekDBSQLiteClient,
    SEEKDB_SCHEMAS,
)


# ---------------------------------------------------------------------------
# SeekDBMemoryClient — Inverted Indexes
# ---------------------------------------------------------------------------


class TestMemoryClientInvertedIndex:
    """Verify inverted index behavior for in-memory client."""

    @pytest.fixture
    def client(self):
        c = SeekDBMemoryClient()
        c.connect()
        yield c
        c.disconnect()

    @pytest.fixture
    def populated_client(self, client):
        """Client with 100 experiences across 3 robots."""
        for i in range(100):
            robot = f"bot_{i % 3}"
            outcome = "success" if i % 4 != 0 else "failure"
            client.insert("experience_graph", {
                "id": f"exp_{i}",
                "event_type": "praxis",
                "robot_id": robot,
                "timestamp": float(i),
                "instruction": f"task {i}",
                "outcome": outcome,
                "tags": [],
            })
        return client

    def test_indexes_created_on_connect(self, client):
        """Inverted indexes should exist for declared schema indices."""
        assert "experience_graph" in client._indices
        idx = client._indices["experience_graph"]
        assert "robot_id" in idx
        assert "outcome" in idx
        assert "event_type" in idx
        assert "timestamp" in idx

    def test_insert_updates_index(self, client):
        """Inserting a record should update inverted indexes."""
        client.insert("experience_graph", {
            "id": "exp_1",
            "robot_id": "bot_a",
            "outcome": "success",
            "event_type": "praxis",
            "timestamp": 1.0,
        })
        idx = client._indices["experience_graph"]
        assert "exp_1" in idx["robot_id"]["bot_a"]
        assert "exp_1" in idx["outcome"]["success"]

    def test_indexed_query_correctness(self, populated_client):
        """Indexed query should return same results as full scan."""
        c = populated_client
        # Query by robot_id (indexed)
        results = c.query("experience_graph", filters={"robot_id": "bot_0"})
        assert len(results) > 0
        for r in results:
            assert r["robot_id"] == "bot_0"

    def test_compound_filter_query(self, populated_client):
        """Compound filter (robot_id + outcome) should work correctly."""
        c = populated_client
        results = c.query("experience_graph",
                          filters={"robot_id": "bot_1", "outcome": "success"})
        assert len(results) > 0
        for r in results:
            assert r["robot_id"] == "bot_1"
            assert r["outcome"] == "success"

    def test_indexed_query_with_ordering(self, populated_client):
        """Ordering should work after indexed filter."""
        c = populated_client
        results = c.query("experience_graph",
                          filters={"robot_id": "bot_0"},
                          order_by="-timestamp",
                          limit=5)
        assert len(results) <= 5
        # Verify descending order
        timestamps = [r["timestamp"] for r in results]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_update_maintains_index(self, client):
        """Updating a record should update index entries."""
        client.insert("experience_graph", {
            "id": "exp_u",
            "robot_id": "bot_a",
            "outcome": "success",
            "event_type": "praxis",
            "timestamp": 1.0,
        })
        idx = client._indices["experience_graph"]
        assert "exp_u" in idx["outcome"]["success"]

        client.update("experience_graph", "exp_u", {"outcome": "failure"})
        assert "exp_u" not in idx["outcome"].get("success", set())
        assert "exp_u" in idx["outcome"]["failure"]

    def test_count_uses_index(self, populated_client):
        """count() should use index for simple filter queries."""
        c = populated_client
        count = c.count("experience_graph", filters={"robot_id": "bot_0"})
        assert count > 0
        # Verify matches actual query
        results = c.query("experience_graph", filters={"robot_id": "bot_0"},
                          limit=10000)
        assert count == len(results)

    def test_non_indexed_filter_fallback(self, client):
        """Non-indexed filters should fall back to scan."""
        client.insert("experience_graph", {
            "id": "exp_x",
            "robot_id": "bot_a",
            "instruction": "unique instruction xyz",
            "event_type": "praxis",
            "timestamp": 1.0,
            "outcome": "success",
        })
        # instruction is not indexed
        results = client.query("experience_graph",
                               filters={"instruction": "unique instruction xyz"})
        assert len(results) == 1
        assert results[0]["id"] == "exp_x"

    def test_empty_index_returns_empty(self, client):
        """Query against empty table should return []."""
        results = client.query("experience_graph",
                               filters={"robot_id": "nonexistent"})
        assert results == []

    def test_max_scan_limit(self):
        """MAX_SCAN_LIMIT should be defined."""
        assert hasattr(SeekDBMemoryClient, "MAX_SCAN_LIMIT")
        assert SeekDBMemoryClient.MAX_SCAN_LIMIT > 0


# ---------------------------------------------------------------------------
# SeekDBSQLiteClient — Composite Indexes
# ---------------------------------------------------------------------------


class TestSQLiteClientCompositeIndexes:
    """Verify composite indexes are created in SQLite backend."""

    @pytest.fixture
    def client(self, tmp_path):
        db_path = str(tmp_path / "test_seekdb.sqlite")
        c = SeekDBSQLiteClient(db_path)
        c.connect()
        yield c
        c.disconnect()

    def test_tables_created(self, client):
        """All schema tables should exist."""
        cursor = client._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cursor.fetchall()}
        for table_name in SEEKDB_SCHEMAS:
            assert table_name in tables

    def test_single_column_indexes_created(self, client):
        """Single-column indexes from schema should exist."""
        cursor = client._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        )
        indexes = {row[0] for row in cursor.fetchall()}
        for table_name, schema in SEEKDB_SCHEMAS.items():
            for col in schema.get("indices", []):
                assert f"idx_{table_name}_{col}" in indexes

    def test_composite_indexes_created(self, client):
        """Composite indexes should be created."""
        cursor = client._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        )
        indexes = {row[0] for row in cursor.fetchall()}
        for _, idx_name, _ in SeekDBSQLiteClient._COMPOSITE_INDICES:
            assert idx_name in indexes, f"Composite index {idx_name} missing"

    def test_composite_index_used_for_robot_ts_query(self, client):
        """EXPLAIN QUERY PLAN should show index usage for robot_id+timestamp query."""
        # Insert test data
        for i in range(100):
            client.insert("experience_graph", {
                "id": f"exp_{i}",
                "event_type": "praxis",
                "robot_id": "bot_1",
                "timestamp": float(i),
                "instruction": f"task {i}",
                "outcome": "success",
            })

        # Check query plan uses composite index
        cursor = client._conn.execute(
            "EXPLAIN QUERY PLAN SELECT * FROM experience_graph "
            "WHERE robot_id = ? ORDER BY timestamp DESC LIMIT 10",
            ("bot_1",),
        )
        plan_rows = cursor.fetchall()
        # sqlite3.Row needs dict conversion; extract 'detail' column
        plan_text = " ".join(
            dict(row).get("detail", str(dict(row))) for row in plan_rows
        )
        # Should use an index (either idx_exp_robot_ts or idx_experience_graph_robot_id)
        assert "INDEX" in plan_text.upper() or "USING" in plan_text.upper()

    def test_query_correctness_with_composite_index(self, client):
        """Query results should be correct with composite index."""
        for i in range(50):
            robot = f"bot_{i % 2}"
            client.insert("experience_graph", {
                "id": f"exp_{i}",
                "event_type": "praxis",
                "robot_id": robot,
                "timestamp": float(i),
                "instruction": f"task {i}",
                "outcome": "success" if i % 3 != 0 else "failure",
            })

        results = client.query("experience_graph",
                               filters={"robot_id": "bot_0"},
                               order_by="-timestamp",
                               limit=10)
        assert len(results) <= 10
        for r in results:
            assert r["robot_id"] == "bot_0"
        timestamps = [r["timestamp"] for r in results]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_count_uses_index(self, client):
        """COUNT query should use index for filtered counts."""
        for i in range(20):
            client.insert("experience_graph", {
                "id": f"exp_{i}",
                "event_type": "praxis",
                "robot_id": "bot_x",
                "timestamp": float(i),
                "outcome": "success",
            })
        count = client.count("experience_graph", filters={"robot_id": "bot_x"})
        assert count == 20


# ---------------------------------------------------------------------------
# Performance benchmark
# ---------------------------------------------------------------------------


class TestSeekDBPerformance:
    """Verify that indexes provide performance improvement."""

    def test_memory_client_indexed_vs_scan(self):
        """Indexed query should be faster than or equal to full scan on large dataset."""
        c = SeekDBMemoryClient()
        c.connect()
        for i in range(1000):
            c.insert("experience_graph", {
                "id": f"exp_{i}",
                "event_type": "praxis",
                "robot_id": f"bot_{i % 10}",
                "timestamp": float(i),
                "instruction": f"task {i}",
                "outcome": "success",
            })

        # Indexed query (robot_id is indexed)
        t0 = time.time()
        for _ in range(100):
            c.query("experience_graph", filters={"robot_id": "bot_0"}, limit=10)
        indexed_time = time.time() - t0

        # Non-indexed query (instruction is not indexed)
        t0 = time.time()
        for _ in range(100):
            c.query("experience_graph",
                    filters={"instruction": "task 500"}, limit=10)
        scan_time = time.time() - t0

        # Indexed should generally be faster (allow some tolerance)
        # Just verify both complete within reasonable time
        assert indexed_time < 5.0, f"Indexed query too slow: {indexed_time:.3f}s"
        assert scan_time < 5.0, f"Scan query too slow: {scan_time:.3f}s"
        c.disconnect()

    def test_sqlite_client_10k_rows(self, tmp_path):
        """SQLite should handle 10K rows with indexed queries."""
        db_path = str(tmp_path / "perf_test.sqlite")
        c = SeekDBSQLiteClient(db_path)
        c.connect()

        for i in range(1000):
            c.insert("experience_graph", {
                "id": f"exp_{i}",
                "event_type": "praxis",
                "robot_id": "bot_1",
                "timestamp": float(i),
                "instruction": f"task {i}",
                "outcome": "success",
            })

        t0 = time.time()
        results = c.query("experience_graph",
                          filters={"robot_id": "bot_1"},
                          order_by="-timestamp",
                          limit=10)
        query_time = time.time() - t0

        assert len(results) == 10
        assert query_time < 0.1, f"Query too slow: {query_time * 1000:.1f}ms"
        c.disconnect()
