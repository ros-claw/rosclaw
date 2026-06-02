"""
Tests for Memory capacity management (forgetting / eviction).

Verifies:
1. delete_experience() removes a single experience
2. forget_old_experiences() removes experiences older than N days
3. enforce_capacity() evicts oldest when over limit
4. get_capacity_info() returns utilization metrics
5. Auto-eviction on store_experience at capacity

P1 Issue 5: https://github.com/ros-claw/rosclaw-v1.0/issues/XXX
"""

import time

import pytest

from rosclaw.memory.interface import MemoryInterface
from rosclaw.memory.seekdb_client import SeekDBMemoryClient, SeekDBSQLiteClient


# ---------------------------------------------------------------------------
# SeekDB delete operations
# ---------------------------------------------------------------------------


class TestSeekDBDelete:
    """Verify delete() and delete_where() on both SeekDB backends."""

    def test_memory_client_delete(self):
        c = SeekDBMemoryClient()
        c.connect()
        c.insert("experience_graph", {"id": "e1", "robot_id": "r1",
                                      "timestamp": 1.0, "event_type": "t"})
        assert c.count("experience_graph") == 1
        assert c.delete("experience_graph", "e1") is True
        assert c.count("experience_graph") == 0

    def test_memory_client_delete_nonexistent(self):
        c = SeekDBMemoryClient()
        c.connect()
        assert c.delete("experience_graph", "nope") is False

    def test_memory_client_delete_where(self):
        c = SeekDBMemoryClient()
        c.connect()
        for i in range(5):
            c.insert("experience_graph", {
                "id": f"e{i}", "robot_id": "r1",
                "outcome": "failure" if i < 3 else "success",
                "timestamp": float(i), "event_type": "t",
            })
        deleted = c.delete_where("experience_graph", {"outcome": "failure"})
        assert deleted == 3
        assert c.count("experience_graph") == 2

    def test_memory_client_delete_updates_index(self):
        c = SeekDBMemoryClient()
        c.connect()
        c.insert("experience_graph", {"id": "e1", "robot_id": "r1",
                                      "outcome": "success", "timestamp": 1.0,
                                      "event_type": "t"})
        idx = c._indices["experience_graph"]
        assert "e1" in idx["robot_id"]["r1"]
        c.delete("experience_graph", "e1")
        assert "e1" not in idx["robot_id"].get("r1", set())

    def test_sqlite_client_delete(self, tmp_path):
        c = SeekDBSQLiteClient(str(tmp_path / "t.sqlite"))
        c.connect()
        c.insert("experience_graph", {"id": "e1", "robot_id": "r1",
                                      "timestamp": 1.0, "event_type": "t"})
        assert c.count("experience_graph") == 1
        assert c.delete("experience_graph", "e1") is True
        assert c.count("experience_graph") == 0
        c.disconnect()

    def test_sqlite_client_delete_where(self, tmp_path):
        c = SeekDBSQLiteClient(str(tmp_path / "t.sqlite"))
        c.connect()
        for i in range(5):
            c.insert("experience_graph", {
                "id": f"e{i}", "robot_id": "r1",
                "outcome": "failure" if i < 3 else "success",
                "timestamp": float(i), "event_type": "t",
            })
        deleted = c.delete_where("experience_graph", {"outcome": "failure"})
        assert deleted == 3
        assert c.count("experience_graph") == 2
        c.disconnect()


# ---------------------------------------------------------------------------
# MemoryInterface capacity management
# ---------------------------------------------------------------------------


@pytest.fixture
def mem():
    m = MemoryInterface("test_bot")
    m.initialize()
    yield m
    m.stop()


class TestDeleteExperience:
    """Verify delete_experience() method."""

    def test_delete_existing(self, mem):
        mem.store_experience("del1", "praxis", "to delete")
        assert mem.get_experience("del1") is not None
        assert mem.delete_experience("del1") is True
        assert mem.get_experience("del1") is None

    def test_delete_nonexistent(self, mem):
        assert mem.delete_experience("nope") is False

    def test_delete_updates_count(self, mem):
        mem.store_experience("d1", "praxis", "a")
        mem.store_experience("d2", "praxis", "b")
        assert mem.get_statistics()["total_experiences"] == 2
        mem.delete_experience("d1")
        assert mem.get_statistics()["total_experiences"] == 1


class TestForgetOldExperiences:
    """Verify forget_old_experiences() method."""

    def test_forget_old(self, mem):
        # Manually insert old experience with past timestamp
        old_ts = time.time() - (40 * 86400)  # 40 days ago
        mem._client.insert("experience_graph", {
            "id": "old1", "event_type": "praxis",
            "robot_id": "test_bot", "timestamp": old_ts,
            "instruction": "old task", "outcome": "success",
        })
        # Insert recent experience
        mem.store_experience("new1", "praxis", "recent task")

        deleted = mem.forget_old_experiences(max_age_days=30)
        assert deleted == 1
        assert mem.get_experience("old1") is None
        assert mem.get_experience("new1") is not None

    def test_forget_with_outcome_filter(self, mem):
        old_ts = time.time() - (40 * 86400)
        # Old failure
        mem._client.insert("experience_graph", {
            "id": "of1", "event_type": "praxis",
            "robot_id": "test_bot", "timestamp": old_ts,
            "instruction": "old failure", "outcome": "failure",
        })
        # Old success
        mem._client.insert("experience_graph", {
            "id": "os1", "event_type": "praxis",
            "robot_id": "test_bot", "timestamp": old_ts,
            "instruction": "old success", "outcome": "success",
        })

        # Only forget old failures
        deleted = mem.forget_old_experiences(max_age_days=30,
                                             outcome_filter="failure")
        assert deleted == 1
        assert mem.get_experience("of1") is None
        assert mem.get_experience("os1") is not None

    def test_forget_nothing_when_all_recent(self, mem):
        mem.store_experience("r1", "praxis", "recent")
        mem.store_experience("r2", "praxis", "also recent")
        deleted = mem.forget_old_experiences(max_age_days=30)
        assert deleted == 0


class TestEnforceCapacity:
    """Verify enforce_capacity() method."""

    def test_evicts_oldest_when_over_capacity(self, mem):
        for i in range(10):
            mem._client.insert("experience_graph", {
                "id": f"e{i}", "event_type": "praxis",
                "robot_id": "test_bot", "timestamp": float(1000 + i),
                "instruction": f"task {i}", "outcome": "success",
            })

        evicted = mem.enforce_capacity(max_experiences=7)
        assert evicted == 3
        # Oldest 3 should be gone (e0, e1, e2)
        assert mem.get_experience("e0") is None
        assert mem.get_experience("e1") is None
        assert mem.get_experience("e2") is None
        # Newest should remain
        assert mem.get_experience("e9") is not None

    def test_no_eviction_when_under_capacity(self, mem):
        mem.store_experience("e1", "praxis", "task")
        evicted = mem.enforce_capacity(max_experiences=100)
        assert evicted == 0

    def test_auto_eviction_on_store(self, mem):
        """Auto-eviction triggers every 100 inserts."""
        # Insert 100 experiences directly to populate the store
        for i in range(100):
            mem._client.insert("experience_graph", {
                "id": f"auto_{i}", "event_type": "praxis",
                "robot_id": "test_bot", "timestamp": float(i),
                "instruction": f"old task {i}", "outcome": "success",
            })

        # Set counter to 99 so the next store_experience triggers eviction
        mem._insert_count = 99

        original = MemoryInterface.DEFAULT_MAX_EXPERIENCES
        MemoryInterface.DEFAULT_MAX_EXPERIENCES = 50
        try:
            mem.store_experience("trigger", "praxis", "101st task")
            # After 100th insert, auto-eviction should fire
            total = mem._client.count("experience_graph",
                                      {"robot_id": "test_bot"})
            assert total <= 50
        finally:
            MemoryInterface.DEFAULT_MAX_EXPERIENCES = original


class TestCapacityInfo:
    """Verify get_capacity_info() method."""

    def test_empty_store(self, mem):
        info = mem.get_capacity_info()
        assert info["total_experiences"] == 0
        assert info["max_experiences"] > 0
        assert info["utilization"] == 0.0

    def test_with_data(self, mem):
        mem.store_experience("c1", "praxis", "task 1")
        mem.store_experience("c2", "praxis", "task 2")
        info = mem.get_capacity_info()
        assert info["total_experiences"] == 2
        assert info["utilization"] > 0
        assert info["newest_timestamp"] >= info["oldest_timestamp"]

    def test_utilization_fraction(self, mem):
        MemoryInterface.DEFAULT_MAX_EXPERIENCES = 100
        for i in range(25):
            mem.store_experience(f"u{i}", "praxis", f"task {i}")
        info = mem.get_capacity_info()
        assert abs(info["utilization"] - 0.25) < 0.01
        MemoryInterface.DEFAULT_MAX_EXPERIENCES = 10_000  # restore
