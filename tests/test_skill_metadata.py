"""
Tests for skill_metadata writing from SkillExecutor.

Verifies that SkillExecutor writes to skill_metadata table
after each execution, accumulating success/failure counts
and average duration.

P1 Issue 4: https://github.com/ros-claw/rosclaw-v1.0/issues/XXX
"""

import time
from pathlib import Path
from typing import Any

import pytest
from pytest import MonkeyPatch

from rosclaw.core.event_bus import EventBus
from rosclaw.memory.seekdb_client import SeekDBMemoryClient
from rosclaw.skill_manager.executor import SkillExecutor
from rosclaw.skill_manager.registry import SkillEntry, SkillRegistry


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Isolate tests from the real ~/.rosclaw workspace."""
    monkeypatch.setenv("HOME", str(tmp_path))


@pytest.fixture
def env() -> Any:
    """Test environment: EventBus + Registry + SeekDB + Executor."""
    bus = EventBus()
    registry = SkillRegistry(event_bus=bus)
    seekdb = SeekDBMemoryClient()
    seekdb.connect()
    executor = SkillExecutor(event_bus=bus, registry=registry, seekdb_client=seekdb)
    registry.initialize()
    executor.initialize()
    yield bus, registry, seekdb, executor
    executor.stop()
    registry.stop()
    seekdb.disconnect()


def _make_skill(name="pick_up", handler=None, skill_type="programmed",
                preconditions=None):
    """Helper to create a SkillEntry."""
    return SkillEntry(
        name=name,
        description=f"Skill: {name}",
        skill_type=skill_type,
        parameters={"target": "cup"},
        preconditions=preconditions or [],
        handler=handler,
    )


class TestSkillMetadataWriting:
    """Verify skill_metadata table is written after execution."""

    def test_successful_execution_writes_metadata(self, env):
        """Successful execution should create skill_metadata row."""
        bus, registry, seekdb, executor = env
        registry.register(_make_skill("pick_up", handler=lambda p: {"ok": True}))

        result = executor.execute("pick_up")
        assert result["status"] == "success"

        rows = seekdb.query("skill_metadata", filters={"skill_id": "pick_up"})
        assert len(rows) == 1
        row = rows[0]
        assert row["name"] == "pick_up"
        assert row["success_count"] == 1
        assert row["failure_count"] == 0
        assert row["category"] == "programmed"
        assert row["source"] == "skill_executor"

    def test_failed_execution_writes_metadata(self, env):
        """Failed execution should record failure_count."""
        bus, registry, seekdb, executor = env

        def failing_handler(params):
            raise RuntimeError("motor timeout")

        registry.register(_make_skill("grasp", handler=failing_handler))

        result = executor.execute("grasp")
        assert result["status"] == "error"

        rows = seekdb.query("skill_metadata", filters={"skill_id": "grasp"})
        assert len(rows) == 1
        assert rows[0]["success_count"] == 0
        assert rows[0]["failure_count"] == 1

    def test_multiple_executions_accumulate_counts(self, env):
        """Multiple executions should accumulate success/failure counts."""
        bus, registry, seekdb, executor = env

        call_count = 0

        def flaky_handler(params):
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:
                raise RuntimeError("intermittent failure")
            return {"ok": True}

        registry.register(_make_skill("place", handler=flaky_handler))

        for _ in range(6):
            executor.execute("place")

        rows = seekdb.query("skill_metadata", filters={"skill_id": "place"})
        assert len(rows) == 1
        row = rows[0]
        # 4 successes (1,2,4,5) and 2 failures (3,6)
        assert row["success_count"] == 4
        assert row["failure_count"] == 2

    def test_avg_duration_is_computed(self, env):
        """avg_duration_sec should be computed from execution times."""
        bus, registry, seekdb, executor = env

        def slow_handler(params):
            time.sleep(0.01)
            return {"ok": True}

        registry.register(_make_skill("scan", handler=slow_handler))

        executor.execute("scan")
        executor.execute("scan")

        rows = seekdb.query("skill_metadata", filters={"skill_id": "scan"})
        assert len(rows) == 1
        avg = rows[0]["avg_duration_sec"]
        assert avg > 0.005  # Should be at least 5ms per execution

    def test_last_used_is_updated(self, env):
        """last_used timestamp should be updated on each execution."""
        bus, registry, seekdb, executor = env
        registry.register(_make_skill("move", handler=lambda p: {"ok": True}))

        before = time.time()
        executor.execute("move")
        after = time.time()

        rows = seekdb.query("skill_metadata", filters={"skill_id": "move"})
        assert before <= rows[0]["last_used"] <= after

    def test_prerequisites_stored(self, env):
        """Prerequisites should be stored in metadata."""
        bus, registry, seekdb, executor = env
        skill = _make_skill(
            "pick_up", handler=lambda p: {"ok": True},
            preconditions=["skill:locate"],
        )
        # Register the prerequisite skill first
        registry.register(_make_skill("locate", handler=lambda p: {"ok": True}))
        registry.register(skill)
        executor.execute("pick_up")

        rows = seekdb.query("skill_metadata", filters={"skill_id": "pick_up"})
        prereqs = rows[0]["prerequisites"]
        assert isinstance(prereqs, list)
        assert "skill:locate" in prereqs

    def test_no_seekdb_client_no_crash(self):
        """Executor without seekdb_client should work fine."""
        bus = EventBus()
        registry = SkillRegistry(event_bus=bus)
        executor = SkillExecutor(event_bus=bus, registry=registry)  # no seekdb
        registry.initialize()
        executor.initialize()

        registry.register(_make_skill("test", handler=lambda p: {"ok": True}))
        result = executor.execute("test")
        assert result["status"] == "success"

        executor.stop()
        registry.stop()

    def test_dispatched_skill_writes_metadata(self, env):
        """Skills without handlers (dispatched) should still write metadata."""
        bus, registry, seekdb, executor = env
        registry.register(_make_skill("remote_task"))  # no handler

        result = executor.execute("remote_task")
        assert result["status"] == "dispatched"

        rows = seekdb.query("skill_metadata", filters={"skill_id": "remote_task"})
        assert len(rows) == 1

    def test_result_includes_duration(self, env):
        """Execution result should include duration_sec."""
        bus, registry, seekdb, executor = env
        registry.register(_make_skill("quick", handler=lambda p: {"ok": True}))

        result = executor.execute("quick")
        assert "duration_sec" in result
        assert result["duration_sec"] >= 0.0

    def test_metadata_field_stored(self, env):
        """Custom metadata dict from SkillEntry should be persisted."""
        bus, registry, seekdb, executor = env
        skill = _make_skill("custom", handler=lambda p: {"ok": True})
        skill.metadata = {"version": "1.0", "author": "test"}
        registry.register(skill)

        executor.execute("custom")

        rows = seekdb.query("skill_metadata", filters={"skill_id": "custom"})
        meta = rows[0].get("metadata", {})
        assert meta.get("version") == "1.0"
