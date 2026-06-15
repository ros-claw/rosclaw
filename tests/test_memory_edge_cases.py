"""Edge-case tests for Memory/How modules.

Covers boundary values, empty inputs, and exception paths not exercised
by the main test suites.
"""

import time

from rosclaw.core.event_bus import Event, EventBus
from rosclaw.how.engine import HeuristicEngine
from rosclaw.how.recovery_loop import RecoveryLoop
from rosclaw.memory.interface import MemoryInterface
from rosclaw.memory.types import ArtifactRef

# ---------------------------------------------------------------------------
# MemoryInterface — empty inputs & boundary values
# ---------------------------------------------------------------------------


class TestStoreExperienceEdgeCases:
    def test_empty_event_id(self):
        mem = MemoryInterface("test_bot")
        mem.initialize()
        # Empty string event_id is allowed but potentially dangerous
        rid = mem.store_experience(event_id="", event_type="praxis",
                                    instruction="task", outcome="success")
        assert rid == ""
        exp = mem.get_experience("")
        assert exp is not None
        assert exp["instruction"] == "task"
        mem.stop()

    def test_duplicate_event_id_overwrites(self):
        mem = MemoryInterface("test_bot")
        mem.initialize()
        mem.store_experience(event_id="dup", event_type="praxis",
                             instruction="first", outcome="success")
        mem.store_experience(event_id="dup", event_type="praxis",
                             instruction="second", outcome="failure")
        exp = mem.get_experience("dup")
        assert exp["instruction"] == "second"
        assert exp["outcome"] == "failure"
        mem.stop()


class TestFindSimilarExperiencesEdgeCases:
    def test_limit_zero_returns_empty(self):
        mem = MemoryInterface("test_bot")
        mem.initialize()
        mem.store_experience("e1", "praxis", "pick up cup", outcome="success")
        results = mem.find_similar_experiences("pick", limit=0)
        assert results == []
        mem.stop()

    def test_special_chars_only_returns_empty(self):
        mem = MemoryInterface("test_bot")
        mem.initialize()
        mem.store_experience("e1", "praxis", "pick up cup", outcome="success")
        results = mem.find_similar_experiences("!@#$%", limit=5)
        assert results == []
        mem.stop()

    def test_query_exceeds_200_experiences(self):
        """find_similar_experiences caps at 200 recent experiences."""
        mem = MemoryInterface("test_bot")
        mem.initialize()
        for i in range(250):
            mem.store_experience(f"e{i}", "praxis", f"task {i}",
                                 outcome="success")
        # Should not crash; only searches 200 most recent
        results = mem.find_similar_experiences("task", limit=5)
        assert len(results) <= 5
        mem.stop()


class TestFindAnalogyEdgeCases:
    def test_empty_error_log_returns_none(self):
        mem = MemoryInterface("test_bot")
        mem.initialize()
        assert mem.find_analogy("") is None
        mem.stop()

    def test_no_failures_returns_none(self):
        mem = MemoryInterface("test_bot")
        mem.initialize()
        # Only successes stored
        mem.store_experience("s1", "praxis", "task", outcome="success")
        assert mem.find_analogy("error") is None
        mem.stop()


class TestCapacityEdgeCases:
    def test_forget_zero_days_deletes_all(self):
        mem = MemoryInterface("test_bot")
        mem.initialize()
        mem.store_experience("e1", "praxis", "task1", outcome="success")
        mem.store_experience("e2", "praxis", "task2", outcome="success")
        deleted = mem.forget_old_experiences(max_age_days=0)
        assert deleted >= 2
        assert mem.get_experience("e1") is None
        mem.stop()

    def test_enforce_capacity_zero_evicts_all(self):
        mem = MemoryInterface("test_bot")
        mem.initialize()
        mem.store_experience("e1", "praxis", "task1", outcome="success")
        mem.store_experience("e2", "praxis", "task2", outcome="success")
        evicted = mem.enforce_capacity(max_experiences=0)
        assert evicted >= 2
        assert mem._client.count("experience_graph") == 0
        mem.stop()

    def test_get_capacity_info_empty_store(self):
        mem = MemoryInterface("test_bot")
        mem.initialize()
        info = mem.get_capacity_info()
        assert info["total_experiences"] == 0
        assert info["utilization"] == 0.0
        mem.stop()


class TestStatisticsEdgeCases:
    def test_statistics_empty_db(self):
        mem = MemoryInterface("test_bot")
        mem.initialize()
        stats = mem.get_statistics()
        assert stats["total_experiences"] == 0
        assert stats["success_rate"] == 0.0
        mem.stop()


# ---------------------------------------------------------------------------
# Sprint 8 API — boundary values
# ---------------------------------------------------------------------------


class TestSprint8EdgeCases:
    def test_explain_last_failure_no_failures(self):
        mem = MemoryInterface("test_bot")
        mem.initialize()
        assert mem.explain_last_failure() is None
        mem.stop()

    def test_retrieve_skill_success_pattern_missing(self):
        mem = MemoryInterface("test_bot")
        mem.initialize()
        assert mem.retrieve_skill_success_pattern("nonexistent") is None
        mem.stop()

    def test_retrieve_robot_capability_empty(self):
        mem = MemoryInterface("test_bot")
        mem.initialize()
        caps = mem.retrieve_robot_capability()
        assert caps == []
        mem.stop()

    def test_write_praxis_event_dict_fallback(self):
        mem = MemoryInterface("test_bot")
        mem.initialize()
        # Verify dict path works (backward compat)
        record = {
            "id": "evt1",
            "event_id": "evt1",
            "robot_id": "test_bot",
            "event_type": "grasp",
            "timestamp": time.time(),
            "payload": {"object": "cup"},
        }
        rid = mem.write_praxis_event(record)
        assert rid is not None
        mem.stop()

    def test_write_failure_memory_dict_fallback(self):
        mem = MemoryInterface("test_bot")
        mem.initialize()
        record = {
            "id": "fail1",
            "failure_id": "fail1",
            "robot_id": "test_bot",
            "failure_type": "collision",
            "root_cause": "obstacle",
        }
        rid = mem.write_failure_memory(record)
        assert rid is not None
        mem.stop()


# ---------------------------------------------------------------------------
# RecoveryLoop — idempotency & resource management
# ---------------------------------------------------------------------------


class TestRecoveryLoopEdgeCases:
    def test_duplicate_retry_intent_is_idempotent(self):
        bus = EventBus()
        mem = MemoryInterface("test_bot", event_bus=bus)
        mem.initialize()
        he = HeuristicEngine(mem.seekdb_client)
        rl = RecoveryLoop(bus, mem, he)
        rl.subscribe()

        # Publish same hint twice
        for _ in range(2):
            rl._on_recovery_hint(Event(
                topic="rosclaw.how.recovery_hint.generated",
                payload={
                    "request_id": "dup_req",
                    "failure_type": "slip",
                    "retry_plan": {
                        "rule_id": "rule_slip",
                        "parameter_patch": {"force": 0.1},
                        "max_retries": 3,
                    },
                },
                source="test",
            ))

        rows = mem.seekdb_client.query("retries", filters={"id": "dup_req"})
        assert len(rows) == 1  # Idempotent: only one record
        rl.unsubscribe()
        mem.stop()

    def test_executor_reused_across_calls(self):
        bus = EventBus()
        mem = MemoryInterface("test_bot", event_bus=bus)
        mem.initialize()
        he = HeuristicEngine(mem.seekdb_client)
        rl = RecoveryLoop(bus, mem, he)
        rl.subscribe()

        # Trigger _run_async via success handler
        he._seekdb.insert("heuristic_rules", {
            "id": "rule_test", "condition": "x", "action": "y",
            "priority": 1, "success_count": 0, "failure_count": 0,
        })
        he._rule_cache = {"rule_test": {"id": "rule_test"}}
        he._cache_valid = True

        rl._on_recovery_hint(Event(
            topic="rosclaw.how.recovery_hint.generated",
            payload={
                "request_id": "req_exec",
                "failure_type": "x",
                "retry_plan": {"rule_id": "rule_test", "max_retries": 3},
            },
            source="test",
        ))
        rl._on_retry_success(Event(
            topic="rosclaw.sandbox.episode.succeeded",
            payload={"request_id": "req_exec", "episode_id": "ep1"},
            source="test",
        ))

        # Executor should be lazily created and reused
        assert rl._executor is not None
        rl.unsubscribe()
        assert rl._executor is None  # Shutdown on unsubscribe
        mem.stop()

    def test_unsubscribe_when_bus_is_none(self):
        mem = MemoryInterface("test_bot")
        mem.initialize()
        he = HeuristicEngine(mem.seekdb_client)
        rl = RecoveryLoop(None, mem, he)
        # Should not raise
        rl.unsubscribe()
        mem.stop()


# ---------------------------------------------------------------------------
# Types — boundary values
# ---------------------------------------------------------------------------


class TestArtifactRefEdgeCases:
    def test_build_uri_with_empty_fields(self):
        uri = ArtifactRef.build_uri("", "", "", "")
        assert uri == "artifact://///"

    def test_to_seekdb_record_with_none(self):
        art = ArtifactRef(
            artifact_id="a1",
            artifact_type="mcap",
            uri="artifact://episodes/2026-06-01/ep1/episode.mcap",
            episode_id=None,
            size_bytes=None,
        )
        rec = art.to_seekdb_record()
        assert rec["episode_id"] is None
        assert rec["size_bytes"] is None
