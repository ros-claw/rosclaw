"""Sprint 8 — Knowledge Plane tests.

Covers:
- PraxisEvent / FailureMemory / ArtifactRef types
- Core query APIs (retrieve_similar_episode, explain_last_failure, etc.)
- Artifact handling
- Event subscriptions
"""


import pytest

from rosclaw.memory.interface import MemoryInterface
from rosclaw.memory.types import PraxisEvent, FailureMemory, ArtifactRef
from rosclaw.core.event_bus import EventBus, Event


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mem():
    m = MemoryInterface("test_bot")
    m.initialize()
    yield m
    m.stop()


@pytest.fixture
def mem_with_bus():
    bus = EventBus()
    m = MemoryInterface("test_bot", event_bus=bus)
    m.initialize()
    yield m
    m.stop()


# ---------------------------------------------------------------------------
# PraxisEvent / FailureMemory types
# ---------------------------------------------------------------------------


class TestPraxisEvent:
    def test_roundtrip(self):
        evt = PraxisEvent(
            event_id="e1",
            robot_id="r1",
            event_type="grasp",
            episode_id="ep1",
            task_id="t1",
            payload={"key": "val"},
        )
        rec = evt.to_seekdb_record()
        restored = PraxisEvent.from_seekdb_record(rec)
        assert restored.event_id == "e1"
        assert restored.robot_id == "r1"
        assert restored.payload == {"key": "val"}


class TestFailureMemory:
    def test_roundtrip(self):
        fm = FailureMemory(
            failure_id="f1",
            robot_id="r1",
            failure_type="grasp_slippage",
            root_cause="force_low",
            recovery_hint="increase_force",
            sandbox_intervened=True,
            category="control",
        )
        rec = fm.to_seekdb_record()
        restored = FailureMemory.from_seekdb_record(rec)
        assert restored.failure_id == "f1"
        assert restored.failure_type == "grasp_slippage"
        assert restored.sandbox_intervened is True
        assert restored.category == "control"


class TestArtifactRef:
    def test_build_uri(self):
        uri = ArtifactRef.build_uri("episodes", "2026-05-29", "ep_0001", "episode.mcap")
        assert uri == "artifact://episodes/2026-05-29/ep_0001/episode.mcap"

    def test_to_seekdb_record(self):
        art = ArtifactRef(
            artifact_id="a1",
            artifact_type="mcap",
            uri="artifact://episodes/2026-05-29/ep_0001/episode.mcap",
            episode_id="ep1",
            size_bytes=1024,
        )
        rec = art.to_seekdb_record()
        assert rec["id"] == "a1"
        assert rec["artifact_type"] == "mcap"
        assert rec["size_bytes"] == 1024


# ---------------------------------------------------------------------------
# Core query APIs
# ---------------------------------------------------------------------------


class TestWritePraxisEvent:
    def test_write_and_retrieve(self, mem):
        evt = PraxisEvent(event_id="pe1", robot_id="test_bot", event_type="step")
        rid = mem.write_praxis_event(evt)
        assert rid == "pe1"
        rows = mem._client.query("praxis_events", filters={"id": "pe1"})
        assert len(rows) == 1
        assert rows[0]["event_type"] == "step"


class TestWriteFailureMemory:
    def test_write_and_retrieve(self, mem):
        fm = FailureMemory(
            failure_id="f1",
            robot_id="test_bot",
            failure_type="collision",
            root_cause="obstacle_unexpected",
            recovery_hint="replan",
        )
        rid = mem.write_failure_memory(fm)
        assert rid == "f1"
        rows = mem._client.query("failures", filters={"id": "f1"})
        assert len(rows) == 1
        assert rows[0]["failure_type"] == "collision"


class TestRetrieveSimilarEpisode:
    def test_empty_returns_empty(self, mem):
        assert mem.retrieve_similar_episode(task_id="t1") == []

    def test_filter_by_task(self, mem):
        mem._client.insert("episodes", {
            "id": "ep1", "task_id": "t1", "robot_id": "test_bot",
            "started_at": 1000.0, "outcome": "success", "artifact_uri": "",
        })
        mem._client.insert("episodes", {
            "id": "ep2", "task_id": "t2", "robot_id": "test_bot",
            "started_at": 2000.0, "outcome": "failure", "artifact_uri": "",
        })
        results = mem.retrieve_similar_episode(task_id="t1")
        assert len(results) == 1
        assert results[0]["id"] == "ep1"

    def test_limit(self, mem):
        for i in range(5):
            mem._client.insert("episodes", {
                "id": f"ep{i}", "task_id": "t1", "robot_id": "test_bot",
                "started_at": float(i), "outcome": "success", "artifact_uri": "",
            })
        results = mem.retrieve_similar_episode(task_id="t1", n=3)
        assert len(results) == 3


class TestExplainLastFailure:
    def test_no_failure_returns_none(self, mem):
        assert mem.explain_last_failure(task_id="t1") is None

    def test_returns_latest(self, mem):
        mem._client.insert("failures", {
            "id": "f1", "task_id": "t1", "robot_id": "test_bot",
            "failure_type": "collision", "root_cause": "old",
            "timestamp": 1000.0, "recovery_hint": "",
        })
        mem._client.insert("failures", {
            "id": "f2", "task_id": "t1", "robot_id": "test_bot",
            "failure_type": "slippage", "root_cause": "recent",
            "timestamp": 2000.0, "recovery_hint": "",
        })
        result = mem.explain_last_failure(task_id="t1")
        assert result is not None
        assert result["id"] == "f2"
        assert result["root_cause"] == "recent"


class TestRetrieveRobotCapability:
    def test_empty_when_no_data(self, mem):
        assert mem.retrieve_robot_capability("test_bot") == []

    def test_returns_capabilities(self, mem):
        mem._client.insert("knowledge_graph", {
            "id": "k1", "subject": "test_bot", "predicate": "has_capability",
            "object": "grasp", "confidence": 0.9, "source": "seed", "timestamp": 1.0,
        })
        caps = mem.retrieve_robot_capability("test_bot")
        assert len(caps) == 1
        assert caps[0]["object"] == "grasp"


class TestRetrieveSkillSuccessPattern:
    def test_missing_returns_none(self, mem):
        assert mem.retrieve_skill_success_pattern("grasp") is None

    def test_returns_pattern(self, mem):
        mem._client.insert("success_patterns", {
            "id": "sp1", "skill_id": "grasp", "robot_id": "test_bot",
            "context_hash": "abc", "success_count": 5, "avg_duration_sec": 2.5,
        })
        pat = mem.retrieve_skill_success_pattern("grasp", "test_bot")
        assert pat is not None
        assert pat["success_count"] == 5


class TestRetrieveSafetyCase:
    def test_filter_by_constraint(self, mem):
        mem._client.insert("heuristic_rules", {
            "id": "h1", "condition": "force_limit", "action": "stop",
            "priority": 5, "success_count": 0, "failure_count": 0, "last_triggered": None,
        })
        cases = mem.retrieve_safety_case(constraint_type="force_limit")
        assert len(cases) == 1
        assert cases[0]["action"] == "stop"


# ---------------------------------------------------------------------------
# Artifact handling
# ---------------------------------------------------------------------------


class TestArtifactHandling:
    def test_store_and_get(self, mem):
        art = ArtifactRef(
            artifact_id="a1",
            artifact_type="mcap",
            uri="artifact://episodes/2026-05-29/ep_0001/episode.mcap",
            episode_id="ep1",
            size_bytes=2048,
        )
        rid = mem.store_artifact(art)
        assert rid == "a1"
        fetched = mem.get_artifact("a1")
        assert fetched is not None
        assert fetched.uri == art.uri
        assert fetched.size_bytes == 2048

    def test_find_by_episode(self, mem):
        mem.store_artifact(ArtifactRef("a1", "mcap", "u1", episode_id="ep1"))
        mem.store_artifact(ArtifactRef("a2", "video", "u2", episode_id="ep1"))
        mem.store_artifact(ArtifactRef("a3", "mcap", "u3", episode_id="ep2"))
        results = mem.find_artifacts_by_episode("ep1")
        assert len(results) == 2
        results = mem.find_artifacts_by_episode("ep1", artifact_type="video")
        assert len(results) == 1
        assert results[0].artifact_type == "video"


# ---------------------------------------------------------------------------
# Event subscriptions
# ---------------------------------------------------------------------------


class TestEventSubscriptions:
    def test_practice_event_created(self, mem_with_bus):
        mem_with_bus.event_bus.publish(Event(
            topic="rosclaw.practice.event.created",
            payload={
                "event_id": "pe1",
                "event_type": "step_completed",
                "robot_id": "test_bot",
                "episode_id": "ep1",
                "task_id": "t1",
            },
            source="test",
        ))
        rows = mem_with_bus._client.query("praxis_events", filters={"id": "pe1"})
        assert len(rows) == 1
        assert rows[0]["event_type"] == "step_completed"

    def test_sandbox_episode_failed(self, mem_with_bus):
        mem_with_bus.event_bus.publish(Event(
            topic="rosclaw.sandbox.episode.failed",
            payload={
                "failure_id": "f1",
                "robot_id": "test_bot",
                "failure_type": "collision",
                "root_cause": "obstacle",
                "sandbox_intervened": True,
            },
            source="test",
        ))
        rows = mem_with_bus._client.query("failures", filters={"id": "f1"})
        assert len(rows) == 1
        meta = rows[0].get("metadata", {})
        assert meta.get("sandbox_intervened") is True

    def test_sandbox_episode_succeeded(self, mem_with_bus):
        mem_with_bus.event_bus.publish(Event(
            topic="rosclaw.sandbox.episode.succeeded",
            payload={
                "pattern_id": "sp1",
                "skill_id": "grasp",
                "robot_id": "test_bot",
                "success_count": 3,
                "avg_duration_sec": 1.2,
            },
            source="test",
        ))
        rows = mem_with_bus._client.query("success_patterns", filters={"id": "sp1"})
        assert len(rows) == 1
        assert rows[0]["success_count"] == 3

    def test_recovery_hint_generated(self, mem_with_bus):
        mem_with_bus._client.insert("failures", {
            "id": "f1", "task_id": "t1", "robot_id": "test_bot",
            "failure_type": "collision", "root_cause": "obs",
            "timestamp": 1000.0, "recovery_hint": "old_hint",
        })
        mem_with_bus.event_bus.publish(Event(
            topic="rosclaw.how.recovery_hint.generated",
            payload={
                "failure_id": "f1",
                "hint": "new_hint: replan",
            },
            source="test",
        ))
        rows = mem_with_bus._client.query("failures", filters={"id": "f1"})
        assert rows[0]["recovery_hint"] == "new_hint: replan"
