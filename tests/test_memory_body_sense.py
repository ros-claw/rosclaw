"""Tests for MemoryInterface body-conditioned failure writing (Phase 7)."""

from __future__ import annotations

import pytest

from rosclaw.core.event_bus import Event, EventBus
from rosclaw.memory.interface import MemoryInterface
from rosclaw.memory.seekdb_client import InMemoryKnowledgeStore
from rosclaw.memory.types import FailureMemory
from rosclaw.sense.config import SenseConfig
from rosclaw.sense.runtime import SenseRuntime


@pytest.fixture
def sense_runtime_kick_not_ready():
    bus = EventBus()
    cfg = SenseConfig(
        robot_id="g1_lab_01",
        collector="mock",
        update_hz=0.0,
        extra={"scenario": "kick_not_ready"},
    )
    runtime = SenseRuntime(cfg, event_bus=bus, robot_id="g1_lab_01")
    runtime.initialize()
    runtime.tick()
    yield runtime
    runtime.stop()


@pytest.fixture
def memory_with_sense(sense_runtime_kick_not_ready):
    client = InMemoryKnowledgeStore()
    client.connect()
    memory = MemoryInterface(
        robot_id="g1_lab_01",
        event_bus=EventBus(),
        seekdb_client=client,
    )
    memory.set_sense_runtime(sense_runtime_kick_not_ready)
    return memory


class TestMemoryBodyConditionFailure:
    def test_set_sense_runtime_initializes_adapter(self, memory_with_sense):
        assert memory_with_sense._memory_writer_adapter is not None

    def test_write_failure_memory_enriches_with_body_sense(self, memory_with_sense):
        failure = FailureMemory(
            failure_id="fail_001",
            robot_id="g1_lab_01",
            episode_id="ep_001",
            task_id="kick_ball",
            failure_type="body_sense_blocked",
            root_cause="joint overheating",
            recovery_hint="cooldown",
        )
        record_id = memory_with_sense.write_failure_memory(failure)
        assert record_id

        stored = memory_with_sense.seekdb_client.query(
            "failures", filters={"id": record_id}, limit=1
        )
        assert len(stored) == 1
        assert stored[0].get("body_condition_failure") is True
        assert "body_sense_evidence" in stored[0]
        evidence = stored[0]["body_sense_evidence"]
        assert evidence["overall_status"] == "not_ready"

    def test_write_failure_memory_dict_enriched(self, memory_with_sense):
        record = {
            "id": "fail_002",
            "failure_type": "blocked",
            "root_cause": "overheat",
        }
        record_id = memory_with_sense.write_failure_memory(record)
        stored = memory_with_sense.seekdb_client.query(
            "failures", filters={"id": record_id}, limit=1
        )
        assert stored[0].get("body_condition_failure") is True
        assert "body_sense_evidence" in stored[0]

    def test_firewall_blocked_handler_writes_body_condition(self, sense_runtime_kick_not_ready):
        client = InMemoryKnowledgeStore()
        client.connect()
        bus = EventBus()
        memory = MemoryInterface(robot_id="g1_lab_01", event_bus=bus, seekdb_client=client)
        memory.set_sense_runtime(sense_runtime_kick_not_ready)
        memory._do_initialize()

        bus.publish(
            Event(
                topic="firewall.action_blocked",
                payload={
                    "episode_id": "ep_fire_001",
                    "reason": "joint limit",
                },
                source="test",
            )
        )

        stored = client.query("failures", filters={"id": "ep_fire_001"}, limit=1)
        assert len(stored) == 1
        assert stored[0].get("body_condition_failure") is True
        assert stored[0].get("body_sense_evidence", {}).get("overall_status") == "not_ready"

    def test_ready_state_does_not_flag_body_condition(self):
        bus = EventBus()
        cfg = SenseConfig(
            robot_id="g1_lab_01", collector="mock", update_hz=0.0, extra={"scenario": "normal"}
        )
        runtime = SenseRuntime(cfg, event_bus=bus, robot_id="g1_lab_01")
        runtime.initialize()
        runtime.tick()

        client = InMemoryKnowledgeStore()
        client.connect()
        memory = MemoryInterface(robot_id="g1_lab_01", event_bus=EventBus(), seekdb_client=client)
        memory.set_sense_runtime(runtime)

        record_id = memory.write_failure_memory(
            {
                "id": "fail_ready",
                "failure_type": "other",
                "root_cause": "unknown",
            }
        )
        stored = client.query("failures", filters={"id": record_id}, limit=1)
        assert stored[0].get("body_condition_failure") is False
        runtime.stop()

    def test_no_sense_runtime_leaves_record_unchanged(self):
        client = InMemoryKnowledgeStore()
        client.connect()
        memory = MemoryInterface(robot_id="g1_lab_01", event_bus=EventBus(), seekdb_client=client)
        record_id = memory.write_failure_memory(
            {
                "id": "fail_no_sense",
                "failure_type": "other",
            }
        )
        stored = client.query("failures", filters={"id": record_id}, limit=1)
        assert "body_condition_failure" not in stored[0]
        assert "body_sense_evidence" not in stored[0]
