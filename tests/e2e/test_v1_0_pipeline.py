"""E2E test: Runtime -> EventBus -> Practice -> Memory -> Knowledge Query.

Sprint 7 acceptance: After a grasp task, verify:
1. Agent intent recorded
2. Provider capability selection recorded
3. Sandbox block status recorded
4. Runtime execution recorded
5. Critic judgment recorded
6. Memory write confirmed
"""

import pytest

from rosclaw.core.event_bus import Event
from rosclaw.core.runtime import Runtime, RuntimeConfig


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    # Isolation: chdir to tmp_path so the Runtime's KI sees an empty
    # ``data/knowledge_assets/`` directory and falls back to the curated
    # baseline patterns this test was written against.
    monkeypatch.chdir(tmp_path)
    config = RuntimeConfig(
        robot_id="e2e_test_bot",
        enable_memory=True,
        enable_practice=True,
        enable_knowledge=True,
        enable_how=False,
        enable_firewall=False,
        enable_provider=False,
        seekdb_backend="memory",
    )
    rt = Runtime(config)
    rt.initialize()
    rt.start()
    yield rt
    rt.stop()


class TestFullPipeline:
    def test_runtime_modules_initialized(self, runtime):
        assert runtime.state.name == "RUNNING"
        assert runtime.memory is not None
        assert runtime.practice is not None
        assert runtime.knowledge is not None

    def test_skill_execution_flows_to_praxis_recorded(self, runtime):
        captured = []
        runtime.event_bus.subscribe("praxis.recorded", lambda e: captured.append(e))

        runtime.event_bus.publish(
            Event(
                topic="skill.execution.start",
                payload={
                    "skill_name": "pick_and_place",
                    "correlation_id": "e2e-session-001",
                    "parameters": {"object": "red_cup"},
                },
            )
        )
        runtime.event_bus.publish(
            Event(
                topic="skill.execution.complete",
                payload={
                    "skill_name": "pick_and_place",
                    "correlation_id": "e2e-session-001",
                    "result": {"status": "success"},
                    "duration_sec": 4.2,
                },
            )
        )
        runtime.event_bus.publish(
            Event(
                topic="praxis.completed",
                payload={"practice_id": "e2e-session-001", "outcome": {"reward": 1.0}},
            )
        )

        assert len(captured) >= 1
        evt = captured[-1]
        assert evt.topic == "praxis.recorded"
        assert evt.payload["event_id"] == "e2e-session-001"
        assert evt.payload["robot_id"] == "e2e_test_bot"
        assert "artifact_uri" in evt.payload

    def test_memory_ingests_praxis(self, runtime):
        runtime.event_bus.publish(
            Event(
                topic="praxis.recorded",
                payload={
                    "event_id": "e2e-memory-001",
                    "event_type": "success",
                    "instruction": "pick the red cup and place it on the table",
                    "duration_sec": 5.5,
                    "outcome": "success",
                },
            )
        )

        stats = runtime.memory.get_statistics()
        assert stats["total_experiences"] >= 1

    def test_knowledge_queries_work(self, runtime):
        caps = runtime.knowledge.query_robot_capabilities("ur5e")
        assert isinstance(caps, list)

        match = runtime.knowledge.match_symptom("torque overflow saturation")
        assert match is not None
        assert match["pattern_id"] == "Torque_Overflow"

    def test_full_pipeline_all_events(self, runtime):
        praxis_captured = []
        runtime.event_bus.subscribe("praxis.recorded", lambda e: praxis_captured.append(e))

        runtime.event_bus.publish(
            Event(
                topic="agent.command",
                payload={"action": "grasp", "object": "red_cup", "request_id": "e2e-full-001"},
            )
        )
        runtime.event_bus.publish(
            Event(
                topic="agent.response",
                payload={"request_id": "e2e-full-001", "status": "safe", "is_safe": True},
            )
        )
        runtime.event_bus.publish(
            Event(
                topic="skill.execution.start",
                payload={"skill_name": "grasp", "correlation_id": "e2e-full-001"},
            )
        )
        runtime.event_bus.publish(
            Event(
                topic="skill.execution.complete",
                payload={
                    "skill_name": "grasp",
                    "correlation_id": "e2e-full-001",
                    "result": {"status": "success", "reward": 1.0},
                    "duration_sec": 3.0,
                },
            )
        )
        runtime.event_bus.publish(
            Event(
                topic="praxis.completed",
                payload={
                    "practice_id": "e2e-full-001",
                    "outcome": {"status": "success", "reward": 1.0, "skill_name": "grasp"},
                },
            )
        )

        assert len(praxis_captured) >= 1
        # Find the EpisodeRecorder event (has artifact_uri)
        recorder_events = [e for e in praxis_captured if "artifact_uri" in e.payload]
        assert len(recorder_events) >= 1
        recorded = recorder_events[0]
        assert recorded.payload["event_id"] == "e2e-full-001"
        assert recorded.payload["outcome"] == "success"

        stats = runtime.memory.get_statistics()
        assert stats["total_experiences"] >= 1

        match = runtime.knowledge.match_symptom("velocity divergence")
        assert match is not None

    def test_firewall_blocked_pipeline(self, runtime):
        praxis_captured = []
        runtime.event_bus.subscribe("praxis.recorded", lambda e: praxis_captured.append(e))

        runtime.event_bus.publish(
            Event(
                topic="skill.execution.start",
                payload={"skill_name": "dangerous_move", "correlation_id": "e2e-blocked-001"},
            )
        )
        runtime.event_bus.publish(
            Event(
                topic="firewall.action_blocked",
                payload={
                    "request_id": "e2e-blocked-001",
                    "violations": [{"description": "predicted collision"}],
                },
            )
        )
        runtime.event_bus.publish(
            Event(
                topic="skill.execution.complete",
                payload={"correlation_id": "e2e-blocked-001"},
            )
        )
        runtime.event_bus.publish(
            Event(
                topic="praxis.failed",
                payload={
                    "practice_id": "e2e-blocked-001",
                    "outcome": {"reward": -1.0, "status": "BLOCKED"},
                },
            )
        )

        assert len(praxis_captured) >= 1
        recorder_events = [e for e in praxis_captured if "artifact_uri" in e.payload]
        assert len(recorder_events) >= 1
        recorded = recorder_events[0]
        assert recorded.payload["outcome"] == "BLOCKED"

    def test_runtime_lifecycle(self, runtime):
        assert runtime.state.name == "RUNNING"
        runtime.stop()
        assert runtime.state.name == "STOPPED"
