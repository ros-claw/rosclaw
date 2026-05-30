"""Tests for EpisodeRecorder — Sprint 7 artifact management and event assembly."""

import json
import os
import tempfile

import pytest

from rosclaw.core.event_bus import EventBus, Event
from rosclaw.practice.episode_recorder import EpisodeRecorder


@pytest.fixture
def temp_artifact_dir():
    with tempfile.TemporaryDirectory() as td:
        yield td


@pytest.fixture
def bus():
    return EventBus()


@pytest.fixture
def recorder(bus, temp_artifact_dir):
    r = EpisodeRecorder("test_bot", bus, artifact_base_dir=temp_artifact_dir)
    r.initialize()
    yield r
    r.stop()


class TestLifecycle:
    def test_initialize_creates_artifact_dir(self, bus, temp_artifact_dir):
        r = EpisodeRecorder("bot", bus, artifact_base_dir=temp_artifact_dir)
        r.initialize()
        assert os.path.isdir(temp_artifact_dir)
        r.stop()

    def test_stop_finalizes_active_buffers(self, bus, temp_artifact_dir):
        r = EpisodeRecorder("bot", bus, artifact_base_dir=temp_artifact_dir)
        r.initialize()
        buf = r._get_or_create_buffer("ep_test")
        buf.received_events.add("skill.execution.start")
        r.stop()
        assert r.active_episode_count == 0


class TestEventBuffering:
    def test_skill_start_creates_buffer(self, recorder):
        recorder._event_bus.publish(Event(
            topic="skill.execution.start",
            payload={"skill_name": "pick", "correlation_id": "ep_001"},
        ))
        assert recorder.active_episode_count == 1
        buf = recorder._buffers["ep_001"]
        assert buf.semantic_intent == "pick"
        assert "skill.execution.start" in buf.received_events

    def test_skill_complete_then_praxis_finalizes(self, recorder, temp_artifact_dir):
        recorder._event_bus.publish(Event(
            topic="skill.execution.start",
            payload={"skill_name": "pick", "correlation_id": "ep_002"},
        ))
        recorder._event_bus.publish(Event(
            topic="skill.execution.complete",
            payload={"correlation_id": "ep_002", "result": {"status": "success"}, "duration_sec": 3.5},
        ))
        recorder._event_bus.publish(Event(
            topic="praxis.completed",
            payload={"practice_id": "ep_002", "outcome": {"reward": 1.0}},
        ))
        assert recorder.active_episode_count == 0
        meta_path = os.path.join(temp_artifact_dir, "episodes", "ep_002", "metadata.json")
        assert os.path.exists(meta_path)

    def test_praxis_completed_sets_status(self, recorder, temp_artifact_dir):
        recorder._event_bus.publish(Event(
            topic="skill.execution.start",
            payload={"skill_name": "pick", "correlation_id": "ep_003"},
        ))
        recorder._event_bus.publish(Event(
            topic="skill.execution.complete",
            payload={"correlation_id": "ep_003"},
        ))
        recorder._event_bus.publish(Event(
            topic="praxis.completed",
            payload={"practice_id": "ep_003", "outcome": {"reward": 0.95}},
        ))
        meta_path = os.path.join(temp_artifact_dir, "episodes", "ep_003", "metadata.json")
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["status"] == "success"
        assert meta["reward"] == 0.95

    def test_praxis_failed_sets_failure(self, recorder, temp_artifact_dir):
        recorder._event_bus.publish(Event(
            topic="skill.execution.start",
            payload={"skill_name": "pick", "correlation_id": "ep_004"},
        ))
        recorder._event_bus.publish(Event(
            topic="skill.execution.complete",
            payload={"correlation_id": "ep_004"},
        ))
        recorder._event_bus.publish(Event(
            topic="praxis.failed",
            payload={"practice_id": "ep_004", "outcome": {"reward": -0.8}, "error_log": "slip"},
        ))
        meta_path = os.path.join(temp_artifact_dir, "episodes", "ep_004", "metadata.json")
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["status"] == "failure"
        assert meta["reward"] == -0.8

    def test_firewall_blocked_sets_blocked(self, recorder, temp_artifact_dir):
        recorder._event_bus.publish(Event(
            topic="skill.execution.start",
            payload={"skill_name": "pick", "correlation_id": "ep_005"},
        ))
        recorder._event_bus.publish(Event(
            topic="firewall.action_blocked",
            payload={
                "request_id": "ep_005",
                "violations": [{"description": "collision detected"}],
            },
        ))
        meta_path = os.path.join(temp_artifact_dir, "episodes", "ep_005", "metadata.json")
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["status"] == "BLOCKED"
        assert meta["sandbox_blocked"] is True
        assert meta["sandbox_block_reason"] == "collision detected"

    def test_safety_violation_sets_blocked(self, recorder, temp_artifact_dir):
        recorder._event_bus.publish(Event(
            topic="skill.execution.start",
            payload={"skill_name": "pick", "correlation_id": "ep_006"},
        ))
        recorder._event_bus.publish(Event(
            topic="safety.violation",
            payload={"request_id": "ep_006", "violations": ["joint limit exceeded"]},
        ))
        meta_path = os.path.join(temp_artifact_dir, "episodes", "ep_006", "metadata.json")
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["status"] == "BLOCKED"

    def test_agent_response_records_provider_trace(self, recorder, temp_artifact_dir):
        recorder._event_bus.publish(Event(
            topic="skill.execution.start",
            payload={"skill_name": "pick", "correlation_id": "ep_007"},
        ))
        recorder._event_bus.publish(Event(
            topic="agent.response",
            payload={"request_id": "ep_007", "status": "safe", "is_safe": True},
        ))
        recorder._event_bus.publish(Event(
            topic="praxis.completed",
            payload={"practice_id": "ep_007", "outcome": {"reward": 1.0}},
        ))
        trace_path = os.path.join(temp_artifact_dir, "episodes", "ep_007", "provider_trace.jsonl")
        assert os.path.exists(trace_path)
        with open(trace_path) as f:
            traces = [json.loads(line) for line in f]
        assert len(traces) == 1
        assert traces[0]["status"] == "safe"

    def test_out_of_order_events(self, recorder, temp_artifact_dir):
        recorder._event_bus.publish(Event(
            topic="skill.execution.complete",
            payload={"correlation_id": "ep_008", "result": {"status": "success"}},
        ))
        recorder._event_bus.publish(Event(
            topic="skill.execution.start",
            payload={"skill_name": "pick", "correlation_id": "ep_008"},
        ))
        recorder._event_bus.publish(Event(
            topic="praxis.completed",
            payload={"practice_id": "ep_008", "outcome": {"reward": 1.0}},
        ))
        meta_path = os.path.join(temp_artifact_dir, "episodes", "ep_008", "metadata.json")
        assert os.path.exists(meta_path)
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["is_complete"] is True


class TestArtifactFiles:
    def test_all_artifact_files_written(self, recorder, temp_artifact_dir):
        recorder._event_bus.publish(Event(
            topic="skill.execution.start",
            payload={"skill_name": "pick", "correlation_id": "ep_009"},
        ))
        recorder._event_bus.publish(Event(
            topic="praxis.completed",
            payload={"practice_id": "ep_009", "outcome": {"reward": 1.0}},
        ))
        ep_dir = os.path.join(temp_artifact_dir, "episodes", "ep_009")
        assert os.path.exists(os.path.join(ep_dir, "metadata.json"))
        assert os.path.exists(os.path.join(ep_dir, "trajectory.jsonl"))
        assert os.path.exists(os.path.join(ep_dir, "provider_trace.jsonl"))
        assert os.path.exists(os.path.join(ep_dir, "sandbox_replay.json"))
        # CRITICAL FIX: agent_request.json is now part of the 7 artifact files
        assert os.path.exists(os.path.join(ep_dir, "agent_request.json"))

    def test_trajectory_jsonl_content(self, recorder, temp_artifact_dir):
        recorder._event_bus.publish(Event(
            topic="skill.execution.start",
            payload={"skill_name": "grasp", "parameters": {"force": 10}, "correlation_id": "ep_010"},
        ))
        recorder._event_bus.publish(Event(
            topic="praxis.completed",
            payload={"practice_id": "ep_010", "outcome": {"reward": 1.0}},
        ))
        traj_path = os.path.join(temp_artifact_dir, "episodes", "ep_010", "trajectory.jsonl")
        with open(traj_path) as f:
            entries = [json.loads(line) for line in f]
        assert len(entries) == 1
        assert entries[0]["phase"] == "start"
        assert entries[0]["skill_name"] == "grasp"

    def test_sandbox_replay_content(self, recorder, temp_artifact_dir):
        recorder._event_bus.publish(Event(
            topic="skill.execution.start",
            payload={"skill_name": "pick", "correlation_id": "ep_011"},
        ))
        recorder._event_bus.publish(Event(
            topic="firewall.action_blocked",
            payload={"request_id": "ep_011", "violations": [{"description": "blocked"}]},
        ))
        replay_path = os.path.join(temp_artifact_dir, "episodes", "ep_011", "sandbox_replay.json")
        with open(replay_path) as f:
            replay = json.load(f)
        assert replay["blocked"] is True
        assert replay["block_reason"] == "blocked"
        assert len(replay["actions"]) == 1


class TestPraxisRecordedEvent:
    def test_publishes_praxis_recorded(self, recorder):
        captured = []
        recorder._event_bus.subscribe("praxis.recorded", lambda e: captured.append(e))

        recorder._event_bus.publish(Event(
            topic="skill.execution.start",
            payload={"skill_name": "pick", "correlation_id": "ep_012"},
        ))
        recorder._event_bus.publish(Event(
            topic="praxis.completed",
            payload={"practice_id": "ep_012", "outcome": {"reward": 1.0}},
        ))

        assert len(captured) == 1
        evt = captured[0]
        assert evt.topic == "praxis.recorded"
        assert evt.payload["event_id"] == "ep_012"
        assert evt.payload["robot_id"] == "test_bot"
        assert "artifact_uri" in evt.payload
        assert evt.payload["artifact_uri"].startswith("rosclaw://artifacts/episodes/ep_012")


class TestPublicAPI:
    def test_list_episodes(self, recorder, temp_artifact_dir):
        for cid in ["ep_013", "ep_014"]:
            recorder._event_bus.publish(Event(
                topic="skill.execution.start",
                payload={"skill_name": "pick", "correlation_id": cid},
            ))
            recorder._event_bus.publish(Event(
                topic="praxis.completed",
                payload={"practice_id": cid, "outcome": {"reward": 1.0}},
            ))

        episodes = recorder.list_episodes()
        assert len(episodes) == 2
        ids = [e["episode_id"] for e in episodes]
        assert "ep_013" in ids
        assert "ep_014" in ids

    def test_get_episode(self, recorder, temp_artifact_dir):
        recorder._event_bus.publish(Event(
            topic="skill.execution.start",
            payload={"skill_name": "pick", "correlation_id": "ep_015"},
        ))
        recorder._event_bus.publish(Event(
            topic="praxis.completed",
            payload={"practice_id": "ep_015", "outcome": {"reward": 1.0}},
        ))

        meta = recorder.get_episode("ep_015")
        assert meta is not None
        assert meta["episode_id"] == "ep_015"

    def test_get_episode_missing(self, recorder):
        assert recorder.get_episode("nonexistent") is None


class TestStubTopics:
    def test_provider_inference_stub(self, recorder, temp_artifact_dir):
        recorder._event_bus.publish(Event(
            topic="rosclaw.provider.inference.completed",
            payload={"episode_id": "ep_016", "intent": "grasp red cup", "cot": "plan A"},
        ))
        recorder._event_bus.publish(Event(
            topic="skill.execution.start",
            payload={"skill_name": "grasp", "correlation_id": "ep_016"},
        ))
        recorder._event_bus.publish(Event(
            topic="praxis.completed",
            payload={"practice_id": "ep_016", "outcome": {"reward": 1.0}},
        ))
        meta_path = os.path.join(temp_artifact_dir, "episodes", "ep_016", "metadata.json")
        with open(meta_path) as f:
            meta = json.load(f)
        assert "rosclaw.provider.inference.completed" in meta["received_events"]

    def test_critic_success_stub(self, recorder, temp_artifact_dir):
        recorder._event_bus.publish(Event(
            topic="skill.execution.start",
            payload={"skill_name": "grasp", "correlation_id": "ep_017"},
        ))
        recorder._event_bus.publish(Event(
            topic="rosclaw.critic.success.detected",
            payload={"episode_id": "ep_017", "success": True, "reward": 0.9},
        ))
        recorder._event_bus.publish(Event(
            topic="praxis.completed",
            payload={"practice_id": "ep_017", "outcome": {"reward": 1.0}},
        ))
        meta_path = os.path.join(temp_artifact_dir, "episodes", "ep_017", "metadata.json")
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["status"] == "SUCCESS"
        assert meta["reward"] == 0.9
