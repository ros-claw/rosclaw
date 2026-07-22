"""A verified sandbox receipt should automatically enter Practice."""

import hashlib
import json

from rosclaw.core.event_bus import Event, EventBus
from rosclaw.practice.bridges import SandboxPracticeBridge


def test_bridge_persists_quality_and_publishes_terminal(tmp_path):
    artifact = tmp_path / "states.json"
    artifact.write_text("{}", encoding="utf-8")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    bus = EventBus()
    succeeded = []
    finished = []
    bus.subscribe("rosclaw.sandbox.episode.succeeded", lambda event: succeeded.append(event))
    bus.subscribe("rosclaw.sandbox.episode.finished", lambda event: finished.append(event))
    bridge = SandboxPracticeBridge("ur5e", bus, str(tmp_path / "practice"))
    bridge.initialize()
    bus.publish(
        Event(
            topic="firewall.action_allowed",
            payload={
                "robot_id": "ur5e",
                "simulation_receipt": {
                    "scenario_id": "scenario-42",
                    "evidence_domain": "SIMULATION",
                    "physics_executed": True,
                    "is_safe": True,
                    "reason": "physics_trajectory_safe",
                    "body_snapshot_hash": "sha256:body",
                    "model_hash": "sha256:model",
                    "action_hash": "sha256:action",
                    "final_qpos": [0.0] * 6,
                    "request": {"trajectory": [[0.0] * 6]},
                    "artifacts": [artifact.as_uri()],
                    "artifact_hashes": {"states.json": digest},
                },
            },
            source="test",
            trace_id="trace-42",
        )
    )
    bridge.stop()

    assert len(succeeded) == len(finished) == 1
    assert succeeded[0].payload["data_quality_pass"] is True
    records = [
        json.loads(line)
        for line in (tmp_path / "practice" / "sandbox" / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert records[0]["event_type"] == "sandbox.physics_rollout.completed"
    assert records[0]["quality"]["artifact_hash_valid"] is True


def test_bridge_ignores_fixture_evidence(tmp_path):
    bus = EventBus()
    bridge = SandboxPracticeBridge("ur5e", bus, str(tmp_path))
    bridge.initialize()
    bus.publish(
        Event(
            topic="rosclaw.runtime.action.receipt",
            payload={"evidence_domain": "FIXTURE", "physics_executed": False},
        )
    )
    bridge.stop()
    assert (tmp_path / "sandbox" / "events.jsonl").read_text(encoding="utf-8") == ""
