"""A verified sandbox receipt should automatically enter Practice."""

import hashlib
import json

from rosclaw.core.event_bus import Event, EventBus
from rosclaw.practice.bridges import SandboxPracticeBridge
from rosclaw.sandbox.backends import ReplayReport
from rosclaw.sandbox.evidence import SimulationEvidenceVerification


def _verified(_receipt: dict) -> SimulationEvidenceVerification:
    replay = ReplayReport(True, True, True, True, 0.0, "strict_replay_verified")
    return SimulationEvidenceVerification(True, replay)


def test_bridge_persists_quality_and_publishes_terminal(tmp_path):
    artifact = tmp_path / "trajectory_states.json"
    artifact.write_text(
        json.dumps(
            {
                "schema_version": "rosclaw.trajectory_states.v1",
                "states": [
                    {
                        "step": 1,
                        "time": 0.002,
                        "qpos": [0.0] * 6,
                        "qvel": [0.0] * 6,
                        "command": [0.0] * 6,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    bus = EventBus()
    succeeded = []
    finished = []
    bus.subscribe("rosclaw.sandbox.episode.succeeded", lambda event: succeeded.append(event))
    bus.subscribe("rosclaw.sandbox.episode.finished", lambda event: finished.append(event))
    bridge = SandboxPracticeBridge(
        "ur5e",
        bus,
        str(tmp_path / "practice"),
        receipt_verifier=_verified,
    )
    bridge.initialize()
    bus.publish(
        Event(
            topic="firewall.action_allowed",
            payload={
                "robot_id": "ur5e",
                "simulation_receipt": {
                    "schema_version": "rosclaw.simulation_receipt.v1",
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
                    "artifact_hashes": {artifact.name: digest},
                    "replay_report": {
                        "verified": True,
                        "environment_match": True,
                        "hashes_verified": True,
                        "deterministic_label": True,
                        "mismatches": [],
                    },
                    "data_quality": {"body_snapshot_match": True},
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


def test_bridge_ignores_untyped_action_simulation_result(tmp_path):
    bus = EventBus()
    finished = []
    bus.subscribe("rosclaw.sandbox.episode.finished", lambda event: finished.append(event))
    bridge = SandboxPracticeBridge("ur5e", bus, str(tmp_path))
    bridge.initialize()
    bus.publish(
        Event(
            topic="rosclaw.runtime.action.receipt",
            payload={
                "simulation_result": {
                    "evidence_domain": "SIMULATION",
                    "physics_executed": True,
                    "is_safe": True,
                }
            },
        )
    )
    bridge.stop()
    assert finished == []
    assert (tmp_path / "sandbox" / "events.jsonl").read_text(encoding="utf-8") == ""


def test_bridge_deduplicates_identical_receipts(tmp_path):
    bus = EventBus()
    failed = []
    bus.subscribe("rosclaw.sandbox.episode.failed", lambda event: failed.append(event))
    bridge = SandboxPracticeBridge("ur5e", bus, str(tmp_path))
    bridge.initialize()
    event = Event(
        topic="firewall.action_blocked",
        payload={
            "simulation_receipt": {
                "schema_version": "rosclaw.simulation_receipt.v1",
                "scenario_id": "duplicate-scenario",
                "action_hash": "sha256:duplicate",
                "seed": 1,
                "evidence_domain": "SIMULATION",
                "physics_executed": False,
            }
        },
    )
    bus.publish(event)
    bus.publish(event)
    bridge.stop()
    assert len(failed) == 1


def test_bridge_never_publishes_success_from_truthy_strings_or_bad_quality(tmp_path):
    bus = EventBus()
    succeeded = []
    failed = []
    bus.subscribe("rosclaw.sandbox.episode.succeeded", lambda event: succeeded.append(event))
    bus.subscribe("rosclaw.sandbox.episode.failed", lambda event: failed.append(event))
    bridge = SandboxPracticeBridge(
        "ur5e",
        bus,
        str(tmp_path),
        receipt_verifier=_verified,
    )
    bridge.initialize()
    bus.publish(
        Event(
            topic="firewall.action_allowed",
            payload={
                "simulation_receipt": {
                    "schema_version": "rosclaw.simulation_receipt.v1",
                    "scenario_id": "forged",
                    "evidence_domain": "SIMULATION",
                    "physics_executed": "true",
                    "is_safe": "true",
                    "artifact_hashes": ["not-a-mapping"],
                    "artifacts": {"not": "a-list"},
                    "replay_report": ["not-a-mapping"],
                    "data_quality": ["not-a-mapping"],
                }
            },
        )
    )
    bridge.stop()

    assert succeeded == []
    assert len(failed) == 1
    assert failed[0].payload["success"] is False
    assert failed[0].payload["data_quality_pass"] is False


def test_bridge_bounds_unverified_persistence_fields(tmp_path):
    bus = EventBus()
    bridge = SandboxPracticeBridge(
        "ur5e",
        bus,
        str(tmp_path),
        receipt_verifier=lambda _receipt: SimulationEvidenceVerification(
            False, ReplayReport(False, False, False, False, None, "invalid")
        ),
    )
    bridge.initialize()
    bus.publish(
        Event(
            topic="firewall.action_blocked",
            payload={
                "simulation_receipt": {
                    "schema_version": "rosclaw.simulation_receipt.v1",
                    "scenario_id": "s" * 10_000,
                    "evidence_domain": "SIMULATION",
                    "physics_executed": False,
                    "reason": "r" * 100_000,
                    "violations": ["v" * 10_000] * 100,
                    "metrics": {"blob": "x" * 1_000_000},
                    "artifacts": ["a" * 10_000] * 100,
                }
            },
        )
    )
    bridge.stop()

    line = (tmp_path / "sandbox" / "events.jsonl").read_text(encoding="utf-8")
    record = json.loads(line)
    assert len(line.encode("utf-8")) < 64 * 1024
    assert len(record["episode_id"]) == 256
    assert len(record["payload"]["reason"]) == 1024
    assert len(record["payload"]["violations"]) == 50
