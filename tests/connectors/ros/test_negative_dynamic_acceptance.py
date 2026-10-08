"""Synthetic end-to-end evidence contracts; these are not D4 physics runs."""

import hashlib
import importlib.util
import json
import sqlite3
import time
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from rosclaw.connectors.ros.diagnosis.coverage_audit import CoverageAuditLog
from rosclaw.connectors.ros.verification.brush_timeline import BrushStateEvent
from rosclaw.connectors.ros.verification.coverage import CoverageVerifier
from rosclaw.connectors.ros.verification.mission import replay_coverage
from rosclaw.connectors.ros.verification.occupancy_geometry import OccupancyProjector
from tests.connectors.ros.test_dynamic_fixture_scenario import binding, source_row
from tests.connectors.ros.test_native_negative_terminal import (
    negative,  # noqa: F401 - shared real migrations
)
from tests.connectors.ros.test_physics_component_packets import packet, parse
from tests.connectors.ros.test_temporal_mission_runtime import GRID, evidence

ROOT = Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance"


@pytest.fixture
def closed(negative, monkeypatch):  # noqa: F811 - imported pytest fixture
    root, db, native, client, bundle = negative
    monkeypatch.syspath_prepend(str(ROOT))
    spec = importlib.util.spec_from_file_location(
        "negative_dynamic_test", ROOT / "negative_dynamic_acceptance.py"
    )
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    b = binding()
    b.update(mission_id="mission", map_world_identity_approved=True, grid=GRID.copy())
    brush = {k: b[k] for k in ("run_id", "body_snapshot_hash", "attachment_hash")}
    brush["producer_id"] = "actor"
    config = {
        "body_id": "body",
        "body_snapshot_hash": b["body_snapshot_hash"],
        "dynamic_fixture_admission": {"mission_id": "mission", "run_id": b["run_id"]},
    }
    (root / "execution_config.json").write_text(json.dumps(config))
    with sqlite3.connect(db) as connection:
        connection.execute("update action_txns set body_hash=?", (b["body_snapshot_hash"],))
    audit = CoverageAuditLog(
        root / "plan-events-source.jsonl", context={"run_id": b["run_id"]}, capacity=512
    )
    samples = []
    confirmation = None
    for i in range(205):
        p = packet()
        p.update(sequence=i, sim_time_sec=i / 10, paused=False, captured_at_unix_ns=time.time_ns())
        p["body"]["world_pose"][:2] = [0.015, 0.005]
        p["obstacles"][0]["world_pose"][:2] = [2.0, 0.0] if i < 11 else [0.0, 0.0]
        decoded = parse(p)
        snapshot = OccupancyProjector(CoverageVerifier(**GRID), decoded["geometry"]).project(
            decoded["model_poses"],
            run_id=b["run_id"],
            mission_id="mission",
            sequence=i,
            frame_id="map",
            sim_time_sec=i / 10,
            ground_truth_age_sec=0.01,
            complete=True,
        )
        payload = asdict(snapshot)
        payload["occupied_cells"] = list(payload["occupied_cells"])
        if i <= 170:
            samples.append(
                {
                    "x": 0.015,
                    "y": 0.005,
                    "yaw": 0.0,
                    "time_sec": i / 10,
                    "cleaning_enabled": 1 <= i < 170,
                    "occupancy": payload,
                    "occupancy_hash": snapshot.artifact_hash(),
                }
            )
        row = source_row(p)
        audit.emit(row["kind"], row["payload"], sim_time=i / 10)
        if i == 11:
            confirmation = row["payload"]["packet_sha256"]
    summary = audit.close()
    assert summary["complete"]
    Path(str(audit.path) + ".summary.json").write_text(json.dumps(summary))
    ev = evidence(samples)
    ev.update(
        body_snapshot_hash=b["body_snapshot_hash"],
        action_ids=["failed_action"],
        occupancy_binding={
            "run_id": b["run_id"],
            "geometry_hash": decoded["geometry"].artifact_hash(),
        },
    )
    artifact = root / "actions/rosevidence_partial.json"
    artifact.write_text(json.dumps(ev))
    coverage, temporal = replay_coverage(ev)
    bundle["receipt"].update(
        body_snapshot_hash=b["body_snapshot_hash"],
        final_state="BLOCKED",
        verification_result={
            "mission_id": "mission",
            "coverage_ratio": coverage["coverage_ratio"],
            "time_paired_accounting": temporal,
            "evidence_artifact": {
                "path": str(artifact),
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            },
        },
    )
    terminal = native.negative_native_progress(root, "D4", client)
    (root / "native-negative-terminal.json").write_text(json.dumps(terminal))
    (root / "physics_binding.json").write_text(json.dumps(b))
    (root / "brush_binding.json").write_text(json.dumps(brush))
    protocol = {
        "schema_version": "rosclaw.dynamic_fixture_scenario.v1",
        "case": "D4",
        "run_id": b["run_id"],
        "mission_id": "mission",
        "obstacle_name": b["obstacle_names"][0],
        "target_xy": [0.0, 0.0],
        "dwell_sim_sec": 10,
        "introduce_after_cleaning_sim_sec": 1,
        "wall_timeout_sec": 900,
    }
    raw = json.dumps(protocol).encode()
    (root / "scenario.json").write_bytes(raw)
    events = []
    for kind in [
        "SCENARIO_STARTED",
        "FIRST_ENABLED_CLEANING",
        "INTRODUCTION_REQUESTED",
        "INTRODUCTION_ACK_REQUIRES_ACTUAL_PACKET",
        "ACTUAL_POSTUPDATE_POSITION_CONFIRMED",
        "TASK_RUNNER_STOP_REQUESTED",
    ]:
        row = {
            "kind": kind,
            "scenario_sha256": hashlib.sha256(raw).hexdigest(),
            "run_id": b["run_id"],
            "mission_id": "mission",
            "physical_acceptance": "NOT_VERIFIED",
        }
        if kind == "ACTUAL_POSTUPDATE_POSITION_CONFIRMED":
            row.update(
                state="CONFIRMING_INTRODUCTION",
                actual_xy=[0.0, 0.0],
                packet_sha256=confirmation,
                sim_time_sec=1.1,
            )
        if kind == "TASK_RUNNER_STOP_REQUESTED":
            row["final_state"] = "OCCUPIED"
        if kind == "FIRST_ENABLED_CLEANING":
            row["sim_time_sec"] = 0.1
        events.append(row)
    (root / "dynamic-scenario-events.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in events)
    )
    base = datetime(2026, 10, 8, tzinfo=UTC)
    stop = {
        "samples": [
            {
                "source": "independent_gazebo_ground_truth_subscription",
                "x": 0.0,
                "y": 0.0,
                "yaw": 0.0,
                "time_sec": 17 + i / 10,
                "captured_at": (base + timedelta(seconds=17 + i / 10)).isoformat(),
            }
            for i in range(31)
        ]
    }
    actor = CoverageAuditLog(
        root / "brush-events-source.jsonl",
        context={**brush, "source": "simulator_owned_actuator", "evidence_domain": "SIMULATION"},
        capacity=512,
    )
    for i in range(411):
        event = BrushStateEvent(
            **brush,
            sequence=i,
            sim_time_sec=i / 20,
            kind="TRANSITION" if i in (2, 340) else "WATERMARK",
            enabled=2 <= i < 340,
            captured_at=(base + timedelta(seconds=i / 20)).isoformat(),
            complete=True,
        )
        actor.emit(
            "brush_state_event",
            {
                "event": asdict(event),
                "artifact_hash": event.artifact_hash(),
                "lease_remaining_sec": -1.0,
                "lease_updates": 3,
            },
            sim_time=i / 20,
        )
    actor_summary = actor.close()
    assert actor_summary["complete"]
    Path(str(actor.path) + ".summary.json").write_text(json.dumps(actor_summary))
    observations = [
        {
            "time_sec": i / 10,
            "observation_complete": True,
            "collision_count": 0,
            "brush_source_binding": brush,
            "physics_source_fault": None,
            "brush_source_fault": None,
            "evidence_domain": "GAZEBO_PHYSICS",
            "ground_truth_age_ms": 1.0,
            "cleaning_enabled": 1 <= i < 170,
        }
        for i in range(205)
    ]
    (root / "witness.jsonl").write_text("".join(json.dumps(r) + "\n" for r in observations))
    return reader, root, db, stop, events, observations


def test_complete_negative_evidence_preserves_real_failed_task_and_source_closure(closed):
    reader, root, db, stop, _, _ = closed
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    answer = reader.validate_d4_negative(root, stop)
    assert answer["status"] == "PASS_EXPECTED_SAFE_FAILURE"
    assert answer["task_state"] == "FAILED" and not answer["task_kernel_succeeded"]
    assert answer["permanently_occupied_unclean_cells"] == [0]
    assert answer["closed_source_replay"]["source_replay_match"]
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before


@pytest.mark.parametrize(
    "fault",
    [
        "success",
        "receipt",
        "scene_failed",
        "withdrawal",
        "confirmation_hash",
        "confirmation_time",
        "target",
        "not_occupied",
        "missing_scene",
        "contact",
        "observer_loss",
        "observer_gap",
        "observer_identity",
        "stale",
        "success_artifact",
        "actor_open",
    ],
)
def test_negative_outcome_does_not_bypass_any_independent_physical_gate(closed, fault):
    reader, root, db, stop, events, observations = closed
    if fault == "success":
        with sqlite3.connect(db) as connection:
            connection.execute("update tasks set state='SUCCEEDED'")
    elif fault == "receipt":
        path = root / "actions/rosevidence_partial.json"
        path.write_text(path.read_text() + " ")
    elif fault == "scene_failed":
        events[4]["kind"] = "SCENARIO_FAILED"
    elif fault == "withdrawal":
        events[4]["kind"] = "WITHDRAWAL_REQUESTED"
    elif fault == "confirmation_hash":
        events[4]["packet_sha256"] = "foreign"
    elif fault == "confirmation_time":
        events[4]["sim_time_sec"] = 2.0
    elif fault == "target":
        events[4]["actual_xy"] = [0.5, 0.5]
    elif fault == "not_occupied":
        events[5]["final_state"] = "WAITING_FOR_CLEANING"
    elif fault == "missing_scene":
        events.pop()
    elif fault == "contact":
        observations[30]["collision_count"] = 1
    elif fault == "observer_loss":
        observations[30]["observation_complete"] = False
    elif fault == "observer_gap":
        del observations[30:40]
    elif fault == "observer_identity":
        observations[30]["brush_source_binding"] = {}
    elif fault == "stale":
        observations[30]["ground_truth_age_ms"] = 300.0
    elif fault == "success_artifact":
        (root / "actions/unexpected.verification.json").write_text("{}")
    else:
        p = root / "brush-events-source.jsonl.summary.json"
        summary = json.loads(p.read_bytes())
        summary["writer_stopped"] = False
        p.write_text(json.dumps(summary))
    (root / "dynamic-scenario-events.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in events)
    )
    (root / "witness.jsonl").write_text("".join(json.dumps(r) + "\n" for r in observations))
    with pytest.raises((ValueError, RuntimeError)):
        reader.validate_d4_negative(root, stop)
