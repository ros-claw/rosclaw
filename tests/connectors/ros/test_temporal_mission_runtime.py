"""Daemon receiver/replay contracts; these fixtures are not Native acceptance."""

import copy
import time
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from rosclaw.connectors.ros.mission.executor import RosCoverageSimulationExecutor
from rosclaw.connectors.ros.verification.coverage import CoverageVerifier
from rosclaw.connectors.ros.verification.mission import replay_coverage, verify_mission
from rosclaw.connectors.ros.verification.occupancy import OccupancyAccounting, OccupancySnapshot

GRID = {
    "width": 2,
    "height": 1,
    "resolution": 0.01,
    "origin": [0, 0],
    "accessible_cells": [0, 1],
    "cleaning_polygon": [[-0.004, -0.004], [0.004, -0.004], [0.004, 0.004], [-0.004, 0.004]],
}
BINDING = {"run_id": "run", "geometry_hash": "geometry"}


def sample(t, seq, x, blocked):
    snapshot = OccupancySnapshot(
        run_id="run",
        mission_id="mission",
        frame_id="map",
        sim_time_sec=t,
        sequence=seq,
        occupied_cells=tuple(blocked),
        geometry_hash="geometry",
        source="independent_gazebo_model_geometry",
        complete=True,
        ground_truth_age_sec=0.01,
    )
    payload = asdict(snapshot)
    payload["occupied_cells"] = list(payload["occupied_cells"])
    return {
        "x": x,
        "y": 0.005,
        "yaw": 0,
        "time_sec": t,
        "cleaning_enabled": True,
        "observation_complete": True,
        "collision_count": 0,
        "occupancy": payload,
        "occupancy_hash": snapshot.artifact_hash(),
    }


def evidence(samples):
    return {
        "schema_version": "rosclaw.time_paired_mission_evidence.v1",
        "mission_id": "mission",
        "body_id": "body",
        "frame_id": "map",
        "grid": copy.deepcopy(GRID),
        "occupancy_binding": BINDING.copy(),
        "trajectory": [
            {k: s[k] for k in ("x", "y", "yaw", "time_sec", "cleaning_enabled")} for s in samples
        ],
        "occupancy_samples": [{k: s[k] for k in ("occupancy", "occupancy_hash")} for s in samples],
        "collision": {"collision_count": 0, "observation_complete": True},
    }


def test_dynamic_replay_never_credits_old_blocked_pose_from_final_free_mask():
    rows = [sample(0, 0, 0.005, [0]), sample(0.1, 1, 0.015, [])]
    ev = evidence(rows)
    coverage, temporal = replay_coverage(ev)
    assert coverage["coverage_ratio"] == 0.5
    assert temporal["complete"] and temporal["fixed_denominator_cells"] == 2
    assert coverage["mask"] == [0, 1]
    ev = evidence(rows + [sample(0.2, 2, 0.005, [])])
    assert replay_coverage(ev)[0]["coverage_ratio"] == 1
    result = verify_mission(ev)
    assert result["calculation_pass"] is True
    assert result["verification_status"] == "NOT_VERIFIED"
    assert result["hardware_verified"] is False
    assert result["time_paired_accounting"]["occupancy_sample_count"] == 3


@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "time", "hash", "pose_override", "schema"]
)
def test_dynamic_evidence_corruption_refused(fault):
    ev = evidence([sample(0, 0, 0.005, [0]), sample(0.1, 1, 0.015, [])])
    if fault == "missing":
        ev["occupancy_samples"].pop()
    elif fault == "duplicate":
        ev["occupancy_samples"][1] = ev["occupancy_samples"][0]
    elif fault == "time":
        ev["trajectory"][1]["time_sec"] = 0.11
    elif fault == "hash":
        ev["occupancy_samples"][1]["occupancy_hash"] = "forged"
    elif fault == "pose_override":
        ev["occupancy_samples"][0]["x"] = 1.5
    else:
        ev.pop("schema_version")
    with pytest.raises(ValueError):
        replay_coverage(ev)


def test_daemon_repair_consumes_same_paired_occupancy_as_final_replay(tmp_path):
    rows = [sample(0, 0, 0.005, [0]), sample(0.1, 1, 0.015, [])]
    binding = BINDING.copy()
    executor = RosCoverageSimulationExecutor(
        owner="daemon_test",
        client=None,
        control=None,
        witness=SimpleNamespace(since=lambda _: rows),
        output=tmp_path,
        body_id="body",
        body_snapshot_hash="bodyhash",
        grid=GRID,
        occupancy_binding=binding,
    )
    binding["geometry_hash"] = "changed-after-configuration"
    verifier = CoverageVerifier(**GRID)
    assert executor._repair(verifier, 0, "action", time.monotonic() + 1, mission_id="mission") == []
    assert set(verifier.visits) == {1}
    assert verifier.result() == replay_coverage(evidence(rows))[0]
    with pytest.raises(TypeError):
        executor.occupancy_binding["run_id"] = "other"


@pytest.mark.parametrize("fault", ["missing", "hash", "complete", "collision", "boolean_collision"])
def test_daemon_transport_fault_latches_before_additional_credit(fault):
    accounting = OccupancyAccounting(CoverageVerifier(**GRID), mission_id="mission", **BINDING)
    accounting.observe_sample(sample(0, 0, 0.005, []))
    bad = sample(0.1, 1, 0.015, [])
    if fault == "missing":
        bad.pop("occupancy")
    elif fault == "hash":
        bad["occupancy_hash"] = "bad"
    elif fault == "complete":
        bad["observation_complete"] = False
    elif fault == "boolean_collision":
        bad["collision_count"] = False
    else:
        bad["collision_count"] = 1
    with pytest.raises(ValueError):
        accounting.observe_sample(bad)
    assert set(accounting.verifier.visits) == {0}
    with pytest.raises(ValueError, match="latched"):
        accounting.observe_sample(sample(0.2, 2, 0.015, []))


def test_negative_first_sim_timestamp_never_enters_credit():
    accounting = OccupancyAccounting(CoverageVerifier(**GRID), mission_id="mission", **BINDING)
    with pytest.raises(ValueError, match="nonnegative"):
        accounting.observe_sample(sample(-0.1, 0, 0.005, []))
    assert not accounting.verifier.visits
    assert accounting.result()["complete"] is False


def brush_evidence():
    from datetime import UTC, datetime

    from rosclaw.connectors.ros.verification.brush_timeline import (
        BrushStateEvent,
        BrushStateTimeline,
    )

    binding = {
        "run_id": "run",
        "body_snapshot_hash": "bodyhash",
        "attachment_hash": "brush",
        "producer_id": "actuator",
    }
    timeline = BrushStateTimeline(**binding)
    seq = 0

    def append(stamp, enabled=False, kind="WATERMARK"):
        nonlocal seq
        event = BrushStateEvent(
            **binding,
            sequence=seq,
            sim_time_sec=stamp,
            kind=kind,
            enabled=enabled,
            captured_at=datetime.now(UTC).isoformat(),
            complete=True,
        )
        seq += 1
        timeline.append(event, artifact_hash=event.artifact_hash())

    append(0)
    append(0.01, True, "TRANSITION")
    rows = [sample(0.05, 0, 0.005, []), sample(0.15, 1, 0.015, [])]
    proofs = []
    for row in rows:
        append(row["time_sec"] + 0.01, True)
        proofs.append(
            {
                "brush_state_pair": timeline.state_at(row["time_sec"]),
                "brush_source_binding": binding.copy(),
                "brush_source_fault": None,
            }
        )
    ev = evidence(rows)
    ev["brush_evidence_binding"], ev["brush_evidence_samples"] = binding, proofs
    return ev


def test_retained_brush_pair_chain_replays_without_sender_or_source_path():
    ev = brush_evidence()
    coverage, temporal = replay_coverage(ev)
    assert coverage["coverage_ratio"] == 1 and temporal["complete"]
    assert verify_mission(ev)["verification_status"] == "NOT_VERIFIED"


@pytest.mark.parametrize(
    "fault", ["missing", "pair_hash", "chain_gap", "body", "same_time", "override"]
)
def test_brush_artifact_corruption_refused_before_credit(fault):
    ev = brush_evidence()
    records = ev["brush_evidence_samples"]
    if fault == "missing":
        records.pop()
    elif fault == "pair_hash":
        records[1]["brush_state_pair"]["pair_chain_hash"] = "corrupted"
    elif fault == "chain_gap":
        records[1]["brush_state_pair"]["previous_pair_chain_hash"] = "GENESIS"
    elif fault == "body":
        records[1]["brush_source_binding"]["body_snapshot_hash"] = "other"
    elif fault == "same_time":
        records[1]["brush_state_pair"]["watermark_sim_time_sec"] = 0.15
    else:
        records[0]["cleaning_enabled"] = True
    with pytest.raises(ValueError):
        replay_coverage(ev)
