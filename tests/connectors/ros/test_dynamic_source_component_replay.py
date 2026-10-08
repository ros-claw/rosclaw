"""Synthetic SDK source/artifact correspondence; no physical or Native claim."""

import importlib.util
import json
import time
from dataclasses import asdict
from pathlib import Path

import pytest

from rosclaw.connectors.ros.diagnosis.coverage_audit import CoverageAuditLog
from rosclaw.connectors.ros.verification.coverage import CoverageVerifier
from rosclaw.connectors.ros.verification.occupancy_geometry import OccupancyProjector
from tests.connectors.ros.test_dynamic_fixture_scenario import binding, source_row
from tests.connectors.ros.test_physics_component_packets import packet, parse
from tests.connectors.ros.test_temporal_mission_runtime import GRID, evidence

ROOT = Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance"


def fixture(root, monkeypatch, fault=None):
    monkeypatch.syspath_prepend(str(ROOT))
    spec = importlib.util.spec_from_file_location(
        "component_replay_test", ROOT / "dynamic_source_replay.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    b = binding()
    b.update(mission_id="mission", map_world_identity_approved=True, grid=GRID.copy())
    path = root / "observer.jsonl"
    audit = CoverageAuditLog(path, context={"run_id": b["run_id"]})
    samples = []
    geometry = None
    for i, x in enumerate([0.005, 0.015, 0.005]):
        p = packet()
        p.update(sequence=i, sim_time_sec=i * 0.1, paused=False, captured_at_unix_ns=time.time_ns())
        p["body"]["world_pose"][:2] = [x, 0.005]
        p["obstacles"][0]["world_pose"][0] = 0 if i == 0 else 2
        decoded = parse(p)
        geometry = decoded["geometry"]
        projector = OccupancyProjector(CoverageVerifier(**GRID), geometry)
        snapshot = projector.project(
            decoded["model_poses"],
            run_id=b["run_id"],
            mission_id="mission",
            sequence=i,
            frame_id="map",
            sim_time_sec=i * 0.1,
            ground_truth_age_sec=0.01,
            complete=True,
        )
        payload = asdict(snapshot)
        payload["occupied_cells"] = list(payload["occupied_cells"])
        samples.append(
            {
                "x": x,
                "y": 0.005,
                "yaw": 0,
                "time_sec": i * 0.1,
                "cleaning_enabled": True,
                "occupancy": payload,
                "occupancy_hash": snapshot.artifact_hash(),
            }
        )
        if i == 1:
            if fault == "actual_pose":
                p["body"]["world_pose"][0] += 0.001
            elif fault == "actual_mask":
                p["obstacles"][0]["world_pose"][0] = 0
            elif fault == "source_gap":
                p["sequence"] = 5
        row = source_row(p)
        if i == 1 and fault == "raw_bytes":
            row["payload"]["raw_packet_utf8"] += " "
        audit.emit(row["kind"], row["payload"], sim_time=p["sim_time_sec"])
    summary = audit.close()
    Path(str(path) + ".summary.json").write_text(json.dumps(summary))
    e = evidence(samples)
    e["body_snapshot_hash"] = b["body_snapshot_hash"]
    e["occupancy_binding"] = {"run_id": b["run_id"], "geometry_hash": geometry.artifact_hash()}
    if fault == "closure":
        summary["complete"] = False
    elif fault == "dropped":
        summary["dropped_events"] = 1
    elif fault == "count":
        summary["events_written"] += 1
    elif fault == "terminal_hash":
        summary["last_event_hash"] = "wrong"
    elif fault == "identity":
        b["map_world_identity_approved"] = False
    elif fault == "grid":
        b["grid"]["resolution"] = 0.02
    elif fault == "typed_closure":
        summary["dropped_events"] = False
    elif fault == "incomplete_line":
        path.write_bytes(path.read_bytes().rstrip(b"\n"))
    Path(str(path) + ".summary.json").write_text(json.dumps(summary))
    return m, path, e, b


def test_all_historical_masks_match_closed_original_component_packets(tmp_path, monkeypatch):
    m, path, e, b = fixture(tmp_path, monkeypatch)
    result = m.replay_component_occupancy(path, e, b)
    assert result["source_replay_match"] and result["matched_canonical_samples"] == 3
    assert result["fixed_denominator_cells"] == 2
    assert len(result["original_packets"]) == 3
    assert result["physical_acceptance"] == "NOT_VERIFIED"


@pytest.mark.parametrize(
    "fault",
    [
        "actual_pose",
        "actual_mask",
        "source_gap",
        "raw_bytes",
        "closure",
        "dropped",
        "count",
        "terminal_hash",
        "identity",
        "incomplete_line",
        "grid",
        "typed_closure",
    ],
)
def test_source_mismatch_or_unclosed_prefix_cannot_pass_replay(tmp_path, monkeypatch, fault):
    m, path, e, b = fixture(tmp_path, monkeypatch, fault)
    with pytest.raises(ValueError):
        m.replay_component_occupancy(path, e, b)


def test_actual_executor_artifact_preserves_body_binding_for_closed_source_replay(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from rosclaw.connectors.ros.mission.executor import RosCoverageSimulationExecutor
    from rosclaw.kernel import ActionState

    m, audit, prepared, binding = fixture(tmp_path, monkeypatch)
    rows = [
        {**pose, **occupancy, "observation_complete": True, "collision_count": 0}
        for pose, occupancy in zip(
            prepared["trajectory"], prepared["occupancy_samples"], strict=True
        )
    ]
    control = SimpleNamespace(
        call_service=lambda *a, **kw: SimpleNamespace(ok=True, data={"values": {"success": True}})
    )
    client = SimpleNamespace(send_goal=lambda **kw: kw["on_result"](4, {}))
    witness = SimpleNamespace(mark=lambda: 0, since=lambda _: rows, fresh=lambda: rows[-1])
    executor = RosCoverageSimulationExecutor(
        owner="daemon_offline_contract",
        client=client,
        control=control,
        witness=witness,
        output=tmp_path / "actions",
        body_id=prepared["body_id"],
        body_snapshot_hash=binding["body_snapshot_hash"],
        grid=prepared["grid"],
        occupancy_binding=prepared["occupancy_binding"],
    )
    monkeypatch.setattr(executor, "_coverage_goal", lambda _: {})
    monkeypatch.setattr(executor, "_repair", lambda *a, **kw: [])
    action = SimpleNamespace(
        capability_id="coverage.execute",
        action_id="fresh_action",
        body_id=prepared["body_id"],
        arguments={"mission_id": "mission"},
        verification_policy=SimpleNamespace(timeout_sec=10),
    )
    result = executor._execute(action)
    assert result.final_state == ActionState.COMPLETED
    artifact = result.verification_result["evidence_artifact"]
    actual = json.loads(Path(artifact["path"]).read_bytes())
    assert actual["body_snapshot_hash"] == binding["body_snapshot_hash"]
    replay = m.replay_component_occupancy(audit, actual, binding)
    assert replay["source_replay_match"] and replay["matched_canonical_samples"] == 3
    assert replay["physical_acceptance"] == "NOT_VERIFIED"
