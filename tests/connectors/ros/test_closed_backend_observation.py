"""Synthetic closed joint tapes; declared CDR decoder/source-loader doubles."""

import copy
import importlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rosclaw.connectors.ros.diagnosis.coverage_audit import digest
from tests.connectors.ros.test_backend_source_gate import retained, sources


@pytest.fixture
def tape(monkeypatch, tmp_path, request):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    module = importlib.import_module("closed_backend_observation")
    joint = importlib.import_module("backend_observer_replay")
    probe = importlib.import_module("closed_backend_probe")

    def synthetic_pose(raw, *args):
        data = json.loads(raw)
        return data["sim"], data["pose"]

    monkeypatch.setattr(joint, "decode_ros_pose", synthetic_pose)
    monkeypatch.setattr(probe, "decode_ros_pose", synthetic_pose)
    calls = []
    monkeypatch.setattr(module, "reopen_native_policy", lambda *a, **kw: calls.append(a))
    robot, robot_policy, instrument, probe_policy = sources()
    spatial = bool(getattr(request, "param", False))
    spatial_options, scene = {}, None
    if spatial:
        from tests.connectors.ros import test_probe_scene_geometry as spatial_contract

        _, scene, _, _, declaration, binding, _ = spatial_contract.fixture.__wrapped__(monkeypatch)
        spatial_options = {"scene_binding": binding, "probe_declaration": declaration}
    engine = joint.BackendObserverReplay(
        robot_policy,
        probe_policy,
        robot_pose_frame="synthetic_robot_world",
        probe_pose_frame="synthetic_probe_world",
        **spatial_options,
    )
    rows = []
    origin = 1_791_504_000_000_000_000

    def emit(kind, payload):
        projection = engine.apply(kind, payload)
        wall = payload["received_monotonic_sec"]
        unix = payload.get("received_unix_ns", origin + round((wall - 100) * 1e9))
        sim = (
            projection["snapshot"]["robot_sim_time_sec"]
            if kind == "backend_observation_sample"
            else projection.get("sim_time_sec")
        )
        rows.append(
            {
                "schema_version": "rosclaw.coverage_audit_event.v1",
                "run_id": engine.binding["run_id"],
                "body_snapshot_hash": engine.binding["body_snapshot_hash"],
                "constraint_policy_hash": engine.gate.policy_hash,
                "evidence_domain": "SIMULATION",
                "source": "independent_SIM_backend_original_sources",
                "kind": kind,
                "captured_at": datetime.fromtimestamp((unix + 100_000) / 1e9, UTC).isoformat(),
                "wall_monotonic_sec": wall + 0.0001,
                "sim_time_sec": sim,
                "payload": {**payload, "projection": projection},
            }
        )

    def frame(i, z=0.05, touching=True):
        sim, wall = i * 0.05, 100 + i * 0.05
        r = copy.deepcopy(robot)
        r.update(
            sequence=i,
            iterations=i,
            physics_step_dt_sec=0.05,
            sim_time_sec=sim,
            captured_at_unix_ns=origin + round(sim * 1e9),
        )
        p = copy.deepcopy(instrument)
        p.update(
            sequence=i,
            iterations=i if spatial else i * 50,
            sim_time_sec=sim,
            captured_at_unix_ns=r["captured_at_unix_ns"],
            body_world_pose=[6, 0, z, 1, 0, 0, 0],
        )
        if not touching:
            p["collision_contacts"][0]["contacts"] = []
        if spatial:
            scene.update(
                sequence=i,
                physics_iteration=i,
                sim_time_sec=sim,
                captured_at_unix_ns=r["captured_at_unix_ns"],
            )
            scene["obstacles"][1]["world_pose"] = p["body_world_pose"].copy()
            emit(
                "backend_scene_components",
                {
                    **retained(
                        json.dumps(scene).encode(), "gazebo_ecm_postupdate_observation_json"
                    ),
                    "received_monotonic_sec": wall,
                    "received_unix_ns": r["captured_at_unix_ns"],
                },
            )
        for offset, prefix, packet, pose in [
            (0, "robot", r, [0, 0, 0, 1, 0, 0, 0]),
            (0.002, "probe", p, p["body_world_pose"]),
        ]:
            raw = json.dumps({"sim": sim, "pose": pose}).encode()
            emit(
                "backend_" + prefix + "_pose",
                {
                    **retained(raw, "tf2_msgs/msg/TFMessage"),
                    "received_monotonic_sec": wall + offset,
                },
            )
            emit(
                "backend_" + prefix + "_components",
                {
                    **retained(json.dumps(packet).encode(), "gazebo_ecm_contact_sensor_data_json"),
                    "received_monotonic_sec": wall + offset + 0.001,
                    "received_unix_ns": packet["captured_at_unix_ns"]
                    + round((offset + 0.001) * 1e9),
                },
            )
        emit("backend_observation_sample", {"received_monotonic_sec": wall + 0.004})

    for i in range(1, 13):
        frame(i)
    request = importlib.import_module("probe_lift_evidence").lift_request(probe_policy)
    emit(
        "backend_probe_lift_ack",
        {
            "request": retained(request, "gz.msgs.Pose_protobuf_text"),
            "response": retained(b"data: true", "gz.msgs.Boolean_protobuf_text"),
            "returncode": 0,
            "received_monotonic_sec": 100.605,
            "received_unix_ns": origin + 605_000_000,
        },
    )
    for i in range(13, 25):
        frame(i, z=10 - (i - 13) * 0.05, touching=False)
    for i in range(25, 37):
        frame(i)
    path = tmp_path / "joint.jsonl"

    def write():
        previous = None
        for i, row in enumerate(rows, 1):
            row.update(sequence=i, previous_hash=previous)
            row.pop("artifact_sha256", None)
            row["artifact_sha256"] = digest(row)
            previous = row["artifact_sha256"]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        Path(str(path) + ".summary.json").write_text(
            json.dumps(
                {
                    "complete": True,
                    "writer_stopped": True,
                    "writer_error": None,
                    "dropped_events": 0,
                    "events_written": len(rows),
                    "last_event_hash": previous,
                }
            )
        )

    def validate():
        return module.closed_backend_observation(
            path,
            robot_policy,
            probe_policy,
            robot_directory=tmp_path,
            probe_directory=tmp_path,
            robot_plugin=tmp_path / "synthetic_robot.so",
            probe_plugin=tmp_path / "synthetic_probe.so",
            robot_pose_frame="synthetic_robot_world",
            probe_pose_frame="synthetic_probe_world",
            **spatial_options,
        )

    write()
    return path, rows, write, validate, calls


def test_closed_original_joint_projection_replays_without_acceptance(tape):
    _, _, _, validate, calls = tape
    result = validate()
    assert result["events_replayed"] == 181 and result["samples_replayed"] == 36
    assert result["probe_completed_cache_cycles"] == 1 and result["robot_collision_count"] == 0
    assert len(calls) == 4
    assert result["physical_acceptance"] == "NOT_VERIFIED" and not result["backend_health_admitted"]


@pytest.mark.parametrize("tape", [True], indirect=True)
def test_closed_spatial_original_tape_requires_all_same_step_geometry(tape):
    _, rows, _, validate, _ = tape
    result = validate()
    assert result["events_replayed"] == 217 and result["completed_exact_scene_joins"] == 36
    assert result["spatial_source_join_required"]
    assert result["probe_completed_cache_cycles"] == 1
    assert not result["backend_health_admitted"] and not result["authorization"]
    assert sum(row["kind"] == "backend_scene_components" for row in rows) == 36


@pytest.mark.parametrize("tape", [True], indirect=True)
@pytest.mark.parametrize("fault", ["missing_scene", "pose_mismatch", "projection", "iteration"])
def test_closed_spatial_tape_refuses_hash_resealed_false_source_or_projection(tape, fault):
    _, rows, write, validate, _ = tape
    source = next(row for row in rows if row["kind"] == "backend_scene_components")
    if fault == "missing_scene":
        rows.remove(source)
    elif fault == "projection":
        source["payload"]["projection"]["joined_spatial_sources"] = [{"invented": True}]
    else:
        import base64

        raw = base64.b64decode(source["payload"]["original_source_base64"])
        packet = json.loads(raw)
        if fault == "pose_mismatch":
            packet["body"]["world_pose"][0] += 0.1
        else:
            packet["physics_iteration"] += 1
        replacement = retained(
            json.dumps(packet).encode(), "gazebo_ecm_postupdate_observation_json"
        )
        source["payload"].update(replacement)
    write()
    with pytest.raises(ValueError):
        validate()


@pytest.mark.parametrize(
    "fault",
    [
        "projection",
        "source_hash",
        "sequence",
        "body",
        "clock",
        "sim",
        "unknown_kind",
        "partial",
        "closure",
        "drops",
        "duplicate_json",
        "nonfinite_json",
    ],
)
def test_closed_tape_rejects_substitution_loss_and_unknown_source(tape, fault):
    path, rows, write, validate, _ = tape
    row = rows[10]
    if fault == "projection":
        row["payload"]["projection"] = {"fabricated": True}
    elif fault == "source_hash":
        rows[1]["payload"]["original_source_sha256"] = "0" * 64
    elif fault == "body":
        row["body_snapshot_hash"] = "foreign"
    elif fault == "clock":
        row["payload"]["received_monotonic_sec"] = 99
    elif fault == "sim":
        row["sim_time_sec"] = 999
    elif fault == "unknown_kind":
        row["kind"] = "invented_source"
    write()
    if fault == "sequence":
        row["sequence"] = 500
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    elif fault == "partial":
        path.write_bytes(path.read_bytes()[:-1])
    elif fault in {"closure", "drops"}:
        summary = Path(str(path) + ".summary.json")
        data = json.loads(summary.read_text())
        data["writer_stopped" if fault == "closure" else "dropped_events"] = (
            False if fault == "closure" else 1
        )
        summary.write_text(json.dumps(data))
    elif fault in {"duplicate_json", "nonfinite_json"}:
        content = path.read_text()
        addition = '"sequence":1,' if fault == "duplicate_json" else '"bad":NaN,'
        path.write_text("{" + addition + content[1:])
    with pytest.raises(ValueError):
        validate()
