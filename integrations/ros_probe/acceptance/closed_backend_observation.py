"""Closed joint original robot/probe correspondence, never physical acceptance."""

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

from backend_observer_replay import BackendObserverReplay
from native_contact_evidence import reopen_native_policy

from rosclaw.connectors.ros.diagnosis.coverage_audit import digest


def object_unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate original joint audit JSON key")
        result[key] = value
    return result


def nonfinite(value):
    raise ValueError("nonfinite original joint audit JSON constant")


def strict_json(raw):
    return json.loads(raw, object_pairs_hook=object_unique, parse_constant=nonfinite)


def closed_backend_observation(
    path,
    robot_policy,
    probe_policy,
    *,
    robot_directory,
    probe_directory,
    robot_plugin,
    probe_plugin,
    robot_pose_frame,
    probe_pose_frame,
    scene_binding=None,
    probe_declaration=None,
    instrument_service_binary=None,
    instrument_service_binary_sha256=None,
    prefix_projection=None,
):
    """Replay complete owned original sources with the online event engine."""
    if (instrument_service_binary is None) != (instrument_service_binary_sha256 is None):
        raise ValueError("complete frozen instrument SDK parser path/SHA required")
    if prefix_projection is not None and (
        type(prefix_projection) is not dict
        or type(prefix_projection.get("snapshot")) is not dict
        or prefix_projection["snapshot"].get("live_source_constraint_satisfied") is not True
        or prefix_projection["snapshot"].get("source_fault") is not None
        or prefix_projection["snapshot"].get("robot_collision_count") != 0
        or prefix_projection["snapshot"].get("probe_completed_cache_cycles", 0) < 1
    ):
        raise ValueError("exact healthy original projection before registered source loss required")
    prefix_reached = False
    replay_count = 0
    sdk_rpc_replays = 0
    engine = BackendObserverReplay(
        robot_policy,
        probe_policy,
        robot_pose_frame=robot_pose_frame,
        probe_pose_frame=probe_pose_frame,
        scene_binding=scene_binding,
        probe_declaration=probe_declaration,
        instrument_service_binary_sha256=instrument_service_binary_sha256,
    )
    for root, policy, plugin in (
        (robot_directory, robot_policy, robot_plugin),
        (probe_directory, probe_policy["native_policy"], probe_plugin),
    ):
        reopen_native_policy(root, policy, plugin_path=plugin)
    path = Path(path)
    summary_path = Path(str(path) + ".summary.json")
    if (
        any(p.is_symlink() or not p.is_file() for p in (path, summary_path))
        or not 0 < path.stat().st_size <= 1_000_000_000
        or not 0 < summary_path.stat().st_size <= 65536
    ):
        raise ValueError("bounded regular closed joint observation tape required")
    identity = (path.stat().st_dev, path.stat().st_ino, path.stat().st_size)
    summary_raw = summary_path.read_bytes()
    summary = strict_json(summary_raw)
    if (
        type(summary) is not dict
        or summary.get("complete") is not True
        or summary.get("writer_stopped") is not True
        or type(summary.get("dropped_events")) is not int
        or summary["dropped_events"] != 0
        or summary.get("writer_error") is not None
    ):
        raise ValueError("stopped complete lossless original joint writer required")
    expected = {
        "run_id": engine.binding["run_id"],
        "body_snapshot_hash": engine.binding["body_snapshot_hash"],
        "constraint_policy_hash": engine.gate.policy_hash,
        "evidence_domain": "SIMULATION",
        "source": "independent_SIM_backend_original_sources",
    }
    previous, count, last_capture, sample_count, latest = None, 0, None, 0, None
    source_hash = hashlib.sha256()
    with path.open("rb") as stream:
        for line in iter(lambda: stream.readline(2_000_001), b""):
            if len(line) > 2_000_000 or not line.endswith(b"\n"):
                raise ValueError("bounded complete original joint audit row required")
            source_hash.update(line)
            row = strict_json(line)
            if type(row) is not dict:
                raise ValueError("original joint event object required")
            saved = row.pop("artifact_sha256", None)
            count += 1
            if (
                count > 2_000_000
                or row.get("schema_version") != "rosclaw.coverage_audit_event.v1"
                or type(row.get("sequence")) is not int
                or row["sequence"] != count
                or row.get("previous_hash") != previous
                or digest(row) != saved
            ):
                raise ValueError("joint original audit genesis/sequence/hash differs")
            previous = saved
            if any(row.get(k) != v for k, v in expected.items()):
                raise ValueError("joint original observation run/Body/policy/source differs")
            captured = datetime.fromisoformat(row["captured_at"])
            if captured.tzinfo is None or (last_capture is not None and captured < last_capture):
                raise ValueError("original joint event capture clock regressed")
            last_capture = captured
            payload = row["payload"]
            wall = payload.get("received_monotonic_sec")
            audit_wall = row.get("wall_monotonic_sec")
            if (
                any(
                    type(v) not in (int, float) or not math.isfinite(v) or v < 0
                    for v in (wall, audit_wall)
                )
                or not 0 <= audit_wall - wall < 0.3
            ):
                raise ValueError("joint original receipt differs from captured audit")
            if row["kind"] in {
                "backend_robot_components",
                "backend_probe_components",
                "backend_probe_lift_ack",
                "backend_probe_lift_begin",
                "backend_scene_components",
            }:
                unix = payload.get("received_unix_ns")
                if (
                    type(unix) is not int
                    or not 0 <= round(captured.timestamp() * 1e9) - unix < 300_000_000
                ):
                    raise ValueError("joint original UNIX receipt differs from captured audit")
            if prefix_reached:
                # Preserve/check the entire original closed audit chain, but no
                # post-fault source is replayed or promoted to healthy evidence.
                continue
            if row["kind"] == "backend_probe_lift_ack" and instrument_service_binary is not None:
                from closed_native_contact_evidence import original_ros_bytes
                from instrument_service_evidence import verify_service_sdk_projection

                raw_service = original_ros_bytes(
                    payload["original_service_record"], "owned_gazebo_instrument_RPC_original_json"
                )
                verify_service_sdk_projection(
                    raw_service,
                    probe_policy,
                    binary_path=instrument_service_binary,
                    binary_sha256=instrument_service_binary_sha256,
                    robot_model_name=robot_policy["contact_policy"]["model_name"],
                    partition="rosclaw_backend_"
                    + hashlib.sha256(engine.binding["run_id"].encode()).hexdigest()[:32],
                )
                sdk_rpc_replays += 1
            actual = engine.apply(row["kind"], payload)
            replay_count += 1
            if digest(payload.get("projection")) != digest(actual):
                raise ValueError("joint retained online projection differs from original replay")
            expected_sim = actual.get("sim_time_sec")
            if row["kind"] == "backend_observation_sample":
                expected_sim = actual["snapshot"]["robot_sim_time_sec"]
                sample_count += 1
                latest = actual
            if row.get("sim_time_sec") != expected_sim:
                raise ValueError("joint audit SIM clock differs from original projection")
            if prefix_projection is not None and actual == prefix_projection:
                if row["kind"] != "backend_observation_sample":
                    raise ValueError("registered healthy prefix must end at an actual observation")
                prefix_reached = True
    if (
        (path.stat().st_dev, path.stat().st_ino, path.stat().st_size) != identity
        or summary_path.read_bytes() != summary_raw
        or type(summary.get("events_written")) is not int
        or summary["events_written"] != count
        or summary.get("last_event_hash") != previous
    ):
        raise ValueError("joint original writer/source closure differs")
    for root, policy, plugin in (
        (robot_directory, robot_policy, robot_plugin),
        (probe_directory, probe_policy["native_policy"], probe_plugin),
    ):
        reopen_native_policy(root, policy, plugin_path=plugin)
    if (
        latest is None
        or not latest["actor_envelope"]["live_source_constraint_satisfied"]
        or engine.gate.fault
        or (prefix_projection is not None and not prefix_reached)
        or (prefix_projection is None and engine.gate.probe.pending)
        or (prefix_projection is None and engine.gate.probe.lift_transaction is not None)
    ):
        raise ValueError("closed joined original source has no fresh qualified final constraint")
    return {
        "original_service_wire_required": instrument_service_binary is not None,
        "original_service_SDK_wire_replays": sdk_rpc_replays,
        "evidence_role": "closed_joint_source_correspondence_not_world_admission_or_stop_or_task_acceptance",
        "original_source_sha256": source_hash.hexdigest(),
        "summary_sha256": hashlib.sha256(summary_raw).hexdigest(),
        "constraint_policy_hash": engine.gate.policy_hash,
        "source_window": (
            "PREFIX_BEFORE_REGISTERED_SOURCE_LOSS"
            if prefix_projection is not None
            else "FULL_CLOSED_SOURCE"
        ),
        "post_prefix_source_health": "UNKNOWN"
        if prefix_projection is not None
        else "FULL_REPLAY_REQUIRED",
        "closed_audit_events_checked": count,
        "events_replayed": replay_count,
        "samples_replayed": sample_count,
        "robot_collision_count": engine.gate.robot.snapshot(engine.last_wall)["collision_count"],
        "probe_completed_cache_cycles": engine.gate.probe.tracker.cycles,
        "spatial_source_join_required": engine.spatial is not None,
        "completed_exact_scene_joins": engine.spatial.completed if engine.spatial else None,
        "actual_world_and_body_admission": "NOT_VERIFIED",
        "backend_health_admitted": False,
        "physical_acceptance": "NOT_VERIFIED",
        "authorization": False,
    }
