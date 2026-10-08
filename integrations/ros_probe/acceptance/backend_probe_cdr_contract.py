"""Actual ROS serialization contracts over declared synthetic probe episodes.

No Node, DDS, Gazebo world, scene service or physical backend is started.
"""

import argparse
import copy
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from backend_probe_fixture import prepare_probe_fixture
from closed_backend_probe import ProbeEventReplay, closed_probe_replay
from geometry_msgs.msg import TransformStamped
from native_contact_cdr_contract import retained, save
from native_contact_evidence import prepare_native_policy
from probe_lift_evidence import lift_request
from rclpy.serialization import serialize_message
from tf2_msgs.msg import TFMessage

from rosclaw.connectors.ros.diagnosis.coverage_audit import digest


def fixture(root, plugin):
    declaration = {
        "schema_version": "rosclaw.backend_probe_fixture_declaration.v1",
        "source": "simulator_operator_fixture_policy",
        "approved": True,
        "evidence_domain": "SIMULATION",
        "run_id": "synthetic_probe_cdr_run",
        "world_name": "fixture_world",
        "robot_model_name": "separate_synthetic_robot",
        "probe_model_name": "instrument_probe",
        "probe_xy": [6, 0],
        "sphere_radius_m": 0.05,
        "ground_z_m": 0,
        "lift_z_m": 10,
        "ground_collision_name": "support_plane::floor_link::floor_collision",
        "cleaning_polygon": [[-1, -1], [1, -1], [1, 1], [-1, 1]],
        "maximum_robot_radius_m": 0.3,
        "pose_topic": "/instrument/pose",
        "component_topic": "/instrument/components",
        "contact_topic": "/instrument/contact",
    }
    artifact = prepare_probe_fixture(root, declaration)
    shutil.copyfile(plugin, root / "libcontacts.so")
    native = prepare_native_policy(
        root,
        plugin_path=root / "libcontacts.so",
        support_topics=["/instrument/contact"],
        ground_collisions=[declaration["ground_collision_name"]],
        pose_topic="/instrument/pose",
        component_topic="/instrument/components",
        component_gz_topic="/rosclaw_sim/backend_probe_components",
    )
    policy = {
        "schema_version": "rosclaw.backend_cache_probe_policy.v1",
        "source": "simulator_operator_fixture_policy",
        "approved": True,
        "evidence_domain": "SIMULATION",
        "native_policy": native,
        "probe_xy": [6, 0],
        "sphere_radius_m": 0.05,
        "ground_z_m": 0,
        "lift_z_m": 10,
        "stable_samples": 10,
        "stable_sim_span_sec": 0.4,
        "refresh_sim_sec": 10,
        "refresh_wall_sec": 20,
    }
    (root / "probe-policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    source = (
        Path(__file__).resolve().parents[3]
        / "tests/connectors/ros/fixtures/passive-native-contact-contract-packets.jsonl"
    )
    template = next(
        row["packet"]
        for row in map(json.loads, source.read_text().splitlines())
        if row["case"] == "native_actual_contact_entity_names"
    )
    template.update(artifact["binding"])
    template["contact_sources"][0].update(
        sensor_name="probe_contact",
        link_name="probe_link",
        gz_topic="/rosclaw_sim/backend_probe_contact",
    )
    collision = "instrument_probe::probe_link::probe_collision"
    template["collision_contacts"][0]["collision_name"] = collision
    template["collision_contacts"][0]["contacts"][0]["collision1_name"] = collision
    replay, rows = ProbeEventReplay(policy, "fixture_world"), []
    origin = 1_791_504_000_000_000_000

    def emit(kind, payload):
        result = replay.apply(kind, payload)
        payload["projection"] = result
        wall = payload["received_monotonic_sec"]
        rows.append(
            {
                "schema_version": "rosclaw.coverage_audit_event.v1",
                **{
                    k: artifact["binding"][k]
                    for k in ("run_id", "body_snapshot_hash", "attachment_hash", "producer_id")
                },
                "source": "owned_SIM_backend_probe_original_sources",
                "evidence_domain": "SIMULATION",
                "probe_policy_hash": digest(policy),
                "pose_frame": "fixture_world",
                "kind": kind,
                "payload": payload,
                "wall_monotonic_sec": wall + 0.0001,
                "captured_at": datetime.fromtimestamp(
                    (origin + round((wall - 100) * 1e9) + 100_000) / 1e9, UTC
                ).isoformat(),
                "sim_time_sec": result.get("sim_time_sec")
                if kind in {"backend_probe_pose", "backend_probe_components"}
                else replay.tracker.previous[2]
                if kind == "backend_probe_snapshot" and replay.tracker.previous
                else None,
            }
        )

    for i in range(1, 37):
        sim, wall = i * 0.05, 100 + i * 0.05
        z, touching = (10 - (i - 13) * 0.05, False) if 13 <= i <= 24 else (0.05, True)
        item = TransformStamped()
        item.header.frame_id, item.child_frame_id = "fixture_world", "instrument_probe"
        item.header.stamp.sec, item.header.stamp.nanosec = divmod(round(sim * 1e9), 1_000_000_000)
        item.transform.translation.x, item.transform.translation.z = 6.0, float(z)
        item.transform.rotation.w = 1.0
        emit(
            "backend_probe_pose",
            {
                **retained(
                    serialize_message(TFMessage(transforms=[item])), "tf2_msgs/msg/TFMessage"
                ),
                "received_monotonic_sec": wall,
            },
        )
        packet = copy.deepcopy(template)
        packet.update(
            sequence=i,
            iterations=i * 50,
            sim_time_sec=sim,
            captured_at_unix_ns=origin + round(sim * 1e9),
            body_world_pose=[6, 0, z, 1, 0, 0, 0],
        )
        if not touching:
            packet["collision_contacts"][0]["contacts"] = []
        emit(
            "backend_probe_components",
            {
                **retained(json.dumps(packet).encode(), "gazebo_ecm_contact_sensor_data_json"),
                "received_monotonic_sec": wall + 0.001,
                "received_unix_ns": packet["captured_at_unix_ns"] + 1_000_000,
            },
        )
        emit("backend_probe_snapshot", {"received_monotonic_sec": wall + 0.002})
        if i == 12:
            emit(
                "backend_probe_lift_ack",
                {
                    "request": retained(lift_request(policy), "gz.msgs.Pose_protobuf_text"),
                    "response": retained(b"data: true\n", "gz.msgs.Boolean_protobuf_text"),
                    "returncode": 0,
                    "received_monotonic_sec": wall + 0.003,
                    "received_unix_ns": packet["captured_at_unix_ns"] + 3_000_000,
                },
            )
    return policy, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--plugin", required=True, type=Path)
    args = parser.parse_args()
    args.directory.mkdir()
    cases = []
    for fault in (
        "valid",
        "invalid_cdr",
        "original_pose_frame",
        "original_sha",
        "projection",
        "ack_robot_target",
        "ack_failed_reply",
        "unix_receipt",
        "source_binding",
        "sim_clock",
        "closure",
        "partial",
    ):
        root = args.directory / fault
        policy, rows = fixture(root, args.plugin)
        pose = next(row for row in rows if row["kind"] == "backend_probe_pose")
        component = next(row for row in rows if row["kind"] == "backend_probe_components")
        ack = next(row for row in rows if row["kind"] == "backend_probe_lift_ack")
        if fault == "invalid_cdr":
            pose["payload"].update(retained(b"invalid serialized pose", "tf2_msgs/msg/TFMessage"))
        elif fault == "original_pose_frame":
            item = TransformStamped()
            item.header.frame_id, item.child_frame_id = "wrong_world", "instrument_probe"
            item.transform.rotation.w = 1.0
            pose["payload"].update(
                retained(serialize_message(TFMessage(transforms=[item])), "tf2_msgs/msg/TFMessage")
            )
        elif fault == "original_sha":
            component["payload"]["original_source_sha256"] = "0" * 64
        elif fault == "projection":
            component["payload"]["projection"]["completed_cache_cycles"] = 9
        elif fault == "ack_robot_target":
            ack["payload"]["request"] = retained(
                b'name: "separate_synthetic_robot"', "gz.msgs.Pose_protobuf_text"
            )
        elif fault == "ack_failed_reply":
            ack["payload"]["response"] = retained(b"data: false", "gz.msgs.Boolean_protobuf_text")
        elif fault == "unix_receipt":
            component["payload"]["received_unix_ns"] += 10_000_000_000
        elif fault == "source_binding":
            component["run_id"] = "wrong_run"
        elif fault == "sim_clock":
            component["sim_time_sec"] += 1
        path = save(root, rows)
        if fault == "closure":
            Path(str(path) + ".summary.json").write_text("{}")
        elif fault == "partial":
            path.write_bytes(path.read_bytes()[:-1])
        try:
            result = closed_probe_replay(
                path, policy, plugin_path=root / "libcontacts.so", pose_frame="fixture_world"
            )
            if fault != "valid":
                raise AssertionError("invalid probe source admitted: " + fault)
            cases.append(
                {"case": fault, "status": "PASS_OFFLINE_ORIGINAL_CDR_REPLAY", "result": result}
            )
        except ValueError as exc:
            if fault == "valid":
                raise
            cases.append({"case": fault, "status": "PASS_EXPECTED_REJECTION", "error": str(exc)})
    summary = {
        "status": "PASS_OFFLINE",
        "cases": cases,
        "actual_ros_serialization": True,
        "native_json_frames": "SYNTHETIC_DERIVED_SDK_FIXTURES",
        "node_instantiated": False,
        "dds_or_gazebo_started": False,
        "service_executed": False,
        "backend_health_admitted": False,
        "physical_acceptance": "NOT_RUN",
    }
    (args.directory / "contract-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
