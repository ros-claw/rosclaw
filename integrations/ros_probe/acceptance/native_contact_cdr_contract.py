"""Synthetic closed-audit contracts with official ROS CDR; no Node or physics.

Native JSON frames are explicitly synthetic projections derived from a retained
SDK fixture. They are never labelled as a running Gazebo producer's frames.
"""

import argparse
import base64
import copy
import hashlib
import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml
from closed_native_contact_evidence import closed_native_contact_window
from geometry_msgs.msg import TransformStamped
from native_contact_evidence import NativeContactEvidence, prepare_native_policy
from rclpy.serialization import deserialize_message, serialize_message
from tf2_msgs.msg import TFMessage

from rosclaw.connectors.ros.diagnosis.coverage_audit import digest


def retained(raw, kind):
    return {
        "original_source_base64": base64.b64encode(raw).decode(),
        "original_source_sha256": hashlib.sha256(raw).hexdigest(),
        "original_size_bytes": len(raw),
        "source_bytes_complete": True,
        "source_type": kind,
    }


def fixture(root, plugin):
    root.mkdir()
    source = (
        Path(__file__).resolve().parents[3]
        / "tests/connectors/ros/fixtures/passive-native-contact-contract-packets.jsonl"
    )
    packet = next(
        row["packet"]
        for row in map(json.loads, source.read_text().splitlines())
        if row["case"] == "native_actual_contact_entity_names"
    )
    (root / "robot.sdf").write_text(
        '<sdf version="1.9"><model name="anonymous_body"><link name="actual_base"><collision name="actual_collision"/><sensor name="native_touch" type="contact"><topic>/actual/contact</topic><contact><collision>actual_collision</collision><topic>/qualified/native_contact</topic></contact></sensor></link><plugin name="gz::sim::systems::PosePublisher"><topic>/actual_pose</topic><publish_model_pose>true</publish_model_pose><use_pose_vector_msg>true</use_pose_vector_msg></plugin></model></sdf>'
    )
    bridge = [
        {
            "ros_topic_name": "/actual/contact",
            "gz_topic_name": "/qualified/native_contact",
            "ros_type_name": "ros_gz_interfaces/msg/Contacts",
            "gz_type_name": "gz.msgs.Contacts",
            "direction": "GZ_TO_ROS",
        },
        {
            "ros_topic_name": "/actual_pose",
            "gz_topic_name": "/actual_pose",
            "ros_type_name": "tf2_msgs/msg/TFMessage",
            "gz_type_name": "gz.msgs.Pose_V",
            "direction": "GZ_TO_ROS",
        },
        {
            "ros_topic_name": "/rosclaw_sim/contact_components",
            "gz_topic_name": "/rosclaw_sim/contact_components",
            "ros_type_name": "std_msgs/msg/String",
            "gz_type_name": "gz.msgs.StringMsg",
            "direction": "GZ_TO_ROS",
        },
    ]
    (root / "bridge.yaml").write_text(yaml.safe_dump(bridge))
    binding = {
        k: packet[k] for k in ("run_id", "body_snapshot_hash", "attachment_hash", "producer_id")
    }
    (root / "brush_binding.json").write_text(json.dumps(binding))
    (root / "physics_binding.json").write_text(
        json.dumps({**binding, "world_name": "fixture_world", "body_model_name": "anonymous_body"})
    )
    shutil.copyfile(plugin, root / "libcontacts.so")
    policy = prepare_native_policy(
        root,
        plugin_path=root / "libcontacts.so",
        support_topics=["/actual/contact"],
        ground_collisions=["support_plane::floor_link::floor_collision"],
        pose_topic="/actual_pose",
    )
    (root / "native-policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    tracker = NativeContactEvidence(policy)
    origin = datetime(2026, 10, 8, 12, tzinfo=UTC)
    rows = []

    def emit(kind, payload, sim, wall, captured):
        rows.append(
            {
                "schema_version": "rosclaw.coverage_audit_event.v1",
                **binding,
                "source": "independent_gazebo_contact_component_subscription",
                "evidence_domain": "SIMULATION",
                "native_policy_hash": digest(policy),
                "pose_frame": "fixture_world",
                "kind": kind,
                "payload": payload,
                "sim_time_sec": sim,
                "wall_monotonic_sec": wall + 0.0001,
                "captured_at": (captured + timedelta(seconds=0.0001)).isoformat(),
            }
        )

    for i in range(41):
        sim, wall = 10 + i / 10, 100 + i / 10
        captured = origin + timedelta(seconds=i / 10)
        ns = round(sim * 1e9)
        transform = TransformStamped()
        transform.header.frame_id = "fixture_world"
        transform.child_frame_id = "anonymous_body"
        transform.header.stamp.sec, transform.header.stamp.nanosec = divmod(ns, 1_000_000_000)
        transform.transform.rotation.w = 1.0
        raw = serialize_message(TFMessage(transforms=[transform]))
        tracker.pose(sim, wall, [0, 0, 0, 1, 0, 0, 0])
        emit(
            "native_contact_pose_received",
            {**retained(raw, "tf2_msgs/msg/TFMessage"), "received_monotonic_sec": wall},
            sim,
            wall,
            captured,
        )
        native = copy.deepcopy(packet)
        native.update(
            sequence=i,
            iterations=100 + i * 100,
            sim_time_sec=sim,
            captured_at_unix_ns=int(captured.timestamp() * 1e9),
        )
        raw = json.dumps(native).encode()
        unix = native["captured_at_unix_ns"] + 10_000
        metadata = tracker.observe(
            raw, received_monotonic_sec=wall + 0.00001, received_unix_ns=unix
        )
        emit(
            "native_contact_source_received",
            {
                **retained(raw, "gazebo_ecm_contact_sensor_data_json"),
                **metadata,
                "received_monotonic_sec": wall + 0.00001,
                "received_unix_ns": unix,
            },
            sim,
            wall,
            captured,
        )
        result = tracker.snapshot(wall + 0.00002)
        emit(
            "native_contact_observation",
            {**result, "snapshot_monotonic_sec": wall + 0.00002},
            sim,
            wall,
            captured,
        )
    return policy, rows, origin


def save(root, rows):
    previous = None
    for index, row in enumerate(rows, 1):
        row.update(sequence=index, previous_hash=previous)
        row.pop("artifact_sha256", None)
        row["artifact_sha256"] = digest(row)
        previous = row["artifact_sha256"]
    path = root / "native-contact-events.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary = {
        "complete": True,
        "writer_stopped": True,
        "dropped_events": 0,
        "writer_error": None,
        "events_written": len(rows),
        "last_event_hash": previous,
    }
    Path(str(path) + ".summary.json").write_text(json.dumps(summary))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--plugin", required=True, type=Path)
    args = parser.parse_args()
    args.directory.mkdir()
    outcomes = []
    faults = [
        "valid",
        "bytes_hash",
        "bytes_size",
        "invalid_pose_cdr",
        "original_pose_frame",
        "frame",
        "projection",
        "source_binding",
        "receipt_unix",
        "sequence",
        "sim_regress",
        "inventory",
        "support_missing",
        "rejection",
        "hash_chain",
        "closure",
        "partial",
    ]
    for fault in faults:
        root = args.directory / fault
        policy, rows, origin = fixture(root, args.plugin)
        native = next(row for row in rows if row["kind"] == "native_contact_source_received")
        payload = native["payload"]
        if fault == "bytes_hash":
            payload["original_source_sha256"] = "f" * 64
        elif fault == "bytes_size":
            payload["original_size_bytes"] += 1
        elif fault == "invalid_pose_cdr":
            rows[0]["payload"].update(
                retained(b"invalid synthetic CDR source", "tf2_msgs/msg/TFMessage")
            )
        elif fault == "original_pose_frame":
            message = deserialize_message(
                base64.b64decode(rows[0]["payload"]["original_source_base64"]), TFMessage
            )
            message.transforms[0].header.frame_id = "foreign"
            rows[0]["payload"].update(
                retained(serialize_message(message), "tf2_msgs/msg/TFMessage")
            )
        elif fault == "frame":
            rows[0]["pose_frame"] = "foreign"
        elif fault == "projection":
            rows[2]["payload"]["backend_health_admitted"] = True
        elif fault == "source_binding":
            rows[4]["body_snapshot_hash"] = "foreign"
        elif fault == "receipt_unix":
            payload["received_unix_ns"] += 1_000_000_000
        elif fault in {"sequence", "sim_regress", "inventory", "support_missing"}:
            target = rows[4]["payload"]
            packet = json.loads(base64.b64decode(target["original_source_base64"]))
            if fault == "sequence":
                packet["sequence"] = 0
            elif fault == "sim_regress":
                packet["sim_time_sec"] = 9
            elif fault == "inventory":
                packet["contact_sources"][0]["sensor_entity_id"] = 999
            else:
                packet["collision_contacts"][0]["contacts"] = []
            target.update(
                retained(json.dumps(packet).encode(), "gazebo_ecm_contact_sensor_data_json")
            )
        elif fault == "rejection":
            rows[3]["kind"] = "native_contact_source_rejected"
        path = save(root, rows)
        if fault == "hash_chain":
            path.write_bytes(path.read_bytes().replace(b'"sequence": 2,', b'"sequence": 999,', 1))
        elif fault == "closure":
            Path(str(path) + ".summary.json").write_text("{}")
        elif fault == "partial":
            path.write_bytes(path.read_bytes()[:-1])
        try:
            result = closed_native_contact_window(
                path,
                policy,
                plugin_path=root / "libcontacts.so",
                pose_frame="fixture_world",
                start_sim=10.2,
                end_sim=13.8,
                start_wall=(origin + timedelta(seconds=0.2)).isoformat(),
                end_wall=(origin + timedelta(seconds=3.8)).isoformat(),
            )
            if fault != "valid":
                raise AssertionError("corrupted source unexpectedly admitted: " + fault)
            assert result["sample_count"] >= 20 and result["backend_health_admitted"] is False
            outcomes.append({"case": fault, "status": "PASS", "result": result})
        except ValueError as exc:
            if fault == "valid":
                raise
            outcomes.append({"case": fault, "status": "PASS_EXPECTED_REJECTION", "error": str(exc)})
    summary = {
        "role": "SYNTHETIC_OFFICIAL_CDR_AND_NATIVE_JSON_REPLAY_NOT_PHYSICS",
        "cases": outcomes,
        "node_instantiated": False,
        "physical_acceptance": "NOT_RUN",
        "backend_health_admitted": False,
    }
    (args.directory / "contract-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "cases": len(outcomes), "physical_acceptance": "NOT_RUN"}))


if __name__ == "__main__":
    main()
