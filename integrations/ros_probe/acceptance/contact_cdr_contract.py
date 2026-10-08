"""Offline official-ROS CDR contracts; no Node, DDS, simulation or motion."""

import argparse
import base64
import copy
import hashlib
import json
import tempfile
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from pathlib import Path

from closed_contact_evidence import closed_contact_window
from contact_evidence import IndependentContacts, prepare_contact_policy
from geometry_msgs.msg import TransformStamped
from rclpy.serialization import serialize_message
from ros_gz_interfaces.msg import Contact, Contacts
from tf2_msgs.msg import TFMessage

from rosclaw.connectors.ros.diagnosis.coverage_audit import digest


def fixture(policy=None):
    policy = policy or {
        "schema_version": "rosclaw.independent_contact_policy.v1",
        "run_id": "synthetic_run",
        "body_snapshot_hash": "synthetic_body",
        "attachment_hash": "synthetic_brush",
        "producer_id": "synthetic_actor",
        "model_name": "anonymous_body",
        "pose_topic": "/independent_pose",
        "contacts": {
            "/contacts/support": ["anonymous_body::support::solid"],
            "/contacts/body": ["anonymous_body::base::solid"],
        },
        "support_topics": ["/contacts/support"],
        "ground_collisions": ["floor::ground::solid"],
        "source_sdf_sha256": "a" * 64,
        "source_bridge_sha256": "b" * 64,
    }
    tracker = IndependentContacts(policy)
    rows = []
    origin = datetime(2026, 10, 8, 12, tzinfo=UTC)

    def emit(kind, payload, sim, wall, captured):
        rows.append(
            {
                "schema_version": "rosclaw.coverage_audit_event.v1",
                **{
                    k: policy[k]
                    for k in ("run_id", "body_snapshot_hash", "attachment_hash", "producer_id")
                },
                "source": "independent_gazebo_contact_subscription",
                "evidence_domain": "SIMULATION",
                "contact_policy_hash": tracker.policy_hash,
                "kind": kind,
                "payload": payload,
                "sim_time_sec": sim,
                "wall_monotonic_sec": wall + 0.0001,
                "captured_at": (captured + timedelta(seconds=0.0001)).isoformat(),
            }
        )

    def serialized(message, type_name, received):
        raw = serialize_message(message)
        return {
            "serialized_message_base64": base64.b64encode(raw).decode(),
            "serialized_message_sha256": hashlib.sha256(raw).hexdigest(),
            "source_bytes_complete": True,
            "original_size_bytes": len(raw),
            "source_type": type_name,
            "received_monotonic_sec": received,
        }

    for i in range(41):
        sim, wall = 10 + i / 10, 100 + i / 10
        captured = origin + timedelta(seconds=i / 10)
        ns = round(sim * 1e9)
        transform = TransformStamped()
        transform.header.stamp.sec, transform.header.stamp.nanosec = divmod(ns, 1_000_000_000)
        transform.child_frame_id = "anonymous_body"
        transform.transform.rotation.w = 1.0
        poses = TFMessage(transforms=[transform])
        tracker.pose(sim, wall)
        emit(
            "independent_contact_pose_received",
            serialized(poses, "tf2_msgs/msg/TFMessage", wall),
            sim,
            wall,
            captured,
        )
        for index, topic in enumerate(policy["contacts"]):
            message = Contacts()
            message.header.stamp = transform.header.stamp
            pairs = []
            if index == 0:
                contact = Contact()
                contact.collision1.name = policy["contacts"][topic][0]
                contact.collision2.name = policy["ground_collisions"][0]
                message.contacts = [contact]
                pairs = [[contact.collision1.name, contact.collision2.name]]
            received = wall + (index + 1) / 1000
            tracker.contacts(topic, sim, received, pairs)
            emit(
                "independent_contact_source_received",
                {
                    **serialized(message, "ros_gz_interfaces/msg/Contacts", received),
                    "topic": topic,
                    "collisions": pairs,
                },
                sim,
                received,
                captured + timedelta(seconds=(index + 1) / 1000),
            )
        sampled = wall + 0.003
        observation = tracker.snapshot(sampled)
        observation["snapshot_monotonic_sec"] = sampled
        observation["captured_at"] = (captured + timedelta(seconds=0.003)).isoformat()
        emit(
            "independent_contact_observation",
            observation,
            sim,
            sampled,
            datetime.fromisoformat(observation["captured_at"]),
        )
    return policy, rows, origin


def prepared_sources(root):
    model = "anonymous_body"
    links = "".join(
        f'<link name="{name}"><collision name="solid"><geometry><box><size>0.1 0.1 0.1</size></box></geometry></collision><sensor name="touch" type="contact"><topic>/contacts/{topic}</topic><contact><collision>solid</collision></contact></sensor></link>'
        for name, topic in (("support", "support"), ("base", "body"))
    )
    pose_plugin = '<plugin name="gz::sim::systems::PosePublisher"><topic>/actual_pose</topic><publish_model_pose>true</publish_model_pose><use_pose_vector_msg>true</use_pose_vector_msg></plugin>'
    (root / "robot.sdf").write_text(
        f'<sdf version="1.9"><model name="{model}">{links}{pose_plugin}</model></sdf>'
    )
    bridge = [
        {
            "ros_topic_name": topic,
            "gz_topic_name": topic,
            "ros_type_name": "ros_gz_interfaces/msg/Contacts",
            "gz_type_name": "gz.msgs.Contacts",
            "direction": "GZ_TO_ROS",
        }
        for topic in ("/contacts/support", "/contacts/body")
    ]
    bridge.append(
        {
            "ros_topic_name": "/rosclaw_sim/ground_truth",
            "gz_topic_name": "/actual_pose",
            "gz_type_name": "gz.msgs.Pose_V",
            "ros_type_name": "tf2_msgs/msg/TFMessage",
            "direction": "GZ_TO_ROS",
        }
    )
    (root / "bridge.yaml").write_text(json.dumps(bridge))
    binding = {
        "run_id": "synthetic_run",
        "body_snapshot_hash": "synthetic_body",
        "attachment_hash": "synthetic_brush",
        "producer_id": "synthetic_actor",
    }
    (root / "brush_binding.json").write_text(json.dumps(binding))
    (root / "physics_binding.json").write_text(
        json.dumps({**binding, "body_model_name": model, "world_name": "fixture_world"})
    )
    return prepare_contact_policy(
        root, support_topics=["/contacts/support"], ground_collisions=["floor::ground::solid"]
    )


def write(path, rows, *, incomplete=False):
    previous = None
    with path.open("w") as stream:
        for sequence, original in enumerate(rows, 1):
            row = {**copy.deepcopy(original), "sequence": sequence, "previous_hash": previous}
            row["artifact_sha256"] = digest(row)
            previous = row["artifact_sha256"]
            stream.write(json.dumps(row) + "\n")
    Path(str(path) + ".summary.json").write_text(
        json.dumps(
            {
                "complete": not incomplete,
                "writer_stopped": True,
                "events_written": len(rows),
                "dropped_events": 0,
                "writer_error": None,
                "last_event_hash": previous,
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory",
        type=Path,
        help="Exclusive owned directory retaining all synthetic source cases",
    )
    args = parser.parse_args()
    if args.directory is not None:
        args.directory.mkdir(parents=True, exist_ok=False)
    cases = [
        "valid",
        "projection",
        "serialized_hash",
        "incomplete_source_bytes",
        "source_binding",
        "source_rejection",
        "incomplete",
        "gap",
        "missing_bracket",
        "pose_projection",
        "pose_source",
        "summary_hash",
        "summary_drop",
        "audit_chain",
        "source_receipt",
        "contact_observation",
    ]
    results = []
    context = (
        nullcontext(str(args.directory))
        if args.directory is not None
        else tempfile.TemporaryDirectory(prefix="independent-contact-cdr-contract-")
    )
    with context as directory:
        policy, rows, origin = fixture(prepared_sources(Path(directory)))
        for case in cases:
            path = Path(directory) / (case + ".jsonl")
            changed = copy.deepcopy(rows)
            if case == "projection":
                changed[41]["payload"]["collisions"] = []
            elif case == "serialized_hash":
                changed[41]["payload"]["serialized_message_sha256"] = "f" * 64
            elif case == "incomplete_source_bytes":
                changed[41]["payload"]["source_bytes_complete"] = False
            elif case == "source_binding":
                changed[40]["run_id"] = "foreign"
            elif case == "source_rejection":
                changed[40]["kind"] = "independent_contact_source_rejected"
            elif case == "gap":
                changed = changed[:40] + changed[56:]
            elif case == "missing_bracket":
                changed = changed[80:]
            elif case == "pose_projection":
                changed[40]["sim_time_sec"] += 1
            elif case == "pose_source":
                changed[40]["payload"]["source_type"] = "foreign"
            elif case == "source_receipt":
                changed[40]["payload"]["received_monotonic_sec"] += 1
            elif case == "contact_observation":
                changed[43]["payload"]["collision_count"] = 1
            write(path, changed, incomplete=case == "incomplete")
            if case in {"summary_hash", "summary_drop"}:
                p = Path(str(path) + ".summary.json")
                summary = json.loads(p.read_text())
                if case == "summary_hash":
                    summary["last_event_hash"] = "f" * 64
                else:
                    summary["dropped_events"] = 1
                p.write_text(json.dumps(summary))
            elif case == "audit_chain":
                path.write_text(
                    path.read_text().replace('"previous_hash": null', '"previous_hash": "old"', 1)
                )
            try:
                report = closed_contact_window(
                    path,
                    policy,
                    start_sim=10.5,
                    end_sim=13.5,
                    start_wall=(origin + timedelta(seconds=0.5)).isoformat(),
                    end_wall=(origin + timedelta(seconds=3.5)).isoformat(),
                )
            except (ValueError, KeyError, TypeError):
                if case == "valid":
                    raise
                results.append({"case": case, "status": "EXPECTED_REJECTION"})
            else:
                if case != "valid":
                    raise AssertionError("invalid original-source replay accepted: " + case)
                assert (
                    report["collision_count"] == 0
                    and report["physical_acceptance"] == "NOT_VERIFIED"
                )
                results.append(
                    {
                        "case": case,
                        "status": "PASS_OFFLINE_CDR_REPLAY",
                        "samples": report["sample_count"],
                    }
                )
    print(
        json.dumps(
            {
                "status": "PASS_OFFLINE_ONLY",
                "official_ros_serialization": True,
                "node_or_dds_started": False,
                "physical_acceptance": "NOT_RUN",
                "cases": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
