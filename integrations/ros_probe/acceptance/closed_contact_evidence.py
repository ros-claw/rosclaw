"""Closed independent contact source replay; no task acceptance or motion."""

import base64
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

from contact_evidence import IndependentContacts, contact_policy, reopen_contact_policy

from rosclaw.connectors.ros.diagnosis.coverage_audit import digest


def decode_ros_contacts(raw):
    """Use the installed official ROS message decoder without starting a Node."""
    from rclpy.serialization import deserialize_message
    from ros_gz_interfaces.msg import Contacts

    message = deserialize_message(raw, Contacts)
    sec, ns = message.header.stamp.sec, message.header.stamp.nanosec
    if type(sec) is not int or type(ns) is not int or sec < 0 or not 0 <= ns < 1_000_000_000:
        raise ValueError("actual serialized ROS contact source stamp required")
    return sec + ns / 1e9, [[c.collision1.name, c.collision2.name] for c in message.contacts]


def decode_ros_pose(raw, model_name):
    from rclpy.serialization import deserialize_message
    from tf2_msgs.msg import TFMessage

    message = deserialize_message(raw, TFMessage)
    matches = [t for t in message.transforms if t.child_frame_id == model_name]
    if len(matches) != 1:
        raise ValueError("one original serialized independent Body pose required")
    item = matches[0]
    p, q = item.transform.translation, item.transform.rotation
    if (
        not all(math.isfinite(v) for v in (p.x, p.y, p.z, q.x, q.y, q.z, q.w))
        or abs(sum(v * v for v in (q.x, q.y, q.z, q.w)) - 1) > 1e-6
    ):
        raise ValueError("finite normalized original serialized pose required")
    sec, ns = item.header.stamp.sec, item.header.stamp.nanosec
    if type(sec) is not int or type(ns) is not int or sec < 0 or not 0 <= ns < 1_000_000_000:
        raise ValueError("actual serialized independent pose stamp required")
    return sec + ns / 1e9


def original_ros_bytes(payload, expected_type):
    raw = base64.b64decode(payload["serialized_message_base64"], validate=True)
    if (
        not 0 < len(raw) <= 262144
        or payload.get("source_bytes_complete") is not True
        or type(payload.get("original_size_bytes")) is not int
        or payload["original_size_bytes"] != len(raw)
        or hashlib.sha256(raw).hexdigest() != payload["serialized_message_sha256"]
        or payload["source_type"] != expected_type
    ):
        raise ValueError("original serialized ROS source bytes differ")
    return raw


def closed_contact_window(path, policy, *, start_sim, end_sim, start_wall, end_wall):
    """Reopen original DDS bytes and require complete zero-contact window brackets."""
    contact_policy(policy)
    if (
        any(
            type(v) not in (int, float) or not math.isfinite(v) or v < 0
            for v in (start_sim, end_sim)
        )
        or end_sim - start_sim < 2.5
    ):
        raise ValueError("bounded advancing independent source window required")
    begin, end = datetime.fromisoformat(start_wall), datetime.fromisoformat(end_wall)
    if begin.tzinfo is None or end.tzinfo is None or (end - begin).total_seconds() < 2.5:
        raise ValueError("timezone-aware independent wall window required")
    path = Path(path)
    reopen_contact_policy(path.parent, policy)
    summary_path = Path(str(path) + ".summary.json")
    if (
        path.is_symlink()
        or not 0 < path.stat().st_size <= 1_000_000_000
        or not 0 < summary_path.stat().st_size <= 65536
    ):
        raise ValueError("bounded original closed independent contact archive required")
    file_identity = (path.stat().st_dev, path.stat().st_ino, path.stat().st_size)
    summary_bytes = summary_path.read_bytes()
    summary = json.loads(summary_bytes)
    if (
        summary.get("complete") is not True
        or summary.get("writer_stopped") is not True
        or type(summary.get("dropped_events")) is not int
        or summary["dropped_events"] != 0
        or summary.get("writer_error") is not None
    ):
        raise ValueError("complete lossless closed independent contact writer required")
    tracker = IndependentContacts(policy)
    previous_hash = None
    sequence = 0
    observations = []
    original_hash = hashlib.sha256()
    last_capture = None
    with path.open("rb") as stream:
        for line in iter(lambda: stream.readline(2_000_001), b""):
            original_hash.update(line)
            if len(line) > 2_000_000 or not line.endswith(b"\n"):
                raise ValueError("bounded complete original contact audit row required")
            row = json.loads(line)
            saved_hash = row.pop("artifact_sha256", None)
            sequence += 1
            if (
                sequence > 2_000_000
                or row.get("schema_version") != "rosclaw.coverage_audit_event.v1"
                or row.get("sequence") != sequence
                or row.get("previous_hash") != previous_hash
                or digest(row) != saved_hash
            ):
                raise ValueError("original contact audit genesis/hash/sequence differs")
            previous_hash = saved_hash
            if (
                any(
                    row.get(k) != policy[k]
                    for k in ("run_id", "body_snapshot_hash", "attachment_hash", "producer_id")
                )
                or row.get("source") != "independent_gazebo_contact_subscription"
                or row.get("evidence_domain") != "SIMULATION"
                or row.get("contact_policy_hash") != tracker.policy_hash
            ):
                raise ValueError("closed independent contact source binding differs")
            captured = datetime.fromisoformat(row["captured_at"])
            if captured.tzinfo is None or (last_capture is not None and captured < last_capture):
                raise ValueError("original contact receipt clock regressed")
            last_capture = captured
            kind, payload = row["kind"], row["payload"]
            if kind == "independent_contact_source_rejected":
                raise ValueError("independent source rejection remains latched in closed evidence")
            if kind == "independent_contact_startup_clock_pending":
                continue
            if kind in {"independent_contact_pose_received", "independent_contact_source_received"}:
                received = payload["received_monotonic_sec"]
                if (
                    type(received) not in (int, float)
                    or not math.isfinite(received)
                    or not 0 <= row["wall_monotonic_sec"] - received < 0.3
                ):
                    raise ValueError(
                        "actual source receipt time differs from retained audit receipt"
                    )
            if kind == "independent_contact_pose_received":
                raw = original_ros_bytes(payload, "tf2_msgs/msg/TFMessage")
                when = decode_ros_pose(raw, policy["model_name"])
                if row["sim_time_sec"] != when:
                    raise ValueError("retained pose clock differs from original ROS bytes")
                tracker.pose(when, payload["received_monotonic_sec"])
            elif kind == "independent_contact_source_received":
                raw = original_ros_bytes(payload, "ros_gz_interfaces/msg/Contacts")
                sim_time, pairs = decode_ros_contacts(raw)
                if row["sim_time_sec"] != sim_time or payload["collisions"] != pairs:
                    raise ValueError("retained contact projection differs from original ROS bytes")
                tracker.contacts(
                    payload["topic"], sim_time, payload["received_monotonic_sec"], pairs
                )
            elif kind == "independent_contact_observation":
                recorded = dict(payload)
                sampled = recorded.pop("snapshot_monotonic_sec")
                if not 0 <= row["wall_monotonic_sec"] - sampled < 0.3:
                    raise ValueError("retained observation sample time differs from actual receipt")
                actual = tracker.snapshot(sampled)
                payload_capture = recorded.pop("captured_at", None)
                if recorded != actual or datetime.fromisoformat(payload_capture) > captured:
                    raise ValueError("retained contact observation differs from source replay")
                if row["sim_time_sec"] is not None:
                    observations.append(
                        (row["sim_time_sec"], datetime.fromisoformat(payload_capture), actual)
                    )
                    if len(observations) > 50000:
                        raise ValueError(
                            "independent observation count exceeds frozen bounded duration"
                        )
            else:
                raise ValueError("unknown closed independent contact event")
    if (path.stat().st_dev, path.stat().st_ino, path.stat().st_size) != file_identity:
        raise ValueError("closed independent contact source replaced or changed during replay")
    if (
        type(summary.get("events_written")) is not int
        or summary["events_written"] != sequence
        or summary.get("last_event_hash") != previous_hash
        or summary_path.read_bytes() != summary_bytes
    ):
        raise ValueError("independent contact writer closure differs from final hash")
    before = [v for v in observations if v[0] <= start_sim and v[1] <= begin]
    after = [v for v in observations if v[0] >= end_sim and v[1] >= end]
    if not before or not after:
        raise ValueError("independent contact window lacks original source brackets")
    first, last = before[-1], after[0]
    selected = [
        v for v in observations if first[0] <= v[0] <= last[0] and first[1] <= v[1] <= last[1]
    ]
    if not 20 <= len(selected) <= 40000:
        raise ValueError("bounded complete independent contact sample window required")
    for index, (sim, wall, result) in enumerate(selected):
        if (
            not result["observation_complete"]
            or result["source_fault"] is not None
            or result["collision_count"] != 0
            or result["active_contact_topics"]
        ):
            raise ValueError("independent contact window is incomplete or contains contact")
        if index:
            sim_delta = sim - selected[index - 1][0]
            wall_delta = (wall - selected[index - 1][1]).total_seconds()
            if not 0 < sim_delta <= 0.3 or not 0 < wall_delta <= 0.3:
                raise ValueError("independent contact observation window has source gaps")
    return {
        "evidence_role": "closed_independent_contact_replay_not_task_acceptance",
        "source": "independent_gazebo_contact_subscription",
        "decoder_source": "installed_ros_gz_interfaces_Contacts_CDR",
        "source_file_sha256": original_hash.hexdigest(),
        "summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
        "contact_policy_hash": tracker.policy_hash,
        "sample_count": len(selected),
        "collision_count": 0,
        "observation_complete": True,
        "physical_acceptance": "NOT_VERIFIED",
        "start_sim": start_sim,
        "end_sim": end_sim,
    }
