"""Closed native contact-cache replay; no backend or task acceptance, no motion."""

import base64
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

from native_contact_evidence import NativeContactEvidence, native_policy, reopen_native_policy

from rosclaw.connectors.ros.diagnosis.coverage_audit import digest


def decode_ros_pose(raw, model_name, pose_frame):
    from rclpy.serialization import deserialize_message
    from tf2_msgs.msg import TFMessage

    message = deserialize_message(raw, TFMessage)
    matches = [t for t in message.transforms if t.child_frame_id == model_name]
    if len(matches) != 1 or matches[0].header.frame_id != pose_frame:
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
    return sec + ns / 1e9, [p.x, p.y, p.z, q.w, q.x, q.y, q.z]


def original_ros_bytes(payload, expected_type):
    raw = base64.b64decode(payload["original_source_base64"], validate=True)
    if (
        not 0 < len(raw) <= 262144
        or payload.get("source_bytes_complete") is not True
        or type(payload.get("original_size_bytes")) is not int
        or payload["original_size_bytes"] != len(raw)
        or hashlib.sha256(raw).hexdigest() != payload["original_source_sha256"]
        or payload["source_type"] != expected_type
    ):
        raise ValueError("original serialized ROS source bytes differ")
    return raw


def closed_native_contact_window(
    path, policy, *, plugin_path, pose_frame, start_sim, end_sim, start_wall, end_wall
):
    """Reopen original DDS bytes and require complete zero-contact window brackets."""
    native_policy(policy)
    base = policy["contact_policy"]
    if type(pose_frame) is not str or not 0 < len(pose_frame) <= 256:
        raise ValueError("explicit independent world frame required")
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
    reopen_native_policy(path.parent, policy, plugin_path=plugin_path)
    summary_path = Path(str(path) + ".summary.json")
    if (
        path.is_symlink()
        or summary_path.is_symlink()
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
    tracker = NativeContactEvidence(policy)
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
                    row.get(k) != base[k]
                    for k in ("run_id", "body_snapshot_hash", "attachment_hash", "producer_id")
                )
                or row.get("source") != "independent_gazebo_contact_component_subscription"
                or row.get("evidence_domain") != "SIMULATION"
                or row.get("native_policy_hash") != digest(policy)
                or row.get("pose_frame") != pose_frame
            ):
                raise ValueError("closed independent contact source binding differs")
            captured = datetime.fromisoformat(row["captured_at"])
            if captured.tzinfo is None or (last_capture is not None and captured < last_capture):
                raise ValueError("original contact receipt clock regressed")
            last_capture = captured
            kind, payload = row["kind"], row["payload"]
            if kind == "native_contact_source_rejected":
                raise ValueError("native source rejection remains latched in closed evidence")
            if kind == "native_contact_startup_pose_pending":
                original_ros_bytes(payload, "gazebo_ecm_contact_sensor_data_json")
                continue
            if kind in {"native_contact_pose_received", "native_contact_source_received"}:
                received = payload["received_monotonic_sec"]
                if (
                    type(received) not in (int, float)
                    or not math.isfinite(received)
                    or not 0 <= row["wall_monotonic_sec"] - received < 0.3
                ):
                    raise ValueError("original source receipt differs from retained audit receipt")
            if kind == "native_contact_pose_received":
                raw = original_ros_bytes(payload, "tf2_msgs/msg/TFMessage")
                when, pose = decode_ros_pose(raw, base["model_name"], pose_frame)
                if row["sim_time_sec"] != when:
                    raise ValueError("retained pose clock differs from original ROS bytes")
                tracker.pose(when, payload["received_monotonic_sec"], pose)
            elif kind == "native_contact_source_received":
                raw = original_ros_bytes(payload, "gazebo_ecm_contact_sensor_data_json")
                unix = payload["received_unix_ns"]
                if (
                    type(unix) is not int
                    or not 0 <= int(captured.timestamp() * 1e9) - unix < 300_000_000
                ):
                    raise ValueError("native source original wall receipt differs from audit")
                metadata = tracker.observe(
                    raw, received_monotonic_sec=received, received_unix_ns=unix
                )
                if (
                    any(payload.get(k) != v for k, v in metadata.items())
                    or row["sim_time_sec"] != tracker.tracker.sim_time
                ):
                    raise ValueError(
                        "retained native projection differs from original source bytes"
                    )
            elif kind == "native_contact_observation":
                recorded = dict(payload)
                sampled = recorded.pop("snapshot_monotonic_sec")
                if (
                    type(sampled) not in (int, float)
                    or not math.isfinite(sampled)
                    or not 0 <= row["wall_monotonic_sec"] - sampled < 0.3
                ):
                    raise ValueError("retained native observation sample differs from receipt")
                actual = tracker.snapshot(sampled)
                if recorded != actual:
                    raise ValueError("retained native observation differs from source replay")
                if row["sim_time_sec"] is not None:
                    if row["sim_time_sec"] != actual["sim_time_sec"]:
                        raise ValueError("native observation SIM time differs from original source")
                    observations.append((row["sim_time_sec"], captured, actual))
                    if len(observations) > 50000:
                        raise ValueError("native observation count exceeds bounded duration")
            else:
                raise ValueError("unknown closed native contact event")
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
        "evidence_role": "closed_native_contact_cache_replay_not_backend_or_task_acceptance",
        "source": "independent_gazebo_contact_component_subscription",
        "decoder_source": "installed_tf2_msgs_TFMessage_CDR_and_original_native_JSON",
        "source_file_sha256": original_hash.hexdigest(),
        "summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
        "native_policy_hash": digest(policy),
        "backend_health_admitted": False,
        "sample_count": len(selected),
        "collision_count": 0,
        "observation_complete": True,
        "physical_acceptance": "NOT_VERIFIED",
        "start_sim": start_sim,
        "end_sim": end_sim,
    }
