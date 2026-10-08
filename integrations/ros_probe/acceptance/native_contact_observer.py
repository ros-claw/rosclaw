"""Separate passive native component observer; no actuator or motion service.

Final acceptance requires replay of closed original bytes and a separately
admitted physics backend. This process only supplies source observations.
"""

import argparse
import base64
import hashlib
import json
import signal
import time
from pathlib import Path

from native_contact_evidence import NativeContactEvidence, reopen_native_policy

from rosclaw.connectors.ros.diagnosis.coverage_audit import CoverageAuditLog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--plugin", required=True, type=Path)
    parser.add_argument("--pose-frame", required=True)
    parser.add_argument("--duration", type=int, default=1920)
    args = parser.parse_args()
    if not 60 <= args.duration <= 1920 or not 0 < len(args.pose_frame) <= 256:
        raise ValueError("bounded immutable observer deadline and explicit world frame required")
    if (
        args.policy.is_symlink()
        or not args.policy.is_file()
        or not 0 < args.policy.stat().st_size <= 65536
        or not args.policy.resolve().is_relative_to(args.directory.resolve())
    ):
        raise ValueError("owned bounded regular frozen native contact policy required")
    policy_bytes = args.policy.read_bytes()
    policy = reopen_native_policy(args.directory, json.loads(policy_bytes), plugin_path=args.plugin)
    tracker = NativeContactEvidence(policy)
    import rclpy
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from rclpy.serialization import serialize_message
    from std_msgs.msg import String
    from tf2_msgs.msg import TFMessage

    rclpy.init()
    node = None
    try:
        node = Node(
            "independent_native_contact_observer",
            enable_rosout=False,
            start_parameter_services=False,
            enable_logger_service=False,
            use_global_arguments=False,
            parameter_overrides=[Parameter("start_type_description_service", value=False)],
        )
        # Standard rclpy initialization can emit parameter metadata. Retire its
        # built-in metadata publisher before observing; no actuator publisher or
        # external parameter/logger/type-description service is exposed.
        for publisher in tuple(node.publishers):
            if publisher.topic_name != "/parameter_events":
                raise ValueError("unexpected publisher on passive native observer")
            node.destroy_publisher(publisher)
        if tuple(node.services):
            raise ValueError("unexpected service on passive native observer")
        audit = CoverageAuditLog(
            args.directory / "native-contact-events.jsonl",
            context={
                **{
                    k: policy["contact_policy"][k]
                    for k in ("run_id", "body_snapshot_hash", "attachment_hash", "producer_id")
                },
                "source": "independent_gazebo_contact_component_subscription",
                "evidence_domain": "SIMULATION",
                "native_policy_hash": tracker.snapshot(time.monotonic())["native_policy_hash"],
                "pose_frame": args.pose_frame,
            },
        )
    except BaseException:
        try:
            if node is not None:
                node.destroy_node()
        finally:
            rclpy.shutdown()
        raise
    stop = [False]
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))

    def retained(raw):
        result = {
            "original_source_sha256": hashlib.sha256(raw).hexdigest(),
            "original_size_bytes": len(raw),
            "source_bytes_complete": len(raw) <= 262144,
        }
        if len(raw) <= 262144:
            result["original_source_base64"] = base64.b64encode(raw).decode("ascii")
        return result

    def reject(payload, kind, exc):
        tracker.tracker.fault = tracker.tracker.fault or str(exc)
        audit.emit(
            "native_contact_source_rejected",
            {**payload, "source_kind": kind, "error": str(exc)[:1024]},
        )

    def pose(message):
        payload = {}
        try:
            payload = {
                **retained(serialize_message(message)),
                "source_type": "tf2_msgs/msg/TFMessage",
            }
            if not payload["source_bytes_complete"]:
                raise ValueError("original independent pose exceeds source byte bound")
            candidates = [
                t
                for t in message.transforms
                if t.child_frame_id == policy["contact_policy"]["model_name"]
            ]
            if len(candidates) != 1 or candidates[0].header.frame_id != args.pose_frame:
                raise ValueError("one exact actual Body world frame required")
            transform = candidates[0]
            stamp = transform.header.stamp
            if stamp.sec < 0 or not 0 <= stamp.nanosec < 1_000_000_000:
                raise ValueError("original independent pose timestamp invalid")
            p, q = transform.transform.translation, transform.transform.rotation
            when, wall = stamp.sec + stamp.nanosec / 1e9, time.monotonic()
            tracker.pose(when, wall, [p.x, p.y, p.z, q.w, q.x, q.y, q.z])
            audit.emit(
                "native_contact_pose_received",
                {**payload, "received_monotonic_sec": wall},
                sim_time=when,
            )
        except ValueError as exc:
            reject(payload, "pose", exc)

    def components(message):
        payload = {}
        try:
            raw = message.data.encode("utf-8")
            received, unix = time.monotonic(), time.time_ns()
            payload = {
                **retained(raw),
                "source_type": "gazebo_ecm_contact_sensor_data_json",
                "received_monotonic_sec": received,
                "received_unix_ns": unix,
            }
            if tracker.world_pose is None:
                audit.emit("native_contact_startup_pose_pending", payload)
                return
            metadata = tracker.observe(raw, received_monotonic_sec=received, received_unix_ns=unix)
            audit.emit(
                "native_contact_source_received",
                {**payload, **metadata},
                sim_time=tracker.tracker.sim_time,
            )
        except (ValueError, UnicodeError) as exc:
            reject(payload, "components", exc)

    try:
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        node.create_subscription(TFMessage, policy["contact_policy"]["pose_topic"], pose, qos)
        node.create_subscription(String, policy["component_topic"], components, qos)
        deadline, next_tick, last_sim = time.monotonic() + args.duration, 0, None
        while not stop[0] and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.02)
            sampled = time.monotonic()
            if sampled < next_tick:
                continue
            if args.policy.read_bytes() != policy_bytes:
                raise ValueError("frozen native contact policy changed")
            result = tracker.snapshot(sampled)
            result["snapshot_monotonic_sec"] = sampled
            temporary = args.directory / "native-contact-latest.json.tmp"
            with temporary.open("x") as stream:
                stream.write(json.dumps(result) + "\n")
            temporary.replace(args.directory / "native-contact-latest.json")
            if not result["observation_complete"] or result["sim_time_sec"] != last_sim:
                audit.emit("native_contact_observation", result, sim_time=result["sim_time_sec"])
                last_sim = result["sim_time_sec"]
            next_tick = sampled + 0.05
    finally:
        try:
            reopen_native_policy(args.directory, policy, plugin_path=args.plugin)
        except ValueError as exc:
            reject({}, "prepared_source", exc)
        try:
            node.destroy_node()
        finally:
            try:
                rclpy.shutdown()
            finally:
                summary = audit.close()
                with Path(str(audit.path) + ".summary.json").open("x") as stream:
                    stream.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
