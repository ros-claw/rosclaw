"""Independent passive contact process for owned dynamic-fault SIM acceptance.

No publishers, services, controller leases or robot commands. A prepared frozen
source policy and fresh messages from every declared stream are required.
"""

import argparse
import base64
import hashlib
import json
import math
import signal
import time
from datetime import UTC, datetime
from pathlib import Path

from contact_evidence import IndependentContacts, reopen_contact_policy

from rosclaw.connectors.ros.diagnosis.coverage_audit import CoverageAuditLog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("/evidence"))
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--duration", type=int, default=1920)
    args = parser.parse_args()
    if not 60 <= args.duration <= 1920 or not 0 < args.policy.stat().st_size <= 65536:
        raise ValueError("bounded immutable source policy and observer deadline required")
    if args.policy.is_symlink() or not args.policy.resolve().is_relative_to(
        args.directory.resolve()
    ):
        raise ValueError("owned regular frozen contact policy required")
    policy_bytes = args.policy.read_bytes()
    policy = reopen_contact_policy(args.directory, json.loads(policy_bytes))
    tracker = IndependentContacts(policy)
    # Imports are intentionally inside the fixture process, never the agent.
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from rclpy.serialization import serialize_message
    from ros_gz_interfaces.msg import Contacts
    from tf2_msgs.msg import TFMessage

    rclpy.init()
    node = Node("independent_contact_observer")
    audit = CoverageAuditLog(
        args.directory / "independent-contact-events.jsonl",
        context={
            **{
                k: policy[k]
                for k in ("run_id", "body_snapshot_hash", "attachment_hash", "producer_id")
            },
            "source": "independent_gazebo_contact_subscription",
            "evidence_domain": "SIMULATION",
            "contact_policy_hash": tracker.policy_hash,
        },
    )
    stop = [False]
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))

    def stamp(message):
        sec, ns = message.header.stamp.sec, message.header.stamp.nanosec
        if type(sec) is not int or type(ns) is not int or sec < 0 or not 0 <= ns < 1_000_000_000:
            raise ValueError("actual typed independent source stamp required")
        return sec + ns / 1e9

    def source_bytes(message, source_type):
        raw = serialize_message(message)
        payload = {
            "serialized_message_sha256": hashlib.sha256(raw).hexdigest(),
            "original_size_bytes": len(raw),
            "source_type": source_type,
            "source_bytes_complete": len(raw) <= 262144,
        }
        if len(raw) <= 262144:
            payload["serialized_message_base64"] = base64.b64encode(raw).decode("ascii")
        return payload

    def observe_pose(message):
        payload = {}
        try:
            payload = source_bytes(message, "tf2_msgs/msg/TFMessage")
            if not payload["source_bytes_complete"]:
                raise ValueError("actual pose message exceeds bounded original-byte budget")
            matches = [t for t in message.transforms if t.child_frame_id == policy["model_name"]]
            if len(matches) != 1:
                raise ValueError("one actual independent Body pose source required")
            item = matches[0]
            p, q = item.transform.translation, item.transform.rotation
            if (
                not all(math.isfinite(v) for v in (p.x, p.y, p.z, q.x, q.y, q.z, q.w))
                or abs(sum(v * v for v in (q.x, q.y, q.z, q.w)) - 1) > 1e-6
            ):
                raise ValueError("finite normalized actual Body pose required")
            when, received = stamp(item), time.monotonic()
            tracker.pose(when, received)
            payload["received_monotonic_sec"] = received
            audit.emit("independent_contact_pose_received", payload, sim_time=when)
        except ValueError as exc:
            tracker.fault = tracker.fault or str(exc)
            audit.emit(
                "independent_contact_source_rejected",
                {**payload, "error": str(exc), "source_kind": "pose"},
            )

    def observe_contacts(topic, message):
        payload = {}
        try:
            payload = {**source_bytes(message, "ros_gz_interfaces/msg/Contacts"), "topic": topic}
            if not payload["source_bytes_complete"]:
                raise ValueError("actual contact message exceeds bounded original-byte budget")
            if tracker.sim_time is None:
                audit.emit("independent_contact_startup_clock_pending", payload)
                return
            when = stamp(message)
            pairs = [[c.collision1.name, c.collision2.name] for c in message.contacts]
            received = time.monotonic()
            tracker.contacts(topic, when, received, pairs)
            payload["received_monotonic_sec"] = received
            audit.emit(
                "independent_contact_source_received",
                {**payload, "collisions": pairs},
                sim_time=when,
            )
        except ValueError as exc:
            tracker.fault = tracker.fault or str(exc)
            audit.emit(
                "independent_contact_source_rejected",
                {**payload, "error": str(exc), "source_kind": "contacts", "topic": topic},
            )

    try:
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        node.create_subscription(TFMessage, policy["pose_topic"], observe_pose, qos)
        for topic in policy["contacts"]:
            node.create_subscription(
                Contacts, topic, lambda message, t=topic: observe_contacts(t, message), qos
            )
        until = time.monotonic() + args.duration
        next_snapshot = 0
        last_snapshot_sim = None
        while not stop[0] and time.monotonic() < until:
            rclpy.spin_once(node, timeout_sec=0.02)
            if time.monotonic() >= next_snapshot:
                if args.policy.read_bytes() != policy_bytes:
                    raise ValueError("frozen independent contact policy changed")
                sampled = time.monotonic()
                result = tracker.snapshot(sampled)
                result["snapshot_monotonic_sec"] = sampled
                result["captured_at"] = datetime.now(UTC).isoformat()
                temporary = args.directory / "independent-contact-latest.json.tmp"
                with temporary.open("x") as stream:
                    stream.write(json.dumps(result) + "\n")
                temporary.replace(args.directory / "independent-contact-latest.json")
                if not result["observation_complete"] or tracker.sim_time != last_snapshot_sim:
                    audit.emit("independent_contact_observation", result, sim_time=tracker.sim_time)
                    last_snapshot_sim = tracker.sim_time
                next_snapshot = time.monotonic() + 0.05
    finally:
        try:
            reopen_contact_policy(args.directory, policy)
        except ValueError as exc:
            audit.emit(
                "independent_contact_source_rejected",
                {"error": str(exc), "source_kind": "prepared_source"},
            )
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
