"""Independent passive Gazebo pose stream for observer-process failure tests."""

import argparse
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path

from runtime_policy import load_frozen_sim_runtime_policy


def observer_binding(root):
    """Reopen generic source policy; never guess a profile or transport endpoint."""
    policy = load_frozen_sim_runtime_policy(root)
    if policy is not None:
        return {
            "model": policy["policy"]["body_model_name"],
            "topic": policy["policy"]["topics"]["independent_pose"],
            "run_id": policy["run_id"],
            "runtime_policy_hash": policy["artifact_hash"],
        }
    from profiles import PROFILES

    saved = json.loads((Path(root) / "fixture_profile.json").read_text())
    profile = PROFILES[saved["name"]]
    if saved != profile.to_dict():
        raise ValueError("independent observer profile differs from supported geometry")
    return {"model": profile.simulation_model, "topic": "/rosclaw_sim/ground_truth"}


def pose_sample(message, binding, *, captured_at=None):
    """Select exactly one actual model transform and reject malformed geometry."""
    matches = [t for t in message.transforms if t.child_frame_id == binding["model"]]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError("ambiguous independent actual model transform")
    transform = matches[0]
    p, q = transform.transform.translation, transform.transform.rotation
    values = (p.x, p.y, p.z, q.x, q.y, q.z, q.w)
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError("finite independent actual pose required")
    if abs(sum(v * v for v in (q.x, q.y, q.z, q.w)) - 1) > 1e-6:
        raise ValueError("normalized independent actual orientation required")
    stamp = transform.header.stamp
    if (
        type(stamp.sec) is not int
        or stamp.sec < 0
        or type(stamp.nanosec) is not int
        or not 0 <= stamp.nanosec < 1_000_000_000
    ):
        raise ValueError("actual bounded simulation timestamp required")
    return {
        "x": p.x,
        "y": p.y,
        "yaw": math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)),
        "captured_at": (captured_at or datetime.now(UTC)).isoformat(),
        "time_sec": stamp.sec + stamp.nanosec / 1e9,
        "source": "independent_gazebo_ground_truth_subscription",
        "model_name": binding["model"],
        "source_topic": binding["topic"],
        **{k: binding[k] for k in ("run_id", "runtime_policy_hash") if k in binding},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--duration", type=int, default=90)
    args = parser.parse_args()
    if not 60 <= args.duration <= 1920:
        parser.error("owned passive observation duration must be between 60 and 1920 seconds")
    binding = observer_binding(Path("/evidence"))
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from tf2_msgs.msg import TFMessage

    rclpy.init()
    node = Node("independent_stop_observer")
    with open(args.output, "w", buffering=1) as trace:

        def observe(message):
            sample = pose_sample(message, binding)
            if sample is not None:
                trace.write(json.dumps(sample) + "\n")

        node.create_subscription(
            TFMessage,
            binding["topic"],
            observe,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
        )
        until = time.monotonic() + args.duration
        try:
            while time.monotonic() < until:
                rclpy.spin_once(node, timeout_sec=0.1)
        finally:
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
