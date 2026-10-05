"""Verify real clock-pause diagnosis from the native read-only fixture probe.

Only an inactive owned Gazebo world is paused/resumed. No motor command or
robot configuration write is sent; every mutation requires world-service ACK.
"""

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

from faults import observed, service


def main():
    import rclpy
    from rclpy.parameter import Parameter
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from std_msgs.msg import String

    if os.environ.get("ROS_DOMAIN_ID") != "173":
        raise ValueError("the owned Golden fixture domain 173 is required")
    initial = observed()
    if initial["cleaning_enabled"] or initial["lease_remaining_sec"] > 0:
        raise RuntimeError("pause diagnosis requires an inactive fixture lease")
    rclpy.init()
    node = rclpy.create_node(
        "rosclaw_probe_pause_acceptance",
        parameter_overrides=[Parameter("use_sim_time", value=True)],
    )
    latest = []
    node.create_subscription(
        String,
        "/rosclaw_probe/snapshot",
        lambda m: latest.append(json.loads(m.data)),
        QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
    )

    def wait(after, advancing, timeout=12):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if latest:
                snapshot = latest[-1]
                stamp = datetime.fromisoformat(snapshot["captured_at"])
                if stamp >= after and snapshot["observations"]["clock_advancing"] is advancing:
                    return snapshot
        raise TimeoutError("native clock observation did not reflect the real world transition")

    output = Path("/evidence/probe-pause-acceptance.json")
    record = {"status": "FAIL", "evidence_domain": "GAZEBO_NATIVE_DDS", "motor_commands": 0}
    paused = False
    try:
        record["before"] = wait(datetime.now(UTC), True)
        record["pause_ack"] = service("control", "gz.msgs.WorldControl", "pause: true")
        paused = True
        after = datetime.now(UTC)
        # Require a new wall-time envelope after the historical clock window
        # has expired; a retained pre-pause sample cannot satisfy this gate.
        record["paused"] = wait(after, False)
        if datetime.fromisoformat(record["paused"]["captured_at"]) <= after:
            raise RuntimeError("pause envelope was not newly captured")
        record["resume_ack"] = service("control", "gz.msgs.WorldControl", "pause: false")
        paused = False
        record["resumed"] = wait(datetime.now(UTC), True)
        record["status"] = "PASS"
    except Exception as exc:
        record["error"] = str(exc)
        raise
    finally:
        try:
            if paused:
                record["cleanup_resume_ack"] = service(
                    "control", "gz.msgs.WorldControl", "pause: false"
                )
        finally:
            node.destroy_node()
            rclpy.shutdown()
            output.write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": record["status"],
                "clock_advancing": [
                    record[k]["observations"]["clock_advancing"]
                    for k in ["before", "paused", "resumed"]
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
