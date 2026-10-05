"""Native ROS1 probe acceptance against real ROS1 synthetic publishers.

This tests graph/subscription/action discovery, not robot motion or coverage.
"""

import argparse
import json
import time
from pathlib import Path

import actionlib
import rospy
from actionlib.msg import TestAction, TestResult
from probe import ReadOnlyProbe
from sensor_msgs.msg import LaserScan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rospy.init_node("ros1_probe_acceptance")
    publisher = rospy.Publisher("/fixture/scan", LaserScan, queue_size=10)
    action = actionlib.SimpleActionServer("/fixture/test", TestAction, auto_start=False)
    action.register_goal_callback(
        lambda: (action.accept_new_goal(), action.set_succeeded(TestResult()))
    )
    action.start()
    probe = ReadOnlyProbe()
    try:
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            scan = LaserScan()
            scan.header.stamp = rospy.Time.now()
            scan.header.frame_id = "laser"
            scan.range_min, scan.range_max, scan.ranges = 0.1, 10, [1.0] * 10
            publisher.publish(scan)
            probe.snapshot()
            time.sleep(0.05)
        snapshot = probe.snapshot()
        signal = next(s for s in snapshot["signals"] if s["topic"] == "/fixture/scan")
        assert 10 <= signal["rate_hz"] <= 30, signal
        assert signal["last_message_age_ms"] < 200, signal
        assert any(a["action_type"] == "actionlib/TestAction" for a in snapshot["graph"]["actions"])
        assert snapshot["qos"]["supported"] is False
        assert snapshot["completeness"]["lifecycle"] is False
        args.output.write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "evidence_domain": "NATIVE_ROS1_SYNTHETIC_PUBLISHERS",
                    "hardware_verified": False,
                    "snapshot": snapshot,
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        publisher.unregister()
        probe.close()


if __name__ == "__main__":
    main()
