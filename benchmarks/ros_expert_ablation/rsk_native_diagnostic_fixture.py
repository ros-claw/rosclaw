"""Fixed isolated SDK role: frozen sensor endpoints and two read-only captures.

Run only by SensorOnlyDiagnosticOperator. No robot driver, Runtime, actuator
publisher, action server, parameter repair or QoS reset is present. Producer
stopping, when configured, completes before the observation window begins.
"""

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path


def main():
    if (
        not Path("/.dockerenv").is_file()
        or os.environ.get("ROSCLAW_RSK_FIXTURE") != "REGISTERED_SENSOR_ONLY_READONLY_V1"
        or os.environ.get("ROS_LOCALHOST_ONLY") != "1"
        or os.environ.get("RMW_IMPLEMENTATION") != "rmw_cyclonedds_cpp"
    ):
        raise RuntimeError("fixed isolated sensor-only SDK role required")
    run, source = os.environ["RSK_RUN_ID"], os.environ["RSK_SOURCE_COMMIT"]
    if re.fullmatch(r"[0-9a-f]{32}", run) is None or re.fullmatch(r"[0-9a-f]{40}", source) is None:
        raise ValueError("original run/source identity required")
    raw_config = Path("/fixture-config.json").read_bytes()
    config = json.loads(raw_config)
    if (
        set(config)
        != {
            "schema_version",
            "namespace",
            "topic",
            "offered_reliability",
            "requested_reliability",
            "producer_schedule",
        }
        or config["schema_version"] != "rosclaw.rsk_sensor_observation_config.v1"
        or re.fullmatch(r"/rsk_fixture_[a-z0-9]{8,32}", config["namespace"]) is None
        or not config["topic"].startswith(config["namespace"] + "/sensor/")
        or re.fullmatch(r"/rsk_fixture_[a-z0-9]{8,32}/sensor/[a-z0-9_]{1,32}", config["topic"])
        is None
        or config["offered_reliability"] not in {"RELIABLE", "BEST_EFFORT"}
        or config["requested_reliability"] not in {"RELIABLE", "BEST_EFFORT"}
        or config["producer_schedule"] not in {"CONTINUOUS", "STOP_BEFORE_WINDOW"}
    ):
        raise ValueError("closed frozen sensor observation config required")
    sys.path[:0] = ["/repo/src", "/repo/integrations/ros_probe/ros2"]
    import rclpy
    from probe import ReadOnlyProbe
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import LaserScan

    out = Path("/evidence")

    def save(name, value):
        with (out / name).open("x") as stream:
            json.dump(value, stream, indent=2)

    rclpy.init(args=[])
    emitter = Node("emitter", namespace=config["namespace"])
    receiver = Node("receiver", namespace=config["namespace"])
    probe = ReadOnlyProbe(snapshot_period_sec=0.1)
    reliability = {
        "RELIABLE": ReliabilityPolicy.RELIABLE,
        "BEST_EFFORT": ReliabilityPolicy.BEST_EFFORT,
    }
    publisher = emitter.create_publisher(
        LaserScan,
        config["topic"],
        QoSProfile(depth=10, reliability=reliability[config["offered_reliability"]]),
    )
    received = [0]
    last = [None]

    def observe(message):
        received[0] += 1
        last[0] = time.monotonic()

    receiver.create_subscription(
        LaserScan,
        config["topic"],
        observe,
        QoSProfile(depth=10, reliability=reliability[config["requested_reliability"]]),
    )
    executor = SingleThreadedExecutor()
    for node in [emitter, receiver, probe]:
        executor.add_node(node)
    began = time.monotonic()
    next_send = began
    sent = 0
    before = None
    window_started = None

    def consumer():
        return {
            "count": received[0],
            "sent": sent,
            "last_receive_age_ms": None if last[0] is None else (time.monotonic() - last[0]) * 1000,
        }

    try:
        while time.monotonic() - began < 15:
            steady = time.monotonic()
            if steady >= next_send:
                next_send = steady + 0.05
                if config["producer_schedule"] == "CONTINUOUS" or steady - began < 2.0:
                    message = LaserScan()
                    message.header.frame_id = "fixture_sensor"
                    message.header.stamp = emitter.get_clock().now().to_msg()
                    message.angle_min = -0.1
                    message.angle_max = 0.1
                    message.angle_increment = 0.1
                    message.range_min = 0.05
                    message.range_max = 10.0
                    message.ranges = [1.0, 1.0, 1.0]
                    publisher.publish(message)
                    sent += 1
            executor.spin_once(timeout_sec=0.005)
            if before is None and time.monotonic() - began >= 3:
                snapshot_start = time.monotonic()
                before = probe.snapshot()
                before_call_sec = time.monotonic() - snapshot_start
                save("native-before.json", before)
                before_consumer = consumer()
                window_started = time.monotonic()
            if before is not None and time.monotonic() - window_started >= 5.2:
                snapshot_start = time.monotonic()
                after = probe.snapshot()
                after_call_sec = time.monotonic() - snapshot_start
                save("native-after.json", after)
                after_consumer = consumer()
                save(
                    "owner-observations.json",
                    {
                        "schema_version": "rosclaw.rsk_sensor_observation_owner.v1",
                        "run_id": run,
                        "source": source,
                        "execution_config_sha256": hashlib.sha256(raw_config).hexdigest(),
                        "native_before_sha256": hashlib.sha256(
                            (out / "native-before.json").read_bytes()
                        ).hexdigest(),
                        "native_after_sha256": hashlib.sha256(
                            (out / "native-after.json").read_bytes()
                        ).hexdigest(),
                        "publisher_policy_changed_during_window": False,
                        "native_snapshot_call_durations_sec": [before_call_sec, after_call_sec],
                        "observation_steady_sec": time.monotonic() - window_started,
                        "consumer_before": before_consumer,
                        "consumer_after": after_consumer,
                        "robot_driver_processes": [],
                        "navigation_goals": 0,
                        "control_commands": 0,
                        "body_admission": "NOT_RUN",
                        "physics": "NOT_RUN",
                        "source_file_sha256": {
                            name: hashlib.sha256((Path("/repo") / name).read_bytes()).hexdigest()
                            for name in [
                                "integrations/ros_probe/ros2/probe.py",
                                "benchmarks/ros_expert_ablation/rsk_native_diagnostic_fixture.py",
                            ]
                        },
                    },
                )
                break
        else:
            raise TimeoutError("bounded sensor-only observation fixture did not finish")
    finally:
        executor.shutdown()
        for node in [probe, receiver, emitter]:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
