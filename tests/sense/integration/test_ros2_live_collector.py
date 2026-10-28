"""Trusted isolated ROS fixture only: no hardware, models, or robot actions.

Run in a private container/network namespace with ROS_DOMAIN_ID=32 and
RMW_IMPLEMENTATION=rmw_cyclonedds_cpp; ordinary Python CI skips without ROS.
"""

import os
import time

import pytest

rclpy = pytest.importorskip("rclpy")
from rclpy.context import Context  # noqa: E402
from rclpy.qos import qos_profile_sensor_data  # noqa: E402
from sensor_msgs.msg import BatteryState, Imu  # noqa: E402

from rosclaw.core.event_bus import EventBus  # noqa: E402
from rosclaw.sense.config import SenseConfig  # noqa: E402
from rosclaw.sense.runtime import SenseRuntime  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.deployment]


def test_actual_ros_telemetry_silence_replay_and_owned_context_restart():
    if os.environ.get("ROS_DOMAIN_ID") != "32":
        pytest.skip("Explicit isolated fixture domain32 required; never discover a live robot")
    context = Context()
    rclpy.init(context=context)
    node = None
    collector = None
    try:
        node = rclpy.create_node("trusted_sense_fixture", context=context)
        ns = "/isolated_sense_fixture"
        bpub = node.create_publisher(BatteryState, ns + "/battery", qos_profile_sensor_data)
        ipub = node.create_publisher(Imu, ns + "/imu", qos_profile_sensor_data)
        foreign = node.create_publisher(BatteryState, "/foreign/battery", qos_profile_sensor_data)
        runtime = SenseRuntime(
            SenseConfig(
                robot_id="fixture-only",
                collector="ros2",
                update_hz=0.0,
                publish_events=False,
                extra={
                    "ros2_topics": {
                        "battery": {"topic": ns + "/battery", "max_age_s": 0.5},
                        "imu": {"topic": ns + "/imu", "max_age_s": 0.5},
                    }
                },
            ),
            EventBus(),
        )
        collector = runtime._collector
        runtime.initialize()
        b = BatteryState()
        b.present = True
        b.percentage = 0.73
        b.voltage = 14.0
        b.current = -0.2
        b.power_supply_status = 2
        b.header.stamp = node.get_clock().now().to_msg()
        foreign.publish(b)
        time.sleep(0.15)
        assert collector.collect().energy.battery_percent is None
        deadline = time.monotonic() + 8
        while True:
            b.header.stamp = node.get_clock().now().to_msg()
            bpub.publish(b)
            imu = Imu()
            imu.header.stamp = node.get_clock().now().to_msg()
            imu.orientation.w = 1.0
            imu.orientation_covariance[0] = 0.01
            imu.angular_velocity.z = 0.2
            imu.angular_velocity_covariance[0] = 0.01
            ipub.publish(imu)
            sample = collector.collect()
            if sample.energy.battery_percent is not None and sample.imu.yaw_deg is not None:
                break
            assert time.monotonic() < deadline, "DDS telemetry not observed within fixture deadline"
            time.sleep(0.04)
        time.sleep(0.12)
        sample = collector.collect()
        assert sample.energy.battery_percent == pytest.approx(73.0, abs=1e-4)
        assert sample.imu.yaw_deg == pytest.approx(0.0, abs=1e-6)
        assert sample.imu.angular_velocity[2] == pytest.approx(0.2, abs=1e-6)
        original = sample.timestamp
        header = sample.raw["roles"]["battery"]["original_header_stamp_s"]
        assert header == pytest.approx(
            b.header.stamp.sec + b.header.stamp.nanosec / 1e9, abs=1e-6, rel=0
        )
        time.sleep(0.06)
        assert collector.collect().timestamp == original
        time.sleep(0.55)
        stale = collector.collect()
        assert stale.energy.battery_percent is None and stale.imu.yaw_deg is None
        assert stale.timestamp == original and all(
            not r["fresh"] for r in stale.raw["roles"].values()
        )
        for _ in range(3):
            bpub.publish(b)
            time.sleep(0.04)
        assert collector.collect().energy.battery_percent is None
        assert runtime.tick().overall_status != "ready"
        old = collector._context
        epoch = collector.collect().raw["observation_epoch"]
        collector.stop()
        assert (
            not old.ok()
            and collector._thread is None
            and collector.collect().energy.battery_percent is None
        )
        collector.start()
        assert collector.collect().energy.battery_percent is None
        assert collector.collect().raw["observation_epoch"] > epoch
        deadline = time.monotonic() + 8
        while True:
            b.header.stamp = node.get_clock().now().to_msg()
            bpub.publish(b)
            if collector.collect().energy.battery_percent is not None:
                break
            assert time.monotonic() < deadline, "DDS did not recover after new owned context"
            time.sleep(0.04)
        assert not sample.raw["body_identity_verified"] and not sample.raw["action_authority"]
    finally:
        if collector is not None:
            collector.stop()
        if node is not None:
            node.destroy_node()
        context.shutdown()
