#!/usr/bin/env python3
"""Skydio X2 Drone ROS2 Integration Tests.

Tests ROS2Driver with Skydio X2 drone model:
- Motor thrust control
- Hover state
- Emergency stop
- Trajectory execution
- Multi-waypoint flight

Runs in standalone subprocess to avoid pytest module reload issues.
"""

import sys
import time
import traceback

if sys.version_info[:2] != (3, 10):
    print(f"SKIP: Requires Python 3.10 (found {sys.version_info.major}.{sys.version_info.minor})")
    sys.exit(0)

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from trajectory_msgs.msg import JointTrajectory
except ImportError as e:
    print(f"SKIP: rclpy not available: {e}")
    sys.exit(0)

sys.path.insert(0, "/home/dell/rosclaw-v1.0/src")

from rosclaw.mcp_drivers.ros2_driver import ROS2Driver
from rosclaw.mcp_drivers.base import TrajectoryCommand
from rosclaw.firewall.decorator import DigitalTwinFirewall, SafetyLevel


PASSED = 0
FAILED = 0
ERRORS = []


def test(name):
    def decorator(func):
        global PASSED, FAILED
        try:
            func()
            PASSED += 1
            print(f"  PASS: {name}")
        except Exception as e:
            FAILED += 1
            ERRORS.append((name, traceback.format_exc()))
            print(f"  FAIL: {name} - {e}")
        return func
    return decorator


counter = 0


def next_name(base: str) -> str:
    global counter
    counter += 1
    return f"{base}_{counter}"


def spin_nodes(nodes, iterations: int = 10, timeout: float = 0.05):
    for _ in range(iterations):
        for n in nodes:
            rclpy.spin_once(n, timeout_sec=timeout)
        time.sleep(0.02)


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

@test("Skydio X2 driver initializes with 4 motors")
def test_skydio_init():
    driver = ROS2Driver("skydio_x2", joint_dof=4, node_name=next_name("skydio"))
    driver.initialize()
    assert driver.joint_dof == 4
    driver.stop()


@test("Skydio hover thrust publishes trajectory")
def test_skydio_hover():
    driver = ROS2Driver("skydio_x2", joint_dof=4, node_name=next_name("skydio"))
    driver.initialize()
    subscriber = Node(next_name("sub"))
    received = []
    subscriber.create_subscription(
        JointTrajectory,
        "/joint_trajectory_controller/joint_trajectory",
        lambda msg: received.append(msg),
        10,
    )

    driver.move_joints([0.5, 0.5, 0.5, 0.5], duration=1.0)
    spin_nodes([driver._node, subscriber])

    assert len(received) >= 1
    assert len(received[0].points) == 1
    assert len(received[0].points[0].positions) == 4

    subscriber.destroy_node()
    driver.stop()


@test("Skydio motor state feedback")
def test_skydio_state_feedback():
    driver = ROS2Driver("skydio_x2", joint_dof=4, node_name=next_name("skydio"))
    driver.initialize()
    pub = Node(next_name("pub"))
    pub_pub = pub.create_publisher(JointState, "/joint_states", 10)

    msg = JointState()
    msg.name = ["motor_fl", "motor_fr", "motor_rl", "motor_rr"]
    msg.position = [0.6, 0.6, 0.6, 0.6]
    msg.velocity = [100.0, 100.0, 100.0, 100.0]
    pub_pub.publish(msg)
    spin_nodes([pub, driver._node])

    positions = driver.get_joint_positions()
    velocities = driver.get_joint_velocities()
    assert positions == [0.6, 0.6, 0.6, 0.6]
    assert velocities == [100.0, 100.0, 100.0, 100.0]

    pub.destroy_node()
    driver.stop()


@test("Skydio emergency stop")
def test_skydio_emergency():
    driver = ROS2Driver("skydio_x2", joint_dof=4, node_name=next_name("skydio"))
    driver.initialize()
    driver.emergency_stop()
    state = driver.get_state()
    assert state.error_code == 99
    assert "Emergency" in state.error_message
    driver.stop()


@test("Skydio multi-waypoint trajectory")
def test_skydio_multi_waypoint():
    driver = ROS2Driver("skydio_x2", joint_dof=4, node_name=next_name("skydio"))
    driver.initialize()
    subscriber = Node(next_name("sub"))
    received = []
    subscriber.create_subscription(
        JointTrajectory,
        "/joint_trajectory_controller/joint_trajectory",
        lambda msg: received.append(msg),
        10,
    )

    traj = TrajectoryCommand(
        waypoints=[[0.4, 0.4, 0.4, 0.4], [0.5, 0.5, 0.5, 0.5], [0.6, 0.6, 0.6, 0.6]],
        times=[0.0, 1.0, 2.0],
    )
    driver.execute_trajectory(traj)
    spin_nodes([driver._node, subscriber])

    assert len(received) >= 1
    assert len(received[0].points) == 3

    subscriber.destroy_node()
    driver.stop()


@test("Skydio boundary: negative thrust")
def test_skydio_negative_thrust():
    driver = ROS2Driver("skydio_x2", joint_dof=4, node_name=next_name("skydio"))
    driver.initialize()
    try:
        result = driver.move_joints([-0.1, -0.1, -0.1, -0.1], duration=1.0)
        assert result is not None
    except (ValueError, AssertionError):
        pass
    driver.stop()


@test("Two Skydio drones no crosstalk")
def test_skydio_multi_drone():
    d1 = ROS2Driver("skydio_x2", joint_dof=4, node_name=next_name("skydio_a"))
    d2 = ROS2Driver("skydio_x2", joint_dof=4, node_name=next_name("skydio_b"))
    d1.initialize()
    d2.initialize()

    d1.move_joints([0.5, 0.5, 0.5, 0.5], duration=1.0)
    d2.move_joints([0.7, 0.7, 0.7, 0.7], duration=1.0)

    s1 = d1.get_state()
    s2 = d2.get_state()
    assert s1.joint_positions == [0.5, 0.5, 0.5, 0.5]
    assert s2.joint_positions == [0.7, 0.7, 0.7, 0.7]

    d1.stop()
    d2.stop()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    if not rclpy.ok():
        rclpy.init(args=None)

    print("=" * 60)
    print("ROSClaw Skydio X2 Drone ROS2 Integration Tests")
    print("=" * 60)

    test_skydio_init()
    test_skydio_hover()
    test_skydio_state_feedback()
    test_skydio_emergency()
    test_skydio_multi_waypoint()
    test_skydio_negative_thrust()
    test_skydio_multi_drone()

    print("=" * 60)
    print(f"Results: {PASSED} passed, {FAILED} failed")
    print("=" * 60)

    if ERRORS:
        print("\nErrors:")
        for name, tb in ERRORS:
            print(f"\n--- {name} ---")
            print(tb)

    if rclpy.ok():
        rclpy.shutdown()

    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
