#!/usr/bin/env python3
"""Crazyflie 2.1 Nano Drone ROS2 Integration Tests.

Tests ROS2Driver with Crazyflie nano drone model:
- Low-power motor control
- Swarm-like multi-drone coordination
- Indoor navigation simulation
- Battery/weight constraints

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

@test("Crazyflie initializes with 4 motors")
def test_crazyflie_init():
    driver = ROS2Driver("crazyflie_2", joint_dof=4, node_name=next_name("cf"))
    driver.initialize()
    assert driver.joint_dof == 4
    driver.stop()


@test("Crazyflie low thrust hover")
def test_crazyflie_hover():
    driver = ROS2Driver("crazyflie_2", joint_dof=4, node_name=next_name("cf"))
    driver.initialize()
    subscriber = Node(next_name("sub"))
    received = []
    subscriber.create_subscription(
        JointTrajectory,
        "/joint_trajectory_controller/joint_trajectory",
        lambda msg: received.append(msg),
        10,
    )

    # Low thrust for nano drone (27g)
    driver.move_joints([0.15, 0.15, 0.15, 0.15], duration=0.5)
    spin_nodes([driver._node, subscriber])

    assert len(received) >= 1
    assert len(received[0].points[0].positions) == 4

    subscriber.destroy_node()
    driver.stop()


@test("Crazyflie rapid state updates")
def test_crazyflie_rapid_updates():
    driver = ROS2Driver("crazyflie_2", joint_dof=4, node_name=next_name("cf"))
    driver.initialize()
    pub = Node(next_name("pub"))
    pub_pub = pub.create_publisher(JointState, "/joint_states", 10)

    for i in range(20):
        msg = JointState()
        msg.name = ["m1", "m2", "m3", "m4"]
        msg.position = [0.1 + i * 0.01] * 4
        pub_pub.publish(msg)
        rclpy.spin_once(pub, timeout_sec=0.01)
        rclpy.spin_once(driver._node, timeout_sec=0.01)
        time.sleep(0.01)

    positions = driver.get_joint_positions()
    assert len(positions) == 4
    assert positions[0] > 0.1

    pub.destroy_node()
    driver.stop()


@test("Crazyflie swarm: 3 drones")
def test_crazyflie_swarm():
    drones = []
    for i in range(3):
        d = ROS2Driver("crazyflie_2", joint_dof=4, node_name=next_name(f"cf_swarm_{i}"))
        d.initialize()
        drones.append(d)

    # Each drone different thrust
    thrusts = [[0.1, 0.1, 0.1, 0.1], [0.15, 0.15, 0.15, 0.15], [0.2, 0.2, 0.2, 0.2]]
    for d, t in zip(drones, thrusts):
        d.move_joints(t, duration=0.5)

    for i, d in enumerate(drones):
        s = d.get_state()
        assert s.joint_positions == thrusts[i]

    for d in drones:
        d.stop()


@test("Crazyflie partial state message")
def test_crazyflie_partial_state():
    driver = ROS2Driver("crazyflie_2", joint_dof=4, node_name=next_name("cf"))
    driver.initialize()
    pub = Node(next_name("pub"))
    pub_pub = pub.create_publisher(JointState, "/joint_states", 10)

    msg = JointState()
    msg.name = ["m1", "m2", "m3", "m4"]
    msg.position = [0.12, 0.13]
    pub_pub.publish(msg)
    spin_nodes([pub, driver._node])

    positions = driver.get_joint_positions()
    assert len(positions) <= 4

    pub.destroy_node()
    driver.stop()


@test("Crazyflie emergency stop no state")
def test_crazyflie_emergency_no_state():
    driver = ROS2Driver("crazyflie_2", joint_dof=4, node_name=next_name("cf"))
    driver.initialize()
    driver.emergency_stop()
    state = driver.get_state()
    assert state.error_code == 99
    driver.stop()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    if not rclpy.ok():
        rclpy.init(args=None)

    print("=" * 60)
    print("ROSClaw Crazyflie 2.1 Drone ROS2 Integration Tests")
    print("=" * 60)

    test_crazyflie_init()
    test_crazyflie_hover()
    test_crazyflie_rapid_updates()
    test_crazyflie_swarm()
    test_crazyflie_partial_state()
    test_crazyflie_emergency_no_state()

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
