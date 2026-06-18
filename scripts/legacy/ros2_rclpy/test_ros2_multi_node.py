#!/usr/bin/env python3
"""Multi-node concurrency and stress tests for ROS2 drivers.

Tests multiple ROS2Driver instances running simultaneously.
Verifies no cross-talk between instances and high-frequency message handling.
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


# ------------------------------------------------------------------
# Test framework
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def spin_nodes(nodes, iterations: int = 20, timeout: float = 0.05):
    for _ in range(iterations):
        for n in nodes:
            rclpy.spin_once(n, timeout_sec=timeout)
        time.sleep(0.02)


counter = 0


def next_name(base: str) -> str:
    global counter
    counter += 1
    return f"{base}_{counter}"


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

@test("Two drivers run simultaneously without interference")
def test_two_drivers_no_crosstalk():
    driver_a = ROS2Driver("bot_a", joint_dof=6, node_name=next_name("drv"))
    driver_b = ROS2Driver("bot_b", joint_dof=6, node_name=next_name("drv"))
    driver_a.initialize()
    driver_b.initialize()

    # Create separate publishers on same topic
    pub_a = Node(next_name("pub"))
    pub_b = Node(next_name("pub"))
    pub_a_pub = pub_a.create_publisher(JointState, "/joint_states", 10)
    pub_b_pub = pub_b.create_publisher(JointState, "/joint_states", 10)

    # Publish different positions
    msg_a = JointState()
    msg_a.name = [f"joint_{i}" for i in range(6)]
    msg_a.position = [0.1] * 6
    msg_b = JointState()
    msg_b.name = [f"joint_{i}" for i in range(6)]
    msg_b.position = [0.9] * 6

    pub_a_pub.publish(msg_a)
    spin_nodes([pub_a, driver_a._node, driver_b._node])

    # Both drivers should have received the same message (same topic)
    pos_a = driver_a.get_joint_positions()
    pos_b = driver_b.get_joint_positions()
    assert abs(pos_a[0] - 0.1) < 0.01
    assert abs(pos_b[0] - 0.1) < 0.01

    pub_b_pub.publish(msg_b)
    spin_nodes([pub_b, driver_a._node, driver_b._node])

    pos_a2 = driver_a.get_joint_positions()
    pos_b2 = driver_b.get_joint_positions()
    assert abs(pos_a2[0] - 0.9) < 0.01
    assert abs(pos_b2[0] - 0.9) < 0.01

    pub_a.destroy_node()
    pub_b.destroy_node()
    driver_a.stop()
    driver_b.stop()


@test("Multiple init/stop cycles are stable")
def test_multiple_init_stop_cycles():
    for i in range(5):
        driver = ROS2Driver(f"bot_{i}", joint_dof=6, node_name=next_name("drv"))
        driver.initialize()
        assert driver._node is not None
        assert driver.move_joints([0.1] * 6, duration=1.0) is True
        driver.stop()
        assert driver._node is None


@test("Driver publishes trajectory while receiving state")
def test_publish_and_subscribe_concurrent():
    driver = ROS2Driver("bot", joint_dof=6, node_name=next_name("drv"))
    driver.initialize()

    # Subscriber for trajectory
    traj_sub = Node(next_name("traj_sub"))
    received_traj = []
    traj_sub.create_subscription(
        JointTrajectory,
        "/joint_trajectory_controller/joint_trajectory",
        lambda msg: received_traj.append(msg),
        10,
    )

    # Publisher for joint state
    state_pub = Node(next_name("state_pub"))
    state_pub_pub = state_pub.create_publisher(JointState, "/joint_states", 10)

    # Publish state and command simultaneously
    msg = JointState()
    msg.name = [f"joint_{i}" for i in range(6)]
    msg.position = [0.5] * 6
    state_pub_pub.publish(msg)
    driver.move_joints([0.6] * 6, duration=1.0)

    spin_nodes([state_pub, traj_sub, driver._node], iterations=30)

    assert abs(driver.get_joint_positions()[0] - 0.5) < 0.01
    assert len(received_traj) >= 1

    traj_sub.destroy_node()
    state_pub.destroy_node()
    driver.stop()


@test("High-frequency JointState messages (100Hz burst)")
def test_high_frequency_messages():
    driver = ROS2Driver("bot", joint_dof=6, node_name=next_name("drv"))
    driver.initialize()

    pub = Node(next_name("hf_pub"))
    pub_pub = pub.create_publisher(JointState, "/joint_states", 100)

    # Burst of 50 messages at ~100Hz
    positions = []
    for i in range(50):
        msg = JointState()
        msg.name = [f"joint_{i}" for i in range(6)]
        p = i * 0.01
        msg.position = [p] * 6
        pub_pub.publish(msg)
        positions.append(p)
        rclpy.spin_once(pub, timeout_sec=0.005)
        rclpy.spin_once(driver._node, timeout_sec=0.005)
        time.sleep(0.01)  # ~100Hz

    # Final position should be close to last published
    final_pos = driver.get_joint_positions()
    assert abs(final_pos[0] - positions[-1]) < 0.02

    pub.destroy_node()
    driver.stop()


@test("Different DOF drivers coexist")
def test_different_dof_drivers():
    driver_6 = ROS2Driver("arm6", joint_dof=6, node_name=next_name("drv"))
    driver_3 = ROS2Driver("gripper3", joint_dof=3, node_name=next_name("drv"))
    driver_6.initialize()
    driver_3.initialize()

    assert driver_6.move_joints([0.1] * 6, duration=1.0) is True
    assert driver_3.move_joints([0.2] * 3, duration=1.0) is True

    # 6-DOF should reject 3 positions
    try:
        driver_6.move_joints([0.1] * 3, duration=1.0)
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # 3-DOF should reject 6 positions
    try:
        driver_3.move_joints([0.1] * 6, duration=1.0)
        assert False, "Expected ValueError"
    except ValueError:
        pass

    driver_6.stop()
    driver_3.stop()


@test("Driver state isolation between instances")
def test_driver_state_isolation():
    driver1 = ROS2Driver("bot1", joint_dof=6, node_name=next_name("drv"))
    driver2 = ROS2Driver("bot2", joint_dof=6, node_name=next_name("drv"))
    driver1.initialize()
    driver2.initialize()

    driver1.set_gripper(0.5)
    driver2.set_gripper(0.8)

    assert abs(driver1.get_state().gripper_state - 0.5) < 0.001
    assert abs(driver2.get_state().gripper_state - 0.8) < 0.001

    driver1.emergency_stop()
    assert driver1.get_state().error_code == 99
    assert driver2.get_state().error_code == 0  # Not affected

    driver1.stop()
    driver2.stop()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    if not rclpy.ok():
        rclpy.init(args=None)

    print("=" * 60)
    print("ROSClaw Multi-Node Concurrency and Stress Tests")
    print("=" * 60)

    test_two_drivers_no_crosstalk()
    test_multiple_init_stop_cycles()
    test_publish_and_subscribe_concurrent()
    test_high_frequency_messages()
    test_different_dof_drivers()
    test_driver_state_isolation()

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
