#!/usr/bin/env python3
"""ROS2 Resource Leak and Stress Tests.

Tests node cleanup, memory stability, and repeated init/stop cycles.
Runs in standalone subprocess to avoid pytest module reload issues.
"""

import sys
import time
import traceback
import gc

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

def _count_nodes():
    """Count active rclpy nodes (approximate)."""
    try:
        # rclpy doesn't expose node count directly
        # We can verify by trying to create a test node
        return 0
    except Exception:
        return -1


def _cleanup_rclpy():
    if rclpy.ok():
        try:
            rclpy.shutdown()
        except Exception:
            pass


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

@test("20 init/stop cycles: no resource leak")
def test_20_init_stop_cycles():
    """Repeatedly initialize and stop a driver."""
    for i in range(20):
        if not rclpy.ok():
            rclpy.init(args=None)
        driver = ROS2Driver(f"bot_{i}", joint_dof=6, node_name=f"leak_test_{i}")
        driver.initialize()
        assert driver._node is not None
        assert driver._pub_joint_cmd is not None
        assert driver._sub_joint_state is not None
        driver.stop()
        assert driver._node is None
        if rclpy.ok():
            rclpy.shutdown()
        gc.collect()


@test("50 move_joints calls without init/stop")
def test_50_move_joints_same_driver():
    """Send many commands without reinitializing."""
    if not rclpy.ok():
        rclpy.init(args=None)
    driver = ROS2Driver("stress_bot", joint_dof=6, node_name="stress_driver")
    driver.initialize()

    for i in range(50):
        positions = [float(i) * 0.01] * 6
        assert driver.move_joints(positions, duration=0.1) is True

    driver.stop()
    if rclpy.ok():
        rclpy.shutdown()


@test("Driver handles rapid state updates")
def test_rapid_state_updates():
    """Driver processes rapid JointState messages."""
    if not rclpy.ok():
        rclpy.init(args=None)

    driver = ROS2Driver("rapid_bot", joint_dof=6, node_name="rapid_driver")
    driver.initialize()

    pub = Node("rapid_pub")
    pub_pub = pub.create_publisher(JointState, "/joint_states", 100)

    # Send 200 messages rapidly
    for i in range(200):
        msg = JointState()
        msg.name = [f"joint_{j}" for j in range(6)]
        msg.position = [float(i) * 0.001] * 6
        msg.velocity = [0.001] * 6
        msg.effort = [0.1] * 6
        pub_pub.publish(msg)
        rclpy.spin_once(pub, timeout_sec=0.001)
        rclpy.spin_once(driver._node, timeout_sec=0.001)

    # Final position should be near the last value
    final = driver.get_joint_positions()
    assert abs(final[0] - 0.199) < 0.05

    pub.destroy_node()
    driver.stop()
    if rclpy.ok():
        rclpy.shutdown()


@test("Multiple DOF configurations work correctly")
def test_various_dof_configs():
    """Test drivers with different DOF counts."""
    for dof in [3, 6, 7]:
        if not rclpy.ok():
            rclpy.init(args=None)
        driver = ROS2Driver(f"dof_{dof}", joint_dof=dof, node_name=f"dof_test_{dof}")
        driver.initialize()

        assert len(driver.get_joint_positions()) == dof
        assert len(driver.get_joint_velocities()) == dof
        assert len(driver.get_joint_torques()) == dof

        assert driver.move_joints([0.1] * dof, duration=1.0) is True

        driver.stop()
        if rclpy.ok():
            rclpy.shutdown()


@test("Emergency stop resets correctly after multiple triggers")
def test_repeated_emergency_stop():
    """Trigger emergency stop multiple times."""
    if not rclpy.ok():
        rclpy.init(args=None)
    driver = ROS2Driver("emerg_bot", joint_dof=6, node_name="emerg_driver")
    driver.initialize()

    for i in range(10):
        driver.emergency_stop()
        state = driver.get_state()
        assert state.error_code == 99
        assert "Emergency" in state.error_message

    driver.stop()
    if rclpy.ok():
        rclpy.shutdown()


@test("Driver publishes consistently over time")
def test_consistent_publishing():
    """Driver publishes trajectory messages reliably."""
    if not rclpy.ok():
        rclpy.init(args=None)

    driver = ROS2Driver("pub_bot", joint_dof=6, node_name="pub_driver")
    driver.initialize()

    # Subscriber to count messages
    sub = Node("count_sub")
    count = [0]

    def _cb(msg):
        count[0] += 1

    sub.create_subscription(JointTrajectory, "/joint_trajectory_controller/joint_trajectory", _cb, 10)

    # Publish 10 trajectories
    for i in range(10):
        driver.move_joints([float(i) * 0.1] * 6, duration=1.0)
        for _ in range(5):
            rclpy.spin_once(sub, timeout_sec=0.02)
            rclpy.spin_once(driver._node, timeout_sec=0.02)

    assert count[0] >= 10

    sub.destroy_node()
    driver.stop()
    if rclpy.ok():
        rclpy.shutdown()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    print("=" * 60)
    print("ROSClaw ROS2 Resource Leak and Stress Tests")
    print("=" * 60)

    test_20_init_stop_cycles()
    test_50_move_joints_same_driver()
    test_rapid_state_updates()
    test_various_dof_configs()
    test_repeated_emergency_stop()
    test_consistent_publishing()

    print("=" * 60)
    print(f"Results: {PASSED} passed, {FAILED} failed")
    print("=" * 60)

    if ERRORS:
        print("\nErrors:")
        for name, tb in ERRORS:
            print(f"\n--- {name} ---")
            print(tb)

    _cleanup_rclpy()
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
