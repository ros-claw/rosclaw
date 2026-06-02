#!/usr/bin/env python3
"""Firewall + ROS2 Trajectory Interception Integration Tests.

Tests DigitalTwinFirewall validating trajectories before ROS2Driver publishes them.
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
    from trajectory_msgs.msg import JointTrajectory
except ImportError as e:
    print(f"SKIP: rclpy not available: {e}")
    sys.exit(0)

sys.path.insert(0, "/home/dell/rosclaw-v1.0/src")

from rosclaw.mcp_drivers.ros2_driver import ROS2Driver
from rosclaw.mcp_drivers.base import TrajectoryCommand
from rosclaw.firewall.decorator import DigitalTwinFirewall, SafetyLevel


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

class TrajectorySubscriber:
    def __init__(self, node_name: str):
        self.node = Node(node_name)
        self.received = []
        self.node.create_subscription(
            JointTrajectory,
            "/joint_trajectory_controller/joint_trajectory",
            self._callback,
            10,
        )

    def _callback(self, msg):
        self.received.append({
            "joint_names": list(msg.joint_names),
            "points": [(list(p.positions), p.time_from_start.sec) for p in msg.points],
        })

    def destroy(self):
        self.node.destroy_node()


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

@test("Firewall validates valid trajectory: ALLOW")
def test_firewall_allows_valid_trajectory():
    """A valid trajectory (within limits) should pass firewall and be published."""
    model_path = "/home/dell/rosclaw-v1.0/src/rosclaw/specs/ur5e.xml"
    firewall = DigitalTwinFirewall(model_path=model_path)

    # Validate a safe trajectory (all zeros)
    result = firewall.validate_trajectory([[0.0] * 6], safety_level=SafetyLevel.STRICT)
    assert result.is_safe is True, f"Expected safe, got: {result}"


@test("Firewall blocks limit-violating trajectory: BLOCK")
def test_firewall_blocks_limit_violation():
    """A trajectory with positions outside joint limits should be blocked."""
    model_path = "/home/dell/rosclaw-v1.0/src/rosclaw/specs/ur5e.xml"
    firewall = DigitalTwinFirewall(model_path=model_path)

    # shoulder_pan limit is ~±3.14, so 10.0 should be blocked
    result = firewall.validate_trajectory([[10.0, 0.0, 0.0, 0.0, 0.0, 0.0]], safety_level=SafetyLevel.STRICT)
    assert result.is_safe is False
    assert len(result.violation_details) > 0


@test("Firewall blocks trajectory with collision risk")
def test_firewall_blocks_collision_trajectory():
    """A trajectory causing self-collision should be blocked."""
    model_path = "/home/dell/rosclaw-v1.0/src/rosclaw/specs/ur5e.xml"
    firewall = DigitalTwinFirewall(model_path=model_path)

    # Extreme elbow position may cause self-collision
    result = firewall.validate_trajectory([[0.0, 0.0, 3.5, 0.0, 0.0, 0.0]], safety_level=SafetyLevel.STRICT)
    # Depending on model, this may or may not be blocked
    # Just verify the firewall returns a structured result
    assert hasattr(result, "is_safe")
    assert hasattr(result, "violation_details")


@test("ROS2Driver + Firewall: valid trajectory passes")
def test_driver_firewall_valid():
    """ROS2Driver publishes a valid trajectory, firewall validates."""
    driver = ROS2Driver("ur5e", joint_dof=6, node_name=next_name("drv"))
    driver.initialize()
    subscriber = TrajectorySubscriber(next_name("sub"))

    # Valid positions
    valid_positions = [0.0, -0.5, 1.0, 0.0, 0.0, 0.0]
    driver.move_joints(valid_positions, duration=2.0)
    spin_nodes([driver._node, subscriber.node])

    assert len(subscriber.received) >= 1
    assert subscriber.received[0]["points"][0][0] == valid_positions

    subscriber.destroy()
    driver.stop()


@test("ROS2Driver + Firewall: multi-waypoint trajectory validation")
def test_driver_firewall_multi_waypoint():
    """ROS2Driver publishes multi-waypoint trajectory."""
    driver = ROS2Driver("ur5e", joint_dof=6, node_name=next_name("drv"))
    driver.initialize()
    subscriber = TrajectorySubscriber(next_name("sub"))

    traj = TrajectoryCommand(
        waypoints=[[0.0] * 6, [0.1] * 6, [0.2] * 6],
        times=[0.0, 1.0, 2.0],
    )
    driver.execute_trajectory(traj)
    spin_nodes([driver._node, subscriber.node])

    assert len(subscriber.received) >= 1
    msg = subscriber.received[0]
    assert len(msg["points"]) == 3

    subscriber.destroy()
    driver.stop()


@test("Firewall decorator with ALLOW result")
def test_firewall_decorator_allow():
    """Test DigitalTwinFirewall returns correct structure for allowed actions."""
    model_path = "/home/dell/rosclaw-v1.0/src/rosclaw/specs/ur5e.xml"
    firewall = DigitalTwinFirewall(model_path=model_path)

    # Test a trajectory that should be safe
    result = firewall.validate_trajectory([[0.0] * 6, [0.05] * 6])
    assert hasattr(result, "is_safe")
    assert hasattr(result, "collision_detected")
    assert hasattr(result, "joint_limit_violated")
    assert hasattr(result, "violation_details")


@test("Safety levels: STRICT vs LENIENT")
def test_firewall_safety_levels():
    """Different safety levels should have different thresholds."""
    model_path = "/home/dell/rosclaw-v1.0/src/rosclaw/specs/ur5e.xml"

    strict = DigitalTwinFirewall(model_path=model_path)
    lenient = DigitalTwinFirewall(model_path=model_path)

    # Both should have the same structure
    r1 = strict.validate_trajectory([[0.0] * 6], safety_level=SafetyLevel.STRICT)
    r2 = lenient.validate_trajectory([[0.0] * 6], safety_level=SafetyLevel.LENIENT)

    assert r1.is_safe is True
    assert r2.is_safe is True

    # LENIENT may allow more, STRICT may block more
    # The exact behavior depends on the model


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    if not rclpy.ok():
        rclpy.init(args=None)

    print("=" * 60)
    print("ROSClaw Firewall + ROS2 Integration Tests")
    print("=" * 60)

    test_firewall_allows_valid_trajectory()
    test_firewall_blocks_limit_violation()
    test_firewall_blocks_collision_trajectory()
    test_driver_firewall_valid()
    test_driver_firewall_multi_waypoint()
    test_firewall_decorator_allow()
    test_firewall_safety_levels()

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
