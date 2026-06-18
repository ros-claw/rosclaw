#!/usr/bin/env python3
"""Runtime + ROS2Driver Closed-Loop Integration Tests.

Tests the full pipeline: Runtime → EventBus → ROS2Driver → ROS2 Topic.
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

from rosclaw.core.runtime import Runtime, RuntimeConfig
from rosclaw.core.event_bus import EventBus, Event, EventPriority
from rosclaw.mcp_drivers.ros2_driver import ROS2Driver
from rosclaw.mcp_drivers.base import TrajectoryCommand


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
# Helper classes
# ------------------------------------------------------------------

class JointStatePublisher:
    def __init__(self, node_name: str):
        self.node = Node(node_name)
        self.pub = self.node.create_publisher(JointState, "/joint_states", 10)
        self.timer = self.node.create_timer(0.05, self._publish)
        self.positions = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]
        self.velocities = [0.01] * 6
        self.efforts = [1.0] * 6

    def _publish(self):
        msg = JointState()
        msg.name = [f"joint_{i}" for i in range(6)]
        msg.position = self.positions
        msg.velocity = self.velocities
        msg.effort = self.efforts
        self.pub.publish(msg)

    def destroy(self):
        self.node.destroy_node()


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

@test("Runtime registers ROS2Driver and get_driver returns it")
def test_runtime_register_driver():
    config = RuntimeConfig(
        robot_id="test_bot",
        enable_firewall=False,
        enable_memory=False,
        enable_practice=False,
        enable_how=False,
        enable_provider=False,
    )
    runtime = Runtime(config)
    driver = ROS2Driver("test_bot", joint_dof=6, node_name=next_name("driver"))
    driver.initialize()

    runtime.register_driver("ros2", driver)
    retrieved = runtime.get_driver("ros2")
    assert retrieved is driver
    assert "ros2" in runtime.status["drivers"]

    driver.stop()


@test("Runtime emergency_stop propagates to ROS2Driver")
def test_runtime_emergency_stop():
    config = RuntimeConfig(
        robot_id="test_bot",
        enable_firewall=False,
        enable_memory=False,
        enable_practice=False,
        enable_how=False,
        enable_provider=False,
    )
    runtime = Runtime(config)
    runtime.initialize()  # Required to set up EventBus subscriptions
    driver = ROS2Driver("test_bot", joint_dof=6, node_name=next_name("driver"))
    driver.initialize()
    runtime.register_driver("ros2", driver)

    # Publish some joint states first
    robot = JointStatePublisher(next_name("robot"))
    spin_nodes([robot.node, driver._node], iterations=10)

    # Trigger emergency stop via EventBus
    runtime.event_bus.publish(Event(
        topic="robot.emergency_stop",
        payload={"reason": "test"},
        source="test",
    ))
    # Give EventBus time to process (async handlers)
    time.sleep(0.2)

    state = driver.get_state()
    assert state.error_code == 99
    assert "Emergency" in state.error_message

    robot.destroy()
    driver.stop()


@test("Runtime + ROS2Driver full loop: command -> topic -> subscriber")
def test_runtime_driver_full_loop():
    config = RuntimeConfig(
        robot_id="ur5e",
        enable_firewall=False,
        enable_memory=False,
        enable_practice=False,
        enable_how=False,
        enable_provider=False,
    )
    runtime = Runtime(config)
    driver = ROS2Driver("ur5e", joint_dof=6, node_name=next_name("driver"))
    driver.initialize()
    runtime.register_driver("ros2", driver)

    # Robot publishes current state
    robot = JointStatePublisher(next_name("robot"))
    robot.positions = [0.0, -1.57, 1.57, 0.0, 0.0, 0.0]
    spin_nodes([robot.node, driver._node])

    # Verify driver reads state
    current = driver.get_joint_positions()
    assert abs(current[0]) < 0.01

    # Controller subscribes to trajectory
    controller = TrajectorySubscriber(next_name("controller"))

    # Send move command via driver (simulating Runtime -> driver flow)
    target = [0.1, -1.47, 1.47, 0.1, 0.0, 0.0]
    driver.move_joints(target, duration=2.0)
    spin_nodes([driver._node, controller.node])

    # Verify trajectory was published
    assert len(controller.received) >= 1
    traj = controller.received[0]
    assert traj["points"][0][0] == target

    # Simulate robot reaching target
    robot.positions = target
    spin_nodes([robot.node, driver._node])
    updated = driver.get_joint_positions()
    assert abs(updated[0] - 0.1) < 0.01

    robot.destroy()
    controller.destroy()
    driver.stop()


@test("Runtime + ROS2Driver trajectory execution")
def test_runtime_driver_trajectory():
    config = RuntimeConfig(
        robot_id="ur5e",
        enable_firewall=False,
        enable_memory=False,
        enable_practice=False,
        enable_how=False,
        enable_provider=False,
    )
    runtime = Runtime(config)
    driver = ROS2Driver("ur5e", joint_dof=6, node_name=next_name("driver"))
    driver.initialize()
    runtime.register_driver("ros2", driver)

    controller = TrajectorySubscriber(next_name("controller"))

    traj = TrajectoryCommand(
        waypoints=[[0.0] * 6, [0.1] * 6, [0.2] * 6],
        times=[0.0, 1.0, 2.0],
    )
    driver.execute_trajectory(traj)
    spin_nodes([driver._node, controller.node])

    assert len(controller.received) >= 1
    assert len(controller.received[0]["points"]) == 3

    controller.destroy()
    driver.stop()


@test("EventBus safety.violation triggers driver emergency_stop")
def test_eventbus_safety_to_emergency():
    config = RuntimeConfig(
        robot_id="test_bot",
        enable_firewall=False,
        enable_memory=False,
        enable_practice=False,
        enable_how=False,
        enable_provider=False,
    )
    runtime = Runtime(config)
    # Must initialize Runtime to set up internal EventBus subscriptions
    runtime.initialize()

    driver = ROS2Driver("test_bot", joint_dof=6, node_name=next_name("driver"))
    driver.initialize()
    runtime.register_driver("ros2", driver)

    # safety.violation should trigger robot.emergency_stop
    # which triggers driver.emergency_stop
    runtime.event_bus.publish(Event(
        topic="safety.violation",
        payload={"description": "joint limit exceeded"},
        source="test",
    ))
    time.sleep(0.2)

    state = driver.get_state()
    assert state.error_code == 99

    driver.stop()


@test("Multiple drivers registered with Runtime")
def test_runtime_multiple_drivers():
    config = RuntimeConfig(
        robot_id="test_bot",
        enable_firewall=False,
        enable_memory=False,
        enable_practice=False,
        enable_how=False,
        enable_provider=False,
    )
    runtime = Runtime(config)

    driver1 = ROS2Driver("bot1", joint_dof=6, node_name=next_name("driver"))
    driver2 = ROS2Driver("bot2", joint_dof=6, node_name=next_name("driver"))
    driver1.initialize()
    driver2.initialize()

    runtime.register_driver("bot1", driver1)
    runtime.register_driver("bot2", driver2)

    assert runtime.get_driver("bot1") is driver1
    assert runtime.get_driver("bot2") is driver2
    assert len(runtime.status["drivers"]) == 2

    driver1.stop()
    driver2.stop()


@test("Runtime status includes driver state")
def test_runtime_status_with_driver():
    config = RuntimeConfig(
        robot_id="test_bot",
        enable_firewall=False,
        enable_memory=False,
        enable_practice=False,
        enable_how=False,
        enable_provider=False,
    )
    runtime = Runtime(config)
    driver = ROS2Driver("test_bot", joint_dof=6, node_name=next_name("driver"))
    driver.initialize()
    runtime.register_driver("ros2", driver)

    status = runtime.status
    assert "drivers" in status
    assert "ros2" in status["drivers"]
    assert "modules" in status

    driver.stop()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    if not rclpy.ok():
        rclpy.init(args=None)

    print("=" * 60)
    print("ROSClaw Runtime + ROS2Driver Closed-Loop Tests")
    print("=" * 60)

    test_runtime_register_driver()
    test_runtime_emergency_stop()
    test_runtime_driver_full_loop()
    test_runtime_driver_trajectory()
    test_eventbus_safety_to_emergency()
    test_runtime_multiple_drivers()
    test_runtime_status_with_driver()

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
