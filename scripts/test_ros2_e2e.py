#!/usr/bin/env python3
"""ROS2 End-to-End Closed-Loop Integration Test Runner.

Runs in a standalone Python process to avoid pytest module reload conflicts
with rclpy C extensions.

Usage:
    export LD_LIBRARY_PATH="/tmp/ros2-local/opt/ros/humble/lib:$LD_LIBRARY_PATH"
    export PYTHONPATH="/tmp/ros2-local/opt/ros/humble/local/lib/python3.10/dist-packages:$PYTHONPATH"
    python3 scripts/test_ros2_e2e.py
"""

import sys
import time
import traceback

# Check Python version
if sys.version_info[:2] != (3, 10):
    print(f"SKIP: ROS2 tests require Python 3.10 (found {sys.version_info.major}.{sys.version_info.minor})")
    sys.exit(0)

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from trajectory_msgs.msg import JointTrajectory
except ImportError as e:
    print(f"SKIP: rclpy not available: {e}")
    sys.exit(0)

# Import rosclaw modules after rclpy is ready
sys.path.insert(0, "/home/dell/rosclaw-v1.0/src")
from rosclaw.mcp_drivers.ros2_driver import ROS2Driver
from rosclaw.mcp_drivers.base import TrajectoryCommand


# ------------------------------------------------------------------
# Test framework (minimal)
# ------------------------------------------------------------------

PASSED = 0
FAILED = 0
ERRORS = []


def test(name):
    """Decorator to run a test function."""
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


# ------------------------------------------------------------------
# Test suite
# ------------------------------------------------------------------

counter = 0

def next_name(base: str) -> str:
    global counter
    counter += 1
    return f"{base}_{counter}"


@test("ROS2Driver receives JointState from real publisher")
def test_driver_receives_joint_state():
    driver = ROS2Driver("test_bot", joint_dof=6, node_name=next_name("driver"))
    driver.initialize()
    publisher = JointStatePublisher(next_name("jspub"))
    spin_nodes([publisher.node, driver._node])
    positions = driver.get_joint_positions()
    assert len(positions) == 6
    assert abs(positions[0] - 0.1) < 0.01
    publisher.destroy()
    driver.stop()


@test("ROS2Driver publishes JointTrajectory to real subscriber")
def test_driver_publishes_trajectory():
    driver = ROS2Driver("test_bot", joint_dof=6, node_name=next_name("driver"))
    driver.initialize()
    subscriber = TrajectorySubscriber(next_name("trajsub"))
    driver.move_joints([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], duration=2.0)
    spin_nodes([driver._node, subscriber.node])
    assert len(subscriber.received) >= 1
    msg = subscriber.received[0]
    assert len(msg["joint_names"]) == 6
    assert msg["points"][0][1] == 2
    subscriber.destroy()
    driver.stop()


@test("ROS2Driver executes multi-waypoint trajectory")
def test_driver_multi_waypoint():
    driver = ROS2Driver("test_bot", joint_dof=6, node_name=next_name("driver"))
    driver.initialize()
    subscriber = TrajectorySubscriber(next_name("trajsub"))
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


@test("ROS2Driver state reflects received JointState")
def test_driver_state_updated():
    driver = ROS2Driver("test_bot", joint_dof=6, node_name=next_name("driver"))
    driver.initialize()
    publisher = JointStatePublisher(next_name("jspub"))
    publisher.positions = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    publisher.velocities = [0.1] * 6
    publisher.efforts = [10.0] * 6
    spin_nodes([publisher.node, driver._node])
    state = driver.get_state()
    assert abs(state.joint_positions[0] - 1.0) < 0.01
    assert abs(state.joint_velocities[0] - 0.1) < 0.01
    assert abs(state.joint_torques[0] - 10.0) < 0.01
    publisher.destroy()
    driver.stop()


@test("ROS2Driver emergency stop")
def test_driver_emergency_stop():
    driver = ROS2Driver("test_bot", joint_dof=6, node_name=next_name("driver"))
    driver.initialize()
    publisher = JointStatePublisher(next_name("jspub"))
    spin_nodes([publisher.node, driver._node], iterations=10)
    driver.emergency_stop()
    state = driver.get_state()
    assert state.error_code == 99
    assert "Emergency" in state.error_message
    publisher.destroy()
    driver.stop()


@test("ROS2Driver get_joint_torques")
def test_driver_get_torques():
    driver = ROS2Driver("test_bot", joint_dof=6, node_name=next_name("driver"))
    driver.initialize()
    publisher = JointStatePublisher(next_name("jspub"))
    publisher.efforts = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    spin_nodes([publisher.node, driver._node])
    torques = driver.get_joint_torques()
    assert len(torques) == 6
    assert abs(torques[0] - 5.0) < 0.01
    publisher.destroy()
    driver.stop()


@test("ROS2Driver gripper control")
def test_driver_gripper():
    driver = ROS2Driver("test_bot", joint_dof=6, node_name=next_name("driver"))
    driver.initialize()
    assert driver.set_gripper(0.75, force=0.8) is True
    assert abs(driver.get_state().gripper_state - 0.75) < 0.001
    driver.stop()


@test("ROS2Driver multiple messages")
def test_driver_multiple_messages():
    driver = ROS2Driver("test_bot", joint_dof=6, node_name=next_name("driver"))
    driver.initialize()
    publisher = JointStatePublisher(next_name("jspub"))
    publisher.positions = [0.0] * 6
    spin_nodes([publisher.node, driver._node])
    assert abs(driver.get_joint_positions()[0]) < 0.01
    publisher.positions = [0.5] * 6
    spin_nodes([publisher.node, driver._node])
    assert abs(driver.get_joint_positions()[0] - 0.5) < 0.01
    publisher.destroy()
    driver.stop()


@test("ROS2Driver move_joints updates driver_state")
def test_driver_move_updates_state():
    driver = ROS2Driver("test_bot", joint_dof=6, node_name=next_name("driver"))
    driver.initialize()
    target = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    assert driver.move_joints(target, duration=1.0) is True
    # In real rclpy mode, get_joint_positions() returns from _latest_joint_state
    # which only updates when JointState messages are received.
    # driver_state.joint_positions is updated directly by move_joints.
    assert driver.get_state().joint_positions == target
    driver.stop()


@test("ROS2Driver execute_trajectory publishes to subscriber")
def test_driver_execute_updates_state():
    driver = ROS2Driver("test_bot", joint_dof=6, node_name=next_name("driver"))
    driver.initialize()
    subscriber = TrajectorySubscriber(next_name("trajsub"))
    traj = TrajectoryCommand(
        waypoints=[[0.0] * 6, [0.5] * 6, [1.0] * 6],
        times=[0.0, 1.0, 2.0],
    )
    assert driver.execute_trajectory(traj) is True
    spin_nodes([driver._node, subscriber.node])
    assert len(subscriber.received) >= 1
    assert len(subscriber.received[0]["points"]) == 3
    subscriber.destroy()
    driver.stop()


@test("ROS2Driver wrong DOF raises ValueError")
def test_driver_wrong_dof():
    driver = ROS2Driver("test_bot", joint_dof=6, node_name=next_name("driver"))
    driver.initialize()
    try:
        driver.move_joints([0.1] * 5, duration=1.0)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "Expected 6 joint positions, got 5" in str(e)
    driver.stop()


@test("ROS2Driver lifecycle: init creates node, stop destroys it")
def test_driver_lifecycle():
    driver = ROS2Driver("test_bot", joint_dof=6, node_name=next_name("driver"))
    driver.initialize()
    assert driver._node is not None
    driver.stop()
    assert driver._node is None


@test("ROS2Driver multiple init/stop cycles")
def test_driver_multiple_cycles():
    driver = ROS2Driver("test_bot", joint_dof=6, node_name=next_name("driver"))
    for _ in range(3):
        driver.initialize()
        assert driver._node is not None
        driver.stop()
        assert driver._node is None


@test("Full closed-loop: robot state -> driver -> trajectory -> controller")
def test_full_closed_loop():
    driver = ROS2Driver("ur5e", joint_dof=6, node_name=next_name("driver"))
    driver.initialize()
    robot = JointStatePublisher(next_name("robot"))
    robot.positions = [0.0, -1.57, 1.57, 0.0, 0.0, 0.0]
    spin_nodes([robot.node, driver._node])
    current = driver.get_joint_positions()
    assert abs(current[0]) < 0.01
    target = [0.1, -1.47, 1.47, 0.1, 0.0, 0.0]
    controller = TrajectorySubscriber(next_name("controller"))
    driver.move_joints(target, duration=2.0)
    spin_nodes([driver._node, controller.node])
    assert len(controller.received) >= 1
    traj = controller.received[0]
    assert traj["points"][0][0] == target
    robot.positions = target
    spin_nodes([robot.node, driver._node])
    updated = driver.get_joint_positions()
    assert abs(updated[0] - 0.1) < 0.01
    robot.destroy()
    controller.destroy()
    driver.stop()


@test("Emergency stop propagation")
def test_emergency_stop_propagation():
    driver = ROS2Driver("ur5e", joint_dof=6, node_name=next_name("driver"))
    driver.initialize()
    robot = JointStatePublisher(next_name("robot"))
    spin_nodes([robot.node, driver._node], iterations=10)
    driver.emergency_stop()
    state = driver.get_state()
    assert state.error_code == 99
    assert "Emergency" in state.error_message
    robot.destroy()
    driver.stop()


@test("Sensor feedback loop: position + velocity + effort")
def test_sensor_feedback():
    driver = ROS2Driver("ur5e", joint_dof=6, node_name=next_name("driver"))
    driver.initialize()
    robot = JointStatePublisher(next_name("robot"))
    robot.positions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    robot.velocities = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    robot.efforts = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    spin_nodes([robot.node, driver._node])
    assert driver.get_joint_positions() == robot.positions
    assert driver.get_joint_velocities() == robot.velocities
    assert driver.get_joint_torques() == robot.efforts
    robot.destroy()
    driver.stop()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    global counter
    if not rclpy.ok():
        rclpy.init(args=None)

    print("=" * 60)
    print("ROSClaw ROS2 End-to-End Closed-Loop Integration Tests")
    print("=" * 60)

    # Run all tests
    test_driver_receives_joint_state()
    test_driver_publishes_trajectory()
    test_driver_multi_waypoint()
    test_driver_state_updated()
    test_driver_emergency_stop()
    test_driver_get_torques()
    test_driver_gripper()
    test_driver_multiple_messages()
    test_driver_move_updates_state()
    test_driver_execute_updates_state()
    test_driver_wrong_dof()
    test_driver_lifecycle()
    test_driver_multiple_cycles()
    test_full_closed_loop()
    test_emergency_stop_propagation()
    test_sensor_feedback()

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
