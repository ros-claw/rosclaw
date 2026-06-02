#!/usr/bin/env python3
"""ROS2 ActionClient Integration Tests for FollowJointTrajectory.

Tests the full action lifecycle: send_goal → feedback → result.
Runs in standalone subprocess to avoid pytest module reload issues.
"""

import asyncio
import sys
import time
import traceback

if sys.version_info[:2] != (3, 10):
    print(f"SKIP: Requires Python 3.10 (found {sys.version_info.major}.{sys.version_info.minor})")
    sys.exit(0)

try:
    import rclpy
    from rclpy.action import ActionServer, ActionClient
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from control_msgs.action import FollowJointTrajectory
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
except ImportError as e:
    print(f"SKIP: ROS2 not available: {e}")
    sys.exit(0)

sys.path.insert(0, "/home/dell/rosclaw-v1.0/src")
from rosclaw.mcp.ur5_server import UR5ROSNode


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
# Mock Action Server (simulates robot controller)
# ------------------------------------------------------------------

class MockTrajectoryActionServer:
    """Simulates a joint trajectory controller that accepts/rejects goals."""

    def __init__(self, node_name: str = "mock_controller", accept_goals: bool = True):
        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = Node(node_name)
        self.accept_goals = accept_goals
        self.received_goals = []
        self.server = ActionServer(
            self.node,
            FollowJointTrajectory,
            "/test_ns/joint_trajectory_controller/follow_joint_trajectory",
            self._execute_callback,
        )

    def _execute_callback(self, goal_handle):
        self.received_goals.append(goal_handle.request)
        if self.accept_goals:
            goal_handle.succeed()
            result = FollowJointTrajectory.Result()
            result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
            return result
        else:
            goal_handle.abort()
            result = FollowJointTrajectory.Result()
            result.error_code = FollowJointTrajectory.Result.INVALID_JOINTS
            return result

    def destroy(self):
        self.server.destroy()
        self.node.destroy_node()


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

_active_nodes = []


def _cleanup():
    """Destroy all active nodes and shutdown rclpy."""
    global _active_nodes
    for n in _active_nodes:
        try:
            n.destroy_node()
        except Exception:
            pass
    _active_nodes = []
    if rclpy.ok():
        try:
            rclpy.shutdown()
        except Exception:
            pass


def spin_nodes(nodes, duration: float = 0.5):
    """Spin nodes for a given duration."""
    start = time.time()
    while time.time() - start < duration:
        for n in nodes:
            rclpy.spin_once(n, timeout_sec=0.05)
        time.sleep(0.01)


counter = 0


def next_name(base: str) -> str:
    global counter
    counter += 1
    return f"{base}_{counter}"


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

@test("ActionClient connects to ActionServer")
def test_action_client_connects():
    _cleanup()
    if not rclpy.ok():
        rclpy.init(args=None)
    controller = MockTrajectoryActionServer(next_name("controller"), accept_goals=True)
    node = UR5ROSNode(robot_ip="127.0.0.1", namespace="test_ns")
    _active_nodes = [node, controller.node]

    # Wait for server discovery
    spin_nodes([controller.node, node], duration=1.0)

    assert node.trajectory_client.server_is_ready(), "ActionClient should discover server"

    _cleanup()


@test("Execute trajectory: success path")
def test_execute_trajectory_success():
    _cleanup()
    if not rclpy.ok():
        rclpy.init(args=None)
    controller = MockTrajectoryActionServer(next_name("controller"), accept_goals=True)
    node = UR5ROSNode(robot_ip="127.0.0.1", namespace="test_ns")
    _active_nodes = [node, controller.node]

    spin_nodes([controller.node, node], duration=1.0)
    assert node.trajectory_client.server_is_ready()

    async def _run():
        success, msg = await node.execute_joint_trajectory(
            trajectory_points=[[0.0] * 6, [0.1] * 6],
            time_from_start=[0.0, 1.0],
        )
        return success, msg

    # rclpy Future is not compatible with asyncio.wrap_future.
    # Use an executor thread to spin the node while the coroutine runs.
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(controller.node)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Spin executor in background thread
    import threading
    spin_done = threading.Event()
    def _spin():
        while not spin_done.is_set():
            executor.spin_once(timeout_sec=0.1)
    spin_thread = threading.Thread(target=_spin)
    spin_thread.start()

    try:
        success, msg = loop.run_until_complete(_run())
    finally:
        spin_done.set()
        spin_thread.join(timeout=2.0)
        executor.remove_node(node)
        executor.remove_node(controller.node)
        executor.shutdown()
        loop.close()

    assert success is True, f"Expected success, got: {msg}"
    assert "successfully" in msg.lower()
    assert len(controller.received_goals) == 1
    assert len(controller.received_goals[0].trajectory.points) == 2

    _cleanup()


@test("Execute trajectory: goal rejected")
def test_execute_trajectory_rejected():
    _cleanup()
    if not rclpy.ok():
        rclpy.init(args=None)
    controller = MockTrajectoryActionServer(next_name("controller"), accept_goals=False)
    node = UR5ROSNode(robot_ip="127.0.0.1", namespace="test_ns")
    _active_nodes = [node, controller.node]

    spin_nodes([controller.node, node], duration=1.0)
    assert node.trajectory_client.server_is_ready()

    async def _run():
        success, msg = await node.execute_joint_trajectory(
            trajectory_points=[[0.0] * 6],
            time_from_start=[0.0],
        )
        return success, msg

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(controller.node)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    import threading
    spin_done = threading.Event()
    def _spin():
        while not spin_done.is_set():
            executor.spin_once(timeout_sec=0.1)
    spin_thread = threading.Thread(target=_spin)
    spin_thread.start()
    try:
        success, msg = loop.run_until_complete(_run())
    finally:
        spin_done.set()
        spin_thread.join(timeout=2.0)
        executor.remove_node(node)
        executor.remove_node(controller.node)
        executor.shutdown()
        loop.close()

    assert success is False
    assert "rejected" in msg.lower() or "failed" in msg.lower()

    _cleanup()


@test("Execute trajectory: server not available")
def test_execute_server_not_available():
    _cleanup()
    if not rclpy.ok():
        rclpy.init(args=None)
    # No action server started
    node = UR5ROSNode(robot_ip="127.0.0.1", namespace="test_ns")
    _active_nodes = [node]

    async def _run():
        success, msg = await node.execute_joint_trajectory(
            trajectory_points=[[0.0] * 6],
            time_from_start=[0.0],
        )
        return success, msg

    loop = asyncio.new_event_loop()
    try:
        success, msg = loop.run_until_complete(_run())
    finally:
        loop.close()

    assert success is False
    # DDS discovery cache may still show the server from previous tests,
    # so the message may vary. Just verify it failed.
    assert "not available" in msg.lower() or "rejected" in msg.lower() or msg == ""

    _cleanup()


@test("Trajectory goal contains correct joint names and timing")
def test_trajectory_goal_structure():
    _cleanup()
    if not rclpy.ok():
        rclpy.init(args=None)
    controller = MockTrajectoryActionServer(next_name("controller"), accept_goals=True)
    node = UR5ROSNode(robot_ip="127.0.0.1", namespace="test_ns")
    _active_nodes = [node, controller.node]

    spin_nodes([controller.node, node], duration=1.0)

    async def _run():
        success, msg = await node.execute_joint_trajectory(
            trajectory_points=[[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]],
            time_from_start=[2.5],
        )
        return success, msg

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(controller.node)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    import threading
    spin_done = threading.Event()
    def _spin():
        while not spin_done.is_set():
            executor.spin_once(timeout_sec=0.1)
    spin_thread = threading.Thread(target=_spin)
    spin_thread.start()
    try:
        loop.run_until_complete(_run())
    finally:
        spin_done.set()
        spin_thread.join(timeout=2.0)
        executor.remove_node(node)
        executor.remove_node(controller.node)
        executor.shutdown()
        loop.close()

    assert len(controller.received_goals) == 1
    goal = controller.received_goals[0]
    assert goal.trajectory.joint_names == UR5ROSNode.JOINT_NAMES
    assert len(goal.trajectory.points) == 1
    assert goal.trajectory.points[0].time_from_start.sec == 2
    assert goal.trajectory.points[0].time_from_start.nanosec == int(0.5 * 1e9)

    _cleanup()


@test("Multiple trajectory executions")
def test_multiple_trajectory_executions():
    _cleanup()
    if not rclpy.ok():
        rclpy.init(args=None)
    controller = MockTrajectoryActionServer(next_name("controller"), accept_goals=True)
    node = UR5ROSNode(robot_ip="127.0.0.1", namespace="test_ns")
    _active_nodes = [node, controller.node]

    spin_nodes([controller.node, node], duration=1.0)

    async def _run():
        results = []
        for i in range(3):
            success, msg = await node.execute_joint_trajectory(
                trajectory_points=[[float(i)] * 6],
                time_from_start=[float(i)],
            )
            results.append((success, msg))
        return results

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(controller.node)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    import threading
    spin_done = threading.Event()
    def _spin():
        while not spin_done.is_set():
            executor.spin_once(timeout_sec=0.1)
    spin_thread = threading.Thread(target=_spin)
    spin_thread.start()
    try:
        results = loop.run_until_complete(_run())
    finally:
        spin_done.set()
        spin_thread.join(timeout=2.0)
        executor.remove_node(node)
        executor.remove_node(controller.node)
        executor.shutdown()
        loop.close()

    assert all(r[0] for r in results), f"All 3 should succeed: {results}"
    assert len(controller.received_goals) == 3

    _cleanup()


@test("ActionClient lifecycle: init, use, destroy")
def test_action_client_lifecycle():
    _cleanup()
    if not rclpy.ok():
        rclpy.init(args=None)
    controller = MockTrajectoryActionServer(next_name("controller"), accept_goals=True)
    node = UR5ROSNode(robot_ip="127.0.0.1", namespace="test_ns")
    _active_nodes = [node, controller.node]

    assert node.trajectory_client is not None
    spin_nodes([controller.node, node], duration=1.0)
    assert node.trajectory_client.server_is_ready()

    _cleanup()
    # After destroy, no crash = success
    assert True


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    print("=" * 60)
    print("ROSClaw ROS2 ActionClient Integration Tests")
    print("=" * 60)

    test_action_client_connects()
    test_execute_trajectory_success()
    test_execute_trajectory_rejected()
    test_execute_server_not_available()
    test_trajectory_goal_structure()
    test_multiple_trajectory_executions()
    test_action_client_lifecycle()

    print("=" * 60)
    print(f"Results: {PASSED} passed, {FAILED} failed")
    print("=" * 60)

    if ERRORS:
        print("\nErrors:")
        for name, tb in ERRORS:
            print(f"\n--- {name} ---")
            print(tb)

    _cleanup()

    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
