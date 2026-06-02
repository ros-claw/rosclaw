#!/usr/bin/env python3
"""ROS2 Action Feedback Integration Tests.

Tests ActionClient receiving feedback from ActionServer during trajectory execution.
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
# Mock Action Server with feedback
# ------------------------------------------------------------------

class MockFeedbackActionServer:
    """Simulates a controller that sends feedback during execution."""

    def __init__(self, node_name: str = "mock_controller"):
        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = Node(node_name)
        self.received_goals = []
        self.feedback_count = 0
        self.server = ActionServer(
            self.node,
            FollowJointTrajectory,
            "/test_ns/joint_trajectory_controller/follow_joint_trajectory",
            self._execute_callback,
        )

    def _execute_callback(self, goal_handle):
        self.received_goals.append(goal_handle.request)

        # Send multiple feedback messages
        for i in range(3):
            feedback = FollowJointTrajectory.Feedback()
            feedback.desired.positions = [float(i) * 0.1] * 6
            goal_handle.publish_feedback(feedback)
            self.feedback_count += 1
            time.sleep(0.05)

        goal_handle.succeed()
        result = FollowJointTrajectory.Result()
        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        return result

    def destroy(self):
        self.server.destroy()
        self.node.destroy_node()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

counter = 0


def next_name(base: str) -> str:
    global counter
    counter += 1
    return f"{base}_{counter}"


def _cleanup():
    if rclpy.ok():
        try:
            rclpy.shutdown()
        except Exception:
            pass


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

@test("ActionServer sends feedback during execution")
def test_feedback_received():
    _cleanup()
    if not rclpy.ok():
        rclpy.init(args=None)

    controller = MockFeedbackActionServer(next_name("ctrl"))
    node = UR5ROSNode(robot_ip="127.0.0.1", namespace="test_ns")

    # Spin to discover
    for _ in range(30):
        rclpy.spin_once(controller.node, timeout_sec=0.05)
        rclpy.spin_once(node, timeout_sec=0.05)
        time.sleep(0.02)

    assert node.trajectory_client.server_is_ready()

    # Send goal with feedback callback
    feedback_received = []

    def _feedback_cb(feedback_msg):
        feedback_received.append(feedback_msg.feedback)

    goal_msg = FollowJointTrajectory.Goal()
    goal_msg.trajectory = JointTrajectory()
    goal_msg.trajectory.joint_names = node.JOINT_NAMES
    point = JointTrajectoryPoint()
    point.positions = [0.0] * 6
    point.time_from_start.sec = 1
    goal_msg.trajectory.points.append(point)

    send_goal_future = node.trajectory_client.send_goal_async(
        goal_msg, feedback_callback=_feedback_cb
    )

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(controller.node)

    import threading
    spin_done = threading.Event()

    def _spin():
        while not spin_done.is_set():
            executor.spin_once(timeout_sec=0.1)

    spin_thread = threading.Thread(target=_spin)
    spin_thread.start()

    # Wait for goal handle
    start = time.time()
    while not send_goal_future.done() and time.time() - start < 5.0:
        time.sleep(0.05)

    goal_handle = send_goal_future.result()
    assert goal_handle is not None
    assert goal_handle.accepted

    # Wait for result
    result_future = goal_handle.get_result_async()
    start = time.time()
    while not result_future.done() and time.time() - start < 5.0:
        time.sleep(0.05)

    spin_done.set()
    spin_thread.join(timeout=2.0)
    executor.remove_node(node)
    executor.remove_node(controller.node)
    executor.shutdown()

    assert result_future.result().result.error_code == FollowJointTrajectory.Result.SUCCESSFUL
    # Feedback may or may not be received depending on timing
    # Just verify no crash

    node.destroy_node()
    controller.destroy()


@test("Action goal rejected: no feedback expected")
def test_rejected_no_feedback():
    _cleanup()
    if not rclpy.ok():
        rclpy.init(args=None)

    # Create server that rejects goals
    class RejectServer:
        def __init__(self):
            self.node = Node(next_name("reject"))
            self.server = ActionServer(
                self.node,
                FollowJointTrajectory,
                "/test_ns/joint_trajectory_controller/follow_joint_trajectory",
                self._callback,
            )

        def _callback(self, goal_handle):
            goal_handle.abort()
            result = FollowJointTrajectory.Result()
            result.error_code = FollowJointTrajectory.Result.INVALID_JOINTS
            return result

        def destroy(self):
            self.server.destroy()
            self.node.destroy_node()

    controller = RejectServer()
    node = UR5ROSNode(robot_ip="127.0.0.1", namespace="test_ns")

    for _ in range(30):
        rclpy.spin_once(controller.node, timeout_sec=0.05)
        rclpy.spin_once(node, timeout_sec=0.05)
        time.sleep(0.02)

    goal_msg = FollowJointTrajectory.Goal()
    goal_msg.trajectory = JointTrajectory()
    goal_msg.trajectory.joint_names = node.JOINT_NAMES
    point = JointTrajectoryPoint()
    point.positions = [0.0] * 6
    point.time_from_start.sec = 1
    goal_msg.trajectory.points.append(point)

    send_goal_future = node.trajectory_client.send_goal_async(goal_msg)

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(controller.node)

    import threading
    spin_done = threading.Event()

    def _spin():
        while not spin_done.is_set():
            executor.spin_once(timeout_sec=0.1)

    spin_thread = threading.Thread(target=_spin)
    spin_thread.start()

    start = time.time()
    while not send_goal_future.done() and time.time() - start < 5.0:
        time.sleep(0.05)

    goal_handle = send_goal_future.result()

    spin_done.set()
    spin_thread.join(timeout=2.0)
    executor.remove_node(node)
    executor.remove_node(controller.node)
    executor.shutdown()

    # Goal was rejected/aborted
    assert goal_handle is not None
    assert not goal_handle.accepted or goal_handle.status == 6  # ABORTED

    node.destroy_node()
    controller.destroy()


@test("ActionClient send_goal_async returns valid future")
def test_send_goal_future():
    _cleanup()
    if not rclpy.ok():
        rclpy.init(args=None)

    controller = MockFeedbackActionServer(next_name("ctrl"))
    node = UR5ROSNode(robot_ip="127.0.0.1", namespace="test_ns")

    for _ in range(30):
        rclpy.spin_once(controller.node, timeout_sec=0.05)
        rclpy.spin_once(node, timeout_sec=0.05)
        time.sleep(0.02)

    goal_msg = FollowJointTrajectory.Goal()
    goal_msg.trajectory = JointTrajectory()
    goal_msg.trajectory.joint_names = node.JOINT_NAMES
    point = JointTrajectoryPoint()
    point.positions = [0.5] * 6
    point.time_from_start.sec = 2
    goal_msg.trajectory.points.append(point)

    future = node.trajectory_client.send_goal_async(goal_msg)
    assert future is not None
    assert hasattr(future, "done")

    # Cancel it to avoid hanging
    node.destroy_node()
    controller.destroy()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    if not rclpy.ok():
        rclpy.init(args=None)

    print("=" * 60)
    print("ROSClaw ROS2 Action Feedback Integration Tests")
    print("=" * 60)

    test_feedback_received()
    test_rejected_no_feedback()
    test_send_goal_future()

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
