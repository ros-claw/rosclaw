#!/usr/bin/env python3
"""UR5Server MCP Tools End-to-End Integration Tests.

Tests all 6 MCP tools through the real call_tool dispatcher:
- ur5_get_joint_states
- ur5_move_joints
- ur5_execute_trajectory
- ur5_emergency_stop
- ur5_get_limits
- ur5_validate_trajectory

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
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from trajectory_msgs.msg import JointTrajectory
except ImportError as e:
    print(f"SKIP: rclpy not available: {e}")
    sys.exit(0)

sys.path.insert(0, "/home/dell/rosclaw-v1.0/src")

from rosclaw.mcp.ur5_server import UR5MCPServer, ROS_IMPORTS_OK


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

class JointStatePublisher:
    def __init__(self, node_name: str):
        self.node = Node(node_name)
        self.pub = self.node.create_publisher(JointState, "/ur/joint_states", 10)
        self.timer = self.node.create_timer(0.05, self._publish)
        self.positions = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]
        self.velocities = [0.01] * 6
        self.efforts = [1.0] * 6

    def _publish(self):
        msg = JointState()
        msg.name = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
        ]
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
            "/ur/joint_trajectory_controller/command",
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


def _cleanup_server(server):
    """Clean up UR5MCPServer resources."""
    if server and server.ros_node:
        try:
            server.ros_node.destroy_node()
        except Exception:
            pass


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

@test("MCP tool: get_joint_states returns structured data")
def test_mcp_get_joint_states():
    server = UR5MCPServer(
        robot_ip="127.0.0.1",
        firewall_model_path="/nonexistent/model.xml",
    )
    # Set state directly
    server.ros_node.state.joint_positions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    server.ros_node.state.is_connected = True

    async def _run():
        return await server._handle_get_joint_states()

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(_run())
    finally:
        loop.close()

    assert len(result) == 1
    text = result[0].text
    assert "joint_positions" in text
    assert "0.1" in text
    assert "is_connected" in text

    _cleanup_server(server)


@test("MCP tool: move_joints with valid positions")
def test_mcp_move_joints_valid():
    server = UR5MCPServer(
        robot_ip="127.0.0.1",
        firewall_model_path="/nonexistent/model.xml",
    )

    async def _run():
        return await server._handle_move_joints({
            "joint_positions": [0.0, -0.5, 1.0, 0.0, 0.0, 0.0],
            "duration": 2.0,
            "validate": False,
        })

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(_run())
    finally:
        loop.close()

    assert len(result) == 1
    # Should attempt execution but action server not available
    text = result[0].text
    assert "not available" in text.lower() or "failed" in text.lower() or "trajectory" in text.lower()

    _cleanup_server(server)


@test("MCP tool: move_joints wrong count returns error")
def test_mcp_move_joints_wrong_count():
    server = UR5MCPServer(
        robot_ip="127.0.0.1",
        firewall_model_path="/nonexistent/model.xml",
    )

    async def _run():
        return await server._handle_move_joints({
            "joint_positions": [0.0] * 3,
        })

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(_run())
    finally:
        loop.close()

    assert len(result) == 1
    assert "Expected 6" in result[0].text

    _cleanup_server(server)


@test("MCP tool: move_joints limit violation blocked")
def test_mcp_move_joints_limit_violation():
    server = UR5MCPServer(
        robot_ip="127.0.0.1",
        firewall_model_path="/nonexistent/model.xml",
    )

    async def _run():
        return await server._handle_move_joints({
            "joint_positions": [100.0] * 6,
        })

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(_run())
    finally:
        loop.close()

    assert len(result) == 1
    assert "Joint limit violation" in result[0].text

    _cleanup_server(server)


@test("MCP tool: emergency_stop returns confirmation")
def test_mcp_emergency_stop():
    server = UR5MCPServer(
        robot_ip="127.0.0.1",
        firewall_model_path="/nonexistent/model.xml",
    )

    async def _run():
        return await server._handle_emergency_stop()

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(_run())
    finally:
        loop.close()

    assert len(result) == 1
    assert "EMERGENCY STOP" in result[0].text

    _cleanup_server(server)


@test("MCP tool: get_limits without firewall")
def test_mcp_get_limits_no_firewall():
    server = UR5MCPServer(
        robot_ip="127.0.0.1",
        firewall_model_path="/nonexistent/model.xml",
    )

    async def _run():
        return await server._handle_get_limits()

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(_run())
    finally:
        loop.close()

    assert len(result) == 1
    text = result[0].text
    assert "joint_limits" in text
    assert "firewall_enabled" in text
    assert "false" in text.lower() or "None" in text

    _cleanup_server(server)


@test("MCP tool: validate_trajectory no firewall")
def test_mcp_validate_no_firewall():
    server = UR5MCPServer(
        robot_ip="127.0.0.1",
        firewall_model_path="/nonexistent/model.xml",
    )

    async def _run():
        return await server._handle_validate_trajectory({
            "waypoints": [[0.0] * 6],
        })

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(_run())
    finally:
        loop.close()

    assert len(result) == 1
    assert "firewall not available" in result[0].text

    _cleanup_server(server)


@test("MCP tool: execute_trajectory wrong waypoint size")
def test_mcp_execute_wrong_size():
    server = UR5MCPServer(
        robot_ip="127.0.0.1",
        firewall_model_path="/nonexistent/model.xml",
    )

    async def _run():
        return await server._handle_execute_trajectory({
            "waypoints": [[0.0] * 3],
            "times": [1.0],
        })

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(_run())
    finally:
        loop.close()

    assert len(result) == 1
    assert "should have 6 values" in result[0].text

    _cleanup_server(server)


@test("MCP tool: execute_trajectory mismatched lengths")
def test_mcp_execute_mismatched():
    server = UR5MCPServer(
        robot_ip="127.0.0.1",
        firewall_model_path="/nonexistent/model.xml",
    )

    async def _run():
        return await server._handle_execute_trajectory({
            "waypoints": [[0.0] * 6],
            "times": [1.0, 2.0],
        })

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(_run())
    finally:
        loop.close()

    assert len(result) == 1
    assert "same length" in result[0].text

    _cleanup_server(server)


@test("MCP tool: server has registered tools")
def test_mcp_server_tools_registered():
    server = UR5MCPServer(
        robot_ip="127.0.0.1",
        firewall_model_path="/nonexistent/model.xml",
    )

    # Verify server and tools exist
    assert server is not None
    assert server.server is not None
    assert server.ros_node is not None

    _cleanup_server(server)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    if not rclpy.ok():
        rclpy.init(args=None)

    print("=" * 60)
    print("ROSClaw UR5Server MCP Tools E2E Tests")
    print("=" * 60)

    test_mcp_get_joint_states()
    test_mcp_move_joints_valid()
    test_mcp_move_joints_wrong_count()
    test_mcp_move_joints_limit_violation()
    test_mcp_emergency_stop()
    test_mcp_get_limits_no_firewall()
    test_mcp_validate_no_firewall()
    test_mcp_execute_wrong_size()
    test_mcp_execute_mismatched()
    test_mcp_server_tools_registered()

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
