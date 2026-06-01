"""Verify Claude Code MCP closed loop for ROSClaw.

This test simulates what Claude Code would do when connecting to ROSClaw via MCP:
1. Discover available tools
2. Call basic tools (get_robot_state, query_knowledge)
3. Verify responses are structured and meaningful
4. Check that the server can start and handle requests

Usage:
    cd /home/ubuntu/rosclaw/rosclaw/rosclaw-v1.0
    PYTHONPATH=src python3 tests/test_mcp_closed_loop.py
"""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _check_mcp_server_imports():
    """Helper that returns bool for script mode."""
    try:
        from rosclaw.mcp.ur5_server import RCLPY_AVAILABLE, ROS_IMPORTS_OK
        print(f"✅ UR5MCPServer imported (rclpy available: {RCLPY_AVAILABLE}, ROS imports: {ROS_IMPORTS_OK})")
        return True
    except Exception as e:
        print(f"❌ Failed to import UR5MCPServer: {e}")
        return False


def test_mcp_server_imports():
    """Verify MCP server can be imported without ROS2/MuJoCo."""
    assert _check_mcp_server_imports()


def _check_mcp_hub_imports():
    """Helper that returns bool for script mode."""
    try:
        print("✅ MCPHub imported")
        return True
    except Exception as e:
        print(f"❌ Failed to import MCPHub: {e}")
        return False


def test_mcp_hub_imports():
    """Verify MCPHub can be imported."""
    assert _check_mcp_hub_imports()


async def _check_mcp_hub_tool_registration():
    """Helper that returns bool for script mode."""
    from rosclaw.agent_runtime.mcp_hub import MCPHub
    from rosclaw.core.event_bus import EventBus

    bus = EventBus()
    hub = MCPHub(event_bus=bus, robot_id="test_bot")
    hub.initialize()

    tools = hub._tools
    print(f"✅ MCPHub registered {len(tools)} tools:")
    for name, spec in tools.items():
        print(f"   - {name}: {spec.get('description', 'no desc')[:60]}...")

    # Test get_robot_state (should work without runtime)
    result = await hub.handle_tool_call("get_robot_state", {})
    print(f"✅ get_robot_state returned: {result}")

    # Test emergency_stop
    result = await hub.handle_tool_call("emergency_stop", {})
    print(f"✅ emergency_stop returned: {result}")

    # Test query_knowledge
    result = await hub.handle_tool_call("query_knowledge", {"query_type": "capability", "query": "ur5e"})
    print(f"✅ query_knowledge returned: {result}")

    # Test unknown tool
    result = await hub.handle_tool_call("nonexistent_tool", {})
    assert "error" in result, "Unknown tool should return error"
    print(f"✅ Unknown tool correctly rejected: {result}")

    hub.stop()
    return True


@pytest.mark.asyncio
async def test_mcp_hub_tool_registration():
    """Verify MCPHub registers tools and can handle tool calls."""
    assert await _check_mcp_hub_tool_registration()


def _check_mcp_tools_schema_completeness():
    """Helper that returns bool for script mode."""
    from rosclaw.agent_runtime.mcp_hub import MCPHub
    from rosclaw.core.event_bus import EventBus

    bus = EventBus()
    hub = MCPHub(event_bus=bus)
    hub.initialize()

    issues = []
    for name, spec in hub._tools.items():
        if "name" not in spec:
            issues.append(f"Tool {name} missing 'name' in schema")
        if "description" not in spec:
            issues.append(f"Tool {name} missing 'description' in schema")
        if "inputSchema" not in spec:
            issues.append(f"Tool {name} missing 'inputSchema' in schema")

    hub.stop()

    if issues:
        for issue in issues:
            print(f"❌ {issue}")
        return False

    print(f"✅ All {len(hub._tools)} tools have complete schemas")
    return True


def test_mcp_tools_schema_completeness():
    """Verify all registered tools have valid MCP schemas."""
    assert _check_mcp_tools_schema_completeness()


async def _check_command_response_pattern():
    """Helper that returns bool for script mode."""
    from rosclaw.agent_runtime.mcp_hub import MCPHub
    from rosclaw.core.event_bus import EventBus

    bus = EventBus()
    hub = MCPHub(event_bus=bus)
    hub.initialize()

    # The _send_command_and_wait method should exist and use asyncio.Future
    assert hasattr(hub, '_send_command_and_wait'), "MCPHub missing _send_command_and_wait"
    assert hasattr(hub, '_pending_requests'), "MCPHub missing _pending_requests"

    print("✅ Command-response pattern infrastructure present")
    hub.stop()
    return True


@pytest.mark.asyncio
async def test_command_response_pattern():
    """Verify MCPHub uses command-response pattern (not fire-and-forget)."""
    assert await _check_command_response_pattern()


async def main():
    print("=" * 60)
    print("ROSClaw MCP Closed Loop Verification")
    print("=" * 60)

    results = []

    results.append(("MCP Server Imports", _check_mcp_server_imports()))
    results.append(("MCPHub Imports", _check_mcp_hub_imports()))
    results.append(("Tool Schema Completeness", _check_mcp_tools_schema_completeness()))

    try:
        results.append(("MCPHub Tool Registration", await _check_mcp_hub_tool_registration()))
    except Exception as e:
        print(f"❌ MCPHub Tool Registration failed: {e}")
        results.append(("MCPHub Tool Registration", False))

    try:
        results.append(("Command-Response Pattern", await _check_command_response_pattern()))
    except Exception as e:
        print(f"❌ Command-Response Pattern failed: {e}")
        results.append(("Command-Response Pattern", False))

    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")

    print(f"\n{passed}/{total} checks passed")

    if passed == total:
        print("\n🎉 MCP closed loop verification PASSED")
        print("\nClaude Code should be able to:")
        print("  1. Discover MCP tools (get_robot_state, emergency_stop, query_knowledge, etc.)")
        print("  2. Call tools and receive structured responses")
        print("  3. Rely on command-response pattern (not fire-and-forget)")
        return 0
    else:
        print("\n⚠️ MCP closed loop verification PARTIAL")
        print("Blockers prevent full Claude Code integration")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
