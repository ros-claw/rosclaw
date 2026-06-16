"""Integration test: verify ROSClaw MCP server over stdio (real Claude Code path).

This test spawns the minimal MCP server as a subprocess and communicates
via stdin/stdout, exactly as Claude Code would.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def server_path():
    return str(Path(__file__).parent.parent / "src" / "rosclaw" / "mcp" / "minimal_server.py")


def test_mcp_server_starts_and_lists_tools(server_path):
    """Verify the MCP server starts and returns a valid tool list."""
    proc = subprocess.Popen(
        [sys.executable, server_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**dict(subprocess.os.environ), "PYTHONPATH": str(Path(__file__).parent.parent / "src")},
    )

    try:
        # Send JSON-RPC initialize request (simplified MCP handshake)
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0"},
            },
        }
        proc.stdin.write(json.dumps(init_request) + "\n")
        proc.stdin.flush()

        # Read initialize response
        response_line = proc.stdout.readline()
        response = json.loads(response_line)
        assert "result" in response or "error" not in response, f"Initialize failed: {response}"
        print(f"✅ Initialize response: {response.get('result', {}).get('serverInfo', {})}")

        # Send initialized notification
        initialized = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        proc.stdin.write(json.dumps(initialized) + "\n")
        proc.stdin.flush()

        # List tools
        tools_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        }
        proc.stdin.write(json.dumps(tools_request) + "\n")
        proc.stdin.flush()

        tools_response_line = proc.stdout.readline()
        tools_response = json.loads(tools_response_line)
        assert "result" in tools_response, f"tools/list failed: {tools_response}"
        tools = tools_response["result"]["tools"]
        assert len(tools) > 0, "No tools registered"
        tool_names = {t["name"] for t in tools}
        print(f"✅ Discovered {len(tools)} tools: {tool_names}")

        # Verify essential tools exist
        assert "get_robot_state" in tool_names
        assert "emergency_stop" in tool_names
        assert "system.list_robots" in tool_names
        assert "system.get_version" in tool_names

        # Call get_robot_state
        call_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "get_robot_state", "arguments": {}},
        }
        proc.stdin.write(json.dumps(call_request) + "\n")
        proc.stdin.flush()

        call_response_line = proc.stdout.readline()
        call_response = json.loads(call_response_line)
        assert "result" in call_response, f"tools/call failed: {call_response}"
        content = call_response["result"]["content"]
        assert len(content) > 0
        result_text = content[0]["text"]
        result_json = json.loads(result_text)
        assert result_json["status"] == "ok"
        print(f"✅ get_robot_state returned: {result_json}")

        # Call system.get_version
        version_request = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "system.get_version", "arguments": {}},
        }
        proc.stdin.write(json.dumps(version_request) + "\n")
        proc.stdin.flush()

        version_response_line = proc.stdout.readline()
        version_response = json.loads(version_response_line)
        assert "result" in version_response
        version_text = version_response["result"]["content"][0]["text"]
        version_json = json.loads(version_text)
        assert version_json["name"] == "rosclaw"
        assert version_json["status"] == "ready"
        print(f"✅ system.get_version returned: {version_json}")

    finally:
        proc.stdin.close()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

        stderr = proc.stderr.read()
        if stderr:
            print(f"[Server stderr]: {stderr[:500]}")
        proc.stdout.close()
        proc.stderr.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
