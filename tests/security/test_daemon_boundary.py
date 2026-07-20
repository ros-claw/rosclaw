"""Architecture checks for the agent-to-rosclawd privilege boundary."""

from __future__ import annotations

import ast
from pathlib import Path

from rosclaw.agent.tool_catalog import P0_AGENT_MCP_TOOLS

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_agent_tool_surface_has_no_raw_southbound_primitive() -> None:
    forbidden = {
        "serial_write",
        "raw_modbus_write",
        "publish_any_topic",
        "call_any_service",
        "register_driver",
        "register_executor",
    }

    assert forbidden.isdisjoint(P0_AGENT_MCP_TOOLS)


def test_p0_mcp_adapters_do_not_import_hardware_drivers() -> None:
    adapters = PROJECT_ROOT / "src" / "rosclaw" / "mcp" / "adapters"
    forbidden_prefixes = (
        "serial",
        "rclpy",
        "rosclaw.mcp_drivers",
        "rosclaw.body.rh56.transport",
        "rosclaw.connectors.ros.transport",
    )
    violations: list[tuple[str, str]] = []

    for path in adapters.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            for module in modules:
                if module.startswith(forbidden_prefixes):
                    violations.append((path.name, module))

    assert violations == []


def test_physical_mcp_adapter_names_daemon_not_runtime() -> None:
    safety_source = (
        PROJECT_ROOT / "src" / "rosclaw" / "mcp" / "adapters" / "safety_client.py"
    ).read_text(encoding="utf-8")

    assert "request_emergency_stop" not in safety_source
    assert "Daemon" in safety_source or "daemon" in safety_source


def test_legacy_agent_runtime_mcp_physical_handlers_also_use_daemon() -> None:
    source = (PROJECT_ROOT / "src" / "rosclaw" / "agent_runtime" / "mcp_hub.py").read_text(
        encoding="utf-8"
    )

    assert "self.runtime.submit_action" not in source
    assert "self.runtime.request_emergency_stop" not in source
    assert "self._daemon_client.request_action" in source
    assert "self._daemon_client.emergency_stop" in source


def test_ros_connector_mcp_emergency_stop_also_uses_daemon() -> None:
    source = (
        PROJECT_ROOT / "src" / "rosclaw" / "connectors" / "ros" / "mcp" / "tools.py"
    ).read_text(encoding="utf-8")

    assert "runtime.request_emergency_stop" not in source
    assert "daemon.emergency_stop" in source


def test_legacy_ur5_mcp_owns_no_ros_command_primitive() -> None:
    source = (PROJECT_ROOT / "src" / "rosclaw" / "mcp" / "ur5_server.py").read_text(
        encoding="utf-8"
    )

    assert "create_publisher" not in source
    assert "ActionClient(" not in source
    assert ".publish(" not in source
    assert "self._daemon_client.emergency_stop" in source


def test_ros_connector_cli_has_no_direct_emergency_publish_fallback() -> None:
    source = (
        PROJECT_ROOT / "src" / "rosclaw" / "connectors" / "ros" / "cli" / "ros_cli.py"
    ).read_text(encoding="utf-8")

    assert "transport.publish" not in source
    assert "daemon.emergency_stop" in source
