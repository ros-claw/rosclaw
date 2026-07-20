"""Tests for ROS MCP tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rosclaw.connectors.ros.compiler import (
    CapabilityManifest,
    RosCapability,
    RosCapabilityRisk,
    RosInterface,
    SafetyContractCompiler,
)
from rosclaw.connectors.ros.mcp import register_ros_tools
from rosclaw.connectors.ros.provider import RosCapabilityProvider
from rosclaw.provider.core.manifest import ProviderManifest


@dataclass
class FakeMCP:
    """Fake MCP server that records decorated tool functions."""

    tools: dict[str, Any] = field(default_factory=dict)

    def tool(self, description: str = ""):
        def decorator(func):
            self.tools[func.__name__] = func
            func.description = description
            return func

        return decorator


def test_register_ros_tools_exposes_safe_tools_only():
    mcp = FakeMCP()
    register_ros_tools(mcp)
    names = set(mcp.tools.keys())
    assert "ros_ping" in names
    assert "ros_discover" in names
    assert "ros_compile_manifest" in names
    assert "ros_list_capabilities" in names
    assert "ros_inspect_capability" in names
    assert "ros_validate_capability" in names
    assert "ros_execute_capability" in names
    assert "ros_emergency_stop" in names
    # Dangerous primitives are NOT exposed.
    assert "ros_publish_once" not in names
    assert "ros_call_any_service" not in names


def test_ros_ping_returns_structured_error_without_server():
    mcp = FakeMCP()
    register_ros_tools(mcp)
    result = mcp.tools["ros_ping"](endpoint="ws://127.0.0.1:9998")
    assert result["ok"] is False
    assert "endpoint" in result


def test_ros_validate_capability_uses_runtime_registry():
    # Build a fake runtime with provider registry.
    manifest = ProviderManifest(
        name="ros_capability_provider",
        version="0.1.0",
        type="ros",
        runtime={"endpoint": "ws://127.0.0.1:9090"},
        extra={"robot_id": "turtlesim", "dry_run": True, "auto_discover": False},
    )
    provider = RosCapabilityProvider(manifest)
    cap = RosCapability(
        id="turtlesim.base.velocity_command",
        kind="actuation",
        interface=RosInterface(
            ros_kind="topic", name="/turtle1/cmd_vel", msg_type="geometry_msgs/msg/Twist"
        ),
        risk=RosCapabilityRisk(
            level="high",
            read_only=False,
            destructive=True,
            requires_sandbox=True,
            requires_runtime_guard=True,
            requires_stop_guard=True,
            max_duration_sec=1.0,
        ),
        safety={"constraints": {"linear.x": [-0.2, 0.2]}},
    )
    provider._manifest = CapabilityManifest(robot_id="turtlesim", capabilities=[cap])
    provider._contract = SafetyContractCompiler().compile(provider._manifest)

    class FakeRegistry:
        def get(self, name):
            if name == "ros_capability_provider":
                return provider
            raise KeyError(name)

    class FakeRuntime:
        provider_registry = FakeRegistry()

    mcp = FakeMCP()
    register_ros_tools(mcp, runtime=FakeRuntime())

    result = mcp.tools["ros_validate_capability"](
        capability_id="turtlesim.base.velocity_command",
        args={"linear": {"x": 0.1}, "duration": 0.5},
    )
    assert result["ok"] is True
    assert result["decision"] == "ALLOW"

    result = mcp.tools["ros_validate_capability"](
        capability_id="turtlesim.base.velocity_command",
        args={"linear": {"x": 1.0}, "duration": 0.5},
    )
    assert result["ok"] is False
    assert result["decision"] == "BLOCK"


def test_ros_emergency_stop_returns_runtime_evidence():
    class FakeDaemon:
        def emergency_stop(self, reason, *, source):
            assert reason == "MCP emergency stop for turtlesim"
            assert source == "ros_mcp_tools"
            return {
                "request_dispatched": True,
                "driver_acknowledged": True,
                "physical_stop_observed": False,
                "stopped": False,
                "final_status": "ACKNOWLEDGED",
                "trust_level": "UNVERIFIED",
            }

    mcp = FakeMCP()
    register_ros_tools(mcp, daemon_client=FakeDaemon())

    result = mcp.tools["ros_emergency_stop"](robot_id="turtlesim")

    assert result["ok"] is False
    assert result["request_dispatched"] is True
    assert result["driver_acknowledged"] is True
    assert result["stopped"] is False
    assert result["trust_level"] == "UNVERIFIED"


def test_ros_emergency_stop_never_uses_attached_runtime():
    class RuntimeTrap:
        def request_emergency_stop(self, *_args, **_kwargs):
            raise AssertionError("legacy ROS MCP bypassed rosclawd")

    class Daemon:
        def emergency_stop(self, reason, *, source):
            return {
                "reason": reason,
                "source": source,
                "request_dispatched": False,
                "stopped": False,
            }

    mcp = FakeMCP()
    register_ros_tools(
        mcp,
        runtime=RuntimeTrap(),
        daemon_client=Daemon(),
    )

    result = mcp.tools["ros_emergency_stop"](robot_id="turtlesim")

    assert result["request_dispatched"] is False
    assert result["stopped"] is False


def test_ros_emergency_stop_requires_physical_observation_for_success():
    class Daemon:
        def emergency_stop(self, _reason, *, source):
            assert source == "ros_mcp_tools"
            return {
                "request_dispatched": True,
                "driver_acknowledged": True,
                "physical_stop_observed": False,
                "stopped": True,
            }

    mcp = FakeMCP()
    register_ros_tools(mcp, daemon_client=Daemon())

    result = mcp.tools["ros_emergency_stop"](robot_id="turtlesim")

    assert result["ok"] is False
