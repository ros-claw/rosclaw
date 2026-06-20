"""Tests for ROS capability provider."""

from __future__ import annotations

import pytest

from rosclaw.connectors.ros.compiler import (
    CapabilityManifest,
    RosCapability,
    RosCapabilityRisk,
    RosInterface,
)
from rosclaw.connectors.ros.provider import RosCapabilityProvider
from rosclaw.connectors.ros.transport import MockTransport
from rosclaw.provider.core.manifest import ProviderManifest
from rosclaw.provider.core.request import ProviderRequest


def _make_manifest_provider(
    capabilities: list[RosCapability] | None = None,
    dry_run: bool = True,
) -> RosCapabilityProvider:
    manifest = ProviderManifest(
        name="ros_capability_provider",
        version="0.1.0",
        type="ros",
        runtime={"backend": "rosbridge", "endpoint": "ws://127.0.0.1:9090", "protocol": "websocket"},
        extra={"robot_id": "turtlesim", "dry_run": dry_run, "auto_discover": False},
    )
    provider = RosCapabilityProvider(manifest)
    provider._manifest = CapabilityManifest(robot_id="turtlesim", capabilities=capabilities or [])
    from rosclaw.connectors.ros.compiler import SafetyContractCompiler
    provider._contract = SafetyContractCompiler().compile(provider._manifest)
    provider.capabilities = [cap.id for cap in provider._manifest.capabilities]
    provider._transport = provider._create_transport()
    return provider


def _cmd_vel_capability() -> RosCapability:
    return RosCapability(
        id="turtlesim.base.velocity_command",
        kind="actuation",
        interface=RosInterface(ros_kind="topic", name="/turtle1/cmd_vel", msg_type="geometry_msgs/msg/Twist"),
        risk=RosCapabilityRisk(
            level="high",
            read_only=False,
            destructive=True,
            requires_sandbox=True,
            requires_runtime_guard=True,
            requires_stop_guard=True,
            max_duration_sec=1.0,
        ),
        safety={"constraints": {"linear.x": [-0.2, 0.2], "linear.y": [-0.1, 0.1], "angular.z": [-0.5, 0.5]}},
    )


def _pose_capability() -> RosCapability:
    return RosCapability(
        id="turtlesim.observe.pose",
        kind="observation",
        interface=RosInterface(ros_kind="topic", name="/turtle1/pose", msg_type="turtlesim/msg/Pose"),
        risk=RosCapabilityRisk(level="low", read_only=True, destructive=False),
    )


@pytest.mark.asyncio
async def test_safe_velocity_command_allowed():
    provider = _make_manifest_provider(capabilities=[_cmd_vel_capability()], dry_run=False)
    provider._transport = MockTransport()  # avoid real websocket
    request = ProviderRequest(
        request_id="r1",
        capability="turtlesim.base.velocity_command",
        inputs={
            "linear": {"x": 0.1, "y": 0.0, "z": 0.0},
            "angular": {"z": 0.0},
            "duration": 0.5,
        },
    )
    response = await provider.infer(request)
    assert response.status == "ok"
    assert response.result["ok"]
    assert response.result["stop_guard_triggered"]


@pytest.mark.asyncio
async def test_unsafe_velocity_command_blocked():
    provider = _make_manifest_provider(capabilities=[_cmd_vel_capability()])
    request = ProviderRequest(
        request_id="r2",
        capability="turtlesim.base.velocity_command",
        inputs={
            "linear": {"x": 1.0, "y": 0.0, "z": 0.0},
            "angular": {"z": 0.0},
            "duration": 0.5,
        },
    )
    response = await provider.infer(request)
    assert response.status == "blocked"


@pytest.mark.asyncio
async def test_velocity_without_duration_blocked():
    provider = _make_manifest_provider(capabilities=[_cmd_vel_capability()])
    request = ProviderRequest(
        request_id="r3",
        capability="turtlesim.base.velocity_command",
        inputs={"linear": {"x": 0.1, "y": 0.0, "z": 0.0}, "angular": {"z": 0.0}},
    )
    response = await provider.infer(request)
    assert response.status == "blocked"


@pytest.mark.asyncio
async def test_read_only_observation_allowed():
    provider = _make_manifest_provider(capabilities=[_pose_capability()])
    request = ProviderRequest(
        request_id="r4",
        capability="turtlesim.observe.pose",
        inputs={},
    )
    response = await provider.infer(request)
    assert response.status == "ok"


@pytest.mark.asyncio
async def test_missing_capability_raises():
    provider = _make_manifest_provider(capabilities=[])
    request = ProviderRequest(
        request_id="r5",
        capability="turtlesim.unknown",
        inputs={},
    )
    response = await provider.infer(request)
    assert response.status == "failed"


@pytest.mark.asyncio
async def test_dry_run_does_not_invoke_ros():
    provider = _make_manifest_provider(capabilities=[_cmd_vel_capability()], dry_run=True)
    request = ProviderRequest(
        request_id="r6",
        capability="turtlesim.base.velocity_command",
        inputs={
            "linear": {"x": 0.1, "y": 0.0, "z": 0.0},
            "angular": {"z": 0.0},
            "duration": 0.5,
        },
    )
    response = await provider.infer(request)
    assert response.status == "ok"
    assert response.result.get("ros_response", {}).get("dry_run")


def test_rosbridge_endpoint_from_url():
    from rosclaw.connectors.ros.transport import RosbridgeEndpoint
    ep = RosbridgeEndpoint.from_url("ws://example.com:9091")
    assert ep.scheme == "ws"
    assert ep.host == "example.com"
    assert ep.port == 9091
    assert ep.url == "ws://example.com:9091"
