"""Turtlesim integration test against a live rosbridge endpoint.

This test requires a running ROS 2 turtlesim + rosbridge. Use the provided
Docker Compose stack to start one:

    docker compose -f docker-compose.ros-test.yml up --build -d

Then run:

    pytest tests/connectors/ros/test_turtlesim_integration.py -v

The test is marked with ``@pytest.mark.integration`` so it is skipped by
default in the regular unit-test suite.
"""

from __future__ import annotations

import os

import pytest

from rosclaw.connectors.ros.compiler import CapabilityManifestCompiler
from rosclaw.connectors.ros.discovery import RosGraphDiscovery
from rosclaw.connectors.ros.provider import RosCapabilityProvider
from rosclaw.connectors.ros.transport import RosbridgeEndpoint, RosbridgeTransport
from rosclaw.provider.core.manifest import ProviderManifest
from rosclaw.provider.core.request import ProviderRequest

ENDPOINT_URL = os.environ.get("ROSCLAW_ROS_TEST_ENDPOINT", "ws://127.0.0.1:9090")
ROBOT_ID = "turtlesim"


pytestmark = pytest.mark.integration


def _is_reachable() -> bool:
    try:
        ep = RosbridgeEndpoint.from_url(ENDPOINT_URL)
        transport = RosbridgeTransport(endpoint=ep, max_retries=0)
        result = transport.call_service("/rosapi/topics", {})
        transport.close()
        return result.ok
    except Exception:
        return False


@pytest.fixture(scope="module")
def transport():
    ep = RosbridgeEndpoint.from_url(ENDPOINT_URL)
    t = RosbridgeTransport(endpoint=ep)
    yield t
    t.close()


@pytest.mark.skipif(not _is_reachable(), reason="rosbridge turtlesim not reachable")
def test_ping_reaches_rosbridge(transport: RosbridgeTransport):
    result = transport.call_service("/rosapi/topics", {})
    assert result.ok, result.error
    assert isinstance(result.data, dict)


@pytest.mark.skipif(not _is_reachable(), reason="rosbridge turtlesim not reachable")
def test_discovery_finds_turtlesim_topics(transport: RosbridgeTransport):
    discovery = RosGraphDiscovery(transport)
    snapshot = discovery.discover()
    topic_names = {t.name for t in snapshot.topics}
    assert "/turtle1/pose" in topic_names
    assert "/turtle1/cmd_vel" in topic_names


def _find_velocity_capability(capabilities: set[str]) -> str | None:
    """Return any velocity-command capability ID, accounting for multi-turtle IDs."""
    for cap_id in capabilities:
        if cap_id.startswith("turtlesim.base.velocity_command"):
            return cap_id
    return None


@pytest.mark.skipif(not _is_reachable(), reason="rosbridge turtlesim not reachable")
def test_compile_manifest_for_turtlesim(transport: RosbridgeTransport):
    discovery = RosGraphDiscovery(transport)
    snapshot = discovery.discover()
    manifest = CapabilityManifestCompiler(robot_id=ROBOT_ID).compile(snapshot)
    cap_ids = {cap.id for cap in manifest.capabilities}
    assert _find_velocity_capability(cap_ids), f"No velocity capability in {cap_ids}"
    assert any(cid.startswith("turtlesim.observe.pose") for cid in cap_ids), f"No pose observation in {cap_ids}"


@pytest.mark.skipif(not _is_reachable(), reason="rosbridge turtlesim not reachable")
def test_provider_executes_safe_velocity_command():
    manifest = ProviderManifest(
        name="ros_capability_provider",
        version="0.1.0",
        type="ros",
        runtime={"endpoint": ENDPOINT_URL},
        extra={"robot_id": ROBOT_ID, "dry_run": False, "auto_discover": True},
    )
    provider = RosCapabilityProvider(manifest)
    import asyncio

    asyncio.run(provider.load())

    velocity_cap = _find_velocity_capability(set(provider.capabilities))
    assert velocity_cap, f"No velocity capability found in {provider.capabilities}"

    request = ProviderRequest(
        request_id="integration_safe_velocity",
        capability=velocity_cap,
        inputs={
            "linear": {"x": 0.1, "y": 0.0, "z": 0.0},
            "angular": {"z": 0.2},
            "duration": 0.5,
        },
        context={"dry_run": False},
    )
    response = asyncio.run(provider.infer(request))
    asyncio.run(provider.unload())

    assert response.status == "ok", response.errors
    assert response.result.get("ok") is True


@pytest.mark.skipif(not _is_reachable(), reason="rosbridge turtlesim not reachable")
def test_provider_blocks_unsafe_velocity_command():
    manifest = ProviderManifest(
        name="ros_capability_provider",
        version="0.1.0",
        type="ros",
        runtime={"endpoint": ENDPOINT_URL},
        extra={"robot_id": ROBOT_ID, "dry_run": False, "auto_discover": True},
    )
    provider = RosCapabilityProvider(manifest)
    import asyncio

    asyncio.run(provider.load())

    velocity_cap = _find_velocity_capability(set(provider.capabilities))
    assert velocity_cap, f"No velocity capability found in {provider.capabilities}"

    request = ProviderRequest(
        request_id="integration_unsafe_velocity",
        capability=velocity_cap,
        inputs={
            "linear": {"x": 1.0, "y": 0.0, "z": 0.0},
            "angular": {"z": 1.0},
            "duration": 2.0,
        },
        context={"dry_run": False},
    )
    response = asyncio.run(provider.infer(request))
    asyncio.run(provider.unload())

    assert response.status == "blocked"
    assert response.errors


def _get_pose_msg_type(transport: RosbridgeTransport) -> str:
    """Return the actual message type for /turtle1/pose from the live graph.

    ROS1 uses ``turtlesim/Pose``; ROS2 uses ``turtlesim/msg/Pose``. Querying
    rosapi lets the same test run against either distribution.
    """
    result = transport.call_service("/rosapi/topics", {})
    if result.ok and isinstance(result.data, dict):
        # /rosapi/topics nests the arrays under "values".
        values = result.data.get("values", {})
        topics = values.get("topics", [])
        types = values.get("types", [])
        for topic, msg_type in zip(topics, types, strict=False):
            if topic == "/turtle1/pose":
                return msg_type
    # Fallback to ROS2 convention if the query fails.
    return "turtlesim/msg/Pose"


@pytest.mark.skipif(not _is_reachable(), reason="rosbridge turtlesim not reachable")
def test_subscribe_once_reads_pose(transport: RosbridgeTransport):
    msg_type = _get_pose_msg_type(transport)
    result = transport.subscribe_once("/turtle1/pose", msg_type=msg_type, timeout_sec=5.0)
    assert result.ok, result.error
    assert result.data is not None
    msg = result.data
    # rosbridge wraps published messages in {op: publish, topic, msg}.
    payload = msg.get("msg", msg)
    assert "x" in payload
    assert "y" in payload
    assert "theta" in payload
