"""Live canonical guarded-motion chain against turtlesim over rosbridge."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rosclaw.connectors.ros.transport import RosbridgeEndpoint, RosbridgeTransport
from rosclaw.core.runtime import Runtime, RuntimeConfig
from rosclaw.daemon.client import DaemonClient
from rosclaw.daemon.server import RosclawDaemon
from rosclaw.daemon.service import DaemonControlPlane
from rosclaw.kernel import ExecutionMode
from rosclaw.mcp.adapters.runtime_client import RuntimeClient
from rosclaw.simforge.tasks.guarded_base import (
    GenericMobileBaseSimulationExecutor,
    RosbridgeMobileBaseSink,
)

ENDPOINT_URL = os.environ.get("ROSCLAW_ROS_TEST_ENDPOINT", "ws://127.0.0.1:9090")

pytestmark = pytest.mark.integration


def _is_reachable() -> bool:
    transport = RosbridgeTransport(
        endpoint=RosbridgeEndpoint.from_url(ENDPOINT_URL),
        max_retries=0,
    )
    try:
        return transport.call_service("/rosapi/topics", {}).ok
    except Exception:
        return False
    finally:
        transport.close()


def _pose_message_type(transport: RosbridgeTransport) -> str:
    result = transport.call_service("/rosapi/topics", {})
    if result.ok and isinstance(result.data, dict):
        values = result.data.get("values", {})
        if isinstance(values, dict):
            topics = values.get("topics", [])
            types = values.get("types", [])
            for topic, message_type in zip(topics, types, strict=False):
                if topic == "/turtle1/pose" and isinstance(message_type, str):
                    return message_type
    return "turtlesim/msg/Pose"


@pytest.mark.skipif(not _is_reachable(), reason="rosbridge turtlesim not reachable")
async def test_mcp_daemon_gateway_is_the_only_live_ros_command_path(tmp_path: Path) -> None:
    """A live Twist is emitted only inside the daemon-owned SHADOW executor."""

    transport = RosbridgeTransport(endpoint=RosbridgeEndpoint.from_url(ENDPOINT_URL))
    daemon_id = "daemon_simforge_ros_live"
    sink = RosbridgeMobileBaseSink(
        transport,
        daemon_owner_id=daemon_id,
        pose_message_type=_pose_message_type(transport),
    )
    runtime = Runtime(
        RuntimeConfig(
            robot_id="sim_mobile_base",
            enable_firewall=False,
            enable_memory=False,
            enable_practice=False,
            enable_skill_manager=False,
            enable_knowledge=False,
            enable_how=False,
            enable_auto=False,
            enable_provider=False,
            enable_sense=False,
            enable_event_persistence=False,
            enable_tracing=False,
        )
    )
    runtime.action_gateway.register_executor(
        "mobile_base.guarded_move",
        ExecutionMode.SHADOW,
        GenericMobileBaseSimulationExecutor(sink, daemon_instance_id=daemon_id),
    )
    daemon = RosclawDaemon(
        service=DaemonControlPlane(runtime=runtime),
        socket_path=tmp_path / "rosclawd.sock",
    )
    daemon.start()
    try:
        mcp = RuntimeClient(
            project_root=tmp_path,
            robot_id="sim_mobile_base",
            runtime_profile={},
            daemon_client=DaemonClient(socket_path=daemon.socket_path, timeout_sec=10),
        )
        result = await mcp.request_action(
            capability_id="mobile_base.guarded_move",
            arguments={"linear_x_mps": 0.2, "angular_z_radps": 0.0, "duration_sec": 0.5},
            execution_mode="SHADOW",
            body_snapshot_hash="sha256:" + "a" * 64,
            body_id="sim_mobile_base",
            action_id="mcp-daemon-live-guarded-base",
            required_evidence="TASK_VERIFIED",
            wait_timeout_sec=10,
        )
    finally:
        daemon.stop()
        sink.stop()
        transport.close()

    receipt = result["receipt"]
    assert result["state"] == "FINISHED"
    assert receipt["final_state"] == "COMPLETED"
    assert receipt["evidence_level"] == "TASK_VERIFIED"
    assert receipt["dispatch_result"]["owner"] == daemon_id
    assert receipt["verification_result"]["stop_confirmed"] is True
    assert receipt["verification_result"]["actual_displacement_m"] == pytest.approx(
        0.1,
        abs=0.05,
    )
