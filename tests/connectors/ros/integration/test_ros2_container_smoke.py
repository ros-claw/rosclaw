"""Deployment smoke test for the ROS2 Humble + rosbridge Docker stack.

This test actually runs ``docker compose`` to bring up the stack, verifies
that rosbridge and turtlesim are reachable, and then tears the stack down.

It is marked with ``deployment`` and ``integration`` so it is excluded from
the default fast unit-test run.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

import pytest

from rosclaw.connectors.ros.cli.ros_cli import cmd_doctor_ros
from rosclaw.connectors.ros.compiler import CapabilityManifestCompiler
from rosclaw.connectors.ros.discovery import RosGraphDiscovery
from rosclaw.connectors.ros.transport import RosbridgeEndpoint, RosbridgeTransport

pytestmark = [pytest.mark.deployment, pytest.mark.integration]

COMPOSE_FILE = Path(__file__).parents[4] / "docker-compose.ros-test.yml"
STACK_READY_TIMEOUT = 180.0


def _docker_available() -> bool:
    """Return True if the ``docker compose`` CLI is usable."""
    if shutil.which("docker") is None:
        return False
    try:
        subprocess.run(
            ["docker", "compose", "version"],
            capture_output=True,
            check=True,
            timeout=10.0,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return False
    return True


def _free_port() -> int:
    """Ask the OS for an available TCP port on 127.0.0.1."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _configure_build_proxy(env: dict[str, str]) -> None:
    """Forward host proxy settings to BuildKit without persisting them.

    A proxy bound to loopback is only reachable from a build step when the
    build uses host networking.  Compose consumes these ROSClaw-specific
    variables in ``docker-compose.ros-test.yml``.
    """
    http_proxy = env.get("http_proxy") or env.get("HTTP_PROXY")
    https_proxy = env.get("https_proxy") or env.get("HTTPS_PROXY")
    no_proxy = env.get("no_proxy") or env.get("NO_PROXY")

    if http_proxy:
        env["ROSCLAW_DOCKER_BUILD_HTTP_PROXY"] = http_proxy
    if https_proxy:
        env["ROSCLAW_DOCKER_BUILD_HTTPS_PROXY"] = https_proxy
    if no_proxy:
        env["ROSCLAW_DOCKER_BUILD_NO_PROXY"] = no_proxy

    proxy_hosts = {urlparse(proxy).hostname for proxy in (http_proxy, https_proxy) if proxy}
    if proxy_hosts & {"127.0.0.1", "localhost", "::1"}:
        env["ROSCLAW_DOCKER_BUILD_NETWORK"] = "host"


def _wait_for_rosbridge(endpoint: str, timeout: float = 30.0) -> bool:
    """Poll the rosbridge /rosapi/topics service until turtlesim topics appear."""
    ep = RosbridgeEndpoint.from_url(endpoint)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        transport = RosbridgeTransport(endpoint=ep, max_retries=0)
        try:
            result = transport.call_service("/rosapi/topics", {})
            if result.ok and isinstance(result.data, dict):
                topics = result.data.get("values", {}).get("topics", [])
                if "/turtle1/cmd_vel" in topics and "/turtle1/pose" in topics:
                    return True
        except Exception:
            pass
        finally:
            transport.close()
        time.sleep(1.0)
    return False


@pytest.fixture(scope="module")
def deployed_ros_stack():
    """Bring up the Docker Compose stack on a free host port and tear it down."""
    if not COMPOSE_FILE.exists():
        pytest.fail(f"Compose file not found: {COMPOSE_FILE}")

    if not _docker_available():
        pytest.skip("docker compose is not available")

    host_port = _free_port()
    container_name = f"rosclaw-ros2-humble-bridge-smoke-{host_port}"
    project_name = f"rosclaw-smoke-{host_port}"
    endpoint = f"ws://127.0.0.1:{host_port}"

    env = os.environ.copy()
    env["ROSBRIDGE_HOST_PORT"] = str(host_port)
    env["ROSBRIDGE_CONTAINER_NAME"] = container_name
    env["COMPOSE_PROJECT_NAME"] = project_name
    _configure_build_proxy(env)

    def _run(args: list[str], **kwargs) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["docker", "compose", "-f", str(COMPOSE_FILE), *args],
            env=env,
            capture_output=True,
            text=True,
            **kwargs,
        )

    up = _run(["up", "--build", "-d", "--wait"], timeout=300.0)
    if up.returncode != 0:
        pytest.fail(f"docker compose up failed:\n{up.stdout}\n{up.stderr}")

    # --wait waits for the container healthcheck, but turtlesim needs a moment
    # to register its topics after rosbridge is already listening.
    if not _wait_for_rosbridge(endpoint, timeout=STACK_READY_TIMEOUT):
        _run(["down", "-v", "--remove-orphans"])
        pytest.fail(f"rosbridge did not become reachable at {endpoint}")

    yield endpoint

    _run(["down", "-v", "--remove-orphans"])


def test_stack_deploys_and_rosbridge_reachable(deployed_ros_stack: str) -> None:
    """rosbridge answers /rosapi/topics and turtlesim topics are present."""
    ep = RosbridgeEndpoint.from_url(deployed_ros_stack)
    transport = RosbridgeTransport(endpoint=ep, max_retries=0)
    try:
        result = transport.call_service("/rosapi/topics", {})
        assert result.ok, result.error
        assert isinstance(result.data, dict)
        topics = result.data.get("values", {}).get("topics", [])
        assert "/turtle1/cmd_vel" in topics
        assert "/turtle1/pose" in topics
    finally:
        transport.close()


def test_doctor_reports_healthy(deployed_ros_stack: str) -> None:
    """``rosclaw doctor --ros`` reports a healthy connector against the stack."""
    args = SimpleNamespace(endpoint=deployed_ros_stack)
    rc = cmd_doctor_ros(args)
    assert rc == 0, f"rosclaw doctor --ros returned {rc}"


def test_discovery_finds_turtlesim_capabilities(deployed_ros_stack: str) -> None:
    """RosGraphDiscovery sees the turtlesim command and observation topics."""
    ep = RosbridgeEndpoint.from_url(deployed_ros_stack)
    transport = RosbridgeTransport(endpoint=ep)
    try:
        discovery = RosGraphDiscovery(transport)
        snapshot = discovery.discover()
        topic_names = {t.name for t in snapshot.topics}
        assert "/turtle1/pose" in topic_names
        assert "/turtle1/cmd_vel" in topic_names
    finally:
        transport.close()


def test_compile_manifest_for_turtlesim(deployed_ros_stack: str) -> None:
    """CapabilityManifestCompiler produces the expected turtlesim capabilities."""
    ep = RosbridgeEndpoint.from_url(deployed_ros_stack)
    transport = RosbridgeTransport(endpoint=ep)
    try:
        discovery = RosGraphDiscovery(transport)
        snapshot = discovery.discover()
        manifest = CapabilityManifestCompiler(robot_id="turtlesim").compile(snapshot)
        cap_ids = {cap.id for cap in manifest.capabilities}
        assert any(cid.startswith("turtlesim.base.velocity_command") for cid in cap_ids), cap_ids
        assert any(cid.startswith("turtlesim.observe.pose") for cid in cap_ids), cap_ids
    finally:
        transport.close()
