"""Docker helper utilities for ROSClaw tests and examples."""

from __future__ import annotations

import os
import subprocess


def get_container_ip(container_name: str) -> str | None:
    """Return the first Docker network IP for *container_name*.

    Returns ``None`` if the container is not running, has no IP, or if Docker is
    unavailable.
    """
    try:
        result = subprocess.run(
            [
                "docker",
                "inspect",
                "-f",
                "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
                container_name,
            ],
            capture_output=True,
            text=True,
            timeout=10.0,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    ip = result.stdout.strip().split("\n")[0]
    if not ip or "<no value>" in ip:
        return None
    return ip


def resolve_rosbridge_ip(
    default: str = "127.0.0.1",
    container_name: str | None = None,
) -> str:
    """Return the IP address to use when connecting to rosbridge.

    Resolution order:

    1. ``ROSBRIDGE_IP`` environment variable, unless its value is ``"auto"``.
    2. If ``ROSBRIDGE_IP=auto`` or no explicit IP was given, discover the
       container IP from ``ROSBRIDGE_CONTAINER_NAME`` (or *container_name*).
    3. Fall back to *default* if auto-discovery fails.
    """
    env_ip = os.environ.get("ROSBRIDGE_IP")
    if env_ip and env_ip != "auto":
        return env_ip

    candidate = container_name or os.environ.get(
        "ROSBRIDGE_CONTAINER_NAME", "rosclaw-ros1-noetic-bridge"
    )
    ip = get_container_ip(candidate)
    if ip:
        return ip

    return default


def resolve_rosbridge_endpoint(
    default: str = "ws://127.0.0.1:9090",
    container_name: str | None = None,
) -> str:
    """Return a WebSocket endpoint URL, auto-discovering the host if requested."""
    env_endpoint = os.environ.get("ROSCLAW_ROS_TEST_ENDPOINT")
    endpoint = env_endpoint or default

    env_ip = os.environ.get("ROSBRIDGE_IP")
    if env_ip and env_ip != "auto":
        return endpoint.replace("127.0.0.1", env_ip)

    ip = resolve_rosbridge_ip(default="127.0.0.1", container_name=container_name)
    return endpoint.replace("127.0.0.1", ip)
