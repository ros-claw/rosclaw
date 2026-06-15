"""Unified subprocess wrapper for all ROS2 integration tests.

ROS2 tests require rclpy C extensions which cannot load correctly when
pytest's module assertion-rewrite causes module reloads. This wrapper
runs all ROS2 test files in isolated subprocesses with correct env vars.

Covered test files:
- tests/test_ros2_driver_ros2.py (23 tests)
- tests/test_ur5_server_ros2.py (52 tests)
- tests/test_mcp_drivers_init_ros2.py (8 tests)
- tests/test_ros2_e2e_wrapper.py (1 wrapper -> 16 E2E tests)
"""

import os
import subprocess
import sys

import pytest

from tests._ros2_env import ros2_available

# ROS2 environment paths
_ROS2_PYTHONPATH = (
    "/tmp/ros2-local/opt/ros/humble/local/lib/python3.10/dist-packages"
    ":/opt/ros/humble/local/lib/python3.10/dist-packages"
)
_ROS2_LD_LIBRARY_PATH = "/tmp/ros2-local/opt/ros/humble/lib:/opt/ros/humble/lib"

# Test files to run in subprocess
_ROS2_TEST_FILES = [
    "tests/test_ros2_driver_ros2.py",
    "tests/test_ur5_server_ros2.py",
    "tests/test_mcp_drivers_init_ros2.py",
]


def _build_env():
    """Build environment dict with ROS2 paths prepended."""
    env = dict(os.environ)
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{_ROS2_PYTHONPATH}:{existing_pp}:src" if existing_pp else f"{_ROS2_PYTHONPATH}:src"
    existing_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{_ROS2_LD_LIBRARY_PATH}:{existing_ld}" if existing_ld else _ROS2_LD_LIBRARY_PATH
    return env


@pytest.mark.skipif(
    sys.version_info[:2] != (3, 10) or not ros2_available(),
    reason="Requires Python 3.10 and ROS2 environment",
)
def test_ros2_driver_unit():
    """Run test_ros2_driver_ros2.py in subprocess."""
    result = subprocess.run(
        ["/tmp/ros2-venv/bin/python", "-m", "pytest", "tests/test_ros2_driver_ros2.py", "-q"],
        capture_output=True,
        text=True,
        cwd="/home/dell/rosclaw-v1.0",
        env=_build_env(),
        timeout=300,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    assert result.returncode == 0, f"ROS2 driver tests failed:\n{result.stdout}\n{result.stderr}"


@pytest.mark.skipif(
    sys.version_info[:2] != (3, 10) or not ros2_available(),
    reason="Requires Python 3.10 and ROS2 environment",
)
def test_ros2_ur5_server_unit():
    """Run test_ur5_server_ros2.py in subprocess."""
    result = subprocess.run(
        ["/tmp/ros2-venv/bin/python", "-m", "pytest", "tests/test_ur5_server_ros2.py", "-q"],
        capture_output=True,
        text=True,
        cwd="/home/dell/rosclaw-v1.0",
        env=_build_env(),
        timeout=600,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    assert result.returncode == 0, f"UR5Server ROS2 tests failed:\n{result.stdout}\n{result.stderr}"
    assert "52 passed" in result.stdout or "passed" in result.stdout


@pytest.mark.skipif(
    sys.version_info[:2] != (3, 10) or not ros2_available(),
    reason="Requires Python 3.10 and ROS2 environment",
)
def test_ros2_mcp_drivers_init():
    """Run test_mcp_drivers_init_ros2.py in subprocess."""
    result = subprocess.run(
        ["/tmp/ros2-venv/bin/python", "-m", "pytest", "tests/test_mcp_drivers_init_ros2.py", "-q"],
        capture_output=True,
        text=True,
        cwd="/home/dell/rosclaw-v1.0",
        env=_build_env(),
        timeout=300,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    assert result.returncode == 0, f"MCP drivers init tests failed:\n{result.stdout}\n{result.stderr}"
