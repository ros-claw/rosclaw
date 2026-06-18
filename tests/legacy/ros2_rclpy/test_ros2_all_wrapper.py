"""Unified subprocess wrapper for all legacy rclpy-based ROS2 integration tests.

These legacy tests require rclpy C extensions which cannot load correctly when
pytest's module assertion-rewrite causes module reloads. This wrapper runs all
legacy ROS2 test files in isolated subprocesses with correct env vars.

Covered test files:
- tests/legacy/ros2_rclpy/test_ros2_driver_ros2.py (23 tests)
- tests/legacy/ros2_rclpy/test_ur5_server_ros2.py (52 tests)
- tests/legacy/ros2_rclpy/test_mcp_drivers_init_ros2.py (8 tests)
- tests/legacy/ros2_rclpy/test_ros2_e2e_wrapper.py (1 wrapper -> E2E tests)
"""

import subprocess
import sys

import pytest

from tests._ros2_env import build_ros2_env, repo_root, ros2_available

# Legacy test files to run in subprocess
_ROS2_TEST_FILES = [
    "tests/legacy/ros2_rclpy/test_ros2_driver_ros2.py",
    "tests/legacy/ros2_rclpy/test_ur5_server_ros2.py",
    "tests/legacy/ros2_rclpy/test_mcp_drivers_init_ros2.py",
]


@pytest.mark.legacy_rclpy
@pytest.mark.skipif(
    not ros2_available(),
    reason="ROS2 environment not available",
)
def test_ros2_driver_unit():
    """Run test_ros2_driver_ros2.py in subprocess."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/legacy/ros2_rclpy/test_ros2_driver_ros2.py", "-q"],
        capture_output=True,
        text=True,
        cwd=repo_root(),
        env=build_ros2_env(),
        timeout=300,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    assert result.returncode == 0, f"ROS2 driver tests failed:\n{result.stdout}\n{result.stderr}"


@pytest.mark.legacy_rclpy
@pytest.mark.skipif(
    not ros2_available(),
    reason="ROS2 environment not available",
)
def test_ros2_ur5_server_unit():
    """Run test_ur5_server_ros2.py in subprocess."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/legacy/ros2_rclpy/test_ur5_server_ros2.py", "-q"],
        capture_output=True,
        text=True,
        cwd=repo_root(),
        env=build_ros2_env(),
        timeout=600,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    assert result.returncode == 0, f"UR5Server ROS2 tests failed:\n{result.stdout}\n{result.stderr}"
    assert "52 passed" in result.stdout or "passed" in result.stdout


@pytest.mark.legacy_rclpy
@pytest.mark.skipif(
    not ros2_available(),
    reason="ROS2 environment not available",
)
def test_ros2_mcp_drivers_init():
    """Run test_mcp_drivers_init_ros2.py in subprocess."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/legacy/ros2_rclpy/test_mcp_drivers_init_ros2.py", "-q"],
        capture_output=True,
        text=True,
        cwd=repo_root(),
        env=build_ros2_env(),
        timeout=300,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    assert result.returncode == 0, f"MCP drivers init tests failed:\n{result.stdout}\n{result.stderr}"
