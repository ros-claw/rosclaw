"""Pytest wrapper for ROS2 boundary and fault injection tests."""

import os
import subprocess
import sys

import pytest

from tests._ros2_env import build_ros2_env, repo_root, ros2_available


@pytest.mark.legacy_rclpy
@pytest.mark.skipif(
    not ros2_available(),
    reason="ROS2 environment not available",
)
def test_ros2_boundary_fault():
    """Run boundary and fault injection tests in subprocess."""
    env = build_ros2_env()
    result = subprocess.run(
        [sys.executable, "scripts/legacy/ros2_rclpy/test_ros2_boundary_fault.py"],
        capture_output=True,
        text=True,
        cwd=repo_root(),
        env={**dict(os.environ), **env},
        timeout=300,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    assert result.returncode == 0, f"Boundary fault tests failed:\n{result.stdout}\n{result.stderr}"
    assert "passed" in result.stdout
