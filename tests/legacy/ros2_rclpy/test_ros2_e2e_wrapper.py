"""Pytest wrapper for ROS2 E2E integration tests.

Runs scripts/test_ros2_e2e.py in a subprocess to avoid pytest module
reload conflicts with rclpy C extensions.
"""

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
def test_ros2_e2e_closed_loop():
    """Run the full ROS2 closed-loop integration test suite."""
    env = build_ros2_env()
    result = subprocess.run(
        [sys.executable, "scripts/legacy/ros2_rclpy/test_ros2_e2e.py"],
        capture_output=True,
        text=True,
        cwd=repo_root(),
        env={**dict(os.environ), **env},
        timeout=300,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    assert result.returncode == 0, f"ROS2 E2E tests failed:\n{result.stdout}\n{result.stderr}"
    assert "16 passed, 0 failed" in result.stdout or "passed" in result.stdout
