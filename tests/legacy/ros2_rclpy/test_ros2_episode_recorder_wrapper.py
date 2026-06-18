"""Pytest wrapper for EpisodeRecorder + ROS2Driver tests."""

import os
import subprocess
import sys

import pytest

from tests._ros2_env import build_ros2_env, repo_root, ros2_available


@pytest.mark.skipif(
    not ros2_available(),
    reason="ROS2 environment not available",
)
def test_ros2_episode_recorder():
    """Run EpisodeRecorder + ROS2Driver tests in subprocess."""
    env = build_ros2_env()
    result = subprocess.run(
        [sys.executable, "scripts/test_ros2_episode_recorder.py"],
        capture_output=True,
        text=True,
        cwd=repo_root(),
        env={**dict(os.environ), **env},
        timeout=300,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    assert result.returncode == 0, f"EpisodeRecorder tests failed:\n{result.stdout}\n{result.stderr}"
    assert "passed" in result.stdout
