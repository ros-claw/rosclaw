"""Tests for RealSense MCP runtime tools (graceful degradation when ROS2 unavailable)."""
from __future__ import annotations

import pytest

from rosclaw.mcp.servers import run_tool, CAMERAS


def test_run_tool_get_camera_status() -> None:
    result = run_tool("get_camera_status", camera_key="realsense_d405")
    assert result["camera"] == "Intel RealSense D405"
    assert "online" in result
    assert "topics" in result
    assert "driver_loaded" in result


def test_run_tool_list_camera_topics() -> None:
    result = run_tool("list_camera_topics")
    assert "topics" in result
    assert "typed" in result


def test_run_tool_capture_rgb_frame_no_ros2() -> None:
    result = run_tool("capture_rgb_frame", output_path="/tmp/test_rgb.jpg")
    # When rclpy is unavailable, the tool should return a graceful error
    assert "error" in result
    assert result.get("path") is None


def test_run_tool_capture_depth_frame_no_ros2() -> None:
    result = run_tool("capture_depth_frame", output_path="/tmp/test_depth.png")
    assert "error" in result
    assert result.get("path") is None


def test_run_tool_capture_rgbd_pair_no_ros2() -> None:
    result = run_tool("capture_rgbd_pair", output_dir="/tmp")
    assert "errors" in result
    assert result["rgb_path"] is None
    assert result["depth_path"] is None


def test_run_tool_check_depth_validity_no_ros2() -> None:
    result = run_tool("check_depth_validity")
    assert "error" in result
    assert result.get("valid") is False


def test_run_tool_get_imu_sample_no_ros2() -> None:
    result = run_tool("get_imu_sample", camera_key="realsense_d435i")
    assert "error" in result


def test_run_tool_capture_pointcloud_snapshot_no_ros2() -> None:
    result = run_tool("capture_pointcloud_snapshot", output_path="/tmp/test.pcd")
    assert "error" in result
    assert result.get("path") is None


def test_run_tool_unknown() -> None:
    with pytest.raises(ValueError):
        run_tool("unknown_tool")
