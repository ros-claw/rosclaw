"""Tests for RealSense ROS2 topic adapter (without real ROS2)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rosclaw.practice.adapters.ros2_topic_adapter import Ros2TopicAdapter
from rosclaw.practice.schemas import PracticeEventEnvelope


class FakeImage:
    encoding = "rgb8"
    height = 480
    width = 640
    data = np.zeros((480, 640, 3), dtype=np.uint8).tobytes()
    header = MagicMock()
    header.stamp = MagicMock()
    header.stamp.sec = 0
    header.stamp.nanosec = 0
    header.frame_id = "camera_color_optical_frame"


class FakeDepthImage:
    encoding = "16UC1"
    height = 480
    width = 640
    data = np.zeros((480, 640), dtype=np.uint16).tobytes()
    header = MagicMock()
    header.stamp = MagicMock()
    header.stamp.sec = 0
    header.stamp.nanosec = 0
    header.frame_id = "camera_depth_optical_frame"


class FakeCameraInfo:
    header = MagicMock()
    header.stamp = MagicMock()
    header.stamp.sec = 0
    header.stamp.nanosec = 0
    header.frame_id = "camera_color_optical_frame"
    height = 480
    width = 640
    distortion_model = "plumb_bob"
    d = [0.0] * 5
    k = [0.0] * 9
    r = [0.0] * 9
    p = [0.0] * 12
    binning_x = 1
    binning_y = 1
    roi = MagicMock()
    roi.x_offset = 0
    roi.y_offset = 0
    roi.height = 480
    roi.width = 640
    roi.do_rectify = False


class FakeImu:
    orientation = MagicMock(x=0.0, y=0.0, z=0.0, w=1.0)
    angular_velocity = MagicMock(x=0.0, y=0.0, z=0.0)
    linear_acceleration = MagicMock(x=0.0, y=0.0, z=9.81)
    header = MagicMock()
    header.stamp = MagicMock()
    header.stamp.sec = 0
    header.stamp.nanosec = 0
    header.frame_id = "imu_link"


@pytest.fixture
def tmp_session(tmp_path: Path):
    return MagicMock(practice_id="prac_test", session_dir=str(tmp_path), robot_id="r1")


def test_ros2_topic_adapter_no_ros2_warns(tmp_path: Path):
    """When rclpy is unavailable, adapter should emit a warning event."""
    with patch("rosclaw.practice.adapters.ros2_topic_adapter._HAS_ROS2", False):
        adapter = Ros2TopicAdapter(
            robot_id="r1",
            ros2_topics=[{"topic": "/camera/color/image_raw", "msg_type": MagicMock}],
            output_root=str(tmp_path),
        )
        adapter.start(MagicMock(practice_id="prac_test", session_dir=str(tmp_path)))
        events = list(adapter.poll())
        adapter.stop()
    assert len(events) == 1
    assert events[0].source == "system"
    assert events[0].event_type == "system.camera_missing"


def test_rgb_frame_saved(tmp_session):
    """RGB image should be saved as JPEG and referenced."""
    tmp_path = Path(tmp_session.session_dir)
    with patch("rosclaw.practice.adapters.ros2_topic_adapter._HAS_ROS2", True), \
         patch("rosclaw.practice.adapters.ros2_topic_adapter.Image", FakeImage), \
         patch("rosclaw.practice.adapters.ros2_topic_adapter.CameraInfo", FakeCameraInfo):
        adapter = Ros2TopicAdapter(
            robot_id="r1",
            ros2_topics=[{"topic": "/camera/color/image_raw", "msg_type": FakeImage}],
            sample_camera_hz=1000.0,  # always poll
            output_root=str(tmp_path),
        )
        # bypass start() to avoid real rclpy dependency; set internal state directly
        adapter._practice_id = tmp_session.practice_id
        adapter._session_dir = Path(tmp_session.session_dir)
        adapter._running = True
        adapter._last_camera_poll = 0  # force poll
        adapter._node = MagicMock()
        adapter._node.take = MagicMock(return_value=FakeImage())
        events = list(adapter.poll())
    adapter.stop()
    assert len(events) == 1
    ev = events[0]
    assert ev.source == "camera"
    assert ev.event_type == "camera.rgb_frame"
    assert "rgb_ref" in ev.payload_ref
    rgb_path = tmp_path / ev.payload_ref["rgb_ref"]
    assert rgb_path.exists()
    assert rgb_path.suffix == ".jpg"


def test_depth_frame_saved(tmp_session):
    """Depth image should be saved as PNG16 and referenced."""
    tmp_path = Path(tmp_session.session_dir)
    with patch("rosclaw.practice.adapters.ros2_topic_adapter._HAS_ROS2", True), \
         patch("rosclaw.practice.adapters.ros2_topic_adapter.Image", FakeDepthImage), \
         patch("rosclaw.practice.adapters.ros2_topic_adapter.CameraInfo", FakeCameraInfo):
        adapter = Ros2TopicAdapter(
            robot_id="r1",
            ros2_topics=[{"topic": "/camera/depth/image_rect_raw", "msg_type": FakeDepthImage}],
            sample_camera_hz=1000.0,
            output_root=str(tmp_path),
        )
        adapter._practice_id = tmp_session.practice_id
        adapter._session_dir = Path(tmp_session.session_dir)
        adapter._running = True
        adapter._last_camera_poll = 0
        adapter._node = MagicMock()
        adapter._node.take = MagicMock(return_value=FakeDepthImage())
        events = list(adapter.poll())
    adapter.stop()
    assert len(events) == 1
    ev = events[0]
    assert ev.source == "camera"
    assert ev.event_type == "camera.depth_frame"
    assert "depth_ref" in ev.payload_ref
    depth_path = tmp_path / ev.payload_ref["depth_ref"]
    assert depth_path.exists()
    assert depth_path.suffix == ".png16"


def test_camera_info_saved(tmp_session):
    """CameraInfo should be saved as JSON and referenced."""
    tmp_path = Path(tmp_session.session_dir)
    with patch("rosclaw.practice.adapters.ros2_topic_adapter._HAS_ROS2", True), \
         patch("rosclaw.practice.adapters.ros2_topic_adapter.Image", FakeImage), \
         patch("rosclaw.practice.adapters.ros2_topic_adapter.CameraInfo", FakeCameraInfo):
        adapter = Ros2TopicAdapter(
            robot_id="r1",
            ros2_topics=[{"topic": "/camera/color/camera_info", "msg_type": FakeCameraInfo}],
            sample_camera_hz=1000.0,
            output_root=str(tmp_path),
        )
        adapter._practice_id = tmp_session.practice_id
        adapter._session_dir = Path(tmp_session.session_dir)
        adapter._running = True
        adapter._last_camera_poll = 0
        adapter._node = MagicMock()
        adapter._node.take = MagicMock(return_value=FakeCameraInfo())
        events = list(adapter.poll())
    adapter.stop()
    assert len(events) == 1
    ev = events[0]
    assert ev.source == "ros2"
    assert ev.event_type == "ros2.camera_info"
    assert "camera_info_ref" in ev.payload_ref
    info_path = tmp_path / ev.payload_ref["camera_info_ref"]
    assert info_path.exists()
    data = json.loads(info_path.read_text())
    assert data["width"] == 640


def test_imu_sample_appended(tmp_session):
    """IMU sample should be appended to JSONL and referenced."""
    tmp_path = Path(tmp_session.session_dir)
    with patch("rosclaw.practice.adapters.ros2_topic_adapter._HAS_ROS2", True), \
         patch("rosclaw.practice.adapters.ros2_topic_adapter.Imu", FakeImu):
        adapter = Ros2TopicAdapter(
            robot_id="r1",
            ros2_topics=[{"topic": "/camera/imu", "msg_type": FakeImu}],
            sample_imu_hz=1000.0,
            output_root=str(tmp_path),
        )
        adapter._practice_id = tmp_session.practice_id
        adapter._session_dir = Path(tmp_session.session_dir)
        adapter._running = True
        adapter._last_imu_poll = 0
        adapter._node = MagicMock()
        adapter._node.take = MagicMock(return_value=FakeImu())
        events = list(adapter.poll())
    adapter.stop()
    assert len(events) == 1
    ev = events[0]
    assert ev.source == "ros2"
    assert ev.event_type == "imu.sample"
    assert "imu_ref" in ev.payload_ref
    imu_path = tmp_path / ev.payload_ref["imu_ref"]
    assert imu_path.exists()
    lines = imu_path.read_text().strip().split("\n")
    assert len(lines) == 1
    sample = json.loads(lines[0])
    assert sample["linear_acceleration"]["z"] == 9.81


def test_missing_topic_warns_once(tmp_session):
    """Missing topic should emit a warning only once."""
    tmp_path = Path(tmp_session.session_dir)
    with patch("rosclaw.practice.adapters.ros2_topic_adapter._HAS_ROS2", True), \
         patch("rosclaw.practice.adapters.ros2_topic_adapter.Image", FakeImage), \
         patch("rosclaw.practice.adapters.ros2_topic_adapter.CameraInfo", FakeCameraInfo):
        adapter = Ros2TopicAdapter(
            robot_id="r1",
            ros2_topics=[{"topic": "/camera/color/image_raw", "msg_type": FakeImage}],
            sample_camera_hz=1000.0,
            output_root=str(tmp_path),
        )
        adapter._practice_id = tmp_session.practice_id
        adapter._session_dir = Path(tmp_session.session_dir)
        adapter._running = True
        adapter._last_camera_poll = 0
        adapter._node = MagicMock()
        adapter._node.take = MagicMock(return_value=None)
        events1 = list(adapter.poll())
        events2 = list(adapter.poll())
    adapter.stop()
    assert len(events1) == 1
    assert events1[0].event_type == "system.camera_missing"
    assert len(events2) == 0
