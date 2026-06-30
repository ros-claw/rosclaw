"""ROS2 topic adapter for RealSense camera and IMU data.

Subscribes to configurable ROS2 topics (Image, CameraInfo, Imu) and emits
PracticeEventEnvelope events with file-backed payload references.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.practice.adapters.base import SourceAdapter, SourceHealth
from rosclaw.practice.schemas import PracticeEventEnvelope
from rosclaw.dashboard.client import record_realsense_frame_async, set_realsense_online

logger = logging.getLogger("rosclaw.practice.adapters.ros2_topic")

# Optional ROS2 imports — degrade gracefully if not available
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image, Imu

    _HAS_ROS2 = True
except Exception as _ros2_err:  # noqa: BLE001
    _HAS_ROS2 = False
    Node = Any  # type: ignore[misc,assignment]
    CameraInfo = Any  # type: ignore[misc,assignment]
    Image = Any  # type: ignore[misc,assignment]
    Imu = Any  # type: ignore[misc,assignment]
    qos_profile_sensor_data = None  # type: ignore[misc,assignment]


def _camera_id_from_topic(topic: str) -> str:
    """Derive a camera_id from a topic string, e.g. /camera/d405/color/image_raw -> d405."""
    parts = topic.strip("/").split("/")
    # Heuristic: /camera/<camera_id>/... -> camera_id, otherwise first token.
    if len(parts) >= 2 and parts[0] == "camera":
        return parts[1]
    return parts[0] if parts else "camera"


def _topic_suffix(topic: str) -> str:
    """Return the last meaningful token, e.g. image_raw -> image_raw."""
    return topic.strip("/").split("/")[-1]


class _RosNode(Node):  # type: ignore[misc]
    """Internal rclpy node that buffers latest messages per topic."""

    def __init__(self, topics: list[dict[str, Any]]):
        super().__init__("rosclaw_practice_ros2_adapter")
        self._lock = threading.Lock()
        self._latest: dict[str, Any] = {}
        self._subs: list[Any] = []
        for t in topics:
            topic = t["topic"]
            msg_type = t["msg_type"]
            qos = qos_profile_sensor_data if msg_type is Imu else 1
            self._subs.append(
                self.create_subscription(msg_type, topic, self._make_cb(topic), qos)
            )

    def _make_cb(self, topic: str):
        def cb(msg):
            with self._lock:
                self._latest[topic] = msg
        return cb

    def latest(self, topic: str) -> Any | None:
        with self._lock:
            return self._latest.get(topic)

    def take(self, topic: str) -> Any | None:
        with self._lock:
            return self._latest.pop(topic, None)


class Ros2TopicAdapter(SourceAdapter):
    """Adapter that captures RealSense ROS2 topics and emits practice events."""

    source_name = "ros2"

    def __init__(
        self,
        robot_id: str,
        ros2_topics: list[dict[str, Any]] | None = None,
        sample_camera_hz: float = 5.0,
        sample_imu_hz: float = 50.0,
        output_root: str | None = None,
    ):
        self._robot_id = robot_id
        self._sample_camera_hz = sample_camera_hz
        self._sample_imu_hz = sample_imu_hz
        self._output_root = output_root
        self._ros2_topics = ros2_topics or []
        self._practice_id: str | None = None
        self._session_dir: Path | None = None
        self._running = False
        self._node: _RosNode | None = None
        self._thread: threading.Thread | None = None
        self._last_camera_poll: float = 0.0
        self._last_imu_poll: float = 0.0
        self._camera_missing_warned: set[str] = set()
        self._imu_missing_warned: set[str] = set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, session: Any) -> None:
        self._practice_id = getattr(session, "practice_id", None)
        self._session_dir = getattr(session, "session_dir", None)
        self._running = True
        if not _HAS_ROS2:
            logger.warning("rclpy not available; ROS2 adapter will emit warning events only.")
            return
        if not self._ros2_topics:
            logger.warning("No ROS2 topics configured.")
            return
        try:
            import rclpy
            if not rclpy.ok():
                rclpy.init(args=None)
            self._node = _RosNode(self._ros2_topics)
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
        except Exception as e:
            logger.error("Failed to start ROS2 node: %s", e)
            self._running = False

    def stop(self) -> None:
        self._running = False
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception as e:
                logger.warning("Error destroying ROS2 node: %s", e)
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        # Do NOT shutdown rclpy here — other nodes may be running in the process.

    def health(self) -> SourceHealth:
        healthy = self._running and (_HAS_ROS2 or not self._ros2_topics)
        msg = "ok" if healthy else "ROS2 unavailable or not running"
        return SourceHealth(source=self.source_name, healthy=healthy, message=msg)

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    def poll(self) -> Iterable[PracticeEventEnvelope]:
        if not self._running or self._practice_id is None:
            return

        if not _HAS_ROS2:
            # Only emit the warning once per adapter lifetime
            if not getattr(self, '_warned_missing_ros2', False):
                self._warned_missing_ros2 = True
                yield PracticeEventEnvelope(
                    practice_id=self._practice_id,
                    robot_id=self._robot_id,
                    source="system",
                    event_type="system.camera_missing",
                    timestamp_ns=time.monotonic_ns(),
                    payload={"reason": "rclpy or sensor_msgs not available"},
                )
            return

        now = time.monotonic()
        camera_dt = 1.0 / self._sample_camera_hz if self._sample_camera_hz > 0 else float("inf")
        imu_dt = 1.0 / self._sample_imu_hz if self._sample_imu_hz > 0 else float("inf")

        if now - self._last_camera_poll >= camera_dt:
            self._last_camera_poll = now
            yield from self._poll_camera_topics()

        if now - self._last_imu_poll >= imu_dt:
            self._last_imu_poll = now
            yield from self._poll_imu_topics()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _spin(self) -> None:
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            executor = SingleThreadedExecutor()
            executor.add_node(self._node)
            while self._running and rclpy.ok():
                executor.spin_once(timeout_sec=0.1)
        except Exception as e:
            logger.error("ROS2 spin error: %s", e)

    def _poll_camera_topics(self) -> Iterable[PracticeEventEnvelope]:
        for spec in self._ros2_topics:
            topic = spec["topic"]
            msg_type = spec.get("msg_type")
            if msg_type not in (Image, CameraInfo):
                continue
            msg = self._node.take(topic) if self._node else None  # type: ignore[union-attr]
            if msg is None:
                if topic not in self._camera_missing_warned:
                    self._camera_missing_warned.add(topic)
                    yield PracticeEventEnvelope(
                        practice_id=self._practice_id,
                        robot_id=self._robot_id,
                        source="system",
                        event_type="system.camera_missing",
                        timestamp_ns=time.monotonic_ns(),
                        payload={"topic": topic, "reason": "no message yet"},
                    )
                continue
            self._camera_missing_warned.discard(topic)
            yield from self._handle_image_or_camera_info(topic, msg)

    def _poll_imu_topics(self) -> Iterable[PracticeEventEnvelope]:
        for spec in self._ros2_topics:
            topic = spec["topic"]
            msg_type = spec.get("msg_type")
            if msg_type is not Imu:
                continue
            msg = self._node.take(topic) if self._node else None  # type: ignore[union-attr]
            if msg is None:
                if topic not in self._imu_missing_warned:
                    self._imu_missing_warned.add(topic)
                    yield PracticeEventEnvelope(
                        practice_id=self._practice_id,
                        robot_id=self._robot_id,
                        source="system",
                        event_type="system.camera_missing",
                        timestamp_ns=time.monotonic_ns(),
                        payload={"topic": topic, "reason": "no message yet"},
                    )
                continue
            self._imu_missing_warned.discard(topic)
            yield from self._handle_imu(topic, msg)

    def _handle_image_or_camera_info(self, topic: str, msg: Any) -> Iterable[PracticeEventEnvelope]:
        camera_id = _camera_id_from_topic(topic)
        suffix = _topic_suffix(topic)
        ts_ns = time.monotonic_ns()

        if isinstance(msg, Image):
            if msg.encoding == "rgb8":
                yield self._save_rgb(camera_id, msg, ts_ns)
            elif msg.encoding in ("16UC1", "mono16"):
                yield self._save_depth(camera_id, msg, ts_ns)
            else:
                logger.debug("Unhandled image encoding %s on %s", msg.encoding, topic)
        elif isinstance(msg, CameraInfo):
            yield self._save_camera_info(camera_id, topic, msg, ts_ns)

    def _save_rgb(self, camera_id: str, msg: Any, ts_ns: int) -> PracticeEventEnvelope:
        """Decode rgb8 Image and save as JPEG."""
        from PIL import Image as PILImage

        h, w = msg.height, msg.width
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((h, w, 3))
        rel_path = f"frames/{camera_id}/color_{ts_ns}.jpg"
        out_path = self._resolve_path(rel_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        PILImage.fromarray(arr).save(out_path, format="JPEG", quality=90)
        record_realsense_frame_async(camera_id, "color", str(out_path), latency_ms=33.3)
        return PracticeEventEnvelope(
            practice_id=self._practice_id,
            robot_id=self._robot_id,
            source="camera",
            event_type="camera.rgb_frame",
            timestamp_ns=ts_ns,
            payload={"camera_id": camera_id, "width": w, "height": h, "encoding": "rgb8"},
            payload_ref={"rgb_ref": rel_path},
        )

    def _save_depth(self, camera_id: str, msg: Any, ts_ns: int) -> PracticeEventEnvelope:
        """Decode 16UC1 Image and save as PNG16."""
        from PIL import Image as PILImage

        h, w = msg.height, msg.width
        arr = np.frombuffer(msg.data, dtype=np.uint16).reshape((h, w))
        rel_path = f"frames/{camera_id}/depth_{ts_ns}.png16"
        out_path = self._resolve_path(rel_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        PILImage.fromarray(arr).save(out_path, format="PNG")
        record_realsense_frame_async(camera_id, "depth", str(out_path), latency_ms=33.3)
        return PracticeEventEnvelope(
            practice_id=self._practice_id,
            robot_id=self._robot_id,
            source="camera",
            event_type="camera.depth_frame",
            timestamp_ns=ts_ns,
            payload={"camera_id": camera_id, "width": w, "height": h, "encoding": "16UC1"},
            payload_ref={"depth_ref": rel_path},
        )

    def _save_camera_info(self, camera_id: str, topic: str, msg: Any, ts_ns: int) -> PracticeEventEnvelope:
        """Serialize CameraInfo to JSON and reference it."""
        info = {
            "header": {"stamp": {"sec": msg.header.stamp.sec, "nanosec": msg.header.stamp.nanosec}, "frame_id": msg.header.frame_id},
            "height": msg.height,
            "width": msg.width,
            "distortion_model": msg.distortion_model,
            "d": list(msg.d),
            "k": list(msg.k),
            "r": list(msg.r),
            "p": list(msg.p),
            "binning_x": msg.binning_x,
            "binning_y": msg.binning_y,
            "roi": {"x_offset": msg.roi.x_offset, "y_offset": msg.roi.y_offset, "height": msg.roi.height, "width": msg.roi.width, "do_rectify": msg.roi.do_rectify},
        }
        kind = "color" if "color" in topic else "depth" if "depth" in topic else "camera"
        rel_path = f"frames/{camera_id}/camera_info_{kind}.json"
        out_path = self._resolve_path(rel_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
        set_realsense_online(camera_id, True, info={"width": msg.width, "height": msg.height})
        return PracticeEventEnvelope(
            practice_id=self._practice_id,
            robot_id=self._robot_id,
            source="ros2",
            event_type="ros2.camera_info",
            timestamp_ns=ts_ns,
            payload={"camera_id": camera_id, "width": msg.width, "height": msg.height},
            payload_ref={"camera_info_ref": rel_path},
        )

    def _handle_imu(self, topic: str, msg: Any) -> Iterable[PracticeEventEnvelope]:
        camera_id = _camera_id_from_topic(topic)
        ts_ns = time.monotonic_ns()
        sample = {
            "timestamp_ns": ts_ns,
            "orientation": {"x": msg.orientation.x, "y": msg.orientation.y, "z": msg.orientation.z, "w": msg.orientation.w},
            "angular_velocity": {"x": msg.angular_velocity.x, "y": msg.angular_velocity.y, "z": msg.angular_velocity.z},
            "linear_acceleration": {"x": msg.linear_acceleration.x, "y": msg.linear_acceleration.y, "z": msg.linear_acceleration.z},
        }
        rel_path = f"imu/{camera_id}_imu.jsonl"
        out_path = self._resolve_path(rel_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        set_realsense_online(camera_id, True, info={"last_imu_sample": sample})
        yield PracticeEventEnvelope(
            practice_id=self._practice_id,
            robot_id=self._robot_id,
            source="ros2",
            event_type="imu.sample",
            timestamp_ns=ts_ns,
            payload={"camera_id": camera_id},
            payload_ref={"imu_ref": rel_path},
        )

    def _resolve_path(self, rel_path: str) -> Path:
        if self._session_dir is not None:
            return Path(self._session_dir) / rel_path
        if self._output_root is not None:
            return Path(self._output_root) / rel_path
        return Path(rel_path)

    def on_event(self, callback: Any) -> None:
        pass
