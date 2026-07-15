"""Camera frame source abstractions."""
from __future__ import annotations

import os
import queue
import socket
import struct
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

from ..gesture_schema import CameraFrame


class CameraSource(ABC):
    @abstractmethod
    def read(self) -> Optional[CameraFrame]:
        ...

    @abstractmethod
    def release(self) -> None:
        ...


class MockCameraSource(CameraSource):
    """Returns blank frames for mock mode."""

    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height

    def read(self) -> Optional[CameraFrame]:
        return CameraFrame(
            color=np.zeros((self.height, self.width, 3), dtype=np.uint8),
            timestamp=time.time(),
        )

    def release(self) -> None:
        pass


class USBCameraSource(CameraSource):
    """Standard USB webcam via OpenCV."""

    def __init__(self, device_id: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self._cap = None
        self._device_id = device_id

    def _ensure_open(self):
        import cv2

        if self._cap is None:
            self._cap = cv2.VideoCapture(self._device_id)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)

    def read(self) -> Optional[CameraFrame]:
        self._ensure_open()
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return None
        return CameraFrame(color=frame, timestamp=time.time())

    def release(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None


class RealSenseSource(CameraSource):
    """Intel RealSense D435 via pyrealsense2."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self._pipeline = None
        self._align = None

    def _ensure_open(self):
        if self._pipeline is not None:
            return
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError("pyrealsense2 not installed") from exc

        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self._pipeline.start(config)
        self._align = rs.align(rs.stream.color)

    def read(self) -> Optional[CameraFrame]:
        self._ensure_open()

        frames = self._pipeline.wait_for_frames(timeout_ms=1000)
        aligned = self._align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame:
            return None
        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data()) if depth_frame else None
        return CameraFrame(color=color, depth=depth, timestamp=time.time())

    def release(self) -> None:
        if self._pipeline:
            self._pipeline.stop()
            self._pipeline = None


class ROS2CameraSource(CameraSource):
    """ROS 2 image topic via a system-Python cv_bridge helper process."""

    def __init__(
        self,
        topic: str = "/camera/d435i/color/image_raw",
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ):
        self.topic = topic
        self.width = width
        self.height = height
        self.fps = fps
        self._proc: Optional[subprocess.Popen] = None
        self._sock: Optional[socket.socket] = None
        self._reader: Optional[threading.Thread] = None
        self._frame_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=2)
        self._running = False

    def _ensure_open(self):
        if self._proc is not None:
            return
        helper = Path(__file__).resolve().parent / "ros2_camera_helper.py"
        cmd = [
            sys.executable,
            str(helper),
            self.topic,
            str(self.width),
            str(self.height),
            str(self.fps),
        ]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            bufsize=1,
        )
        # Read the helper's advertised port.
        port_line = self._proc.stdout.readline().strip()
        if not port_line.startswith("PORT "):
            err = self._proc.stderr.read(4096) if self._proc.stderr else ""
            self.release()
            raise RuntimeError(
                f"ROS2 helper did not advertise port: {port_line!r}; stderr={err!r}"
            )
        port = int(port_line.split()[1])
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(10.0)
        try:
            self._sock.connect(("127.0.0.1", port))
        except OSError as exc:
            self.release()
            raise RuntimeError(f"Could not connect to ROS2 helper on port {port}") from exc
        self._sock.settimeout(None)
        self._running = True
        self._reader = threading.Thread(target=self._receive, daemon=True)
        self._reader.start()

    def _receive(self):
        expected = self.height * self.width * 3
        try:
            while self._running:
                header = self._recv_all(4)
                if header is None:
                    break
                length = struct.unpack(">I", header)[0]
                if length != expected:
                    # Helper is speaking a different protocol; drop.
                    continue
                data = self._recv_all(length)
                if data is None:
                    break
                frame = np.frombuffer(data, dtype=np.uint8).reshape(
                    (self.height, self.width, 3)
                )
                try:
                    self._frame_queue.put_nowait(frame)
                except queue.Full:
                    try:
                        self._frame_queue.get_nowait()
                        self._frame_queue.put_nowait(frame)
                    except queue.Empty:
                        pass
        except OSError:
            pass

    def _recv_all(self, n: int) -> Optional[bytes]:
        sock = self._sock
        buf = b""
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    def read(self) -> Optional[CameraFrame]:
        self._ensure_open()
        try:
            frame = self._frame_queue.get(timeout=1.0)
        except queue.Empty:
            return None
        return CameraFrame(color=frame, timestamp=time.time())

    def release(self) -> None:
        self._running = False
        if self._reader and self._reader.is_alive():
            self._reader.join(timeout=1.0)
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None


def build_camera_source(config: dict) -> CameraSource:
    source = config.get("source", "mock")
    width = int(config.get("width", 640))
    height = int(config.get("height", 480))
    fps = int(config.get("fps", 30))
    if source == "realsense":
        return RealSenseSource(width=width, height=height, fps=fps)
    if source == "usb":
        return USBCameraSource(device_id=0, width=width, height=height, fps=fps)
    if source == "ros2":
        return ROS2CameraSource(
            topic=config.get("topic", "/camera/d435i/color/image_raw"),
            width=width,
            height=height,
            fps=fps,
        )
    return MockCameraSource(width=width, height=height)
