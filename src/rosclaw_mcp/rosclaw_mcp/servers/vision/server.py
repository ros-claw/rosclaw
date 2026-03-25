"""
ROSClaw Vision MCP Server

Intel RealSense RGB-D Camera MCP Server using ROS2.
Part of the ROSClaw Embodied Intelligence Operating System.

Features:
- ROS2 communication via rclpy
- Semantic-HAL: exposes only compressed semantic data to LLM (never raw streams)
- Tools: capture_scene_snapshot, get_object_3d_coordinates, check_workspace_clear
- Data flywheel: start/stop rosbag2 recording for LeRobot training
- Safety: LLM never sees raw point clouds or 30Hz streams
- Thread-safe: rclpy spin in daemon thread, FastMCP in main event loop

Hardware: Intel RealSense D415/D435/D455 (USB3)
Protocol: ROS2 (Robot Operating System 2) via realsense2_camera driver

Architecture:
  RealSense → realsense2_camera (ROS2) → VisionROS2Bridge → FastMCP → LLM
  Fast lane (30Hz):  raw topics → rosbag2 / VLA (LLM never sees this)
  Slow lane (1Hz/on-demand): snapshot + inference → MCP tool response
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import struct
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
    from PIL import Image as PILImage
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, CameraInfo
    _HAS_ROS2 = True
except ImportError:
    _HAS_ROS2 = False
    # Stub for environments without ROS2
    class Node:  # type: ignore
        pass

from mcp.server.fastmcp import FastMCP

# Initialize MCP Server
mcp = FastMCP("rosclaw-vision")


# ============ State & Buffer ============

@dataclass
class VisionState:
    """Vision sensor state — internal representation, never sent raw to LLM"""
    timestamp: float

    # Raw image bytes from ROS sensor_msgs/Image (internal only)
    color_image: Optional[bytes] = None    # RGB8 encoding
    depth_image: Optional[bytes] = None    # Z16 encoding (uint16 millimeters)

    # Image dimensions
    image_width: int = 640
    image_height: int = 480

    # Camera intrinsics from /camera/color/camera_info
    fx: float = 615.0    # focal length x (pixels)
    fy: float = 615.0    # focal length y (pixels)
    cx: float = 320.0    # principal point x (pixels)
    cy: float = 240.0    # principal point y (pixels)

    # Metadata
    color_encoding: str = "rgb8"
    depth_encoding: str = "16UC1"
    frame_id: str = "camera_color_optical_frame"

    # Frame counters (for fps estimation)
    color_frame_count: int = 0
    depth_frame_count: int = 0


@dataclass
class StateBuffer:
    """Thread-safe ring buffer for vision frames"""
    max_size: int = 10
    _buffer: deque = field(default_factory=lambda: deque(maxlen=10))
    _lock: threading.Lock = field(default_factory=threading.Lock)
    # Latest intrinsics tracked separately for fast access
    _latest_intrinsics: Optional[Dict] = None

    def append(self, state: VisionState):
        with self._lock:
            self._buffer.append(state)

    def get_latest(self) -> Optional[VisionState]:
        with self._lock:
            return self._buffer[-1] if self._buffer else None

    def update_intrinsics(self, fx: float, fy: float, cx: float, cy: float, w: int, h: int, frame_id: str):
        with self._lock:
            self._latest_intrinsics = {"fx": fx, "fy": fy, "cx": cx, "cy": cy,
                                        "width": w, "height": h, "frame_id": frame_id}

    def get_intrinsics(self) -> Optional[Dict]:
        with self._lock:
            return self._latest_intrinsics


# ============ Vision Bridge ============

class VisionROS2Bridge:
    """
    Intel RealSense ROS2 Communication Bridge (composition pattern)

    Holds an internal rclpy.Node (_node) when ROS2 is available,
    so this class remains a plain Python object and is fully testable
    without ROS2 installed.

    Subscribes to:
    - /camera/color/image_raw        (sensor_msgs/Image, RGB8, 30Hz)
    - /camera/aligned_depth_to_color/image_raw  (sensor_msgs/Image, Z16, 30Hz)
    - /camera/color/camera_info      (sensor_msgs/CameraInfo, intrinsics)

    Exposes only semantic operations to MCP:
    - get_jpeg_base64()     — latest RGB frame as compressed JPEG Base64
    - get_depth_meters()    — depth at a specific pixel
    - pixel_to_3d()         — back-project pixel+depth to 3D XYZ
    - check_volume_clear()  — obstacle detection in a 3D bounding volume
    """

    TOPICS = {
        "color": "/camera/color/image_raw",
        "depth": "/camera/aligned_depth_to_color/image_raw",
        "info":  "/camera/color/camera_info",
    }

    # Configurable depth limits (meters)
    MIN_DEPTH_M = 0.1    # RealSense minimum reliable depth
    MAX_DEPTH_M = 10.0   # RealSense maximum reliable depth

    def __init__(self, node_name: str = "vision_mcp_bridge"):
        self._connected = False
        self._node = None        # rclpy.Node instance (set in _init_ros2_node)
        self._node_name = node_name
        self._state_buffer = StateBuffer(max_size=10)

        # Separate mutable state for color+depth pairing
        self._pending_color: Optional[bytes] = None
        self._pending_color_w: int = 640
        self._pending_color_h: int = 480
        self._pending_color_enc: str = "rgb8"
        self._pending_depth: Optional[bytes] = None
        self._pending_frame_id: str = "camera_color_optical_frame"
        self._frame_lock = threading.Lock()
        self._color_count: int = 0
        self._depth_count: int = 0

    def init_ros2_node(self):
        """Create the internal rclpy.Node and register subscriptions. Call after rclpy.init()."""
        if not _HAS_ROS2:
            return
        bridge = self

        class _VisionNode(Node):  # type: ignore[misc]
            def __init__(self):
                super().__init__(bridge._node_name)
                self.create_subscription(Image, VisionROS2Bridge.TOPICS["color"],
                                         bridge._color_callback, 1)
                self.create_subscription(Image, VisionROS2Bridge.TOPICS["depth"],
                                         bridge._depth_callback, 1)
                self.create_subscription(CameraInfo, VisionROS2Bridge.TOPICS["info"],
                                         bridge._info_callback, 1)
                self.get_logger().info("Vision MCP Bridge initialized")

        self._node = _VisionNode()

    def destroy(self):
        """Clean up ROS2 node"""
        if self._node is not None and _HAS_ROS2:
            self._node.destroy_node()
            self._node = None

    def _color_callback(self, msg: "Image"):
        """Receive RGB frame, update buffer"""
        with self._frame_lock:
            self._pending_color = bytes(msg.data)
            self._pending_color_w = msg.width
            self._pending_color_h = msg.height
            self._pending_color_enc = msg.encoding
            self._pending_frame_id = msg.header.frame_id
            self._color_count += 1
            self._flush_state()

    def _depth_callback(self, msg: "Image"):
        """Receive depth frame, update buffer"""
        with self._frame_lock:
            self._pending_depth = bytes(msg.data)
            self._depth_count += 1
            self._flush_state()

    def _flush_state(self):
        """Commit current pending frames to state buffer (called under _frame_lock)"""
        if self._pending_color is None:
            return
        intrinsics = self._state_buffer.get_intrinsics()
        fx = intrinsics["fx"] if intrinsics else 615.0
        fy = intrinsics["fy"] if intrinsics else 615.0
        cx = intrinsics["cx"] if intrinsics else 320.0
        cy = intrinsics["cy"] if intrinsics else 240.0

        state = VisionState(
            timestamp=time.time(),
            color_image=self._pending_color,
            depth_image=self._pending_depth,
            image_width=self._pending_color_w,
            image_height=self._pending_color_h,
            fx=fx, fy=fy, cx=cx, cy=cy,
            color_encoding=self._pending_color_enc,
            frame_id=self._pending_frame_id,
            color_frame_count=self._color_count,
            depth_frame_count=self._depth_count,
        )
        self._state_buffer.append(state)

    def _info_callback(self, msg: "CameraInfo"):
        """Receive camera intrinsics"""
        self._state_buffer.update_intrinsics(
            fx=msg.k[0], fy=msg.k[4],
            cx=msg.k[2], cy=msg.k[5],
            w=msg.width, h=msg.height,
            frame_id=msg.header.frame_id,
        )

    def get_jpeg_base64(self, quality: int = 80) -> Optional[str]:
        """
        Convert latest RGB frame to JPEG and return as Base64 ASCII string.
        Returns None if no frame available or PIL not installed.
        """
        state = self._state_buffer.get_latest()
        if state is None or state.color_image is None:
            return None
        if not _HAS_PIL:
            return None
        try:
            data = state.color_image
            w, h = state.image_width, state.image_height
            enc = state.color_encoding.lower()
            if enc in ("rgb8", "rgb"):
                arr = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)
                pil_img = PILImage.fromarray(arr, "RGB")
            elif enc in ("bgr8", "bgr"):
                arr = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)
                arr = arr[:, :, ::-1]  # BGR → RGB
                pil_img = PILImage.fromarray(arr, "RGB")
            else:
                return None
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=quality)
            return base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:
            return None

    def get_depth_meters(self, u: int, v: int) -> Optional[float]:
        """
        Get depth at pixel (u, v) in meters.
        Z16 encoding: uint16 value = depth in millimeters.
        Returns None if pixel out of bounds or no depth data.
        """
        state = self._state_buffer.get_latest()
        if state is None or state.depth_image is None:
            return None
        w, h = state.image_width, state.image_height
        if not (0 <= u < w and 0 <= v < h):
            return None
        try:
            offset = (v * w + u) * 2
            raw_mm = struct.unpack_from("<H", state.depth_image, offset)[0]
            if raw_mm == 0:
                return None  # Invalid depth
            depth_m = raw_mm / 1000.0
            if not (self.MIN_DEPTH_M <= depth_m <= self.MAX_DEPTH_M):
                return None
            return depth_m
        except Exception:
            return None

    def pixel_to_3d(self, u: int, v: int, depth_m: float) -> Dict[str, float]:
        """
        Back-project a 2D pixel coordinate + depth value to 3D XYZ in camera frame.
        Uses standard pinhole camera model.
        """
        state = self._state_buffer.get_latest()
        fx = state.fx if state else 615.0
        fy = state.fy if state else 615.0
        cx = state.cx if state else 320.0
        cy = state.cy if state else 240.0
        frame_id = state.frame_id if state else "camera_color_optical_frame"

        X = (u - cx) * depth_m / fx
        Y = (v - cy) * depth_m / fy
        Z = depth_m
        return {"x": round(X, 4), "y": round(Y, 4), "z": round(Z, 4), "frame_id": frame_id}

    def check_volume_clear(self, x_min: float, x_max: float,
                            y_min: float, y_max: float,
                            z_min: float, z_max: float) -> Tuple[bool, int]:
        """
        Check if a 3D bounding volume (in camera frame) is clear of obstacles.
        Samples a grid of pixels and checks if any depth point falls inside the volume.

        Returns:
            (is_clear, obstacle_count)
        """
        state = self._state_buffer.get_latest()
        if state is None or state.depth_image is None:
            return True, 0  # No data → assume clear

        w, h = state.image_width, state.image_height
        fx, fy = state.fx, state.fy
        cx, cy = state.cx, state.cy

        obstacle_count = 0
        step = 8  # Sample every 8th pixel for speed

        for v in range(0, h, step):
            for u in range(0, w, step):
                depth_m = self.get_depth_meters(u, v)
                if depth_m is None:
                    continue
                X = (u - cx) * depth_m / fx
                Y = (v - cy) * depth_m / fy
                Z = depth_m
                if (x_min <= X <= x_max and y_min <= Y <= y_max and z_min <= Z <= z_max):
                    obstacle_count += 1

        return obstacle_count == 0, obstacle_count

    def get_status_info(self) -> Dict:
        """Get current camera status metadata"""
        state = self._state_buffer.get_latest()
        intrinsics = self._state_buffer.get_intrinsics()
        if state is None:
            return {"connected": self._connected, "has_frames": False}
        return {
            "connected": self._connected,
            "has_frames": True,
            "timestamp": state.timestamp,
            "resolution": f"{state.image_width}x{state.image_height}",
            "color_encoding": state.color_encoding,
            "color_frames": state.color_frame_count,
            "depth_frames": state.depth_frame_count,
            "has_depth": state.depth_image is not None,
            "intrinsics": intrinsics or {"fx": state.fx, "fy": state.fy, "cx": state.cx, "cy": state.cy},
            "frame_id": state.frame_id,
        }


# ============ Global State ============

_bridge: Optional[VisionROS2Bridge] = None
_recording_process: Optional[subprocess.Popen] = None
_recording_path: Optional[str] = None


def _init_ros():
    """Initialize ROS2 context"""
    if _HAS_ROS2 and not rclpy.ok():
        rclpy.init()


# ============ MCP Tools ============

@mcp.tool()
async def connect_vision(node_name: str = "vision_mcp_bridge") -> str:
    """
    Connect to RealSense camera via ROS2

    Initializes ROS2 node and starts subscribing to camera topics.
    Requires realsense2_camera node to be running.

    Args:
        node_name: ROS2 node name for this bridge
    """
    global _bridge

    if _bridge is not None:
        return "Already connected to vision system"

    if not _HAS_ROS2:
        return "Error: rclpy not available. Source ROS2: source /opt/ros/humble/setup.bash"

    if not _HAS_PIL:
        return "Error: Pillow not installed. Run: pip install Pillow numpy"

    try:
        _init_ros()
        _bridge = VisionROS2Bridge(node_name=node_name)
        _bridge.init_ros2_node()
        _bridge._connected = True

        def spin_node():
            while rclpy.ok() and _bridge and _bridge._node is not None:
                rclpy.spin_once(_bridge._node, timeout_sec=0.1)

        spin_thread = threading.Thread(target=spin_node, daemon=True, name="VisionROSSpin")
        spin_thread.start()

        return (
            f"✓ Connected to vision system (node: {node_name})\n"
            f"  Subscribing to:\n"
            f"    {VisionROS2Bridge.TOPICS['color']}\n"
            f"    {VisionROS2Bridge.TOPICS['depth']}\n"
            f"    {VisionROS2Bridge.TOPICS['info']}\n"
            f"  Waiting for camera frames..."
        )
    except Exception as e:
        return f"✗ Failed to connect: {e}"


@mcp.tool()
async def disconnect_vision() -> str:
    """Disconnect from RealSense camera and clean up ROS2 node"""
    global _bridge

    if _bridge is None:
        return "Not connected"

    try:
        _bridge._connected = False
        if _HAS_ROS2:
            _bridge.destroy_node()
        _bridge = None
        return "✓ Disconnected from vision system"
    except Exception as e:
        return f"✗ Error during disconnect: {e}"


@mcp.tool()
async def capture_scene_snapshot(quality: int = 80) -> str:
    """
    Capture a single RGB snapshot of the current scene.

    Returns the image as a Base64-encoded JPEG string suitable for
    multimodal LLM vision (Claude, GPT-4o). This is the primary tool
    for giving the LLM "eyes" — call it when you need to understand
    the visual environment.

    Args:
        quality: JPEG compression quality (1-95). Default 80 gives good
                 quality with reasonable size (~50-150KB per frame).

    Returns:
        Base64-encoded JPEG image string, or error message.
    """
    global _bridge

    if _bridge is None or not _bridge._connected:
        return "Error: Not connected to vision system"

    if not (1 <= quality <= 95):
        return "Error: quality must be between 1 and 95"

    # Wait briefly for first frame if just connected
    for _ in range(20):
        b64 = _bridge.get_jpeg_base64(quality=quality)
        if b64 is not None:
            break
        await asyncio.sleep(0.1)

    if b64 is None:
        return "Error: No camera frames received. Is realsense2_camera running?"

    state = _bridge._state_buffer.get_latest()
    size_kb = len(b64) * 3 // 4 // 1024  # approximate decoded size
    return (
        f"image/jpeg;base64,{b64}\n"
        f"[Snapshot: {state.image_width}x{state.image_height}, "
        f"~{size_kb}KB JPEG, frame #{state.color_frame_count}]"
    )


@mcp.tool()
async def get_depth_at_pixel(u: int, v: int) -> str:
    """
    Get the depth (distance from camera) at a specific image pixel.

    Use this to measure how far away a specific point in the scene is.
    Combine with capture_scene_snapshot to identify pixels of interest.

    Args:
        u: Horizontal pixel coordinate (0 = left, increases right)
        v: Vertical pixel coordinate (0 = top, increases down)

    Returns:
        Depth in meters, or error if pixel is invalid or no depth data.
    """
    global _bridge

    if _bridge is None or not _bridge._connected:
        return "Error: Not connected to vision system"

    state = _bridge._state_buffer.get_latest()
    if state is None:
        return "Error: No camera frames available"

    w, h = state.image_width, state.image_height
    if not (0 <= u < w and 0 <= v < h):
        return f"Error: Pixel ({u},{v}) out of bounds. Image is {w}x{h}"

    depth_m = _bridge.get_depth_meters(u, v)

    if depth_m is None:
        return f"No valid depth at pixel ({u},{v}) — may be out of RealSense range or invalid"

    xyz = _bridge.pixel_to_3d(u, v, depth_m)
    return (
        f"✓ Depth at pixel ({u},{v}): {depth_m:.3f} m\n"
        f"  3D position in camera frame: X={xyz['x']:.3f}m, Y={xyz['y']:.3f}m, Z={xyz['z']:.3f}m"
    )


@mcp.tool()
async def get_object_3d_coordinates(target_object: str) -> str:
    """
    Find the 3D position (X, Y, Z) of a named object in the scene.

    This is the core "semantic grounding" tool. It runs local object detection
    to find the object in the RGB image, then uses the aligned depth map and
    camera intrinsics to compute the exact 3D coordinates.

    Use the returned coordinates to command the robot arm (pick_object, move_ee_to_pose).

    Args:
        target_object: Natural language description of the object to find,
                       e.g. "red cup", "screwdriver", "blue box"

    Returns:
        JSON with x, y, z coordinates in camera frame, or error.
    """
    global _bridge

    if _bridge is None or not _bridge._connected:
        return "Error: Not connected to vision system"

    # Wait for frames
    for _ in range(20):
        state = _bridge._state_buffer.get_latest()
        if state is not None and state.color_image is not None:
            break
        await asyncio.sleep(0.1)

    state = _bridge._state_buffer.get_latest()
    if state is None or state.color_image is None:
        return "Error: No camera frames received"

    # Try YOLO-World for zero-shot object detection
    try:
        from ultralytics import YOLOWorld  # type: ignore
        model = YOLOWorld("yolov8s-world.pt")
        model.set_classes([target_object])

        # Decode image
        data = state.color_image
        arr = np.frombuffer(data, dtype=np.uint8).reshape(state.image_height, state.image_width, 3)

        results = model(arr, verbose=False)
        boxes = results[0].boxes

        if len(boxes) == 0:
            # YOLO found nothing — fall back to returning snapshot for LLM
            b64 = _bridge.get_jpeg_base64(quality=70)
            return (
                f"Object '{target_object}' not found by local detection.\n"
                f"Snapshot provided for visual inspection:\n"
                f"image/jpeg;base64,{b64}"
            )

        # Use the highest-confidence detection
        best_box = boxes[boxes.conf.argmax()]
        x1, y1, x2, y2 = best_box.xyxy[0].tolist()
        confidence = float(best_box.conf[0])

        # Sample depth at bounding box center
        u_center = int((x1 + x2) / 2)
        v_center = int((y1 + y2) / 2)
        depth_m = _bridge.get_depth_meters(u_center, v_center)

        if depth_m is None:
            return (
                f"Found '{target_object}' at pixel ({u_center},{v_center}) "
                f"but depth is invalid at that point. "
                f"Bounding box: [{int(x1)},{int(y1)},{int(x2)},{int(y2)}], "
                f"confidence: {confidence:.2f}"
            )

        xyz = _bridge.pixel_to_3d(u_center, v_center, depth_m)
        result = {
            "status": "found",
            "object": target_object,
            "x": xyz["x"],
            "y": xyz["y"],
            "z": xyz["z"],
            "confidence": round(confidence, 3),
            "depth_m": round(depth_m, 4),
            "pixel_uv": [u_center, v_center],
            "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
            "frame_id": xyz["frame_id"],
        }
        return f"✓ Object located:\n{json.dumps(result, indent=2)}"

    except ImportError:
        # ultralytics not installed — return snapshot so LLM can do visual reasoning
        b64 = _bridge.get_jpeg_base64(quality=70)
        if b64 is None:
            return "Error: Cannot capture snapshot for visual detection"
        return (
            f"Local detection unavailable (install ultralytics for YOLO-World).\n"
            f"Please inspect the snapshot below and identify '{target_object}'.\n"
            f"Then call get_depth_at_pixel(u, v) with the object's pixel coordinates.\n"
            f"image/jpeg;base64,{b64}"
        )


@mcp.tool()
async def get_scene_description() -> str:
    """
    Get a text description of the current camera setup and scene metadata.

    Returns camera resolution, intrinsics, frame counts, and topic info.
    Use capture_scene_snapshot for the actual visual scene.
    """
    global _bridge

    if _bridge is None or not _bridge._connected:
        return "Error: Not connected to vision system"

    info = _bridge.get_status_info()
    if not info.get("has_frames"):
        return (
            "Vision system connected but no frames yet.\n"
            "Ensure realsense2_camera is running:\n"
            "  ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true"
        )

    intrinsics = info.get("intrinsics", {})
    return f"""
Scene Description:
  Resolution:    {info['resolution']}
  Color format:  {info['color_encoding']}
  Has depth:     {info['has_depth']}
  Color frames:  {info['color_frames']}
  Depth frames:  {info['depth_frames']}
  Frame ID:      {info['frame_id']}

Camera Intrinsics:
  fx: {intrinsics.get('fx', 'N/A'):.2f} px
  fy: {intrinsics.get('fy', 'N/A'):.2f} px
  cx: {intrinsics.get('cx', 'N/A'):.2f} px  (principal point)
  cy: {intrinsics.get('cy', 'N/A'):.2f} px

ROS2 Topics:
  Color:    {VisionROS2Bridge.TOPICS['color']}
  Depth:    {VisionROS2Bridge.TOPICS['depth']}
  Info:     {VisionROS2Bridge.TOPICS['info']}
""".strip()


@mcp.tool()
async def check_workspace_clear(
    x_min: float = -0.3, x_max: float = 0.3,
    y_min: float = -0.3, y_max: float = 0.3,
    z_min: float = 0.1,  z_max: float = 0.9,
) -> str:
    """
    Check if a 3D workspace volume (in camera frame) is free of obstacles.

    Use before moving the robot arm into a region to verify it's safe.
    Coordinates are in meters relative to the camera optical center.

    Args:
        x_min: Left boundary (meters, negative = left)
        x_max: Right boundary (meters, positive = right)
        y_min: Top boundary (meters, negative = up)
        y_max: Bottom boundary (meters, positive = down)
        z_min: Near boundary (meters from camera)
        z_max: Far boundary (meters from camera)

    Returns:
        Clear/blocked status with obstacle count.
    """
    global _bridge

    if _bridge is None or not _bridge._connected:
        return "Error: Not connected to vision system"

    if not all(lo < hi for lo, hi in [(x_min, x_max), (y_min, y_max), (z_min, z_max)]):
        return "Error: min values must be less than max values"

    is_clear, count = _bridge.check_volume_clear(x_min, x_max, y_min, y_max, z_min, z_max)

    volume = f"X[{x_min:.2f},{x_max:.2f}] Y[{y_min:.2f},{y_max:.2f}] Z[{z_min:.2f},{z_max:.2f}] m"

    if is_clear:
        return f"✓ Workspace clear in volume {volume}"
    else:
        return (
            f"⚠ Workspace BLOCKED in volume {volume}\n"
            f"  Detected {count} obstacle points. Do NOT move robot into this region."
        )


@mcp.tool()
async def start_data_recording(output_dir: str = "/tmp/rosclaw_recordings") -> str:
    """
    Start recording camera and robot data to a ROS2 bag file.

    Records RGB-D camera streams and joint states for the LeRobot
    data flywheel pipeline. Data is automatically labeled with timestamps.

    Args:
        output_dir: Directory to save the bag file. Default: /tmp/rosclaw_recordings
    """
    global _recording_process, _recording_path

    if _bridge is None or not _bridge._connected:
        return "Error: Not connected to vision system"

    if _recording_process is not None and _recording_process.poll() is None:
        return f"Already recording to: {_recording_path}"

    import os
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    bag_name = f"rosclaw_{timestamp}"
    bag_path = os.path.join(output_dir, bag_name)
    os.makedirs(output_dir, exist_ok=True)

    topics = [
        VisionROS2Bridge.TOPICS["color"],
        VisionROS2Bridge.TOPICS["depth"],
        VisionROS2Bridge.TOPICS["info"],
        "/joint_states",
        "/tf",
    ]

    cmd = ["ros2", "bag", "record", "-o", bag_path] + topics
    try:
        _recording_process = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        _recording_path = bag_path
        return (
            f"✓ Recording started\n"
            f"  Output: {bag_path}\n"
            f"  Topics: {', '.join(topics)}\n"
            f"  PID: {_recording_process.pid}"
        )
    except FileNotFoundError:
        return "Error: ros2 command not found. Source ROS2: source /opt/ros/humble/setup.bash"
    except Exception as e:
        return f"✗ Failed to start recording: {e}"


@mcp.tool()
async def stop_data_recording() -> str:
    """
    Stop the current data recording and finalize the bag file.

    The recorded bag can be converted to LeRobot RLDS format for VLA training.
    """
    global _recording_process, _recording_path

    if _recording_process is None:
        return "Not recording"

    if _recording_process.poll() is not None:
        _recording_process = None
        path = _recording_path
        _recording_path = None
        return f"Recording already stopped. Bag at: {path}"

    _recording_process.terminate()
    try:
        _recording_process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        _recording_process.kill()

    path = _recording_path
    _recording_process = None
    _recording_path = None

    return (
        f"✓ Recording stopped\n"
        f"  Bag file: {path}\n"
        f"  Convert for LeRobot: ros2 bag convert -i {path} -o {path}_rlds"
    )


# ============ MCP Resources ============

@mcp.resource("vision://status")
async def get_camera_status() -> str:
    """
    Get current vision system status

    Returns camera resolution, frame counts, intrinsics, and depth availability.
    """
    global _bridge

    if _bridge is None:
        return "Not connected to vision system"

    info = _bridge.get_status_info()
    if not info.get("has_frames"):
        return (
            "Vision system connected but no frames received yet.\n"
            "Check that realsense2_camera is running and publishing to:\n"
            f"  {VisionROS2Bridge.TOPICS['color']}"
        )

    intrinsics = info.get("intrinsics", {})
    return f"""
Vision System Status:
  Connected:    True
  Resolution:   {info['resolution']}
  Encoding:     {info['color_encoding']}
  Has depth:    {info['has_depth']}

Frame Counts:
  Color frames: {info['color_frames']}
  Depth frames: {info['depth_frames']}

Camera Intrinsics:
  fx={intrinsics.get('fx', 0):.2f}  fy={intrinsics.get('fy', 0):.2f}
  cx={intrinsics.get('cx', 0):.2f}  cy={intrinsics.get('cy', 0):.2f}

Frame ID: {info['frame_id']}
Timestamp: {info.get('timestamp', 0):.3f}
""".strip()


@mcp.resource("vision://topics")
async def get_topic_info() -> str:
    """Get ROS2 topic information for the vision system"""
    return f"""
Vision ROS2 Topics:
  Color stream:  {VisionROS2Bridge.TOPICS['color']}
                 Type: sensor_msgs/Image (RGB8, 30Hz)

  Depth stream:  {VisionROS2Bridge.TOPICS['depth']}
                 Type: sensor_msgs/Image (Z16, depth aligned to color, 30Hz)

  Camera info:   {VisionROS2Bridge.TOPICS['info']}
                 Type: sensor_msgs/CameraInfo (intrinsics, distortion)

Launch realsense2_camera:
  ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true

Verify topics are active:
  ros2 topic hz {VisionROS2Bridge.TOPICS['color']}
""".strip()


@mcp.resource("vision://connection")
async def get_connection_status() -> str:
    """Get vision system connection status"""
    global _bridge, _recording_process

    if _bridge and _bridge._connected:
        recording = ""
        if _recording_process and _recording_process.poll() is None:
            recording = f"\n  Recording to: {_recording_path} (PID {_recording_process.pid})"
        return f"Connected to RealSense via ROS2{recording}"
    return "Disconnected"


if __name__ == "__main__":
    mcp.run(transport="stdio")
