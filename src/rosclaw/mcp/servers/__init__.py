"""Built-in MCP servers for RealSense hardware wrappers.

These servers provide a minimal, read-only stdio MCP interface so that
`rosclaw mcp install` and `rosclaw mcp health` can validate the ROSClaw
onboarding pipeline without requiring PyPI packages or a physical driver.
When librealsense2 / ROS2 are available, tools can be extended to call the
actual device; until then they return diagnostic metadata.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types

from rosclaw.dashboard.client import record_realsense_frame, set_realsense_online


@dataclass(frozen=True)
class CameraSpec:
    name: str
    model: str
    vendor: str
    has_imu: bool
    min_range_m: float
    max_range_m: float


# Specs mirror the e-URDF profiles in e-urdf-zoo/realsense-*.
CAMERAS: dict[str, CameraSpec] = {
    "realsense_d405": CameraSpec(
        name="Intel RealSense D405",
        model="D405",
        vendor="Intel",
        has_imu=False,
        min_range_m=0.07,
        max_range_m=0.50,
    ),
    "realsense_d435i": CameraSpec(
        name="Intel RealSense D435i",
        model="D435i",
        vendor="Intel",
        has_imu=True,
        min_range_m=0.30,
        max_range_m=3.00,
    ),
}


def _tool(name: str, description: str, schema: dict | None = None) -> types.Tool:
    return types.Tool(
        name=name,
        description=description,
        inputSchema=schema or {"type": "object", "properties": {}},
    )


def _ros2_topics(prefix: str = "/camera") -> list[str]:
    """Return a list of ROS2 topics under the camera namespace if ROS2 is available."""
    try:
        import subprocess

        result = subprocess.run(
            ["ros2", "topic", "list"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return [t for t in result.stdout.strip().splitlines() if t.startswith(prefix)]
    except Exception:
        pass
    return []


def _ros2_topic_type(topic: str) -> str | None:
    try:
        import subprocess

        result = subprocess.run(
            ["ros2", "topic", "type", topic],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip().splitlines()[-1]
    except Exception:
        pass
    return None


def _subscribe_once(topic: str, msg_type_str: str, timeout_sec: float = 5.0):
    """Subscribe to a ROS2 topic once and return the message.

    Returns None if rclpy is unavailable or the topic is not publishing.
    """
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
    except Exception:
        return None

    if not rclpy.ok():
        rclpy.init()
    node = Node(f"rosclaw_mcp_subscriber_{uuid.uuid4().hex[:8]}")
    msg_mod_name, msg_class_name = msg_type_str.split("/", 1)
    msg_mod = __import__(f"{msg_mod_name}.msg", fromlist=[msg_class_name])
    msg_class = getattr(msg_mod, msg_class_name)

    result = {"msg": None}

    def _cb(msg):
        if result["msg"] is None:
            result["msg"] = msg

    sub = node.create_subscription(msg_class, topic, _cb, qos_profile_sensor_data)

    end = node.get_clock().now() + rclpy.duration.Duration(seconds=timeout_sec)
    while result["msg"] is None and node.get_clock().now() < end:
        rclpy.spin_once(node, timeout_sec=0.1)

    sub.destroy()
    node.destroy_node()
    return result["msg"]


def _save_image(msg, output_path: str, encoding: str = "rgb8") -> str:
    """Save a sensor_msgs/Image to disk.

    Supports rgb8/bgr8 -> jpg and 16UC1 -> png.
    """
    from PIL import Image

    if encoding in ("rgb8", "bgr8"):
        mode = "RGB" if encoding == "rgb8" else "BGR"
        img = Image.frombytes(mode, (msg.width, msg.height), bytes(msg.data))
        if mode == "BGR":
            img = img.convert("RGB")
        img.save(output_path, "JPEG")
    elif encoding == "16UC1":
        import numpy as np

        arr = np.frombuffer(bytes(msg.data), dtype=np.uint16).reshape((msg.height, msg.width))
        Image.fromarray(arr).save(output_path, "PNG")
    else:
        raise ValueError(f"Unsupported image encoding: {encoding}")
    return output_path


def _depth_stats(msg) -> dict[str, Any]:
    """Compute min/max/mean depth from a 16UC1 depth image message."""
    import numpy as np

    arr = np.frombuffer(bytes(msg.data), dtype=np.uint16).reshape((msg.height, msg.width))
    valid = arr[arr > 0]
    if valid.size == 0:
        return {"min_m": 0.0, "max_m": 0.0, "mean_m": 0.0, "valid": False}
    # RealSense depth is typically in millimeters
    scale = 0.001
    return {
        "min_m": float(valid.min() * scale),
        "max_m": float(valid.max() * scale),
        "mean_m": float(valid.mean() * scale),
        "valid": True,
    }


def _imu_sample(msg) -> dict[str, Any]:
    """Extract a single IMU sample dict."""
    return {
        "accel": {
            "x": getattr(msg.linear_acceleration, "x", 0.0),
            "y": getattr(msg.linear_acceleration, "y", 0.0),
            "z": getattr(msg.linear_acceleration, "z", 0.0),
        },
        "gyro": {
            "x": getattr(msg.angular_velocity, "x", 0.0),
            "y": getattr(msg.angular_velocity, "y", 0.0),
            "z": getattr(msg.angular_velocity, "z", 0.0),
        },
        "orientation": {
            "x": getattr(msg.orientation, "x", 0.0),
            "y": getattr(msg.orientation, "y", 0.0),
            "z": getattr(msg.orientation, "z", 0.0),
            "w": getattr(msg.orientation, "w", 1.0),
        },
    }


def make_server(camera_key: str) -> Server:
    """Create a read-only MCP server for a RealSense camera."""
    spec = CAMERAS[camera_key]
    server = Server(f"rosclaw-mcp-{camera_key}")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        tools = [
            _tool("get_camera_status", "Return online status, topic list, and USB type if known."),
            _tool("list_camera_topics", "List available /camera/<name>/* topics."),
            _tool(
                "capture_rgb_frame",
                "Subscribe to the color topic once, save a JPEG, and return the path.",
                {
                    "type": "object",
                    "properties": {
                        "output_path": {"type": "string", "description": "Path to save the JPEG"},
                        "camera_name": {"type": "string", "description": "Camera namespace (default: d405)"},
                    },
                },
            ),
            _tool(
                "capture_depth_frame",
                "Subscribe to the depth topic once, save a PNG16, and return the path.",
                {
                    "type": "object",
                    "properties": {
                        "output_path": {"type": "string", "description": "Path to save the PNG16"},
                        "camera_name": {"type": "string", "description": "Camera namespace (default: d405)"},
                    },
                },
            ),
            _tool(
                "capture_rgbd_pair",
                "Capture aligned RGB + depth pair and return both paths.",
                {
                    "type": "object",
                    "properties": {
                        "output_dir": {"type": "string", "description": "Directory to save the pair"},
                        "camera_name": {"type": "string", "description": "Camera namespace (default: d405)"},
                    },
                },
            ),
            _tool(
                "capture_pointcloud_snapshot",
                "Attempt to subscribe to the pointcloud topic; return an error if unavailable.",
            ),
            _tool(
                "check_depth_validity",
                "Capture a depth frame and report min/max/mean depth and validity flag.",
                {
                    "type": "object",
                    "properties": {
                        "camera_name": {"type": "string", "description": "Camera namespace (default: d405)"},
                    },
                },
            ),
        ]
        if spec.has_imu:
            tools.append(
                _tool(
                    "get_imu_sample",
                    "Subscribe to the IMU topic once and return a sample.",
                    {
                        "type": "object",
                        "properties": {
                            "camera_name": {"type": "string", "description": "Camera namespace (default: d435i)"},
                        },
                    },
                )
            )
        return tools

    @server.call_tool()
    async def call_tool(name: str, arguments: dict | None = None) -> list[types.TextContent]:
        arguments = arguments or {}
        if name == "get_camera_status":
            payload = _tool_get_camera_status(camera_key, spec)
        elif name == "list_camera_topics":
            payload = _tool_list_camera_topics()
        elif name == "capture_rgb_frame":
            payload = _tool_capture_rgb_frame(arguments)
        elif name == "capture_depth_frame":
            payload = _tool_capture_depth_frame(arguments)
        elif name == "capture_rgbd_pair":
            payload = _tool_capture_rgbd_pair(arguments)
        elif name == "capture_pointcloud_snapshot":
            payload = _tool_capture_pointcloud_snapshot(arguments)
        elif name == "check_depth_validity":
            payload = _tool_check_depth_validity()
        elif name == "get_imu" and spec.has_imu:
            payload = _tool_get_imu_sample(camera_key)
        else:
            raise ValueError(f"Unknown tool: {name}")
        return [types.TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]

    return server


def _tool_get_camera_status(camera_key: str, spec: CameraSpec) -> dict[str, Any]:
    topics = _ros2_topics()
    online = bool(topics)
    usb_type = None
    try:
        import subprocess

        result = subprocess.run(
            ["rs-enumerate-devices", "-s"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if spec.model in line:
                    usb_type = line.split()[-1] if line.split() else None
    except Exception:
        pass
    info = {"usb_type": usb_type, "driver_loaded": _driver_available()}
    set_realsense_online(camera_key, online, info=info)
    return {
        "camera": spec.name,
        "model": spec.model,
        "online": online,
        "topics": topics,
        "usb_type": usb_type,
        "driver_loaded": _driver_available(),
    }


def _tool_list_camera_topics() -> dict[str, Any]:
    topics = _ros2_topics()
    typed = {}
    for t in topics:
        tt = _ros2_topic_type(t)
        if tt:
            typed[t] = tt
    return {"topics": topics, "typed": typed}


def _default_camera(arguments: dict[str, Any], key: str = "camera_name") -> str:
    """Return the camera name to use, defaulting from camera_key if available."""
    camera_name = arguments.get(key)
    if camera_name:
        return camera_name
    camera_key = arguments.get("camera_key", "")
    if "d435i" in camera_key:
        return "d435i"
    if "d405" in camera_key:
        return "d405"
    return "d405"


def _topic(prefix: str, camera_name: str, suffix: str) -> str:
    return f"{prefix}/{camera_name}/{suffix}" if prefix else f"/{camera_name}/{suffix}"


def _tool_capture_rgb_frame(arguments: dict[str, Any]) -> dict[str, Any]:
    output_path = arguments.get("output_path", "/tmp/capture_rgb.jpg")
    camera_name = _default_camera(arguments)
    try:
        import rclpy
    except Exception as exc:
        return {"error": f"rclpy not available: {exc}", "path": None}
    msg = _subscribe_once(f"/camera/{camera_name}/color/image_raw", "sensor_msgs/Image", timeout_sec=5.0)
    if msg is None:
        return {"error": f"No message received on /camera/{camera_name}/color/image_raw", "path": None}
    saved = _save_image(msg, output_path, encoding=getattr(msg, "encoding", "rgb8"))
    record_realsense_frame(camera_name, "color", saved, latency_ms=33.3)
    return {"path": saved, "width": msg.width, "height": msg.height}


def _tool_capture_depth_frame(arguments: dict[str, Any]) -> dict[str, Any]:
    output_path = arguments.get("output_path", "/tmp/capture_depth.png")
    camera_name = _default_camera(arguments)
    try:
        import rclpy
    except Exception as exc:
        return {"error": f"rclpy not available: {exc}", "path": None}
    msg = _subscribe_once(f"/camera/{camera_name}/depth/image_rect_raw", "sensor_msgs/Image", timeout_sec=5.0)
    if msg is None:
        return {"error": f"No message received on /camera/{camera_name}/depth/image_rect_raw", "path": None}
    saved = _save_image(msg, output_path, encoding=getattr(msg, "encoding", "16UC1"))
    record_realsense_frame(camera_name, "depth", saved, latency_ms=33.3)
    return {"path": saved, "width": msg.width, "height": msg.height}


def _tool_capture_rgbd_pair(arguments: dict[str, Any]) -> dict[str, Any]:
    output_dir = arguments.get("output_dir", "/tmp")
    camera_name = _default_camera(arguments)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    rgb_path = str(Path(output_dir) / "capture_rgb.jpg")
    depth_path = str(Path(output_dir) / "capture_depth.png")
    rgb_result = _tool_capture_rgb_frame({"output_path": rgb_path, "camera_name": camera_name})
    depth_result = _tool_capture_depth_frame({"output_path": depth_path, "camera_name": camera_name})
    if rgb_result.get("path"):
        record_realsense_frame(camera_name, "color", rgb_result["path"], latency_ms=33.3)
    if depth_result.get("path"):
        record_realsense_frame(camera_name, "depth", depth_result["path"], latency_ms=33.3)
    errors = []
    if rgb_result.get("error"):
        errors.append(rgb_result["error"])
    if depth_result.get("error"):
        errors.append(depth_result["error"])
    return {
        "rgb_path": rgb_result.get("path"),
        "depth_path": depth_result.get("path"),
        "errors": errors if errors else None,
    }


def _tool_capture_pointcloud_snapshot(arguments: dict[str, Any]) -> dict[str, Any]:
    output_path = arguments.get("output_path", "/tmp/capture_pointcloud.pcd")
    try:
        import rclpy
    except Exception as exc:
        return {"error": f"rclpy not available: {exc}", "path": None}
    msg = _subscribe_once("/camera/depth/color/points", "sensor_msgs/PointCloud2", timeout_sec=5.0)
    if msg is None:
        return {"error": "PointCloud2 topic not available or no message received", "path": None}
    # Simple PCD header write (binary)
    import struct

    fields = getattr(msg, "fields", [])
    width = getattr(msg, "width", 0)
    height = getattr(msg, "height", 0)
    data = bytes(getattr(msg, "data", b""))
    header = (
        "PCD\n"
        "VERSION 0.7\n"
        f"FIELDS {' '.join(f.name for f in fields)}\n"
        f"SIZE {' '.join(str(f.count * 4) for f in fields)}\n"
        f"TYPE {' '.join('F' for _ in fields)}\n"
        f"COUNT {' '.join('1' for _ in fields)}\n"
        f"WIDTH {width}\n"
        f"HEIGHT {height}\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {width * height}\n"
        "DATA binary\n"
    )
    with open(output_path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data)
    return {"path": output_path, "width": width, "height": height, "points": width * height}


def _tool_check_depth_validity(arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    arguments = arguments or {}
    camera_name = _default_camera(arguments)
    try:
        import rclpy
    except Exception as exc:
        return {"error": f"rclpy not available: {exc}", "valid": False}
    msg = _subscribe_once(f"/camera/{camera_name}/depth/image_rect_raw", "sensor_msgs/Image", timeout_sec=5.0)
    if msg is None:
        return {"error": f"No message received on /camera/{camera_name}/depth/image_rect_raw", "valid": False}
    stats = _depth_stats(msg)
    record_realsense_frame(camera_name, "depth", "", latency_ms=33.3)
    return {
        "valid": stats["valid"],
        "min_m": stats["min_m"],
        "max_m": stats["max_m"],
        "mean_m": stats["mean_m"],
    }


def _tool_get_imu_sample(arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    arguments = arguments or {}
    camera_name = _default_camera(arguments, "camera_name")
    try:
        import rclpy
    except Exception as exc:
        return {"error": f"rclpy not available: {exc}"}
    msg = _subscribe_once(f"/camera/{camera_name}/imu", "sensor_msgs/Imu", timeout_sec=5.0)
    if msg is None:
        return {"error": f"No message received on /camera/{camera_name}/imu"}
    sample = _imu_sample(msg)
    set_realsense_online(camera_name, True, info={"last_imu_sample": sample})
    return sample


def _driver_available() -> bool:
    """Return True if librealsense2 ROS2 node appears reachable."""
    try:
        import pyrealsense2 as rs  # type: ignore[import-untyped]
    except Exception:  # noqa: BLE001
        return False
    return True


async def run(camera_key: str) -> None:
    server = make_server(camera_key)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main(camera_key: str) -> None:
    asyncio.run(run(camera_key))


def run_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
    """Direct Python helper to invoke a tool without spawning a subprocess.

    This is used by the ``rosclaw mcp call`` CLI for built-in servers.
    """
    if tool_name == "get_camera_status":
        # camera_key is inferred from kwargs or defaults to d405
        camera_key = kwargs.get("camera_key", "realsense_d405")
        spec = CAMERAS.get(camera_key, CAMERAS["realsense_d405"])
        return _tool_get_camera_status(camera_key, spec)
    if tool_name == "list_camera_topics":
        return _tool_list_camera_topics()
    if tool_name == "capture_rgb_frame":
        return _tool_capture_rgb_frame(kwargs)
    if tool_name == "capture_depth_frame":
        return _tool_capture_depth_frame(kwargs)
    if tool_name == "capture_rgbd_pair":
        return _tool_capture_rgbd_pair(kwargs)
    if tool_name == "capture_pointcloud_snapshot":
        return _tool_capture_pointcloud_snapshot(kwargs)
    if tool_name == "check_depth_validity":
        return _tool_check_depth_validity(kwargs)
    if tool_name == "get_imu_sample":
        camera_key = kwargs.get("camera_key", "realsense_d435i")
        kwargs.setdefault("camera_key", camera_key)
        return _tool_get_imu_sample(kwargs)
    raise ValueError(f"Unknown tool: {tool_name}")
