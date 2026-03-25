"""Vision Tools - MCP tool definitions for camera control.

This module registers all MCP tools for RealSense RGB-D camera access.
"""

import base64
import io
import sys
from typing import TYPE_CHECKING, Optional

import numpy as np
from fastmcp import FastMCP
from mcp.types import ToolAnnotations
from PIL import Image

if TYPE_CHECKING:
    from rosclaw_mcp.servers.vision.config import VisionConfig


class CameraInterface:
    """Camera interface wrapper supporting multiple backends."""

    def __init__(self, config: "VisionConfig"):
        self.config = config
        self.pipeline = None
        self.profile = None
        self.is_mock = config.camera_type == "mock"
        self.mock_frame_count = 0

    def connect(self) -> bool:
        """Connect to camera."""
        try:
            if self.config.camera_type == "realsense":
                import pyrealsense2 as rs

                self.pipeline = rs.pipeline()
                rs_config = rs.config()

                # Enable streams
                if self.config.camera_serial:
                    rs_config.enable_device(self.config.camera_serial)

                rs_config.enable_stream(
                    rs.stream.color,
                    self.config.width,
                    self.config.height,
                    rs.format.bgr8,
                    self.config.fps,
                )

                if self.config.enable_depth:
                    rs_config.enable_stream(
                        rs.stream.depth,
                        self.config.width,
                        self.config.height,
                        rs.format.z16,
                        self.config.fps,
                    )

                self.profile = self.pipeline.start(rs_config)

                # Get depth scale
                if self.config.enable_depth:
                    depth_sensor = self.profile.get_device().first_depth_sensor()
                    self.depth_scale = depth_sensor.get_depth_scale()

                return True

            elif self.config.camera_type == "opencv":
                import cv2

                self.pipeline = cv2.VideoCapture(0)
                self.pipeline.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
                self.pipeline.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
                self.pipeline.set(cv2.CAP_PROP_FPS, self.config.fps)

                return self.pipeline.isOpened()

            elif self.config.camera_type == "mock":
                return True

        except ImportError as e:
            print(f"[Vision] Import error: {e}", file=sys.stderr)
            return False
        except Exception as e:
            print(f"[Vision] Connection error: {e}", file=sys.stderr)
            return False

    def disconnect(self):
        """Disconnect from camera."""
        if self.config.camera_type == "realsense" and self.pipeline:
            self.pipeline.stop()
        elif self.config.camera_type == "opencv" and self.pipeline:
            self.pipeline.release()
        self.pipeline = None

    def get_frame(self) -> Optional[dict]:
        """Get a single frame from camera."""
        if self.is_mock:
            return self._get_mock_frame()

        if self.pipeline is None:
            if not self.connect():
                return None

        try:
            if self.config.camera_type == "realsense":
                import pyrealsense2 as rs

                frames = self.pipeline.wait_for_frames(timeout_ms=5000)

                result = {}

                # Get color frame
                if self.config.enable_color:
                    color_frame = frames.get_color_frame()
                    if color_frame:
                        color_image = np.asanyarray(color_frame.get_data())
                        result["color"] = color_image
                        result["color_shape"] = color_image.shape

                # Get depth frame
                if self.config.enable_depth:
                    depth_frame = frames.get_depth_frame()
                    if depth_frame:
                        depth_image = np.asanyarray(depth_frame.get_data())
                        result["depth"] = depth_image
                        result["depth_shape"] = depth_image.shape
                        result["depth_scale"] = getattr(self, "depth_scale", 0.001)

                return result

            elif self.config.camera_type == "opencv":
                ret, frame = self.pipeline.read()
                if ret:
                    return {
                        "color": frame,
                        "color_shape": frame.shape,
                    }
                return None

        except Exception as e:
            print(f"[Vision] Frame capture error: {e}", file=sys.stderr)
            return None

    def _get_mock_frame(self) -> dict:
        """Generate mock frame for testing."""
        self.mock_frame_count += 1

        # Create gradient image
        x = np.linspace(0, 255, self.config.width)
        y = np.linspace(0, 255, self.config.height)
        xx, yy = np.meshgrid(x, y)
        color = np.stack([xx, yy, 255 - xx], axis=2).astype(np.uint8)

        # Create synthetic depth
        depth = (1000 + 500 * np.sin(self.mock_frame_count * 0.1) + np.random.normal(0, 10, (self.config.height, self.config.width))).astype(np.uint16)

        return {
            "color": color,
            "color_shape": color.shape,
            "depth": depth,
            "depth_shape": depth.shape,
            "depth_scale": 0.001,
            "mock": True,
        }


def encode_image_to_base64(image: np.ndarray, format: str = "JPEG") -> str:
    """Encode numpy image to base64 string."""
    # Convert BGR to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = image[:, :, ::-1]  # BGR to RGB

    pil_image = Image.fromarray(image)
    buffer = io.BytesIO()
    pil_image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def register_vision_tools(mcp: FastMCP, config: "VisionConfig") -> None:
    """Register all Vision MCP tools."""

    camera: Optional[CameraInterface] = None

    def get_camera():
        """Get or create camera interface."""
        nonlocal camera
        if camera is None:
            camera = CameraInterface(config)
            if not camera.connect():
                return None
        return camera

    @mcp.tool(
        description=(
            "Capture a single RGB image from camera.\n"
            "Example: capture_image()\n"
            "Returns base64-encoded JPEG image."
        ),
        annotations=ToolAnnotations(
            title="Capture Image",
            readOnlyHint=True,
        ),
    )
    def capture_image() -> dict:
        """Capture a single RGB image."""
        cam = get_camera()
        if cam is None:
            return {"error": "Camera not available"}

        frame = cam.get_frame()
        if frame is None:
            return {"error": "Failed to capture frame"}

        if "color" not in frame:
            return {"error": "No color image available"}

        color_image = frame["color"]
        encoded = encode_image_to_base64(color_image)

        return {
            "success": True,
            "image_base64": encoded,
            "width": color_image.shape[1],
            "height": color_image.shape[0],
            "channels": color_image.shape[2] if len(color_image.shape) > 2 else 1,
            "format": "JPEG",
            "mock": frame.get("mock", False),
        }

    @mcp.tool(
        description=(
            "Capture depth image from camera.\n"
            "Example: capture_depth()\n"
            "Returns depth data as list of lists (may be large)."
        ),
        annotations=ToolAnnotations(
            title="Capture Depth",
            readOnlyHint=True,
        ),
    )
    def capture_depth() -> dict:
        """Capture depth image."""
        cam = get_camera()
        if cam is None:
            return {"error": "Camera not available"}

        frame = cam.get_frame()
        if frame is None:
            return {"error": "Failed to capture frame"}

        if "depth" not in frame:
            return {"error": "No depth image available"}

        depth_image = frame["depth"]

        # Downsample for transmission (too large otherwise)
        h, w = depth_image.shape
        downsample_factor = max(1, min(h, w) // 100)
        depth_downsampled = depth_image[::downsample_factor, ::downsample_factor]

        return {
            "success": True,
            "depth_data": depth_downsampled.tolist(),
            "original_shape": [h, w],
            "downsampled_shape": list(depth_downsampled.shape),
            "depth_scale_meters": frame.get("depth_scale", 0.001),
            "min_depth_mm": int(depth_image.min()),
            "max_depth_mm": int(depth_image.max()),
            "mock": frame.get("mock", False),
        }

    @mcp.tool(
        description=(
            "Capture synchronized RGB and depth images.\n"
            "Example: capture_rgbd()\n"
            "Returns both color (base64) and depth data."
        ),
        annotations=ToolAnnotations(
            title="Capture RGB-D",
            readOnlyHint=True,
        ),
    )
    def capture_rgbd() -> dict:
        """Capture synchronized RGB-D frames."""
        cam = get_camera()
        if cam is None:
            return {"error": "Camera not available"}

        frame = cam.get_frame()
        if frame is None:
            return {"error": "Failed to capture frame"}

        result = {
            "success": True,
            "mock": frame.get("mock", False),
        }

        if "color" in frame:
            color_image = frame["color"]
            result["image_base64"] = encode_image_to_base64(color_image)
            result["color_width"] = color_image.shape[1]
            result["color_height"] = color_image.shape[0]

        if "depth" in frame:
            depth_image = frame["depth"]
            h, w = depth_image.shape
            downsample_factor = max(1, min(h, w) // 100)
            depth_downsampled = depth_image[::downsample_factor, ::downsample_factor]
            result["depth_data"] = depth_downsampled.tolist()
            result["depth_shape"] = [h, w]
            result["depth_scale_meters"] = frame.get("depth_scale", 0.001)

        return result

    @mcp.tool(
        description=(
            "Get camera configuration and status.\n"
            "Example: get_camera_info()"
        ),
        annotations=ToolAnnotations(
            title="Get Camera Info",
            readOnlyHint=True,
        ),
    )
    def get_camera_info() -> dict:
        """Get camera configuration and status."""
        cam = get_camera()
        connected = cam is not None and cam.pipeline is not None

        return {
            "camera_type": config.camera_type,
            "serial": config.camera_serial or "any",
            "width": config.width,
            "height": config.height,
            "fps": config.fps,
            "enable_depth": config.enable_depth,
            "enable_color": config.enable_color,
            "connected": connected,
            "is_mock": config.camera_type == "mock",
        }

    @mcp.tool(
        description=(
            "Get depth value at specific pixel coordinates.\n"
            "Example: get_depth_at(x=320, y=240)\n"
            "Returns depth in meters at specified pixel."
        ),
        annotations=ToolAnnotations(
            title="Get Depth At",
            readOnlyHint=True,
        ),
    )
    def get_depth_at(x: int, y: int) -> dict:
        """Get depth value at specific pixel coordinates."""
        cam = get_camera()
        if cam is None:
            return {"error": "Camera not available"}

        frame = cam.get_frame()
        if frame is None:
            return {"error": "Failed to capture frame"}

        if "depth" not in frame:
            return {"error": "No depth image available"}

        depth_image = frame["depth"]
        h, w = depth_image.shape

        if not (0 <= x < w and 0 <= y < h):
            return {"error": f"Coordinates ({x}, {y}) out of bounds [{w}x{h}]"}

        depth_mm = depth_image[y, x]
        depth_m = depth_mm * frame.get("depth_scale", 0.001)

        return {
            "success": True,
            "x": x,
            "y": y,
            "depth_meters": float(depth_m),
            "depth_mm": int(depth_mm),
            "mock": frame.get("mock", False),
        }

    @mcp.tool(
        description=(
            "Reconnect to camera.\n"
            "Example: reconnect_camera()"
        ),
        annotations=ToolAnnotations(
            title="Reconnect Camera",
            destructiveHint=True,
        ),
    )
    def reconnect_camera() -> dict:
        """Reconnect to camera."""
        nonlocal camera

        if camera is not None:
            camera.disconnect()
            camera = None

        cam = get_camera()
        return {
            "success": cam is not None,
            "connected": cam is not None and cam.pipeline is not None,
        }
