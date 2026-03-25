"""Vision Configuration - Settings for camera control.

This module defines configuration for RGB-D camera access.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class VisionConfig:
    """Configuration for Vision MCP server."""

    # Camera selection
    camera_serial: str = ""  # Empty for any camera
    camera_type: str = "realsense"  # realsense, opencv, mock

    # Resolution
    width: int = 640
    height: int = 480
    fps: int = 30

    # Depth settings
    enable_depth: bool = True
    enable_color: bool = True
    align_depth_to_color: bool = True

    # Processing options
    enable_pointcloud: bool = False

    def __post_init__(self):
        """Validate configuration."""
        if self.camera_type not in ["realsense", "opencv", "mock"]:
            raise ValueError(f"Invalid camera type: {self.camera_type}")

        if self.width <= 0 or self.height <= 0:
            raise ValueError("Width and height must be positive")

        if self.fps <= 0:
            raise ValueError("FPS must be positive")
