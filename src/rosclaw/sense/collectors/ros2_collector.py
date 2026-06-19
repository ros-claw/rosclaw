"""ROS2 collector stub.

The real implementation will subscribe to ROS2 topics and map them to
BodyState fields.  Lazy imports ensure this file can be loaded without rclpy.
"""

from __future__ import annotations

import logging
import time

from rosclaw.sense.collectors.base import BodyStateCollector
from rosclaw.sense.schemas import BodyState

logger = logging.getLogger("rosclaw.sense.collectors.ros2")


class ROS2Collector(BodyStateCollector):
    """Collect BodyState from ROS2 topics.

    This is a stub for Phase 1.  It imports ``rclpy`` lazily and returns an
    unknown state until a real ROS2 environment is available.
    """

    name = "ros2"

    def __init__(self, robot_id: str = "unknown", topic_prefix: str = "/"):
        self.robot_id = robot_id
        self.topic_prefix = topic_prefix
        self._rclpy = None

    def _ensure_rclpy(self) -> bool:
        if self._rclpy is not None:
            return True
        try:
            import rclpy
            self._rclpy = rclpy
            return True
        except ImportError:
            logger.warning("rclpy not available; ROS2 collector will return unknown state")
            return False

    def start(self) -> None:
        if self._ensure_rclpy():
            logger.info("ROS2 collector started (stub)")

    def stop(self) -> None:
        if self._rclpy is not None:
            logger.info("ROS2 collector stopped (stub)")

    def collect(self) -> BodyState:
        self._ensure_rclpy()
        return BodyState(
            robot_id=self.robot_id,
            timestamp=time.time(),
            source="ros2:stub",
        )
