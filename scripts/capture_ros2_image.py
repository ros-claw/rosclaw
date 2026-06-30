#!/usr/bin/env python3
"""Capture the latest ROS2 image frame from a RealSense color topic.

Usage:
    ./scripts/capture_ros2_image.py --topic /camera/d405/color/image_raw \
                                    --output /tmp/d405_color.jpg
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _save_image(msg, output_path: Path) -> dict:
    import numpy as np
    from PIL import Image as PILImage

    h, w = msg.height, msg.width
    encoding = msg.encoding
    data = np.frombuffer(msg.data, dtype=np.uint8)

    if encoding == "rgb8":
        arr = data.reshape((h, w, 3))
    elif encoding == "bgr8":
        arr = data.reshape((h, w, 3))[:, :, ::-1]
    elif encoding in ("mono8", "8UC1"):
        arr = data.reshape((h, w))
    elif encoding == "16UC1":
        # Save depth as 16-bit PNG if output name suggests PNG
        arr = np.frombuffer(msg.data, dtype=np.uint16).reshape((h, w))
        if output_path.suffix.lower() != ".png":
            output_path = output_path.with_suffix(".png")
        PILImage.fromarray(arr).save(output_path)
        return {"path": str(output_path), "width": w, "height": h, "encoding": encoding}
    else:
        raise ValueError(f"Unsupported image encoding: {encoding}")

    PILImage.fromarray(arr).save(output_path, quality=95)
    return {"path": str(output_path), "width": w, "height": h, "encoding": encoding}


def capture(topic: str, output: Path, timeout_sec: float = 5.0) -> dict:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image

    rclpy.init(args=None)
    node = Node("rosclaw_capture_ros2_image")
    result: dict = {}
    received = False

    def _cb(msg: Image) -> None:
        nonlocal received, result
        if received:
            return
        try:
            result = _save_image(msg, output)
            result["topic"] = topic
            result["timestamp"] = time.time()
            received = True
        except Exception as exc:  # noqa: BLE001
            result = {"error": str(exc)}
            received = True

    node.create_subscription(Image, topic, _cb, qos_profile=1)

    start = time.monotonic()
    while not received and time.monotonic() - start < timeout_sec:
        rclpy.spin_once(node, timeout_sec=0.1)

    node.destroy_node()
    rclpy.shutdown()

    if not received:
        return {"error": f"No image received on {topic} within {timeout_sec}s"}
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture a ROS2 image frame")
    parser.add_argument("--topic", default="/camera/d405/color/image_raw", help="ROS2 image topic")
    parser.add_argument("--output", default="/tmp/realsense_capture.jpg", help="Output image path")
    parser.add_argument("--timeout", type=float, default=5.0, help="Wait timeout in seconds")
    args = parser.parse_args(argv)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    result = capture(args.topic, output, args.timeout)
    if result.get("error"):
        print(f"ERROR: {result['error']}", file=sys.stderr)
        return 1

    print(f"Saved {result['width']}x{result['height']} {result['encoding']} -> {result['path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
