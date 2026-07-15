#!/usr/bin/env python3
"""ROS2 camera frame bridge for the RH56 RPS demo.

This helper is intentionally standalone and must be run with the system ROS 2
Python interpreter (e.g. /usr/bin/python3 on Ubuntu 24.04 + ROS 2 Jazzy).  It
subscribes to a sensor_msgs/Image topic, converts frames with cv_bridge, and
streams raw BGR frames over a local TCP socket to avoid JPEG latency.

Protocol:
    1. Helper prints "PORT <port>\n" on stdout once the server socket is ready.
    2. Client connects to <port>.
    3. Each frame is sent as: 4-byte big-endian length + raw BGR bytes
       (length == width * height * 3).
"""
from __future__ import annotations

import argparse
import queue
import socket
import struct
import sys
import threading

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


def main() -> int:
    parser = argparse.ArgumentParser(description="ROS2 camera frame bridge")
    parser.add_argument("topic", type=str, help="sensor_msgs/Image topic")
    parser.add_argument("width", type=int, help="desired output width")
    parser.add_argument("height", type=int, help="desired output height")
    parser.add_argument("fps", type=int, nargs="?", default=30, help="max stream fps")
    parser.add_argument("--port", type=int, default=0, help="server port (0 = auto)")
    args = parser.parse_args()

    frame_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=2)
    bridge = CvBridge()

    rclpy.init(args=[])
    node = rclpy.create_node("rosclaw_rps_camera_helper")

    # Prefer best-effort QoS for camera topics to avoid dropped frames.
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)

    def on_image(msg: Image) -> None:
        try:
            bgr = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            node.get_logger().warning(f"cv_bridge conversion failed: {exc}")
            return
        if bgr is None or bgr.size == 0:
            return
        if (bgr.shape[1], bgr.shape[0]) != (args.width, args.height):
            bgr = cv2.resize(bgr, (args.width, args.height), interpolation=cv2.INTER_LINEAR)
        try:
            frame_queue.put_nowait(bgr)
        except queue.Full:
            try:
                frame_queue.get_nowait()
                frame_queue.put_nowait(bgr)
            except queue.Empty:
                pass

    subscription = node.create_subscription(Image, args.topic, on_image, qos)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", args.port))
    server.listen(1)
    host, port = server.getsockname()
    print(f"PORT {port}", flush=True)

    spinner = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spinner.start()

    server.settimeout(10.0)
    try:
        conn, addr = server.accept()
    except socket.timeout:
        node.get_logger().error("No client connected within 10 s")
        node.destroy_subscription(subscription)
        rclpy.shutdown()
        return 1
    server.settimeout(None)
    conn.settimeout(5.0)

    period_s = 1.0 / max(args.fps, 1)
    last_frame: np.ndarray | None = None
    expected_len = args.width * args.height * 3
    try:
        while rclpy.ok():
            try:
                frame = frame_queue.get(timeout=period_s)
                last_frame = frame
            except queue.Empty:
                frame = last_frame
            if frame is None:
                continue
            # Ensure contiguous array; .tobytes() returns raw BGR pixels.
            data = np.ascontiguousarray(frame).tobytes()
            if len(data) != expected_len:
                continue
            try:
                conn.sendall(struct.pack(">I", len(data)) + data)
            except (BrokenPipeError, ConnectionResetError, OSError):
                break
    except KeyboardInterrupt:
        pass
    finally:
        conn.close()
        server.close()
        node.destroy_subscription(subscription)
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
