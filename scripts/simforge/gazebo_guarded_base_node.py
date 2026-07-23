#!/usr/bin/env python3
"""ROS 2 nodes used by the simulation-only Gazebo GuardedBase harness.

The deadman is a narrow safety boundary: only fresh commands on the guarded
input topic are forwarded to Gazebo. If the command source, rosbridge, or its
worker disappears, a zero Twist is emitted within ``timeout-sec``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Any


def _append_event(path: Path | None, payload: dict[str, Any]) -> None:
    record = {
        "monotonic_ns": time.monotonic_ns(),
        "wall_time_ns": time.time_ns(),
        **payload,
    }
    line = json.dumps(record, sort_keys=True)
    print(f"ROSCLAW_GUARDED_BASE_EVENT {line}", flush=True)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _is_nonzero(message: Any) -> bool:
    values = (
        message.linear.x,
        message.linear.y,
        message.linear.z,
        message.angular.x,
        message.angular.y,
        message.angular.z,
    )
    return any(abs(float(value)) > 1e-9 for value in values)


def run_deadman(args: argparse.Namespace) -> int:
    import rclpy
    from geometry_msgs.msg import Twist
    from rclpy.node import Node

    event_log = Path(args.event_log) if args.event_log else None

    class GuardedBaseDeadman(Node):
        def __init__(self) -> None:
            super().__init__("rosclaw_guarded_base_deadman")
            self._publisher = self.create_publisher(Twist, args.output_topic, 10)
            self._last_fresh = 0.0
            self._motion_active = False
            self._lock = threading.RLock()
            self.create_subscription(Twist, args.input_topic, self._on_command, 10)
            self.create_timer(min(0.05, args.timeout_sec / 4.0), self._check_deadman)
            _append_event(
                event_log,
                {
                    "event": "deadman_ready",
                    "pid": os.getpid(),
                    "input_topic": args.input_topic,
                    "output_topic": args.output_topic,
                    "timeout_sec": args.timeout_sec,
                },
            )

        def _on_command(self, message: Twist) -> None:
            values = (message.linear.x, message.angular.z)
            if not all(math.isfinite(float(value)) for value in values):
                self._publish_stop("non_finite_command")
                return
            with self._lock:
                was_active = self._motion_active
                self._last_fresh = time.monotonic()
                self._motion_active = _is_nonzero(message)
                self._publisher.publish(message)
            if not was_active or not self._motion_active:
                _append_event(
                    event_log,
                    {
                        "event": "command_forwarded",
                        "linear_x_mps": float(message.linear.x),
                        "angular_z_radps": float(message.angular.z),
                    },
                )

        def _check_deadman(self) -> None:
            with self._lock:
                expired = self._motion_active and (
                    time.monotonic() - self._last_fresh >= args.timeout_sec
                )
            if expired:
                self._publish_stop("fresh_command_timeout")

        def _publish_stop(self, reason: str) -> None:
            with self._lock:
                was_active = self._motion_active
                self._publisher.publish(Twist())
                self._motion_active = False
            if was_active or reason != "shutdown":
                _append_event(
                    event_log,
                    {
                        "event": "deadman_stop",
                        "reason": reason,
                        "timeout_sec": args.timeout_sec,
                    },
                )

        def safe_shutdown(self) -> None:
            self._publish_stop("shutdown")
            for _ in range(2):
                self._publisher.publish(Twist())
                time.sleep(0.02)

    rclpy.init()
    node = GuardedBaseDeadman()
    try:
        rclpy.spin(node)
    finally:
        if rclpy.ok():
            node.safe_shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


def run_command_source(args: argparse.Namespace) -> int:
    """Publish a leased heartbeat command; SIGKILL intentionally leaves no stop."""

    import rclpy
    from geometry_msgs.msg import Twist
    from rclpy.node import Node

    event_log = Path(args.event_log) if args.event_log else None

    class GuardedCommandSource(Node):
        def __init__(self) -> None:
            safe_action_id = "".join(
                character if character.isalnum() or character == "_" else "_"
                for character in args.action_id
            )
            super().__init__(f"rosclaw_command_source_{safe_action_id}")
            self._publisher = self.create_publisher(Twist, args.input_topic, 10)
            self._started = time.monotonic()
            self.create_timer(1.0 / args.rate_hz, self._publish)
            _append_event(
                event_log,
                {
                    "event": "command_source_ready",
                    "pid": os.getpid(),
                    "action_id": args.action_id,
                    "input_topic": args.input_topic,
                },
            )

        def _publish(self) -> None:
            if (
                args.crash_after_sec is not None
                and time.monotonic() - self._started >= args.crash_after_sec
            ):
                _append_event(
                    event_log,
                    {
                        "event": "command_source_crash",
                        "pid": os.getpid(),
                        "action_id": args.action_id,
                        "exit_code": 73,
                    },
                )
                os._exit(73)
            message = Twist()
            message.linear.x = args.linear_x_mps
            message.angular.z = args.angular_z_radps
            self._publisher.publish(message)

    rclpy.init()
    node = GuardedCommandSource()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    deadman = subparsers.add_parser("deadman")
    deadman.add_argument(
        "--input-topic",
        default="/guarded_base/guarded_cmd_vel",
    )
    deadman.add_argument("--output-topic", default="/guarded_base/cmd_vel")
    deadman.add_argument("--timeout-sec", type=float, default=0.35)
    deadman.add_argument("--event-log", default="")

    source = subparsers.add_parser("command-source")
    source.add_argument(
        "--input-topic",
        default="/guarded_base/guarded_cmd_vel",
    )
    source.add_argument("--linear-x-mps", type=float, default=0.2)
    source.add_argument("--angular-z-radps", type=float, default=0.0)
    source.add_argument("--rate-hz", type=float, default=20.0)
    source.add_argument("--action-id", required=True)
    source.add_argument("--crash-after-sec", type=float)
    source.add_argument("--event-log", default="")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.mode == "deadman":
        if not 0.05 <= args.timeout_sec <= 5.0:
            raise SystemExit("--timeout-sec must be in [0.05, 5.0]")
        return run_deadman(args)
    if args.rate_hz <= 0 or not math.isfinite(args.rate_hz):
        raise SystemExit("--rate-hz must be finite and positive")
    return run_command_source(args)


if __name__ == "__main__":
    raise SystemExit(main())
