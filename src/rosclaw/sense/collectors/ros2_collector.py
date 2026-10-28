"""Explicit ROS2 telemetry for an operator-owned Sense process.

This collector grants no action authority. Native agents consume the existing
Sense projection, not this ROS transport. Topic selection must be configured by
an operator; discovering a topic never enrolls it as body telemetry.
"""

from __future__ import annotations

import math
import re
import threading
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from rosclaw.sense.collectors.base import BodyStateCollector
from rosclaw.sense.schemas import BodyState, EnergyState, IMUState


@dataclass(frozen=True)
class ROS2TelemetryBinding:
    topic: str
    max_age_s: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.topic, str)
            or re.fullmatch(r"(?:/[A-Za-z_][A-Za-z0-9_]*)+", self.topic) is None
        ):
            raise ValueError("Explicit absolute telemetry topic required")
        if not math.isfinite(self.max_age_s) or self.max_age_s <= 0:
            raise ValueError("Finite positive telemetry age required")


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _battery(message: Any) -> EnergyState:
    if not message.present:
        return EnergyState()
    percentage = _finite(message.percentage)
    voltage = _finite(message.voltage)
    current = _finite(message.current)
    # sensor_msgs/BatteryState: percentage 0..1; discharge current negative.
    return EnergyState(
        battery_percent=percentage * 100
        if percentage is not None and 0 <= percentage <= 1
        else None,
        voltage=voltage if voltage is not None and voltage >= 0 else None,
        current_draw=-current if current is not None else None,
        power_mode="charging" if message.power_supply_status == 1 else "unknown",
    )


def _imu(message: Any) -> IMUState:
    result = IMUState()
    q = message.orientation
    values = [_finite(v) for v in [q.x, q.y, q.z, q.w]]
    if message.orientation_covariance[0] != -1 and all(v is not None for v in values):
        x, y, z, w = (float(v) for v in values)  # type: ignore[arg-type]
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if abs(norm - 1.0) <= 0.01:
            x, y, z, w = (v / norm for v in [x, y, z, w])
            result.roll_deg = math.degrees(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
            result.pitch_deg = math.degrees(math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))))
            result.yaw_deg = math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    angular = [
        _finite(v)
        for v in [
            message.angular_velocity.x,
            message.angular_velocity.y,
            message.angular_velocity.z,
        ]
    ]
    if message.angular_velocity_covariance[0] != -1 and all(v is not None for v in angular):
        result.angular_velocity = [float(v) for v in angular]  # type: ignore[arg-type]
    return result


class ROS2Collector(BodyStateCollector):
    """Battery/IMU subscriptions; absent, stale and replayed data stay unknown.

    No default topic names, body identity verification, collision safety, joint
    interpretation or camera health is inferred. Lifecycle owns a separate ROS
    context/executor and closes them; stop never leaves a usable cached value.
    """

    name = "ros2"

    def __init__(
        self,
        robot_id: str = "unknown",
        topic_prefix: str = "/",
        *,
        topics: dict[str, dict[str, Any]] | None = None,
        use_sim_time: bool = False,
    ):
        self.robot_id = robot_id
        self.topic_prefix = topic_prefix  # Compatibility only; never enroll topics by name.
        self.bindings = MappingProxyType(
            {role: ROS2TelemetryBinding(**value) for role, value in (topics or {}).items()}
        )
        if set(self.bindings) - {"battery", "imu"}:
            raise ValueError("Only explicitly typed battery/imu telemetry is supported")
        if not isinstance(use_sim_time, bool):
            raise ValueError("Explicit boolean ROS clock configuration required")
        self.use_sim_time = use_sim_time
        self._lock = threading.RLock()
        self._records: dict[str, dict[str, Any]] = {}
        self._last_stamps: dict[str, float] = {}
        self._node: Any = None
        self._context: Any = None
        self._executor: Any = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._last_ros_now: float | None = None
        self._epoch = 0
        self._error: str | None = None

    def _ros_now(self) -> float:
        if self._node is None:
            return 0.0
        return self._node.get_clock().now().nanoseconds / 1e9

    def _check_clock(self, now: float) -> None:
        if self._last_ros_now is not None and now < self._last_ros_now - 0.1:
            self._records.clear()
            self._last_stamps.clear()
            self._epoch += 1
        self._last_ros_now = now

    def _receive(self, role: str, message: Any) -> None:
        with self._lock:
            if not self._running:
                return
            now = self._ros_now()
            self._check_clock(now)
            stamp = message.header.stamp.sec + message.header.stamp.nanosec / 1e9
            limit = self.bindings[role].max_age_s
            if not math.isfinite(stamp) or stamp <= 0 or not -0.1 <= now - stamp <= limit:
                return
            # A repeated old sample must not renew receive freshness.
            if stamp <= self._last_stamps.get(role, -math.inf):
                return
            try:
                value = _battery(message) if role == "battery" else _imu(message)
            except (AttributeError, IndexError, TypeError, ValueError):
                self._records.pop(role, None)
                self._error = "malformed " + role + " telemetry"
                return
            self._last_stamps[role] = stamp
            self._records[role] = {
                "value": value,
                "stamp": stamp,
                "wall": time.time(),
                "mono": time.monotonic(),
            }

    def start(self) -> None:
        if not self.bindings:
            return  # Explicit unconfigured state; never fall back to mock or guessed topics.
        if self._running or self._context is not None:
            raise RuntimeError(
                "ROS2 Sense collector already owns a context; stop it before restart"
            )
        try:
            import rclpy
            from rclpy.context import Context
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.parameter import Parameter
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import BatteryState, Imu
        except ImportError as error:
            raise RuntimeError("Configured ROS2 Sense transport unavailable") from error
        try:
            self._context = Context()
            rclpy.init(context=self._context)
            self._node = rclpy.create_node(
                "rosclaw_sense_" + uuid4().hex[:10],
                context=self._context,
                parameter_overrides=[Parameter("use_sim_time", value=self.use_sim_time)],
            )
            self._executor = SingleThreadedExecutor(context=self._context)
            self._executor.add_node(self._node)
            with self._lock:
                self._records.clear()
                self._last_stamps.clear()
                self._last_ros_now = None
                self._epoch += 1
                self._running = True
                self._error = None
            for role, binding in self.bindings.items():
                self._node.create_subscription(
                    BatteryState if role == "battery" else Imu,
                    binding.topic,
                    lambda msg, role=role: self._receive(role, msg),
                    qos_profile_sensor_data,
                )
            self._thread = threading.Thread(
                target=self._spin, name="rosclaw-sense-ros2", daemon=True
            )
            self._thread.start()
        except BaseException:
            self.stop()
            raise

    def _spin(self) -> None:
        try:
            self._executor.spin()
        except Exception as error:  # Transport loss invalidates cached telemetry.
            with self._lock:
                if self._running:
                    self._error = type(error).__name__
        finally:
            with self._lock:
                if self._running and self._error is None:
                    self._error = "ROS2 executor exited"
                self._running = False
                self._records.clear()

    def stop(self) -> None:
        with self._lock:
            self._running = False
            self._records.clear()
        if self._executor is not None:
            self._executor.shutdown(timeout_sec=2.0)
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                raise RuntimeError("ROS2 Sense executor did not stop; not safe to restart")
        if self._node is not None:
            self._node.destroy_node()
        if self._context is not None and self._context.ok():
            self._context.shutdown()
        self._node = self._context = self._executor = self._thread = None

    def collect(self) -> BodyState:
        with self._lock:
            now = self._ros_now()
            self._check_clock(now)
            wall = time.monotonic()
            roles = {}
            timestamp = max((r["wall"] for r in self._records.values()), default=0.0)
            state = BodyState(
                robot_id=self.robot_id,
                timestamp=timestamp,
                source="ros2:live"
                if self._running
                else "ros2:unconfigured"
                if not self.bindings
                else "ros2:unavailable",
            )
            for role, binding in self.bindings.items():
                row = self._records.get(role)
                header_age = now - row["stamp"] if row else None
                receive_age = wall - row["mono"] if row else None
                fresh = bool(
                    self._running
                    and row is not None
                    and header_age is not None
                    and receive_age is not None
                    and -0.1 <= header_age <= binding.max_age_s
                    and 0 <= receive_age <= binding.max_age_s
                )
                roles[role] = {
                    "topic": binding.topic,
                    "message_type": "sensor_msgs/msg/"
                    + ("BatteryState" if role == "battery" else "Imu"),
                    "fresh": fresh,
                    "original_header_stamp_s": row["stamp"] if row else None,
                    "original_receive_wall_s": row["wall"] if row else None,
                    "header_age_s": header_age,
                    "receive_age_s": receive_age,
                    "max_age_s": binding.max_age_s,
                }
                if fresh:
                    assert row is not None
                    if role == "battery":
                        state.energy = EnergyState(**vars(row["value"]))
                    else:
                        state.imu = IMUState(
                            **{
                                **vars(row["value"]),
                                "angular_velocity": list(row["value"].angular_velocity)
                                if row["value"].angular_velocity is not None
                                else None,
                            }
                        )
            state.raw = {
                "collector_configured": bool(self.bindings),
                "transport_running": self._running,
                "observation_epoch": self._epoch,
                "clock_domain": "ros_sim" if self.use_sim_time else "ros_system",
                "roles": roles,
                "error": self._error,
                "body_identity_verified": False,
                "action_authority": False,
            }
            return state
