from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from rosclaw.connectors.ros.transport import RosTransportResult
from rosclaw.simforge.tasks.guarded_base import (
    RosbridgeMobileBaseSink,
    RosbridgeMobileBaseStopDriver,
    read_mobile_base_pose,
)


@dataclass
class _Transport:
    messages: list[dict[str, Any]]
    published: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def subscribe_once(
        self,
        topic: str,
        msg_type: str | None = None,
        timeout_sec: float = 5.0,
        throttle_rate_ms: int = 0,
    ) -> RosTransportResult:
        assert topic and msg_type and timeout_sec > 0 and throttle_rate_ms == 0
        if not self.messages:
            return RosTransportResult(ok=False, error="no observation")
        return RosTransportResult(ok=True, data={"msg": self.messages.pop(0)})

    def publish(self, topic: str, message: dict[str, Any]) -> RosTransportResult:
        self.published.append((topic, message))
        return RosTransportResult(ok=True)


def _odometry(x_m: float, velocity_mps: float) -> dict[str, Any]:
    return {
        "pose": {"pose": {"position": {"x": x_m, "y": 0.0, "z": 0.0}}},
        "twist": {"twist": {"linear": {"x": velocity_mps, "y": 0.0, "z": 0.0}}},
    }


def test_gazebo_odometry_is_normalized_without_ros_python() -> None:
    transport = _Transport([_odometry(1.25, -0.20)])
    assert read_mobile_base_pose(
        transport,  # type: ignore[arg-type]
        topic="/guarded_base/odom",
        message_type="nav_msgs/msg/Odometry",
        timeout_sec=1.0,
    ) == (1.25, -0.20)


def test_deadman_stop_driver_requires_zero_velocity_observation() -> None:
    transport = _Transport([_odometry(0.5, 0.0)])
    result = RosbridgeMobileBaseStopDriver(
        transport,  # type: ignore[arg-type]
        command_topic="/guarded_base/guarded_cmd_vel",
        pose_topic="/guarded_base/odom",
        pose_message_type="nav_msgs/msg/Odometry",
    ).emergency_stop()

    assert result["acknowledged"] is True
    assert result["physical_stop_observed"] is True
    assert result["observed_velocity"] == 0.0
    assert transport.published[0][1]["linear"]["x"] == 0.0


def test_ros_sink_cannot_be_owned_by_provider_worker() -> None:
    with pytest.raises(ValueError, match="daemon instance id"):
        RosbridgeMobileBaseSink(
            _Transport([]),  # type: ignore[arg-type]
            daemon_owner_id="provider_worker_direct",
            command_topic="/guarded_base/guarded_cmd_vel",
            pose_topic="/guarded_base/odom",
            pose_message_type="nav_msgs/msg/Odometry",
        )
