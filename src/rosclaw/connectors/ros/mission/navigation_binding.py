"""Immutable SIM navigation configuration copied from an already compiled Body."""

import math
import re
from dataclasses import dataclass

from rosclaw.body.schema import EffectiveBody


@dataclass(frozen=True)
class SimulationNavigationBinding:
    body_id: str
    body_snapshot_hash: str
    action_name: str
    lease_service: str
    observation_topic: str
    frame_id: str
    run_id: str
    max_goal_distance_m: float
    max_duration_sec: float
    goal_position_tolerance_m: float
    goal_yaw_tolerance_rad: float
    max_speed_mps: float
    max_yaw_rate_rps: float
    stop_speed_mps: float
    stop_yaw_rate_rps: float

    def __post_init__(self):
        names = (self.action_name, self.lease_service, self.observation_topic)
        if any(
            type(n) is not str
            or not re.fullmatch(r"/(?:[A-Za-z_][A-Za-z_0-9]*/)*[A-Za-z_][A-Za-z_0-9]*", n)
            for n in names
        ):
            raise ValueError("explicit fully qualified ROS navigation endpoints required")
        if len(set(names)) != 3:
            raise ValueError("distinct navigation, lease and observation endpoints required")
        if any(type(s) is not str or not s for s in (self.body_id, self.frame_id, self.run_id)):
            raise ValueError("Body, map frame and SIM run identity required")
        if type(self.body_snapshot_hash) is not str or not re.fullmatch(
            r"[a-f0-9]{64}", self.body_snapshot_hash
        ):
            raise ValueError("compiled Body hash required")
        limits = (
            self.max_goal_distance_m,
            self.max_duration_sec,
            self.goal_position_tolerance_m,
            self.goal_yaw_tolerance_rad,
            self.max_speed_mps,
            self.max_yaw_rate_rps,
            self.stop_speed_mps,
            self.stop_yaw_rate_rps,
        )
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in limits):
            raise ValueError("explicit finite positive navigation limits required")
        if (
            self.stop_speed_mps > self.max_speed_mps
            or self.stop_yaw_rate_rps > self.max_yaw_rate_rps
        ):
            raise ValueError("stop thresholds must not exceed navigation limits")

    @classmethod
    def from_body(cls, body: EffectiveBody):
        if body.effective_body_hash != body.compute_hash():
            raise ValueError("compiled Body hash mismatch")
        if (
            body.safety.get("environment", {}).get("real_robot_execution_allowed") is not False
            or "navigation.navigate_to_pose"
            not in (body.capabilities.get("enabled", []) + body.capabilities.get("degraded", []))
            or "navigation.navigate_to_pose"
            in (body.capabilities.get("blocked", []) + body.capabilities.get("disabled", []))
        ):
            raise ValueError("explicit SIM-only Body with unblocked navigation required")
        bindings = body.provider_interfaces.get("ros_capability_bindings", {})
        nav = bindings.get("navigation.navigate_to_pose", {})
        lease = bindings.get("safety.simulation_lease", {})
        witness = bindings.get("safety.simulation_observation", {})
        limits = body.safety.get("navigation_simulation", {})
        keys = {
            "max_goal_distance_m",
            "max_duration_sec",
            "goal_position_tolerance_m",
            "goal_yaw_tolerance_rad",
            "max_speed_mps",
            "max_yaw_rate_rps",
            "stop_speed_mps",
            "stop_yaw_rate_rps",
        }
        if (
            nav.get("action_type") != "nav2_msgs/action/NavigateToPose"
            or lease.get("srv_type") != "std_srvs/srv/SetBool"
            or witness.get("msg_type") != "std_msgs/msg/String"
            or witness.get("evidence_domain") != "GAZEBO_PHYSICS"
            or type(limits) is not dict
            or set(limits) != keys
        ):
            raise ValueError("typed SIM navigation, deadman, witness and explicit limits required")
        frame = body.frames.get("map")
        if not isinstance(frame, str):
            raise ValueError("explicit map frame required")
        return cls(
            body.body_instance_id,
            body.effective_body_hash,
            nav.get("name"),
            lease.get("name"),
            witness.get("name"),
            frame,
            witness.get("run_id"),
            **limits,
        )

    def validate_observation(self, sample):
        if (
            sample.get("body_id") != self.body_id
            or sample.get("body_snapshot_hash") != self.body_snapshot_hash
            or sample.get("run_id") != self.run_id
            or sample.get("frame_id") != self.frame_id
            or type(sample.get("lease_enabled")) is not bool
            or sample.get("evidence_domain") != "GAZEBO_PHYSICS"
            or sample.get("observation_complete") is not True
            or type(sample.get("collision_count")) is not int
            or sample["collision_count"] != 0
        ):
            raise ValueError("independent SIM witness identity or lease state mismatch")
        if any(
            type(sample.get(k)) not in (int, float) or not math.isfinite(sample[k])
            for k in ("x", "y", "yaw", "time_sec")
        ):
            raise ValueError("finite independent SIM witness pose/time required")
        for key, limit in (
            ("linear_speed_mps", self.max_speed_mps),
            ("angular_speed_rps", self.max_yaw_rate_rps),
        ):
            value = sample.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > limit:
                raise ValueError("independent SIM witness missing or excessive speed: " + key)

    def stopped(self, sample):
        self.validate_observation(sample)
        return (
            sample["lease_enabled"] is False
            and abs(sample["linear_speed_mps"]) <= self.stop_speed_mps
            and abs(sample["angular_speed_rps"]) <= self.stop_yaw_rate_rps
        )
