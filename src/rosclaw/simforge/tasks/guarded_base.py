"""Daemon-instance-bound executor for canonical MCP -> rosclawd simulated motion."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Protocol

from rosclaw.connectors.ros.transport import RosbridgeTransport
from rosclaw.kernel import (
    AcknowledgementStage,
    ActionEnvelope,
    ActionExecutionResult,
    ActionState,
    EvidenceDomain,
    EvidenceLevel,
    ExecutionMode,
)


@dataclass(frozen=True)
class MobileBaseObservation:
    start_x_m: float
    final_x_m: float
    final_velocity_mps: float
    timestamp_monotonic: float

    def __post_init__(self) -> None:
        for name in (
            "start_x_m",
            "final_x_m",
            "final_velocity_mps",
            "timestamp_monotonic",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"mobile base observation {name} must be finite")
        if self.timestamp_monotonic < 0:
            raise ValueError("mobile base observation timestamp must be non-negative")


class DaemonOwnedRosSink(Protocol):
    @property
    def daemon_owner_id(self) -> str: ...

    def publish_velocity(
        self,
        linear_x_mps: float,
        angular_z_radps: float,
        duration_sec: float,
    ) -> bool: ...

    def observe_effect(self, timeout_sec: float) -> MobileBaseObservation | None: ...

    def stop(self) -> bool: ...


class RosbridgeMobileBaseSink:
    """Daemon-scoped rosbridge sink with before/after pose observations.

    This adapter deliberately exposes no generic publish method.  The only
    command it can emit is a bounded Twist followed by a zero-velocity stop,
    and the executor verifies the resulting pose before upgrading evidence.
    """

    def __init__(
        self,
        transport: RosbridgeTransport,
        *,
        daemon_owner_id: str,
        command_topic: str = "/turtle1/cmd_vel",
        pose_topic: str = "/turtle1/pose",
        pose_message_type: str = "turtlesim/msg/Pose",
        observation_timeout_sec: float = 5.0,
    ) -> None:
        if (
            not isinstance(daemon_owner_id, str)
            or not daemon_owner_id.startswith("daemon_")
            or len(daemon_owner_id) > 128
        ):
            raise ValueError("a rosclawd daemon instance id is required")
        if any(
            not isinstance(value, str) or not 1 <= len(value) <= 256
            for value in (command_topic, pose_topic, pose_message_type)
        ):
            raise ValueError("ROS command, pose, and message type names are required")
        if not _simulation_topics_allowed(command_topic, pose_topic):
            raise ValueError(
                "guarded simulation sink accepts turtlesim or explicit simulation namespaces only"
            )
        if (
            isinstance(observation_timeout_sec, bool)
            or not isinstance(observation_timeout_sec, (int, float))
            or not math.isfinite(float(observation_timeout_sec))
            or observation_timeout_sec <= 0
            or observation_timeout_sec > 60
        ):
            raise ValueError("observation timeout must be finite and positive")
        self._transport = transport
        self._daemon_owner_id = daemon_owner_id
        self._command_topic = command_topic
        self._pose_topic = pose_topic
        self._pose_message_type = pose_message_type
        self._observation_timeout_sec = observation_timeout_sec
        self._start_x_m: float | None = None
        self._motion_deadline: float | None = None

    @property
    def daemon_owner_id(self) -> str:
        return self._daemon_owner_id

    def publish_velocity(
        self,
        linear_x_mps: float,
        angular_z_radps: float,
        duration_sec: float,
    ) -> bool:
        values = (linear_x_mps, angular_z_radps, duration_sec)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in values
        ):
            return False
        if (
            abs(float(linear_x_mps)) > 0.5
            or abs(float(angular_z_radps)) > 1.0
            or not 0 < float(duration_sec) <= 30
            or abs(float(linear_x_mps)) * float(duration_sec) > 1.0
        ):
            return False
        start = self._read_pose(self._observation_timeout_sec)
        if start is None:
            return False
        result = self._transport.publish(
            self._command_topic,
            _twist_message(linear_x_mps, angular_z_radps),
        )
        if not result.ok:
            return False
        self._start_x_m = start[0]
        self._motion_deadline = time.monotonic() + duration_sec
        return True

    def observe_effect(self, timeout_sec: float) -> MobileBaseObservation | None:
        if (
            isinstance(timeout_sec, bool)
            or not isinstance(timeout_sec, (int, float))
            or not math.isfinite(float(timeout_sec))
            or timeout_sec <= 0
        ):
            return None
        if self._start_x_m is None or self._motion_deadline is None:
            return None
        remaining = self._motion_deadline - time.monotonic()
        if remaining > timeout_sec:
            return None
        if remaining > 0:
            time.sleep(remaining)
        if not self.stop():
            return None
        final = self._read_pose(min(timeout_sec, self._observation_timeout_sec))
        if final is None:
            return None
        return MobileBaseObservation(
            start_x_m=self._start_x_m,
            final_x_m=final[0],
            final_velocity_mps=final[1],
            timestamp_monotonic=time.monotonic(),
        )

    def stop(self) -> bool:
        result = self._transport.publish(self._command_topic, _twist_message(0.0, 0.0))
        return result.ok

    def _read_pose(self, timeout_sec: float) -> tuple[float, float] | None:
        result = self._transport.subscribe_once(
            self._pose_topic,
            msg_type=self._pose_message_type,
            timeout_sec=timeout_sec,
        )
        if not result.ok or not isinstance(result.data, dict):
            return None
        message = result.data.get("msg", result.data)
        if not isinstance(message, dict):
            return None
        x = message.get("x")
        velocity = message.get("linear_velocity")
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return None
        if isinstance(velocity, bool) or not isinstance(velocity, (int, float)):
            return None
        x_value = float(x)
        velocity_value = float(velocity)
        if not (math.isfinite(x_value) and math.isfinite(velocity_value)):
            return None
        return x_value, velocity_value


class GenericMobileBaseSimulationExecutor:
    """A configured rosclawd SHADOW executor is the canonical command route."""

    capability_id = "mobile_base.guarded_move"

    def __init__(
        self,
        sink: DaemonOwnedRosSink,
        *,
        daemon_instance_id: str,
        max_linear_speed_mps: float = 0.5,
        max_angular_speed_radps: float = 1.0,
        max_distance_m: float = 1.0,
    ) -> None:
        if (
            not isinstance(daemon_instance_id, str)
            or not daemon_instance_id.startswith("daemon_")
            or len(daemon_instance_id) > 128
        ):
            raise ValueError("a rosclawd daemon instance id is required")
        if sink.daemon_owner_id != daemon_instance_id:
            raise PermissionError("ROS command sink is not owned by this rosclawd instance")
        limits = (
            max_linear_speed_mps,
            max_angular_speed_radps,
            max_distance_m,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0
            for value in limits
        ):
            raise ValueError("guarded motion limits must be finite and positive")
        self._sink = sink
        self._daemon_instance_id = daemon_instance_id
        self._max_linear = max_linear_speed_mps
        self._max_angular = max_angular_speed_radps
        self._max_distance = max_distance_m

    def __call__(self, action: ActionEnvelope) -> ActionExecutionResult:
        if action.execution_mode is not ExecutionMode.SHADOW:
            return self._blocked("SIM_EXECUTOR_REQUIRES_SHADOW", "guarded simulator uses SHADOW")
        try:
            linear = _finite_argument(action, "linear_x_mps")
            angular = _finite_argument(action, "angular_z_radps", default=0.0)
            duration = _finite_argument(action, "duration_sec")
        except ValueError as exc:
            return self._blocked("INVALID_MOTION_ARGUMENT", str(exc))
        if (
            abs(linear) > self._max_linear
            or abs(angular) > self._max_angular
            or duration <= 0
            or duration > 30
            or abs(linear) * duration > self._max_distance
        ):
            return self._blocked("MOTION_LIMIT_EXCEEDED", "guarded motion exceeds body limits")
        if abs(angular) > 1e-12:
            return self._blocked(
                "ANGULAR_MOTION_UNVERIFIED",
                "guarded simulator currently verifies straight-line motion only",
            )
        dispatch_started = time.monotonic()
        try:
            dispatched = self._sink.publish_velocity(linear, angular, duration)
        except Exception:  # noqa: BLE001 - ROS adapter exceptions must fail closed
            _safe_stop(self._sink)
            return ActionExecutionResult(
                final_state=ActionState.FAILED,
                evidence_level=EvidenceLevel.REQUESTED,
                evidence_domain=EvidenceDomain.SHADOW,
                policy_decision={"allowed": True, "reason": "within_guarded_limits"},
                dispatch_result={"accepted": False, "command_sent": False},
                errors=[
                    {
                        "code": "ROS_DISPATCH_EXCEPTION",
                        "message": "ROS sink raised while dispatching",
                    }
                ],
            )
        if not dispatched:
            _safe_stop(self._sink)
            return ActionExecutionResult(
                final_state=ActionState.FAILED,
                evidence_level=EvidenceLevel.REQUESTED,
                evidence_domain=EvidenceDomain.SHADOW,
                policy_decision={"allowed": True, "reason": "within_guarded_limits"},
                dispatch_result={"accepted": False, "command_sent": False},
                errors=[{"code": "ROS_DISPATCH_FAILED", "message": "ROS sink rejected command"}],
            )
        observation_error = False
        try:
            observation = self._sink.observe_effect(min(duration + 2.0, 32.0))
        except Exception:  # noqa: BLE001 - every dispatched path must still issue stop
            observation = None
            observation_error = True
        finally:
            stopped = _safe_stop(self._sink)
        if observation is None:
            return ActionExecutionResult(
                final_state=ActionState.DEGRADED,
                evidence_level=EvidenceLevel.DISPATCH_CONFIRMED,
                evidence_domain=EvidenceDomain.SHADOW,
                policy_decision={"allowed": True, "reason": "within_guarded_limits"},
                dispatch_result={
                    "accepted": True,
                    "command_sent": True,
                    "owner": self._daemon_instance_id,
                },
                driver_ack=None,
                acknowledgement_stage=AcknowledgementStage.COMMAND_DISPATCHED,
                verification_result={
                    "success": False,
                    "observation_received": False,
                    "stop_confirmed": stopped,
                },
                errors=[
                    {
                        "code": (
                            "OBSERVATION_FAILED" if observation_error else "OBSERVATION_TIMEOUT"
                        ),
                        "message": (
                            "ROS sink raised while observing"
                            if observation_error
                            else "no fresh pose observation"
                        ),
                    }
                ],
            )
        observation_fresh = bool(
            dispatch_started <= observation.timestamp_monotonic <= time.monotonic() + 0.1
        )
        if not observation_fresh:
            return ActionExecutionResult(
                final_state=ActionState.DEGRADED,
                evidence_level=EvidenceLevel.DISPATCH_CONFIRMED,
                evidence_domain=EvidenceDomain.SHADOW,
                policy_decision={"allowed": True, "reason": "within_guarded_limits"},
                dispatch_result={
                    "accepted": True,
                    "command_sent": True,
                    "owner": self._daemon_instance_id,
                },
                acknowledgement_stage=AcknowledgementStage.COMMAND_DISPATCHED,
                verification_result={
                    "success": False,
                    "observation_received": True,
                    "observation_fresh": False,
                    "stop_confirmed": stopped,
                },
                errors=[
                    {
                        "code": "STALE_OBSERVATION",
                        "message": "pose observation predates command dispatch",
                    }
                ],
            )
        displacement = observation.final_x_m - observation.start_x_m
        expected = linear * duration
        verified = bool(
            stopped
            and abs(observation.final_velocity_mps) <= 0.02
            and abs(displacement - expected) <= max(0.05, abs(expected) * 0.25)
        )
        return ActionExecutionResult(
            final_state=ActionState.COMPLETED if verified else ActionState.DEGRADED,
            evidence_level=(
                EvidenceLevel.TASK_VERIFIED if verified else EvidenceLevel.PHYSICALLY_OBSERVED
            ),
            evidence_domain=EvidenceDomain.SHADOW,
            policy_decision={"allowed": True, "reason": "within_guarded_limits"},
            dispatch_result={
                "accepted": True,
                "command_sent": True,
                "owner": self._daemon_instance_id,
            },
            driver_ack={
                "acknowledged": True,
                "stage": AcknowledgementStage.PROTOCOL_ACKNOWLEDGED.value,
            },
            observations=[
                {
                    "start_x_m": observation.start_x_m,
                    "final_x_m": observation.final_x_m,
                    "final_velocity_mps": observation.final_velocity_mps,
                }
            ],
            verification_result={
                "success": verified,
                "observation_received": True,
                "observation_fresh": True,
                "expected_displacement_m": expected,
                "actual_displacement_m": displacement,
                "stop_confirmed": stopped,
            },
        )

    @staticmethod
    def _blocked(code: str, message: str) -> ActionExecutionResult:
        return ActionExecutionResult(
            final_state=ActionState.BLOCKED,
            evidence_level=EvidenceLevel.REQUESTED,
            evidence_domain=EvidenceDomain.SHADOW,
            policy_decision={"allowed": False, "reason": code.lower()},
            errors=[{"code": code, "message": message}],
        )


def _finite_argument(action: ActionEnvelope, name: str, *, default: float | None = None) -> float:
    value = action.arguments.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _twist_message(linear_x_mps: float, angular_z_radps: float) -> dict[str, object]:
    return {
        "linear": {"x": linear_x_mps, "y": 0.0, "z": 0.0},
        "angular": {"x": 0.0, "y": 0.0, "z": angular_z_radps},
    }


def _safe_stop(sink: DaemonOwnedRosSink) -> bool:
    try:
        return sink.stop() is True
    except Exception:  # noqa: BLE001 - stopping is best effort and never upgrades evidence
        return False


def _simulation_topics_allowed(command_topic: str, pose_topic: str) -> bool:
    if (command_topic, pose_topic) == ("/turtle1/cmd_vel", "/turtle1/pose"):
        return True
    prefixes = ("/sim/", "/simulation/", "/gazebo/")
    return any(
        command_topic.startswith(prefix) and pose_topic.startswith(prefix) for prefix in prefixes
    )


__all__ = [
    "DaemonOwnedRosSink",
    "GenericMobileBaseSimulationExecutor",
    "MobileBaseObservation",
    "RosbridgeMobileBaseSink",
]
