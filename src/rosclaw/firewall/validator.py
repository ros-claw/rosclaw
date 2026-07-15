"""FirewallValidator - EventBus-integrated trajectory validation.

Subscribes to agent.command events, validates against e-URDF soft limits
and MuJoCo collision model, publishes agent.response with request_id.

Sprint 3 of DESIGN_SPRINT3_5.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from rosclaw.core.event_bus import Event, EventBus, EventPriority
from rosclaw.core.lifecycle import LifecycleMixin
from rosclaw.e_urdf.parser import RobotModel

logger = logging.getLogger("rosclaw.firewall.validator")


class ValidationLayer(Enum):
    """3-layer validation pipeline."""

    EURDF_SOFT_LIMITS = "eurdf_soft_limits"
    MUJOCO_COLLISION = "mujoco_collision"
    SEMANTIC_SAFETY = "semantic_safety"


@dataclass
class SafetyEnvelope:
    """Extracted safety boundaries from e-URDF + safety.yaml."""

    joint_soft_limits: list[tuple[float, float]]
    max_velocity: list[float]
    max_torque: list[float]
    keepout_zones: list[dict] = field(default_factory=list)
    allowed_contacts: list[str] = field(default_factory=list)
    safety_level: str = "MODERATE"

    @classmethod
    def from_robot_model(
        cls, robot_model: RobotModel, safety_level: str = "MODERATE"
    ) -> "SafetyEnvelope":
        """Extract envelope from RobotModel (e-URDF parsed data)."""
        soft_factor = {"STRICT": 0.90, "MODERATE": 0.95, "LENIENT": 0.99}
        factor = soft_factor.get(safety_level, 0.95)

        joint_limits = []
        max_velocities = []
        max_torques = []

        for joint in robot_model.joints.values():
            lo = joint.limits.get("lower", -float("inf"))
            hi = joint.limits.get("upper", float("inf"))
            # Apply soft factor inward from limits
            lo = lo * factor if lo < 0 else lo + (hi - lo) * (1 - factor) / 2
            hi = hi * factor if hi > 0 else hi - (hi - lo) * (1 - factor) / 2
            joint_limits.append((lo, hi))
            max_velocities.append(joint.limits.get("velocity", float("inf")) * factor)
            max_torques.append(joint.limits.get("effort", float("inf")) * factor)

        return cls(
            joint_soft_limits=joint_limits,
            max_velocity=max_velocities,
            max_torque=max_torques,
            safety_level=safety_level,
        )


@dataclass
class ValidationRequest:
    """Request to validate a trajectory before execution."""

    request_id: str
    robot_id: str
    trajectory: list[list[float]]
    duration_per_waypoint: list[float] = field(default_factory=list)
    source: str = "agent_runtime"
    metadata: dict = field(default_factory=dict)


@dataclass
class ViolationDetail:
    """Single safety violation found during validation."""

    layer: ValidationLayer
    severity: str
    joint_index: int | None
    description: str
    actual_value: float | None = None
    limit_value: float | None = None


@dataclass
class ValidationResponse:
    """Result of trajectory validation."""

    request_id: str
    is_safe: bool
    layers_checked: list[ValidationLayer]
    violations: list[ViolationDetail] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    simulation_duration_ms: float = 0.0

    @property
    def violation_count(self) -> int:
        return len(self.violations)


class FirewallValidator(LifecycleMixin):
    """
    EventBus-integrated firewall with e-URDF awareness.

    Lifecycle:
        initialize() -> loads SafetyEnvelope from RobotModel, subscribes to EventBus
        start() -> begins processing validation requests
        stop() -> unsubscribes, releases MuJoCo resources

    EventBus:
        Subscribes: agent.command (to intercept movement commands)
        Publishes:  agent.response (validation result with request_id)
                    safety.violation (when violations found)
    """

    def __init__(
        self,
        robot_model: RobotModel,
        event_bus: EventBus,
        mujoco_model_path: str | None = None,
        safety_level: str = "MODERATE",
        tracer: Any | None = None,
    ):
        super().__init__()
        self._robot_model = robot_model
        self._event_bus = event_bus
        self._mujoco_model_path = mujoco_model_path
        self._safety_level = safety_level
        self._envelope: SafetyEnvelope | None = None
        self._mj_model = None
        self._mj_data = None
        if tracer is None:
            from rosclaw.observability.tracer import get_tracer

            tracer = get_tracer(event_bus)
        self._tracer = tracer

    def _do_initialize(self) -> None:
        """Build SafetyEnvelope from e-URDF, optionally load MuJoCo."""
        self._envelope = SafetyEnvelope.from_robot_model(self._robot_model, self._safety_level)

        if self._mujoco_model_path:
            try:
                import mujoco

                self._mj_model = mujoco.MjModel.from_xml_path(self._mujoco_model_path)
                self._mj_data = mujoco.MjData(self._mj_model)
            except (ImportError, FileNotFoundError, Exception) as e:
                logger.warning("MuJoCo unavailable: %s", e)

        self._event_bus.subscribe("agent.command", self._on_agent_command)
        logger.info(
            "Initialized with %s safety, %d joints, MuJoCo=%s",
            self._safety_level,
            len(self._envelope.joint_soft_limits),
            "yes" if self._mj_model else "no",
        )

    def _do_start(self) -> None:
        self._event_bus.publish(
            Event(
                topic="firewall.status",
                payload={"state": "running", "safety_level": self._safety_level},
                source="firewall_validator",
            )
        )

    def _do_stop(self) -> None:
        self._event_bus.unsubscribe("agent.command", self._on_agent_command)
        self._mj_model = None
        self._mj_data = None

    def _on_agent_command(self, event: Event) -> None:
        """Intercept agent commands, validate, publish response."""
        payload = event.payload
        action = payload.get("action", "")

        if action not in ("move_joints", "execute_trajectory"):
            return

        request_id = event.metadata.get("request_id", "unknown")
        request = ValidationRequest(
            request_id=request_id,
            robot_id=payload.get("robot_id", "default"),
            trajectory=payload.get("trajectory", []),
            duration_per_waypoint=payload.get("durations", []),
            source=payload.get("source", "agent_runtime"),
        )

        response = self.validate(request)

        self._event_bus.publish(
            Event(
                topic="agent.response",
                payload={
                    "status": "safe" if response.is_safe else "blocked",
                    "is_safe": response.is_safe,
                    "violations": [
                        {
                            "layer": v.layer.value,
                            "severity": v.severity,
                            "description": v.description,
                        }
                        for v in response.violations
                    ],
                    "warnings": response.warnings,
                    "request_id": request.request_id,
                },
                source="firewall_validator",
                priority=EventPriority.HIGH if response.is_safe else EventPriority.CRITICAL,
                metadata={"request_id": request.request_id},
            )
        )

        if not response.is_safe:
            self._event_bus.publish(
                Event(
                    topic="safety.violation",
                    payload={
                        "request_id": request.request_id,
                        "violations": [v.description for v in response.violations],
                        "action": "BLOCKED",
                    },
                    source="firewall_validator",
                    priority=EventPriority.CRITICAL,
                )
            )
            # Publish firewall.action_blocked for HeuristicEngine recovery
            self._event_bus.publish(
                Event(
                    topic="firewall.action_blocked",
                    payload={
                        "request_id": request.request_id,
                        "robot_id": request.robot_id,
                        "action": action,
                        "violations": [
                            {
                                "layer": v.layer.value,
                                "severity": v.severity,
                                "description": v.description,
                            }
                            for v in response.violations
                        ],
                        "trajectory": request.trajectory,
                    },
                    source="firewall_validator",
                    priority=EventPriority.CRITICAL,
                )
            )

    def validate(self, request: ValidationRequest) -> ValidationResponse:
        """Trace and run the deterministic three-layer firewall."""

        with self._tracer.start_span(
            "firewall.validate",
            "GUARDRAIL",
            source="firewall_validator",
            operation="trajectory.validate",
            trace_id=request.request_id,
            attributes={
                "safety.level": self._safety_level,
                "trajectory.waypoints": len(request.trajectory),
            },
            robot_id=request.robot_id,
        ) as span:
            span.set_input(
                {
                    "trajectory": request.trajectory,
                    "duration_per_waypoint": request.duration_per_waypoint,
                    "metadata": request.metadata,
                }
            )
            response = self._validate(request)
            output = {
                "request_id": response.request_id,
                "is_safe": response.is_safe,
                "layers_checked": [layer.value for layer in response.layers_checked],
                "violations": [
                    {
                        "layer": item.layer.value,
                        "severity": item.severity,
                        "joint_index": item.joint_index,
                        "description": item.description,
                        "actual_value": item.actual_value,
                        "limit_value": item.limit_value,
                    }
                    for item in response.violations
                ],
                "warnings": response.warnings,
                "simulation_duration_ms": response.simulation_duration_ms,
            }
            span.set_output(output)
            if not response.is_safe:
                span.set_status(
                    "BLOCKED", "; ".join(item.description for item in response.violations)
                )
            return response

    def _validate(self, request: ValidationRequest) -> ValidationResponse:
        """Run 3-layer validation pipeline."""
        violations = []
        warnings = []
        layers_checked = []

        layer1_violations = self._check_eurdf_limits(request)
        violations.extend(layer1_violations)
        layers_checked.append(ValidationLayer.EURDF_SOFT_LIMITS)

        if self._mj_model is not None:
            import time

            t0 = time.monotonic()
            layer2_violations = self._check_mujoco_collision(request)
            violations.extend(layer2_violations)
            layers_checked.append(ValidationLayer.MUJOCO_COLLISION)
            sim_ms = (time.monotonic() - t0) * 1000
        else:
            sim_ms = 0.0

        layer3_violations, layer3_warnings = self._check_semantic_safety(request)
        violations.extend(layer3_violations)
        warnings.extend(layer3_warnings)
        layers_checked.append(ValidationLayer.SEMANTIC_SAFETY)

        is_safe = all(v.severity != "critical" for v in violations)

        return ValidationResponse(
            request_id=request.request_id,
            is_safe=is_safe,
            layers_checked=layers_checked,
            violations=violations,
            warnings=warnings,
            simulation_duration_ms=sim_ms,
        )

    def _check_eurdf_limits(self, request: ValidationRequest) -> list[ViolationDetail]:
        """Layer 1: Check trajectory against e-URDF soft limits."""
        violations = []
        if self._envelope is None:
            return violations

        for wp_idx, waypoint in enumerate(request.trajectory):
            for j_idx, value in enumerate(waypoint):
                if j_idx >= len(self._envelope.joint_soft_limits):
                    break
                lo, hi = self._envelope.joint_soft_limits[j_idx]
                if value < lo or value > hi:
                    violations.append(
                        ViolationDetail(
                            layer=ValidationLayer.EURDF_SOFT_LIMITS,
                            severity="critical",
                            joint_index=j_idx,
                            description=f"Joint {j_idx} value {value:.3f} outside "
                            f"soft limit [{lo:.3f}, {hi:.3f}] at waypoint {wp_idx}",
                            actual_value=value,
                            limit_value=hi if value > hi else lo,
                        )
                    )
        return violations

    def _check_mujoco_collision(self, request: ValidationRequest) -> list[ViolationDetail]:
        """Layer 2: Simulate trajectory in MuJoCo using mj_step, check for collisions.

        Replaces the previous static mj_forward-based check with dynamic simulation.
        This detects collisions that occur during motion between waypoints.
        """
        violations = []
        if self._mj_data is None:
            return violations

        import mujoco
        import numpy as np

        # Save original state for restoration
        original_qpos = self._mj_data.qpos.copy()
        original_qvel = self._mj_data.qvel.copy()

        # Reset velocity
        self._mj_data.qvel[:] = 0.0

        dt = self._mj_model.opt.timestep
        steps_per_waypoint = max(1, int(0.02 / dt))  # ~50Hz control

        for wp_idx, waypoint in enumerate(request.trajectory):
            dof = min(len(waypoint), self._mj_model.nq)
            target_qpos = np.array(waypoint[:dof])
            current_qpos = np.array(self._mj_data.qpos[:dof])
            velocity = (target_qpos - current_qpos) / (steps_per_waypoint * dt)

            for step in range(steps_per_waypoint):
                # Apply velocity control
                self._mj_data.qvel[:dof] = velocity
                mujoco.mj_step(self._mj_model, self._mj_data)

                # Check contacts after each physics step
                for i in range(self._mj_data.ncon):
                    contact = self._mj_data.contact[i]
                    if contact.dist < 0.001:
                        geom1_name = (
                            mujoco.mj_id2name(self._mj_model, 6, contact.geom1)
                            or f"geom{contact.geom1}"
                        )
                        geom2_name = (
                            mujoco.mj_id2name(self._mj_model, 6, contact.geom2)
                            or f"geom{contact.geom2}"
                        )

                        if (
                            geom1_name in self._envelope.allowed_contacts
                            or geom2_name in self._envelope.allowed_contacts
                        ):
                            continue

                        violations.append(
                            ViolationDetail(
                                layer=ValidationLayer.MUJOCO_COLLISION,
                                severity="critical",
                                joint_index=None,
                                description=f"Dynamic collision: {geom1_name} <-> {geom2_name} "
                                f"at waypoint {wp_idx}, step {step}",
                            )
                        )
                        # Restore original state
                        self._mj_data.qpos[:] = original_qpos
                        self._mj_data.qvel[:] = original_qvel
                        mujoco.mj_step(self._mj_model, self._mj_data)
                        return violations

        # Restore original state
        self._mj_data.qpos[:] = original_qpos
        self._mj_data.qvel[:] = original_qvel
        mujoco.mj_step(self._mj_model, self._mj_data)
        return violations

    def _check_semantic_safety(
        self, request: ValidationRequest
    ) -> tuple[list[ViolationDetail], list[str]]:
        """Layer 3: Check semantic rules from safety.yaml."""
        violations = []
        warnings = []

        if self._envelope and self._envelope.keepout_zones:
            for zone in self._envelope.keepout_zones:
                warnings.append(
                    f"Keepout zone '{zone.get('name', 'unknown')}' defined "
                    f"but FK not computed — skipped"
                )

        if request.duration_per_waypoint and self._envelope:
            for i, duration in enumerate(request.duration_per_waypoint):
                if duration > 0 and i > 0:
                    prev = np.array(request.trajectory[i - 1])
                    curr = np.array(request.trajectory[i])
                    velocities = np.abs(curr - prev) / duration
                    for j_idx, vel in enumerate(velocities):
                        if (
                            j_idx < len(self._envelope.max_velocity)
                            and vel > self._envelope.max_velocity[j_idx]
                        ):
                            violations.append(
                                ViolationDetail(
                                    layer=ValidationLayer.SEMANTIC_SAFETY,
                                    severity="error",
                                    joint_index=j_idx,
                                    description=f"Joint {j_idx} velocity {vel:.2f} rad/s "
                                    f"exceeds limit {self._envelope.max_velocity[j_idx]:.2f}",
                                    actual_value=vel,
                                    limit_value=self._envelope.max_velocity[j_idx],
                                )
                            )

        return violations, warnings
