"""
SandboxRuntimeAdapter — v1.0 Runtime integration for rosclaw-sandbox.

Provides lifecycle management, health checks, and sandbox services
through the Runtime's module registry.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rosclaw.core.event_bus import EventBus
from rosclaw.core.lifecycle import LifecycleMixin

logger = logging.getLogger("rosclaw.sandbox.runtime_adapter")


class SandboxRuntimeAdapter(LifecycleMixin):
    """
    Adapter that integrates rosclaw-sandbox into the v1.0 Runtime.

    Responsibilities:
    - Own the sandbox physics engine (MuJoCo)
    - Provide trajectory validation via dynamic simulation (mj_step)
    - Publish sandbox events to the v1.0 EventBus
    - Maintain health status for Runtime monitoring
    """

    name = "sandbox"

    def __init__(
        self,
        config: dict[str, Any],
        event_bus: EventBus,
        e_urdf_model: Any | None = None,
        runtime: Any | None = None,
    ):
        super().__init__()
        self._config = config
        self._event_bus = event_bus
        self._e_urdf_model = e_urdf_model
        self._runtime = runtime
        self._sandbox_service: Any | None = None
        self._sandbox_context_adapter: Any | None = None
        self._initialization_error: str | None = None
        self._engine_name = config.get("engine", "mujoco")
        self._world_id = config.get("world_id", "empty")
        self._robot_id = config.get("robot_id", "")
        self._artifact_root = Path(
            config.get("artifact_root") or Path.home() / ".rosclaw" / "artifacts" / "sandbox"
        )
        from rosclaw.observability.tracer import Tracer, get_tracer

        runtime_tracer = getattr(runtime, "tracer", None)
        self._tracer = (
            runtime_tracer if isinstance(runtime_tracer, Tracer) else get_tracer(event_bus)
        )
        if self._runtime is not None:
            sense_runtime = getattr(self._runtime, "sense", None)
            if sense_runtime is not None:
                try:
                    from rosclaw.sense.adapters.sandbox_context import SandboxContextAdapter

                    self._sandbox_context_adapter = SandboxContextAdapter(sense_runtime)
                except Exception:
                    logger.warning("Failed to initialize SandboxContextAdapter", exc_info=True)

    def _do_initialize(self) -> None:
        """Initialize sandbox service."""
        logger.info("Initializing with engine=%s", self._engine_name)

        try:
            from rosclaw.sandbox.events.publisher import RuntimePublisher
            from rosclaw.sandbox.sandbox_api import Sandbox

            publisher = RuntimePublisher(self._event_bus)
            self._sandbox_service = Sandbox.create(
                robot_id=self._robot_id,
                world_id=self._world_id,
                engine=self._engine_name,
                publisher=publisher,
            )
            self._initialization_error = self._sandbox_service.load_error
            if self._sandbox_service.has_physics:
                logger.info("Sandbox created: %s", self._sandbox_service.session.session_id)
            elif self._engine_name.lower() in {"fixture", "mock"}:
                logger.info("Explicit fixture sandbox created (no physics)")
            else:
                logger.warning("Sandbox physics unavailable: %s", self._initialization_error)
        except Exception as exc:  # noqa: BLE001
            self._sandbox_service = None
            self._initialization_error = str(exc)
            logger.warning("Failed to create sandbox: %s", exc)

    def _do_start(self) -> None:
        if self._sandbox_service:
            self._sandbox_service.reset()
            logger.info("Sandbox reset")

    def _do_stop(self) -> None:
        if self._sandbox_service:
            self._sandbox_service.close()
            logger.info("Sandbox closed")

    def validate_trajectory(
        self,
        trajectory: list[list[float]],
        safety_level: str = "MODERATE",
        event_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Trace one digital-twin trajectory validation."""

        with self._tracer.start_span(
            "sandbox.validate_trajectory",
            "SANDBOX",
            source="sandbox",
            operation="trajectory.rollout",
            attributes={
                "sandbox.engine": self._engine_name,
                "sandbox.world": self._world_id,
                "safety.level": safety_level,
                "trajectory.waypoints": len(trajectory),
            },
            robot_id=self._robot_id or None,
        ) as span:
            span.set_input({"trajectory": trajectory})
            result = self._validate_trajectory(trajectory, safety_level, event_context)
            span.set_output(result)
            if not result.get("is_safe", False):
                reason = str(result.get("reason", "unsafe"))
                span.set_status(
                    "ERROR"
                    if "error" in reason.lower() or "not initialized" in reason.lower()
                    else "BLOCKED",
                    reason,
                )
            if result.get("replay_id"):
                span.add_evidence(f"sandbox://replay/{result['replay_id']}")
            return result

    def _validate_trajectory(
        self,
        trajectory: list[list[float]],
        safety_level: str = "MODERATE",
        event_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run static policy checks for a proposed trajectory.

        This legacy endpoint does not advance MuJoCo.  Its result is explicitly
        labeled ``StaticPolicyValidation`` so callers cannot mistake it for a
        simulation rollout.
        """
        if self._sandbox_service is None:
            return {"is_safe": False, "reason": "Sandbox not initialized"}

        try:
            from rosclaw.sandbox.firewall.gate import FirewallGate

            # Build action from trajectory
            if not trajectory:
                return {"is_safe": True, "risk_score": 0.0, "replay_id": None}

            gate = FirewallGate(
                robot_id=self._robot_id,
                world_id=self._world_id,
                engine=self._engine_name,
            )

            # Check first and last waypoint for joint limits
            action = {
                "type": "joint_position",
                "values": trajectory[-1] if trajectory else [],
            }
            # Inject body sense snapshot via sense-aware adapter for sense-aware firewall rules
            if self._sandbox_context_adapter is not None:
                try:
                    action = self._sandbox_context_adapter.apply(action)
                except Exception:
                    logger.warning(
                        "SandboxContextAdapter failed; validating without body sense", exc_info=True
                    )
            decision = gate.check(action)
            # CRITICAL FIX: Publish firewall event for BOTH blocked AND allowed
            if self._event_bus:
                from rosclaw.core.event_bus import Event, EventPriority

                topic = (
                    "firewall.action_blocked"
                    if not decision.is_allowed
                    else "firewall.action_allowed"
                )
                self._event_bus.publish(
                    Event(
                        topic=topic,
                        payload={
                            **(event_context or {}),
                            "robot_id": self._robot_id,
                            "world_id": self._world_id,
                            "action": action,
                            "reason": decision.reason,
                            "risk_score": decision.risk_score,
                            "violations": decision.violated_constraints,
                            "replay_id": decision.replay_id,
                            "safety_level": safety_level,
                            "decision": "BLOCK" if not decision.is_allowed else "ALLOW",
                        },
                        source="sandbox.firewall",
                        priority=EventPriority.HIGH,
                    )
                )

            result = {
                "is_safe": decision.is_allowed,
                "validation_type": "StaticPolicyValidation",
                "simulation_executed": False,
                "risk_score": decision.risk_score,
                "predicted_collision": decision.predicted_collision,
                "reason": decision.reason,
                "violations": decision.violated_constraints,
                "replay_id": decision.replay_id,
                "event_published": self._event_bus is not None,
            }

            gate.close()
            return result

        except Exception as e:
            return {"is_safe": False, "reason": f"Validation error: {e}"}

    def simulate_step(self, joint_positions: list[float]) -> dict[str, Any]:
        """Step the sandbox physics with given joint positions.

        Returns real physics state (qpos, qvel, time) if MuJoCo model is loaded,
        otherwise returns an empty dict.
        """
        if self._sandbox_service is None:
            return {}
        if hasattr(self._sandbox_service, "step"):
            state = self._sandbox_service.step(joint_positions)
            return state or {}
        return {}

    def execute_action(self, action: Any) -> Any:
        """Execute one sandbox ActionEnvelope and return gateway-ready evidence."""

        from rosclaw.sandbox.episode import run_reach_action

        return run_reach_action(
            self._sandbox_service,
            action,
            artifact_root=self._artifact_root,
        )

    def get_observation(self, normalize: bool = True) -> dict[str, Any]:
        """Get rich normalized observation from scene state.

        Args:
            normalize: If True, joint positions are normalized to [-1, 1]
                       using MuJoCo joint limits. Velocities use tanh clipping.

        Returns:
            Observation dict with joint positions (raw + normalized),
            body positions, contacts, and simulation time.
            Empty dict if sandbox has no physics model.
        """
        if self._sandbox_service is None:
            return {}
        if hasattr(self._sandbox_service, "get_observation"):
            obs = self._sandbox_service.get_observation(normalize=normalize)
            return obs or {}
        # Fallback to legacy get_state
        if hasattr(self._sandbox_service, "get_state"):
            state = self._sandbox_service.get_state()
            return state or {}
        return {}

    @property
    def has_physics(self) -> bool:
        """True if the sandbox has a real MuJoCo model loaded."""
        if self._sandbox_service is None:
            return False
        return getattr(self._sandbox_service, "has_physics", False)

    @property
    def model_path(self) -> Path | None:
        """Return the resolved MuJoCo model path when physics loaded."""

        if self._sandbox_service is None:
            return None
        return getattr(self._sandbox_service, "model_path", None)

    def health(self) -> dict[str, Any]:
        """Return health status for Runtime monitoring."""
        explicit_fixture = self._engine_name.lower() in {"fixture", "mock"}
        has_physics = self.has_physics
        if has_physics:
            status = "healthy"
        elif explicit_fixture and self._sandbox_service is not None:
            status = "fixture"
        else:
            status = "unavailable"
        return {
            "status": status,
            "engine": self._engine_name,
            "world": self._world_id,
            "has_physics": has_physics,
            "execution_mode": "FIXTURE" if explicit_fixture else "SIMULATION",
            "trust_level": "SYNTHETIC"
            if explicit_fixture
            else ("SIMULATED" if has_physics else "UNAVAILABLE"),
            "usable_for_real_execution": False,
            "error": self._initialization_error,
            "session_id": self._sandbox_service.session.session_id
            if self._sandbox_service
            else None,
        }

    def _get_body_sense_snapshot(self) -> dict[str, Any] | None:
        """Return latest BodySense snapshot from the runtime if available."""
        if self._runtime is None:
            return None
        sense = getattr(self._runtime, "sense", None)
        if sense is None:
            return None
        try:
            latest = sense.get_latest_sense()
            if latest is None:
                latest = sense.tick()
            return latest.to_dict()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to get body sense snapshot: %s", exc)
            return None
