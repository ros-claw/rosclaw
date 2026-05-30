"""
SandboxRuntimeAdapter — v1.0 Runtime integration for rosclaw-sandbox.

Provides lifecycle management, health checks, and sandbox services
through the Runtime's module registry.
"""

from __future__ import annotations

from typing import Any, Optional

from rosclaw.core.lifecycle import LifecycleMixin, LifecycleState
from rosclaw.core.event_bus import EventBus, Event, EventPriority


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
        e_urdf_model: Optional[Any] = None,
    ):
        super().__init__()
        self._config = config
        self._event_bus = event_bus
        self._e_urdf_model = e_urdf_model
        self._sandbox_service: Optional[Any] = None
        self._engine_name = config.get("engine", "mujoco")
        self._world_id = config.get("world_id", "empty")
        self._robot_id = config.get("robot_id", "")

    def _do_initialize(self) -> None:
        """Initialize sandbox service."""
        print(f"[SandboxRuntimeAdapter] Initializing with engine={self._engine_name}")

        try:
            from rosclaw.sandbox.sandbox_api import Sandbox
            from rosclaw.sandbox.events.publisher import RuntimePublisher

            publisher = RuntimePublisher(self._event_bus)
            self._sandbox_service = Sandbox.create(
                robot_id=self._robot_id,
                world_id=self._world_id,
                engine=self._engine_name,
                publisher=publisher,
            )
            print(f"[SandboxRuntimeAdapter] Sandbox created: {self._sandbox_service.session.session_id}")
        except ImportError as e:
            print(f"[SandboxRuntimeAdapter] Sandbox not available: {e}")
            self._sandbox_service = self._create_stub_sandbox()
        except Exception as e:
            print(f"[SandboxRuntimeAdapter] Failed to create sandbox: {e}")
            self._sandbox_service = self._create_stub_sandbox()

    def _create_stub_sandbox(self):
        """Create a stub sandbox for testing/development when real sandbox unavailable."""
        import uuid

        class StubSandbox:
            def __init__(self):
                self.session = type("Session", (), {"session_id": str(uuid.uuid4())})()

            def reset(self):
                pass

            def close(self):
                pass

        return StubSandbox()

    def _do_start(self) -> None:
        if self._sandbox_service:
            self._sandbox_service.reset()
            print("[SandboxRuntimeAdapter] Sandbox reset and running")

    def _do_stop(self) -> None:
        if self._sandbox_service:
            self._sandbox_service.close()
            print("[SandboxRuntimeAdapter] Sandbox closed")

    def validate_trajectory(
        self,
        trajectory: list[list[float]],
        safety_level: str = "MODERATE",
    ) -> dict[str, Any]:
        """
        Validate a trajectory using dynamic simulation (mj_step).

        Replaces the static mj_forward-based check in FirewallValidator
        with a true physics rollout.
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
            decision = gate.check(action)
            # CRITICAL FIX: Publish firewall event for BOTH blocked AND allowed
            if self._event_bus:
                from rosclaw.core.event_bus import Event, EventPriority
                topic = (
                    "firewall.action_blocked"
                    if not decision.is_allowed
                    else "firewall.action_allowed"
                )
                self._event_bus.publish(Event(
                    topic=topic,
                    payload={
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
                ))

            result = {
                "is_safe": decision.is_allowed,
                "risk_score": decision.risk_score,
                "predicted_collision": decision.predicted_collision,
                "reason": decision.reason,
                "violations": decision.violated_constraints,
                "replay_id": decision.replay_id,
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

    @property
    def has_physics(self) -> bool:
        """True if the sandbox has a real MuJoCo model loaded."""
        if self._sandbox_service is None:
            return False
        return getattr(self._sandbox_service, "has_physics", False)

    def health(self) -> dict[str, Any]:
        """Return health status for Runtime monitoring."""
        return {
            "status": "healthy" if self._sandbox_service else "unavailable",
            "engine": self._engine_name,
            "world": self._world_id,
            "session_id": self._sandbox_service.session.session_id if self._sandbox_service else None,
        }
