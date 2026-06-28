"""Unified facade that isolates MCP tools from direct ROS/hardware access."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rosclaw.body.registry import BodyRegistryManager
from rosclaw.body.resolver import BodyResolver
from rosclaw.firstboot.workspace import get_rosclaw_home
from rosclaw.mcp.adapters.memory_client import MemoryClient
from rosclaw.mcp.adapters.practice_client import PracticeClient
from rosclaw.mcp.adapters.safety_client import SafetyClient
from rosclaw.mcp.adapters.sandbox_client import SandboxClient
from rosclaw.mcp.adapters.skill_registry_client import SkillRegistryClient

logger = logging.getLogger("rosclaw.mcp.adapters.runtime_client")


class RuntimeClient:
    """Lazy-initializing client over the existing ROSClaw Runtime components.

    The client is intentionally defensive: if the runtime cannot be initialized
    (missing model, no ROS, etc.) the tools fall back to fixture responses so
    that the agent still receives a well-formed envelope.

    Thin subsystem adapters (MemoryClient, SandboxClient, etc.) are used to keep
    the Runtime surface small and to make the MCP layer easy to unit-test.
    """

    def __init__(
        self,
        *,
        project_root: Path,
        robot_id: str | None,
        runtime_profile: dict[str, Any],
        fixture_mode: bool = False,
    ):
        self.project_root = project_root
        self.robot_id = robot_id or "rosclaw_default"
        self.runtime_profile = runtime_profile or {}
        self.fixture_mode = fixture_mode
        self._runtime: Any | None = None
        self._runtime_error: str | None = None
        self._adapter_cache: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Runtime lifecycle
    # ------------------------------------------------------------------

    def _ensure_runtime(self) -> Any | None:
        if self.fixture_mode:
            return None
        if self._runtime is not None:
            return self._runtime
        if self._runtime_error is not None:
            return None
        try:
            from rosclaw.core.event_bus import EventBus
            from rosclaw.core.runtime import Runtime, RuntimeConfig

            cfg = RuntimeConfig(
                robot_id=self.robot_id,
                event_bus=EventBus(),
                sense_collector=self.runtime_profile.get("sense", {}).get("collector", "mock"),
                sense_update_hz=self.runtime_profile.get("sense", {}).get("update_hz", 1.0),
            )
            rt = Runtime(cfg)
            rt.initialize()
            self._runtime = rt
            self._adapter_cache = None
            return rt
        except Exception as exc:  # noqa: BLE001
            self._runtime_error = str(exc)
            logger.debug("RuntimeClient could not initialize Runtime: %s", exc)
            return None

    def _adapters(self) -> dict[str, Any]:
        """Return a dict of thin adapters for the current runtime.

        Adapters are rebuilt whenever the runtime instance changes so that tests
        can reinitialize the client with different fixture or live runtimes.
        """
        rt = self._ensure_runtime()
        if rt is None:
            return {}
        if self._adapter_cache is not None and self._adapter_cache.get("_runtime") is rt:
            return self._adapter_cache
        self._adapter_cache = {
            "_runtime": rt,
            "memory": MemoryClient(rt.memory) if rt.memory is not None else None,
            "practice": PracticeClient(rt.episode_recorder) if rt.episode_recorder is not None else None,
            "sandbox": SandboxClient(rt.sandbox) if rt.sandbox is not None else None,
            "skill": SkillRegistryClient(rt.skill_manager) if rt.skill_manager is not None else None,
            "safety": SafetyClient(rt),
        }
        return self._adapter_cache

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fixture_state(self) -> dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "mode": "fixture",
            "note": "ROSClaw runtime is not initialized; returning representative fixture state.",
            "body_state": {
                "joint_positions": [0.0] * 6,
                "joint_velocities": [0.0] * 6,
                "timestamp": 0.0,
            },
            "readiness": {"overall": "UNKNOWN", "factors": {}},
            "risk_summary": {"risk_level": "LOW", "violations": []},
            "is_stale": True,
            "age_ms": 0,
        }

    @staticmethod
    def _safe_dict(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if value is not None and hasattr(value, "to_dict"):
            return value.to_dict()
        return {}

    # ------------------------------------------------------------------
    # S0 read-only tools
    # ------------------------------------------------------------------

    async def get_robot_state(self) -> dict[str, Any]:
        rt = self._ensure_runtime()
        if rt is None or rt.sense is None:
            return self._fixture_state()
        try:
            sense = rt.sense
            latest = sense.get_latest_state()
            body_sense = sense.get_body_sense()
            readiness = sense.get_readiness()
            return {
                "robot_id": self.robot_id,
                "mode": "live" if not getattr(sense, "is_stale", True) else "stale",
                "body_state": self._safe_dict(latest),
                "body_sense": self._safe_dict(body_sense),
                "readiness": self._safe_dict(readiness),
                "is_stale": getattr(sense, "is_stale", True),
                "age_ms": getattr(sense, "state_age_ms", 0),
            }
        except Exception as exc:  # noqa: BLE001
            logger.debug("get_robot_state failed: %s", exc)
            return self._fixture_state()

    async def list_skills(self, *, skill_type: str | None = None, full_ids: bool = False) -> dict[str, Any]:
        adapter = self._adapters().get("skill")
        if adapter is None:
            return {"skills": [], "count": 0, "mode": "fixture"}
        try:
            return adapter.list_skills(skill_type=skill_type, full_ids=full_ids)
        except Exception as exc:  # noqa: BLE001
            logger.debug("list_skills failed: %s", exc)
            return {"skills": [], "count": 0, "mode": "fixture"}

    async def query_memory(
        self,
        instruction: str,
        *,
        limit: int = 5,
        outcome_filter: str | None = None,
    ) -> dict[str, Any]:
        adapter = self._adapters().get("memory")
        if adapter is None:
            return {"experiences": [], "count": 0, "mode": "fixture"}
        try:
            return adapter.find_similar_experiences(
                instruction=instruction,
                limit=limit,
                outcome_filter=outcome_filter,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("query_memory failed: %s", exc)
            return {"experiences": [], "count": 0, "mode": "fixture"}

    # ------------------------------------------------------------------
    # S2 validated-plan tool
    # ------------------------------------------------------------------

    async def validate_trajectory(
        self,
        trajectory: list[list[float]],
        *,
        safety_level: str = "MODERATE",
    ) -> dict[str, Any]:
        adapter = self._adapters().get("sandbox")
        if adapter is None:
            return {
                "is_safe": False,
                "risk_score": 1.0,
                "reason": "Sandbox/runtime unavailable; cannot validate trajectory.",
                "violations": ["runtime_unavailable"],
                "replay_id": None,
            }
        try:
            return adapter.validate_trajectory(trajectory=trajectory, safety_level=safety_level)
        except Exception as exc:  # noqa: BLE001
            logger.debug("validate_trajectory failed: %s", exc)
            return {
                "is_safe": False,
                "risk_score": 1.0,
                "reason": f"Validation error: {exc}",
                "violations": ["validation_exception"],
                "replay_id": None,
            }

    # ------------------------------------------------------------------
    # S1 simulation-only tool
    # ------------------------------------------------------------------

    async def sandbox_run(self, joint_positions: list[float]) -> dict[str, Any]:
        adapter = self._adapters().get("sandbox")
        if adapter is None:
            return {
                "physics_state": {},
                "mode": "fixture",
                "note": "Sandbox runtime unavailable; returning empty fixture.",
            }
        try:
            return adapter.simulate_step(joint_positions)
        except Exception as exc:  # noqa: BLE001
            logger.debug("sandbox_run failed: %s", exc)
            return {
                "physics_state": {},
                "mode": "fixture",
                "note": f"Simulation error: {exc}",
            }

    # ------------------------------------------------------------------
    # S0 practice-query tool
    # ------------------------------------------------------------------

    async def practice_query(
        self,
        *,
        episode_id: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        adapter = self._adapters().get("practice")
        if adapter is None:
            return {"episodes": [], "count": 0, "mode": "fixture"}
        try:
            return adapter.query(episode_id=episode_id, limit=limit)
        except Exception as exc:  # noqa: BLE001
            logger.debug("practice_query failed: %s", exc)
            return {"episodes": [], "count": 0, "mode": "fixture"}

    # ------------------------------------------------------------------
    # Body registry tools (P2)
    # ------------------------------------------------------------------

    def _body_workspace(self) -> Path:
        """Return the ROSClaw workspace used by the body registry."""
        return get_rosclaw_home()

    async def list_bodies(self) -> dict[str, Any]:
        """List all registered bodies in the workspace."""
        try:
            workspace = self._body_workspace()
            manager = BodyRegistryManager(workspace)
            bodies = manager.list_bodies()
            stats = manager.stats()
            return {
                "mode": "live",
                "workspace": str(workspace),
                "current": stats["current"],
                "total": stats["total"],
                "bodies": [b.to_dict() for b in bodies],
                "by_profile": stats.get("by_profile", {}),
            }
        except Exception as exc:  # noqa: BLE001
            logger.debug("list_bodies failed: %s", exc)
            return {"mode": "fixture", "bodies": [], "total": 0, "error": str(exc)}

    async def get_body(self, body_id: str) -> dict[str, Any]:
        """Return registry entry and effective body snapshot for one body."""
        try:
            workspace = self._body_workspace()
            manager = BodyRegistryManager(workspace)
            entry = manager.get_body(body_id)
            if entry is None:
                return {"mode": "fixture", "body_id": body_id, "error": "Body not found"}
            resolver = BodyResolver(workspace, body_id=body_id)
            effective = resolver.get_effective_body(recompile_if_stale=False)
            return {
                "mode": "live",
                "body": entry.to_dict(),
                "effective_body": effective.to_dict(),
            }
        except Exception as exc:  # noqa: BLE001
            logger.debug("get_body failed: %s", exc)
            return {"mode": "fixture", "body_id": body_id, "error": str(exc)}

    async def switch_body(self, body_id: str) -> dict[str, Any]:
        """Switch the active body pointer in the registry."""
        try:
            workspace = self._body_workspace()
            manager = BodyRegistryManager(workspace)
            manager.set_current_body_id(body_id)
            return {
                "mode": "live",
                "current_body_id": manager.get_current_body_id(),
                "note": "Active body pointer updated; no hardware motion was performed.",
            }
        except Exception as exc:  # noqa: BLE001
            logger.debug("switch_body failed: %s", exc)
            return {"mode": "fixture", "body_id": body_id, "error": str(exc)}

    async def list_body_history(self, body_id: str) -> dict[str, Any]:
        """List snapshot history for a body."""
        try:
            workspace = self._body_workspace()
            resolver = BodyResolver(workspace, body_id=body_id)
            snapshots = sorted(resolver.snapshots_dir.glob("body-*.yaml"), reverse=True)
            return {
                "mode": "live",
                "body_id": body_id,
                "snapshots": [
                    {"path": str(p), "name": p.name} for p in snapshots
                ],
            }
        except Exception as exc:  # noqa: BLE001
            logger.debug("list_body_history failed: %s", exc)
            return {"mode": "fixture", "body_id": body_id, "error": str(exc)}

    async def check_skill_compatibility(self) -> dict[str, Any]:
        """Check skill compatibility for the current body."""
        try:
            from rosclaw.body.compatibility import SkillCompatibilityChecker
            from rosclaw.body.fleet import discover_skill_manifests

            workspace = self._body_workspace()
            manifests = discover_skill_manifests(workspace)
            resolver = BodyResolver(workspace)
            effective = resolver.get_effective_body(recompile_if_stale=False)
            report = SkillCompatibilityChecker().check_all(manifests, effective)
            return {"mode": "live", "report": report.to_dict()}
        except Exception as exc:  # noqa: BLE001
            logger.debug("check_skill_compatibility failed: %s", exc)
            return {"mode": "fixture", "report": {}, "error": str(exc)}

    async def fleet_skill_compatibility(self) -> dict[str, Any]:
        """Aggregate skill compatibility across all bodies in the workspace."""
        try:
            from rosclaw.body.fleet import FleetCompatibilityAggregator, discover_skill_manifests

            workspace = self._body_workspace()
            manifests = discover_skill_manifests(workspace)
            report = FleetCompatibilityAggregator(workspace).aggregate(manifests)
            return {"mode": "live", "report": report.to_dict()}
        except Exception as exc:  # noqa: BLE001
            logger.debug("fleet_skill_compatibility failed: %s", exc)
            return {"mode": "fixture", "report": {}, "error": str(exc)}

    # ------------------------------------------------------------------
    # P0 body tools
    # ------------------------------------------------------------------

    async def get_body_profile(self) -> dict[str, Any]:
        """Return a static profile summary of the current body."""
        try:
            from rosclaw.body.mcp_tools import BodyMcpTools

            tools = BodyMcpTools(workspace=self._body_workspace())
            return {"mode": "live", "profile": tools.get_body_profile()}
        except Exception as exc:  # noqa: BLE001
            logger.debug("get_body_profile failed: %s", exc)
            return {"mode": "fixture", "profile": {}, "error": str(exc)}

    async def get_body_state(self, *, include_runtime: bool = True) -> dict[str, Any]:
        """Return current body safety and capability state."""
        try:
            from rosclaw.body.mcp_tools import BodyMcpTools

            tools = BodyMcpTools(workspace=self._body_workspace())
            return {"mode": "live", "state": tools.get_body_state(include_runtime=include_runtime)}
        except Exception as exc:  # noqa: BLE001
            logger.debug("get_body_state failed: %s", exc)
            return {"mode": "fixture", "state": {}, "error": str(exc)}

    async def list_body_capabilities(self, *, status: str = "all") -> dict[str, Any]:
        """List capabilities grouped by status."""
        try:
            from rosclaw.body.mcp_tools import BodyMcpTools

            tools = BodyMcpTools(workspace=self._body_workspace())
            return {"mode": "live", "capabilities": tools.list_body_capabilities(status=status)}
        except Exception as exc:  # noqa: BLE001
            logger.debug("list_body_capabilities failed: %s", exc)
            return {"mode": "fixture", "capabilities": {}, "error": str(exc)}

    async def query_body(self, question: str) -> dict[str, Any]:
        """Answer a natural-language question about the body."""
        try:
            from rosclaw.body.mcp_tools import BodyMcpTools

            tools = BodyMcpTools(workspace=self._body_workspace())
            return {"mode": "live", "result": tools.query_body(question)}
        except Exception as exc:  # noqa: BLE001
            logger.debug("query_body failed: %s", exc)
            return {"mode": "fixture", "result": {"answer": str(exc), "decision": "unknown"}, "error": str(exc)}

    async def validate_body_action(
        self,
        action: str,
        capability_id: str,
        *,
        risk: str = "medium",
    ) -> dict[str, Any]:
        """Validate a proposed physical action against the current body."""
        try:
            from rosclaw.body.mcp_tools import BodyMcpTools

            tools = BodyMcpTools(workspace=self._body_workspace())
            return {"mode": "live", "validation": tools.validate_body_action(action, capability_id, risk=risk)}
        except Exception as exc:  # noqa: BLE001
            logger.debug("validate_body_action failed: %s", exc)
            return {"mode": "fixture", "validation": {"body_check": "unknown", "allowed_to_propose": False, "allowed_to_execute_real_robot": False, "reasons": [str(exc)]}, "error": str(exc)}

    async def get_calibration_status(self, *, component: str | None = None) -> dict[str, Any]:
        """Return calibration status for the body or a component."""
        try:
            from rosclaw.body.mcp_tools import BodyMcpTools

            tools = BodyMcpTools(workspace=self._body_workspace())
            return {"mode": "live", "calibration": tools.get_calibration_status(component=component)}
        except Exception as exc:  # noqa: BLE001
            logger.debug("get_calibration_status failed: %s", exc)
            return {"mode": "fixture", "calibration": {"component": component or "*", "status": "unknown", "confidence": 0.0, "blocks": [str(exc)]}, "error": str(exc)}

    # ------------------------------------------------------------------
    # S4 emergency tool
    # ------------------------------------------------------------------

    async def emergency_stop(self, reason: str) -> dict[str, Any]:
        adapter = self._adapters().get("safety")
        if adapter is not None:
            try:
                return adapter.emergency_stop(reason)
            except Exception as exc:  # noqa: BLE001
                logger.debug("emergency_stop failed: %s", exc)
                return {
                    "stopped": False,
                    "reason": reason,
                    "mode": "error",
                    "note": f"Could not confirm emergency stop: {exc}. Activate physical E-stop immediately.",
                }
        return {
            "stopped": True,
            "reason": reason,
            "mode": "degraded",
            "note": "Runtime not initialized. Emergency stop acknowledged; activate physical E-stop if needed.",
        }
