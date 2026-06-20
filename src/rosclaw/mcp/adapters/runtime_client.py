"""Unified facade that isolates MCP tools from direct ROS/hardware access."""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any

from rosclaw.body.fleet import FleetCompatibilityAggregator
from rosclaw.body.registry import BodyRegistryManager
from rosclaw.body.resolver import BodyResolver
from rosclaw.body.schema import SkillManifest

logger = logging.getLogger("rosclaw.mcp.adapters.runtime_client")


class RuntimeClient:
    """Lazy-initializing client over the existing ROSClaw Runtime components.

    The client is intentionally defensive: if the runtime cannot be initialized
    (missing model, no ROS, etc.) the tools fall back to fixture responses so
    that the agent still receives a well-formed envelope.
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
        self._event_bus: Any | None = None

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
            self._event_bus = rt.event_bus
            return rt
        except Exception as exc:  # noqa: BLE001
            self._runtime_error = str(exc)
            logger.debug("RuntimeClient could not initialize Runtime: %s", exc)
            return None

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
                "mode": "live" if not sense.is_stale else "stale",
                "body_state": latest.to_dict() if latest and hasattr(latest, "to_dict") else latest,
                "body_sense": body_sense.to_dict() if body_sense and hasattr(body_sense, "to_dict") else body_sense,
                "readiness": readiness.to_dict() if readiness and hasattr(readiness, "to_dict") else readiness,
                "is_stale": getattr(sense, "is_stale", True),
                "age_ms": getattr(sense, "state_age_ms", 0),
            }
        except Exception as exc:  # noqa: BLE001
            logger.debug("get_robot_state failed: %s", exc)
            return self._fixture_state()

    async def list_skills(self, *, skill_type: str | None = None, full_ids: bool = False) -> dict[str, Any]:
        rt = self._ensure_runtime()
        if rt is None or rt.skill_manager is None:
            return {"skills": [], "count": 0, "mode": "fixture"}
        try:
            entries = rt.skill_manager.list_skills(
                skill_type=skill_type,
                return_entries=True,
                full_ids=full_ids,
            )
            return {
                "skills": [e.to_dict() if hasattr(e, "to_dict") else str(e) for e in entries],
                "count": len(entries),
                "mode": "live",
            }
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
        rt = self._ensure_runtime()
        if rt is None or rt.memory is None:
            return {"experiences": [], "count": 0, "mode": "fixture"}
        try:
            results = rt.memory.find_similar_experiences(
                instruction=instruction,
                limit=limit,
                outcome_filter=outcome_filter,
            )
            return {"experiences": results, "count": len(results), "mode": "live"}
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
        rt = self._ensure_runtime()
        if rt is None or rt.sandbox is None:
            return {
                "is_safe": False,
                "risk_score": 1.0,
                "reason": "Sandbox/runtime unavailable; cannot validate trajectory.",
                "violations": ["runtime_unavailable"],
                "replay_id": None,
            }
        try:
            result = rt.sandbox.validate_trajectory(
                trajectory=trajectory,
                safety_level=safety_level,
            )
            if isinstance(result, dict):
                return result
            return {
                "is_safe": False,
                "risk_score": 1.0,
                "reason": "Sandbox returned an unexpected validation result.",
                "violations": ["invalid_validation_result"],
                "replay_id": None,
            }
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
        rt = self._ensure_runtime()
        if rt is None or rt.sandbox is None:
            return {
                "physics_state": {},
                "mode": "fixture",
                "note": "Sandbox runtime unavailable; returning empty fixture.",
            }
        try:
            state = rt.sandbox.simulate_step(joint_positions)
            return {"physics_state": state, "mode": "live"}
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
        rt = self._ensure_runtime()
        if rt is None or rt._episode_recorder is None:
            return {"episodes": [], "count": 0, "mode": "fixture"}
        try:
            recorder = rt._episode_recorder
            if episode_id:
                episode = recorder.get_episode(episode_id)
                episodes = [episode] if episode else []
            else:
                episodes = recorder.list_episodes()[:limit]
            return {"episodes": episodes, "count": len(episodes), "mode": "live"}
        except Exception as exc:  # noqa: BLE001
            logger.debug("practice_query failed: %s", exc)
            return {"episodes": [], "count": 0, "mode": "fixture"}

    # ------------------------------------------------------------------
    # Body registry tools (P2)
    # ------------------------------------------------------------------

    def _body_workspace(self) -> Path:
        """Return the ROSClaw workspace used by the body registry."""
        return Path.home() / ".rosclaw"

    def _discover_skill_manifests(self, workspace: Path) -> list[SkillManifest]:
        """Discover skill manifests under workspace/skills."""
        skills_dir = workspace / "skills"
        if not skills_dir.exists():
            return []
        manifests: list[SkillManifest] = []
        for path in skills_dir.rglob("*.skill.yaml"):
            with contextlib.suppress(Exception):
                manifests.append(SkillManifest.from_yaml(path))
        return manifests

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

            workspace = self._body_workspace()
            manifests = self._discover_skill_manifests(workspace)
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
            workspace = self._body_workspace()
            manifests = self._discover_skill_manifests(workspace)
            report = FleetCompatibilityAggregator(workspace).aggregate(manifests)
            return {"mode": "live", "report": report.to_dict()}
        except Exception as exc:  # noqa: BLE001
            logger.debug("fleet_skill_compatibility failed: %s", exc)
            return {"mode": "fixture", "report": {}, "error": str(exc)}

    # ------------------------------------------------------------------
    # S4 emergency tool
    # ------------------------------------------------------------------

    async def emergency_stop(self, reason: str) -> dict[str, Any]:
        try:
            rt = self._ensure_runtime()
            if rt is not None:
                from rosclaw.core.event_bus import Event, EventPriority

                event = Event(
                    topic="robot.emergency_stop",
                    payload={"reason": reason, "source": "mcp.emergency_stop"},
                    source="rosclaw.mcp.server",
                    priority=EventPriority.CRITICAL,
                )
                rt.event_bus.publish(event)
                # Also invoke the runtime handler directly for immediate effect.
                rt._on_emergency_stop(event)
                return {"stopped": True, "reason": reason, "mode": "live"}
            # No runtime: still acknowledge and advise physical E-stop.
            return {
                "stopped": True,
                "reason": reason,
                "mode": "degraded",
                "note": "Runtime not initialized. Emergency stop acknowledged; activate physical E-stop if needed.",
            }
        except Exception as exc:  # noqa: BLE001
            logger.debug("emergency_stop failed: %s", exc)
            return {
                "stopped": False,
                "reason": reason,
                "mode": "error",
                "note": f"Could not confirm emergency stop: {exc}. Activate physical E-stop immediately.",
            }
