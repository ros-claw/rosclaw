"""Unified facade that isolates MCP tools from direct ROS/hardware access."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

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
