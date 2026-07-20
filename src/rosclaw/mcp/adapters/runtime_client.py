"""Unified facade that isolates MCP tools from direct ROS/hardware access."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, NoReturn

from rosclaw.body.registry import BodyRegistryManager
from rosclaw.body.resolver import BodyResolver
from rosclaw.firstboot.workspace import get_rosclaw_home
from rosclaw.mcp.adapters.memory_client import MemoryClient
from rosclaw.mcp.adapters.practice_client import PracticeClient
from rosclaw.mcp.adapters.safety_client import SafetyClient
from rosclaw.mcp.adapters.sandbox_client import SandboxClient
from rosclaw.mcp.adapters.skill_registry_client import SkillRegistryClient
from rosclaw.mcp.schemas.common import MCPError

logger = logging.getLogger("rosclaw.mcp.adapters.runtime_client")


class RuntimeClient:
    """Lazy-initializing client over the existing ROSClaw Runtime components.

    Fixture responses are available only when ``fixture_mode=True``.  Runtime
    initialization and subsystem failures otherwise become structured errors;
    synthetic data never leaks into a live path.

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
        daemon_client: Any | None = None,
    ):
        self.project_root = project_root
        self.robot_id = robot_id or "rosclaw_default"
        self.runtime_profile = runtime_profile or {}
        self.fixture_mode = fixture_mode
        if daemon_client is None:
            from rosclaw.daemon.client import DaemonClient

            daemon_client = DaemonClient()
        self._daemon_client = daemon_client
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
            "practice": PracticeClient(rt.episode_recorder)
            if rt.episode_recorder is not None
            else None,
            "sandbox": SandboxClient(rt.sandbox) if rt.sandbox is not None else None,
            "skill": SkillRegistryClient(rt.skill_manager)
            if rt.skill_manager is not None
            else None,
        }
        return self._adapter_cache

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fixture_state(self) -> dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "mode": "fixture",
            "execution_mode": "FIXTURE",
            "trust_level": "SYNTHETIC",
            "usable_for_real_execution": False,
            "note": "Explicit fixture mode; no robot observation was performed.",
            "body_state": {
                "joint_positions": [0.0] * 6,
                "joint_velocities": [0.0] * 6,
                "timestamp": 0.0,
            },
            "readiness": {"overall": "UNKNOWN", "factors": {}},
            "risk_summary": {
                "risk_level": "UNKNOWN",
                "violations": ["synthetic_state_not_valid_for_risk_assessment"],
            },
            "is_stale": True,
            "age_ms": 0,
        }

    @staticmethod
    def _fixture_payload(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            **payload,
            "mode": "fixture",
            "execution_mode": "FIXTURE",
            "trust_level": "SYNTHETIC",
            "usable_for_real_execution": False,
        }

    def _unavailable(
        self,
        operation: str,
        message: str | None = None,
        *,
        code: str = "RUNTIME_UNAVAILABLE",
    ) -> NoReturn:
        reason = message or self._runtime_error or "ROSClaw runtime is unavailable."
        raise MCPError(
            code,
            f"{operation} unavailable: {reason}",
            details={
                "operation": operation,
                "runtime_error": self._runtime_error,
                "execution_mode": "UNKNOWN",
                "trust_level": "UNAVAILABLE",
                "usable_for_real_execution": False,
            },
        )

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
        if self.fixture_mode:
            return self._fixture_state()
        rt = self._ensure_runtime()
        if rt is None:
            self._unavailable("get_robot_state")
        if rt.sense is None:
            self._unavailable("get_robot_state", "Body Sense subsystem is not initialized.")
        try:
            sense = rt.sense
            latest = sense.get_latest_state()
            body_sense = sense.get_body_sense()
            readiness = sense.get_readiness()
            body_state = self._safe_dict(latest)
            source = str(body_state.get("source", ""))
            collector = str(getattr(getattr(sense, "config", None), "collector", ""))
            if source.startswith("mock:") or collector == "mock":
                self._unavailable(
                    "get_robot_state",
                    "Body Sense is using a synthetic mock collector; enable fixture mode "
                    "explicitly to consume synthetic state.",
                    code="SYNTHETIC_SOURCE_NOT_ALLOWED",
                )
            elif source.startswith("file_replay:"):
                mode = "replay"
                execution_mode = "REPLAY"
                trust_level = "RECORDED"
            elif (
                not source
                or source == "unknown"
                or source.endswith(":stub")
                or source == "sense:degraded"
            ):
                self._unavailable(
                    "get_robot_state",
                    f"Body state has no live provenance (source={source or 'missing'}).",
                    code="STATE_PROVENANCE_UNAVAILABLE",
                )
            else:
                mode = "live" if not getattr(sense, "is_stale", True) else "stale"
                execution_mode = "REAL"
                trust_level = "OBSERVED" if mode == "live" else "STALE"
            return {
                "robot_id": self.robot_id,
                "mode": mode,
                "execution_mode": execution_mode,
                "trust_level": trust_level,
                "usable_for_real_execution": mode == "live",
                "body_state": body_state,
                "body_sense": self._safe_dict(body_sense),
                "readiness": self._safe_dict(readiness),
                "is_stale": getattr(sense, "is_stale", True),
                "age_ms": getattr(sense, "state_age_ms", 0),
            }
        except Exception as exc:  # noqa: BLE001
            logger.debug("get_robot_state failed: %s", exc)
            if isinstance(exc, MCPError):
                raise
            self._unavailable("get_robot_state", str(exc), code="SENSE_READ_FAILED")

    async def list_skills(
        self, *, skill_type: str | None = None, full_ids: bool = False
    ) -> dict[str, Any]:
        if self.fixture_mode:
            return self._fixture_payload({"skills": [], "count": 0})
        adapter = self._adapters().get("skill")
        if adapter is None:
            self._unavailable("list_skills", "Skill registry is not initialized.")
        try:
            return adapter.list_skills(skill_type=skill_type, full_ids=full_ids)
        except Exception as exc:  # noqa: BLE001
            logger.debug("list_skills failed: %s", exc)
            self._unavailable("list_skills", str(exc), code="SKILL_QUERY_FAILED")

    async def query_memory(
        self,
        instruction: str,
        *,
        limit: int = 5,
        outcome_filter: str | None = None,
    ) -> dict[str, Any]:
        if self.fixture_mode:
            return self._fixture_payload({"experiences": [], "count": 0})
        adapter = self._adapters().get("memory")
        if adapter is None:
            self._unavailable("query_memory", "Memory subsystem is not initialized.")
        try:
            return adapter.find_similar_experiences(
                instruction=instruction,
                limit=limit,
                outcome_filter=outcome_filter,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("query_memory failed: %s", exc)
            self._unavailable("query_memory", str(exc), code="MEMORY_QUERY_FAILED")

    # ------------------------------------------------------------------
    # S2 validated-plan tool
    # ------------------------------------------------------------------

    async def validate_trajectory(
        self,
        trajectory: list[list[float]],
        *,
        safety_level: str = "MODERATE",
    ) -> dict[str, Any]:
        if self.fixture_mode:
            return self._fixture_payload(
                {
                    "is_safe": False,
                    "risk_score": 1.0,
                    "reason": "Explicit fixture mode cannot validate a physical trajectory.",
                    "violations": ["fixture_not_valid_for_safety_acceptance"],
                    "replay_id": None,
                    "validation_type": "FixtureOnly",
                    "simulation_executed": False,
                }
            )
        adapter = self._adapters().get("sandbox")
        if adapter is None:
            self._unavailable("validate_trajectory", "Sandbox is not initialized.")
        try:
            return adapter.validate_trajectory(trajectory=trajectory, safety_level=safety_level)
        except Exception as exc:  # noqa: BLE001
            logger.debug("validate_trajectory failed: %s", exc)
            self._unavailable("validate_trajectory", str(exc), code="VALIDATION_FAILED")

    # ------------------------------------------------------------------
    # S1 simulation-only tool
    # ------------------------------------------------------------------

    async def sandbox_run(self, joint_positions: list[float]) -> dict[str, Any]:
        if self.fixture_mode:
            return self._fixture_payload(
                {
                    "physics_state": {},
                    "has_physics": False,
                    "note": "Explicit fixture mode; no physics was executed.",
                }
            )
        adapter = self._adapters().get("sandbox")
        if adapter is None:
            self._unavailable("sandbox_run", "Sandbox is not initialized.")
        try:
            result = adapter.simulate_step(joint_positions)
            if not result.get("has_physics", False):
                self._unavailable("sandbox_run", result.get("note", "Physics unavailable."))
            return result
        except Exception as exc:  # noqa: BLE001
            logger.debug("sandbox_run failed: %s", exc)
            if isinstance(exc, MCPError):
                raise
            self._unavailable("sandbox_run", str(exc), code="SIMULATION_FAILED")

    # ------------------------------------------------------------------
    # S0 practice-query tool
    # ------------------------------------------------------------------

    async def practice_query(
        self,
        *,
        episode_id: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        if self.fixture_mode:
            return self._fixture_payload({"episodes": [], "count": 0})
        adapter = self._adapters().get("practice")
        if adapter is None:
            self._unavailable("practice_query", "Practice recorder is not initialized.")
        try:
            return adapter.query(episode_id=episode_id, limit=limit)
        except Exception as exc:  # noqa: BLE001
            logger.debug("practice_query failed: %s", exc)
            self._unavailable("practice_query", str(exc), code="PRACTICE_QUERY_FAILED")

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
            self._unavailable("list_bodies", str(exc), code="BODY_REGISTRY_FAILED")

    async def get_body(self, body_id: str) -> dict[str, Any]:
        """Return registry entry and effective body snapshot for one body."""
        try:
            workspace = self._body_workspace()
            manager = BodyRegistryManager(workspace)
            entry = manager.get_body(body_id)
            if entry is None:
                self._unavailable(
                    "get_body",
                    f"Body '{body_id}' was not found.",
                    code="BODY_NOT_FOUND",
                )
            resolver = BodyResolver(workspace, body_id=body_id)
            effective = resolver.get_effective_body(recompile_if_stale=False)
            return {
                "mode": "live",
                "body": entry.to_dict(),
                "effective_body": effective.to_dict(),
            }
        except Exception as exc:  # noqa: BLE001
            logger.debug("get_body failed: %s", exc)
            if isinstance(exc, MCPError):
                raise
            self._unavailable("get_body", str(exc), code="BODY_READ_FAILED")

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
            self._unavailable("switch_body", str(exc), code="BODY_SWITCH_FAILED")

    async def list_body_history(self, body_id: str) -> dict[str, Any]:
        """List snapshot history for a body."""
        try:
            workspace = self._body_workspace()
            resolver = BodyResolver(workspace, body_id=body_id)
            snapshots = sorted(resolver.snapshots_dir.glob("body-*.yaml"), reverse=True)
            return {
                "mode": "live",
                "body_id": body_id,
                "snapshots": [{"path": str(p), "name": p.name} for p in snapshots],
            }
        except Exception as exc:  # noqa: BLE001
            logger.debug("list_body_history failed: %s", exc)
            self._unavailable("list_body_history", str(exc), code="BODY_HISTORY_FAILED")

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
            self._unavailable(
                "check_skill_compatibility",
                str(exc),
                code="COMPATIBILITY_CHECK_FAILED",
            )

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
            self._unavailable(
                "fleet_skill_compatibility",
                str(exc),
                code="FLEET_COMPATIBILITY_FAILED",
            )

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
            self._unavailable("get_body_profile", str(exc), code="BODY_PROFILE_FAILED")

    async def get_body_state(self, *, include_runtime: bool = True) -> dict[str, Any]:
        """Return current body safety and capability state."""
        try:
            from rosclaw.body.mcp_tools import BodyMcpTools

            tools = BodyMcpTools(workspace=self._body_workspace())
            return {"mode": "live", "state": tools.get_body_state(include_runtime=include_runtime)}
        except Exception as exc:  # noqa: BLE001
            logger.debug("get_body_state failed: %s", exc)
            self._unavailable("get_body_state", str(exc), code="BODY_STATE_FAILED")

    async def list_body_capabilities(self, *, status: str = "all") -> dict[str, Any]:
        """List capabilities grouped by status."""
        try:
            from rosclaw.body.mcp_tools import BodyMcpTools

            tools = BodyMcpTools(workspace=self._body_workspace())
            return {"mode": "live", "capabilities": tools.list_body_capabilities(status=status)}
        except Exception as exc:  # noqa: BLE001
            logger.debug("list_body_capabilities failed: %s", exc)
            self._unavailable(
                "list_body_capabilities",
                str(exc),
                code="BODY_CAPABILITY_QUERY_FAILED",
            )

    async def query_body(self, question: str) -> dict[str, Any]:
        """Answer a natural-language question about the body."""
        try:
            from rosclaw.body.mcp_tools import BodyMcpTools

            tools = BodyMcpTools(workspace=self._body_workspace())
            return {"mode": "live", "result": tools.query_body(question)}
        except Exception as exc:  # noqa: BLE001
            logger.debug("query_body failed: %s", exc)
            self._unavailable("query_body", str(exc), code="BODY_QUERY_FAILED")

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
            return {
                "mode": "live",
                "validation": tools.validate_body_action(action, capability_id, risk=risk),
            }
        except Exception as exc:  # noqa: BLE001
            logger.debug("validate_body_action failed: %s", exc)
            self._unavailable(
                "validate_body_action",
                str(exc),
                code="BODY_ACTION_VALIDATION_FAILED",
            )

    async def get_calibration_status(self, *, component: str | None = None) -> dict[str, Any]:
        """Return calibration status for the body or a component."""
        try:
            from rosclaw.body.mcp_tools import BodyMcpTools

            tools = BodyMcpTools(workspace=self._body_workspace())
            return {
                "mode": "live",
                "calibration": tools.get_calibration_status(component=component),
            }
        except Exception as exc:  # noqa: BLE001
            logger.debug("get_calibration_status failed: %s", exc)
            self._unavailable(
                "get_calibration_status",
                str(exc),
                code="CALIBRATION_QUERY_FAILED",
            )

    # ------------------------------------------------------------------
    # rosclawd control-plane tools
    # ------------------------------------------------------------------

    async def get_runtime_status(self) -> dict[str, Any]:
        """Return daemon status without initializing an in-process Runtime."""

        try:
            result = await asyncio.to_thread(self._daemon_client.get_runtime_status)
        except Exception as exc:  # noqa: BLE001
            self._raise_daemon_error("get_runtime_status", exc)
        return {
            **result,
            "execution_mode": "NONE",
            "trust_level": "DAEMON_REPORTED",
            "usable_for_real_execution": False,
        }

    async def request_action(
        self,
        *,
        capability_id: str,
        arguments: dict[str, Any],
        execution_mode: str = "SHADOW",
        body_snapshot_hash: str,
        principal_id: str = "",
        approval_id: str | None = None,
        body_id: str | None = None,
        action_id: str | None = None,
        required_evidence: str = "TASK_VERIFIED",
        timeout_sec: float = 30.0,
        wait_timeout_sec: float = 2.0,
    ) -> dict[str, Any]:
        """Submit a structured action request; only rosclawd can authorize REAL."""

        from rosclaw.kernel import (
            ActionEnvelope,
            AuthorizationContext,
            EvidenceLevel,
            ExecutionMode,
            VerificationPolicy,
        )

        if self.fixture_mode:
            self._unavailable(
                "request_action",
                "Fixture mode cannot submit actions to rosclawd.",
                code="FIXTURE_ACTION_FORBIDDEN",
            )
        if not isinstance(arguments, dict):
            raise MCPError("INVALID_ARGUMENT", "arguments must be an object")
        try:
            mode = ExecutionMode(str(execution_mode).upper())
        except ValueError as exc:
            raise MCPError(
                "INVALID_EXECUTION_MODE",
                f"Unsupported execution mode {execution_mode!r}.",
            ) from exc
        if mode not in {ExecutionMode.SHADOW, ExecutionMode.REAL}:
            raise MCPError(
                "INVALID_EXECUTION_MODE",
                "request_action accepts only SHADOW or REAL; use sandbox tools for simulation.",
            )
        try:
            evidence = EvidenceLevel(str(required_evidence).upper())
        except ValueError as exc:
            raise MCPError(
                "INVALID_EVIDENCE_LEVEL",
                f"Unsupported evidence level {required_evidence!r}.",
            ) from exc

        action_kwargs: dict[str, Any] = {
            "actor_id": os.environ.get("ROSCLAW_AGENT_ACTOR", "rosclaw-mcp"),
            "agent_framework": os.environ.get("ROSCLAW_AGENT_CLIENT", "mcp"),
            "session_id": os.environ.get("ROSCLAW_AGENT_SESSION", "mcp-session"),
            "body_id": body_id or self.robot_id,
            "body_snapshot_hash": body_snapshot_hash,
            "capability_id": capability_id,
            "arguments": arguments,
            "execution_mode": mode,
            "authorization": AuthorizationContext(
                principal_id=principal_id,
                approved=False,
                approval_id=approval_id,
                scopes=[],
            ),
            "verification_policy": VerificationPolicy(
                required_evidence=evidence,
                timeout_sec=timeout_sec,
                fail_closed=True,
            ),
        }
        if action_id:
            action_kwargs["action_id"] = action_id
        try:
            action = ActionEnvelope(**action_kwargs)
            ticket = await asyncio.to_thread(self._daemon_client.request_action, action)
            result = ticket
            if wait_timeout_sec > 0:
                result = await asyncio.to_thread(
                    self._daemon_client.wait_for_action,
                    action.action_id,
                    timeout_sec=wait_timeout_sec,
                )
        except MCPError:
            raise
        except Exception as exc:  # noqa: BLE001
            self._raise_daemon_error("request_action", exc)
        return self._action_result_metadata(result, mode.value)

    async def get_action_status(self, action_id: str) -> dict[str, Any]:
        """Read daemon queue state and any terminal ExecutionReceipt."""

        try:
            result = await asyncio.to_thread(
                self._daemon_client.get_action_status,
                action_id,
            )
        except Exception as exc:  # noqa: BLE001
            self._raise_daemon_error("get_action_status", exc)
        return self._action_result_metadata(result, "UNKNOWN")

    async def cancel_action(self, action_id: str) -> dict[str, Any]:
        """Cancel only before dispatch; active motion requires emergency_stop."""

        try:
            result = await asyncio.to_thread(
                self._daemon_client.cancel_action,
                action_id,
            )
        except Exception as exc:  # noqa: BLE001
            self._raise_daemon_error("cancel_action", exc)
        return self._action_result_metadata(result, "UNKNOWN")

    async def emergency_stop(self, reason: str) -> dict[str, Any]:
        if self.fixture_mode:
            return self._fixture_payload(
                {
                    "request_id": None,
                    "reason": reason,
                    "targets": [],
                    "request_dispatched": False,
                    "driver_acknowledged": False,
                    "physical_stop_observed": False,
                    "stopped": False,
                    "final_status": "UNVERIFIED",
                    "note": (
                        "Fixture mode cannot stop hardware. Activate the physical E-stop "
                        "if a robot may be moving."
                    ),
                }
            )
        try:
            return await asyncio.to_thread(
                SafetyClient(self._daemon_client).emergency_stop,
                reason,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("emergency_stop through rosclawd failed: %s", exc)
            self._raise_daemon_error(
                "emergency_stop",
                exc,
                suffix=" Activate the certified physical E-stop immediately.",
            )

    @staticmethod
    def _action_result_metadata(
        result: dict[str, Any],
        fallback_mode: str,
    ) -> dict[str, Any]:
        receipt = result.get("receipt")
        receipt_data = receipt if isinstance(receipt, dict) else {}
        return {
            **result,
            "execution_mode": str(
                receipt_data.get("execution_mode") or result.get("execution_mode") or fallback_mode
            ),
            "trust_level": str(receipt_data.get("trust_level", "UNVERIFIED")),
            "usable_for_real_execution": bool(receipt_data.get("usable_for_real_execution", False)),
        }

    def _raise_daemon_error(
        self,
        operation: str,
        exc: Exception,
        *,
        suffix: str = "",
    ) -> NoReturn:
        code = str(getattr(exc, "code", "ROSCLAWD_UNAVAILABLE"))
        details = getattr(exc, "details", {})
        raise MCPError(
            code,
            f"{operation} through rosclawd failed: {exc}.{suffix}",
            details={
                "operation": operation,
                "daemon_error": details if isinstance(details, dict) else {},
                "execution_mode": "UNKNOWN",
                "trust_level": "UNAVAILABLE",
                "usable_for_real_execution": False,
            },
        ) from exc
