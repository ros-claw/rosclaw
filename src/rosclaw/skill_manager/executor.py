"""Skill Executor - Executes skills with parameter binding and validation."""

from __future__ import annotations

import logging
import time
from typing import Any

from rosclaw.body.resolver import BodyResolver
from rosclaw.core.event_bus import Event, EventBus, EventPriority
from rosclaw.core.lifecycle import LifecycleMixin
from rosclaw.runtime.plugin import get_runtime_plugin
from rosclaw.skill_manager.registry import SkillEntry, SkillRegistry

logger = logging.getLogger("rosclaw.skill_manager.executor")


class SkillExecutor(LifecycleMixin):
    """
    Executes skills by publishing events to the EventBus.

    All skill execution goes through the bus for:
    - Firewall validation
    - Practice recording
    - Memory storage
    - Swarm coordination

    When a SeekDB client is provided, skill execution metadata
    (success/failure counts, avg duration) is persisted to the
    ``skill_metadata`` table after each execution.
    """

    def __init__(
        self,
        event_bus: EventBus,
        registry: SkillRegistry,
        seekdb_client: Any | None = None,
        sense_interface: Any | None = None,
        body_resolver: BodyResolver | None = None,
    ):
        super().__init__()
        self.event_bus = event_bus
        self.registry = registry
        self._seekdb = seekdb_client
        self._sense_interface = sense_interface
        self._skill_requirements_adapter: Any | None = None
        if sense_interface is not None:
            try:
                from rosclaw.sense.adapters.skill_requirements import SkillRequirementsAdapter
                self._skill_requirements_adapter = SkillRequirementsAdapter(sense_interface)
            except Exception:
                logger.warning("Failed to initialize SkillRequirementsAdapter", exc_info=True)
        self._body_resolver = body_resolver

    def _do_initialize(self) -> None:
        logger.info("Initialized")

    def execute(self, skill_name: str, parameters: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Execute a skill by name.

        Args:
            skill_name: Name of registered skill
            parameters: Runtime parameters to override defaults

        Returns:
            Execution result with status and metadata
        """
        skill = self.registry.get(skill_name)
        if skill is None:
            return {"status": "error", "message": f"Skill not found: {skill_name}"}

        # Body compatibility check (new P0)
        body_check = self._check_body_compatibility(skill)
        if body_check.get("status") == "blocked":
            return {
                "status": "blocked",
                "message": body_check.get("reason", "Body compatibility check failed"),
                "body_check": body_check,
            }

        # Merge parameters
        params = {**skill.parameters, **(parameters or {})}

        # Validate preconditions
        precheck = self._check_preconditions(skill)
        if not precheck["ok"]:
            return {"status": "precondition_failed", "message": precheck["reason"]}

        # Body sense readiness gate (new)
        sense_check = self._check_body_sense(skill)
        if not sense_check["ok"]:
            self.event_bus.publish(
                Event(
                    topic="rosclaw.sense.capability.blocked",
                    payload={
                        "capability": skill.name,
                        "reason": sense_check["reason"],
                        "failed_requirements": sense_check.get("failed_requirements", []),
                    },
                    source="skill_executor",
                    priority=EventPriority.HIGH,
                )
            )
            self.event_bus.publish(
                Event(
                    topic="skill.execution.blocked",
                    payload={
                        "skill_name": skill.name,
                        "reason": "blocked_by_body_sense",
                        "message": sense_check["reason"],
                        "failed_requirements": sense_check.get("failed_requirements", []),
                    },
                    source="skill_executor",
                    priority=EventPriority.HIGH,
                )
            )
            return {
                "status": "blocked",
                "reason": "blocked_by_body_sense",
                "message": sense_check["reason"],
                "failed_requirements": sense_check.get("failed_requirements", []),
                **({"body_sense_check": bsc} if (bsc := sense_check.get("body_sense_check")) else {}),
            }

        self._current_skill = skill_name

        # Publish skill execution event
        self.event_bus.publish(
            Event(
                topic="skill.execution.start",
                payload={
                    "skill_name": skill_name,
                    "parameters": params,
                    "skill_type": skill.skill_type,
                },
                source="skill_executor",
                priority=EventPriority.HIGH,
            )
        )

        # Execute handler if available
        result: dict[str, Any] = {
            "status": "executed",
            "skill": skill_name,
            "body_check": body_check,
        }
        bsc = sense_check.get("body_sense_check")
        if bsc is not None:
            result["body_sense_check"] = bsc
        t0 = time.time()
        handler = self._resolve_handler(skill)
        if handler is not None:
            try:
                handler_result = handler(params)
                result["handler_result"] = handler_result
                handler_status = handler_result.get("status") if isinstance(handler_result, dict) else None
                result["status"] = handler_status if handler_status in ("success", "error", "blocked", "degraded") else "success"
                self.registry.update_stats(skill_name, success=(result["status"] == "success"))
            except Exception as e:
                result["status"] = "error"
                result["error"] = str(e)
                self.registry.update_stats(skill_name, success=False)
        else:
            # No handler - event-driven execution
            result["status"] = "dispatched"
        duration_sec = time.time() - t0
        result["duration_sec"] = duration_sec

        # Persist skill metadata to SeekDB
        self._write_skill_metadata(skill, result, duration_sec)

        # Publish completion event
        self.event_bus.publish(
            Event(
                topic="skill.execution.complete",
                payload={
                    "skill_name": skill_name,
                    "result": result,
                },
                source="skill_executor",
                priority=EventPriority.NORMAL,
            )
        )

        self._current_skill = None
        return result

    def _write_skill_metadata(
        self,
        skill: SkillEntry,
        result: dict[str, Any],
        duration_sec: float,
    ) -> None:
        """Write/update skill_metadata in SeekDB after execution."""
        if self._seekdb is None:
            return

        success = result.get("status") == "success"
        failure = result.get("status") == "error"

        # Read existing metadata to accumulate counts
        existing = self._seekdb.query(
            "skill_metadata",
            filters={"skill_id": skill.name},
            limit=1,
        )
        if existing:
            row = existing[0]
            success_count = row.get("success_count", 0) + (1 if success else 0)
            failure_count = row.get("failure_count", 0) + (1 if failure else 0)
            old_avg = row.get("avg_duration_sec", 0.0) or 0.0
            total_exec = success_count + failure_count
            avg_duration = (old_avg * (total_exec - 1) + duration_sec) / total_exec
        else:
            success_count = 1 if success else 0
            failure_count = 1 if failure else 0
            avg_duration = duration_sec

        record = {
            "id": skill.name,
            "skill_id": skill.name,
            "name": skill.name,
            "description": skill.description,
            "category": skill.skill_type,
            "source": "skill_executor",
            "success_count": success_count,
            "failure_count": failure_count,
            "avg_duration_sec": round(avg_duration, 6),
            "last_used": time.time(),
            "prerequisites": skill.preconditions,
            "metadata": skill.metadata,
        }
        self._seekdb.insert("skill_metadata", record)

    def _check_body_compatibility(self, skill: SkillEntry) -> dict[str, Any]:
        """Check skill against the current effective body if body is linked."""
        try:
            resolver = self._body_resolver or BodyResolver()
            if not resolver.is_linked():
                return {
                    "status": "ok"
                }  # no body linked — allow execution with backward compatibility
            result = resolver.check_skill_compatibility(skill.name, skill.version)
            if result.status == "blocked":
                return {"status": "blocked", "reason": result.reason}
            if result.status == "unknown":
                return {
                    "status": "blocked",
                    "reason": "Skill compatibility unknown — run `rosclaw body inspect --skills`",
                }
            return {"status": "ok", "result": result.to_dict()}
        except Exception as exc:
            logger.warning("Body compatibility check failed for %s: %s", skill.name, exc)
            return {
                "status": "blocked",
                "reason": f"Body compatibility check error: {exc}",
            }

    def _check_body_sense(self, skill: SkillEntry) -> dict[str, Any]:
        """Check skill against body sense readiness if the skill declares requirements."""
        requirements = skill.metadata.get("requires_body_sense") if skill.metadata else None
        if not requirements:
            result: dict[str, Any] = {"ok": True}
            return self._enrich_with_body_sense_check(skill, result)
        if self._sense_interface is None:
            result = {
                "ok": False,
                "reason": "Skill requires body sense but Sense module is not available",
                "failed_requirements": [],
            }
            return self._enrich_with_body_sense_check(skill, result)
        readiness = self._sense_interface.get_readiness(
            task=skill.name,
            requirements=requirements,
        )
        item = readiness.capabilities.get(skill.name)
        if item is None or item.status == "ready":
            result = {"ok": True}
            return self._enrich_with_body_sense_check(skill, result)
        if item.status in ("degraded", "unknown"):
            # Low-risk tasks may allow degraded; high-risk block.
            safety_level = (
                skill.metadata.get("safety_level", "MODERATE") if skill.metadata else "MODERATE"
            )
            if safety_level in ("HIGH", "CRITICAL"):
                result = {
                    "ok": False,
                    "reason": f"Body sense degraded for high-risk skill {skill.name}: {item.reasons}",
                    "failed_requirements": [req.to_dict() for req in item.failed_requirements],
                }
            else:
                result = {"ok": True, "note": f"Body sense degraded but allowed: {item.reasons}"}
            return self._enrich_with_body_sense_check(skill, result)
        result = {
            "ok": False,
            "reason": f"Body sense not ready for {skill.name}: {item.reasons}",
            "failed_requirements": [req.to_dict() for req in item.failed_requirements],
        }
        return self._enrich_with_body_sense_check(skill, result)

    def _enrich_with_body_sense_check(self, skill: SkillEntry, result: dict[str, Any]) -> dict[str, Any]:
        """Attach a lightweight body-sense readiness check for observability."""
        if self._skill_requirements_adapter is None:
            return result
        try:
            return self._skill_requirements_adapter.apply({
                "task": skill.name,
                **result,
            })
        except Exception:
            logger.warning("SkillRequirementsAdapter failed; returning body-sense result unchanged", exc_info=True)
            return result

    def _check_preconditions(self, skill: SkillEntry) -> dict[str, Any]:
        """Check if skill preconditions are met."""
        for pre in skill.preconditions:
            # Simple string preconditions - can be extended
            if pre.startswith("skill:"):
                required = pre.split(":", 1)[1]
                if self.registry.get(required) is None:
                    return {"ok": False, "reason": f"Required skill not available: {required}"}
        return {"ok": True}

    def _resolve_handler(self, skill: SkillEntry) -> Any:
        """Resolve the executable handler for a skill.

        Resolution order:
            1. Runtime skill plugin registry (new EventBus-native path).
            2. Legacy ``SkillEntry.handler``.
        """
        runtime_handler = get_runtime_plugin().get_handler(skill.name)
        if runtime_handler is not None:
            return runtime_handler
        return skill.handler

    def is_executing(self) -> bool:
        return self._current_skill is not None

    @property
    def current_skill(self) -> str | None:
        return self._current_skill
