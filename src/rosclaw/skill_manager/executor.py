"""Skill Executor - Executes skills with parameter binding and validation."""

import logging
import time
from typing import Any

from rosclaw.core.event_bus import Event, EventBus, EventPriority
from rosclaw.core.lifecycle import LifecycleMixin
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
    ):
        super().__init__()
        self.event_bus = event_bus
        self.registry = registry
        self._seekdb = seekdb_client
        self._current_skill: str | None = None

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

        # Merge parameters
        params = {**skill.parameters, **(parameters or {})}

        # Validate preconditions
        precheck = self._check_preconditions(skill)
        if not precheck["ok"]:
            return {"status": "precondition_failed", "message": precheck["reason"]}

        self._current_skill = skill_name

        # Publish skill execution event
        self.event_bus.publish(Event(
            topic="skill.execution.start",
            payload={
                "skill_name": skill_name,
                "parameters": params,
                "skill_type": skill.skill_type,
            },
            source="skill_executor",
            priority=EventPriority.HIGH,
        ))

        # Execute handler if available
        result = {"status": "executed", "skill": skill_name}
        t0 = time.time()
        if skill.handler is not None:
            try:
                handler_result = skill.handler(params)
                result["handler_result"] = handler_result
                result["status"] = "success"
                self.registry.update_stats(skill_name, success=True)
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
        self.event_bus.publish(Event(
            topic="skill.execution.complete",
            payload={
                "skill_name": skill_name,
                "result": result,
            },
            source="skill_executor",
            priority=EventPriority.NORMAL,
        ))

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

    def _check_preconditions(self, skill: SkillEntry) -> dict[str, Any]:
        """Check if skill preconditions are met."""
        for pre in skill.preconditions:
            # Simple string preconditions - can be extended
            if pre.startswith("skill:"):
                required = pre.split(":", 1)[1]
                if self.registry.get(required) is None:
                    return {"ok": False, "reason": f"Required skill not available: {required}"}
        return {"ok": True}

    def is_executing(self) -> bool:
        return self._current_skill is not None

    @property
    def current_skill(self) -> str | None:
        return self._current_skill
