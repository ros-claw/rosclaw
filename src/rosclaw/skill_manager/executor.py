"""Skill Executor - Executes skills with parameter binding and validation."""

from typing import Any, Optional

from rosclaw.core.event_bus import EventBus, Event, EventPriority
from rosclaw.core.lifecycle import LifecycleMixin
from rosclaw.skill_manager.registry import SkillRegistry, SkillEntry


class SkillExecutor(LifecycleMixin):
    """
    Executes skills by publishing events to the EventBus.

    All skill execution goes through the bus for:
    - Firewall validation
    - Practice recording
    - Memory storage
    - Swarm coordination
    """

    def __init__(self, event_bus: EventBus, registry: SkillRegistry):
        super().__init__()
        self.event_bus = event_bus
        self.registry = registry
        self._current_skill: Optional[str] = None

    def _do_initialize(self) -> None:
        print("[SkillExecutor] Initialized")

    def execute(self, skill_name: str, parameters: Optional[dict[str, Any]] = None) -> dict[str, Any]:
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
    def current_skill(self) -> Optional[str]:
        return self._current_skill
