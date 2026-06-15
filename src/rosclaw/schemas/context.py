"""Context schemas — AgentContext, RuntimeContext, TaskContext."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskContext:
    """What the agent is trying to do."""

    task_name: str = ""
    task_family: str = ""
    domain: str = ""
    embodiment_type: str = ""
    objective_direction: str = "maximize"  # "maximize" | "minimize"
    metric_name: str = ""
    hard_constraints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "task_family": self.task_family,
            "domain": self.domain,
            "embodiment_type": self.embodiment_type,
            "objective_direction": self.objective_direction,
            "metric_name": self.metric_name,
            "hard_constraints": self.hard_constraints,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TaskContext:
        return cls(
            task_name=d.get("task_name", ""),
            task_family=d.get("task_family", ""),
            domain=d.get("domain", ""),
            embodiment_type=d.get("embodiment_type", ""),
            objective_direction=d.get("objective_direction", "maximize"),
            metric_name=d.get("metric_name", ""),
            hard_constraints=list(d.get("hard_constraints", [])),
        )


@dataclass
class RobotContext:
    """Physical robot context."""

    robot_id: str = ""
    eurdf_profile: str = ""
    active_skill: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "eurdf_profile": self.eurdf_profile,
            "active_skill": self.active_skill,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RobotContext:
        return cls(
            robot_id=d.get("robot_id", ""),
            eurdf_profile=d.get("eurdf_profile", ""),
            active_skill=d.get("active_skill", ""),
        )


@dataclass
class SafetyContext:
    """Safety configuration context."""

    safety_level: str = "MODERATE"  # LOW | MODERATE | HIGH | CRITICAL
    sandbox_required: bool = True
    human_nearby: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "safety_level": self.safety_level,
            "sandbox_required": self.sandbox_required,
            "human_nearby": self.human_nearby,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SafetyContext:
        return cls(
            safety_level=d.get("safety_level", "MODERATE"),
            sandbox_required=d.get("sandbox_required", True),
            human_nearby=d.get("human_nearby", False),
        )


@dataclass
class RuntimeContext:
    """Full runtime context for a ROSClaw execution run.

    Mirrors the shape defined in the ROSClaw 1.0 optimization doc
    (Section 10.2 RuntimeContext).
    """

    run_id: str = ""
    trace_id: str = ""
    task_context: TaskContext = field(default_factory=TaskContext)
    robot_context: RobotContext = field(default_factory=RobotContext)
    safety_context: SafetyContext = field(default_factory=SafetyContext)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "trace_id": self.trace_id,
            "task_context": self.task_context.to_dict(),
            "robot_context": self.robot_context.to_dict(),
            "safety_context": self.safety_context.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RuntimeContext:
        return cls(
            run_id=d.get("run_id", ""),
            trace_id=d.get("trace_id", ""),
            task_context=TaskContext.from_dict(d.get("task_context", {})),
            robot_context=RobotContext.from_dict(d.get("robot_context", {})),
            safety_context=SafetyContext.from_dict(d.get("safety_context", {})),
        )


@dataclass
class AgentContext:
    """Agent identity and capability context (Section 10.1)."""

    agent_id: str = ""
    model: str = ""
    task_id: str = ""
    robot_id: str = ""
    available_capabilities: list[str] = field(default_factory=list)
    memory_refs: list[str] = field(default_factory=list)
    safety_level: str = "high"

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "model": self.model,
            "task_id": self.task_id,
            "robot_id": self.robot_id,
            "available_capabilities": self.available_capabilities,
            "memory_refs": self.memory_refs,
            "safety_level": self.safety_level,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AgentContext:
        return cls(
            agent_id=d.get("agent_id", ""),
            model=d.get("model", ""),
            task_id=d.get("task_id", ""),
            robot_id=d.get("robot_id", ""),
            available_capabilities=list(d.get("available_capabilities", [])),
            memory_refs=list(d.get("memory_refs", [])),
            safety_level=d.get("safety_level", "high"),
        )
