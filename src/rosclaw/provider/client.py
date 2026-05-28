"""CapabilityClient - High-level task orchestration over providers.

Decomposes natural-language tasks into multi-step provider invocation
chains with full trace collection.

Example:
    client = CapabilityClient(runtime.capability_router)
    result = await client.run_task(
        task="pick up the red cup",
        robot="ur5e",
        scene_input={"camera_topic": "/camera/color/image_raw"},
    )
"""

from dataclasses import dataclass, field
from typing import Any

from rosclaw.provider.core.request import ProviderRequest
from rosclaw.provider.core.response import ProviderResponse
from rosclaw.provider.core.router import CapabilityRouter
from rosclaw.provider.core.trace import ProviderTrace


@dataclass
class TaskPlan:
    """Planned capability sequence for a task."""

    task: str
    steps: list[dict[str, Any]]
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """Result of a full task execution."""

    task: str
    status: str  # "success" | "partial" | "failed"
    steps: list[dict[str, Any]]
    trace: dict[str, Any]
    final_result: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


class CapabilityClient:
    """High-level client for composite capability tasks.

    Translates human-friendly task descriptions into provider
    capability sequences, executes them, and returns unified results.
    """

    def __init__(self, router: CapabilityRouter):
        self.router = router

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def run_task(
        self,
        task: str,
        robot: str = "",
        scene_input: dict[str, Any] | None = None,
        safety_level: str = "MODERATE",
    ) -> TaskResult:
        """Execute a high-level task through the provider chain.

        Args:
            task: Natural language task (e.g., "pick up the red cup").
            robot: Robot identifier.
            scene_input: Scene input dict (camera_topic, etc.).
            safety_level: STRICT | MODERATE | LENIENT.

        Returns:
            TaskResult with full execution trace and aggregated outputs.
        """
        plan = self._plan_task(task, robot, scene_input or {}, safety_level)
        trace = ProviderTrace(task_id=task)
        step_results: list[dict[str, Any]] = []

        for step in plan.steps:
            capability = step["capability"]
            inputs = step.get("inputs", {})
            context = step.get("context", {})

            request = ProviderRequest(
                request_id=f"task_{len(step_results)}",
                capability=capability,
                inputs=inputs,
                context=context,
                constraints={"safety_level": safety_level},
            )

            try:
                response = await self.router.invoke(request)
                step_result = {
                    "capability": capability,
                    "status": "ok" if response.is_ok else "failed",
                    "provider": response.provider,
                    "result": response.result,
                    "errors": response.errors,
                }
                trace.add_step(
                    name=capability,
                    provider=response.provider,
                    capability=capability,
                    latency_ms=response.latency_ms or 0,
                    status="success" if response.is_ok else "failed",
                    metadata={"warnings": response.warnings},
                )
            except Exception as e:
                step_result = {
                    "capability": capability,
                    "status": "failed",
                    "error": str(e),
                }
                trace.add_step(
                    name=capability,
                    provider="",
                    capability=capability,
                    latency_ms=0,
                    status="failed",
                    metadata={"error": str(e)},
                )

            step_results.append(step_result)

            # If a critical step fails, abort the chain
            if step_result["status"] == "failed" and step.get("critical", True):
                break

        # Aggregate status
        failed_steps = [s for s in step_results if s["status"] == "failed"]
        if not failed_steps:
            status = "success"
        elif failed_steps and any(s["status"] == "ok" for s in step_results):
            status = "partial"
        else:
            status = "failed"

        # Build final result from the last successful step that produced output
        final_result: dict[str, Any] = {}
        for s in reversed(step_results):
            if s["status"] == "ok" and s.get("result"):
                final_result = s["result"]
                break

        return TaskResult(
            task=task,
            status=status,
            steps=step_results,
            trace=trace.to_dict(),
            final_result=final_result,
            errors=[s.get("error", "") for s in failed_steps if "error" in s],
        )

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------
    async def locate_and_grasp(
        self,
        object_name: str,
        robot: str = "",
        camera_topic: str = "/camera/color/image_raw",
        safety_level: str = "MODERATE",
    ) -> TaskResult:
        """Specialized task: locate object, then grasp it."""
        return await self.run_task(
            task=f"pick up the {object_name}",
            robot=robot,
            scene_input={"camera_topic": camera_topic, "object_name": object_name},
            safety_level=safety_level,
        )

    async def verify_and_retry(
        self,
        task_description: str,
        robot: str = "",
        camera_topic: str = "/camera/color/image_raw",
    ) -> TaskResult:
        """Verify task success and get retry advice if needed."""
        plan = TaskPlan(
            task=task_description,
            steps=[
                {
                    "capability": "critic.success_detection",
                    "inputs": {"task": task_description, "camera_topic": camera_topic},
                    "context": {"robot": robot},
                },
            ],
        )
        # Reuse run_task with a pre-built plan
        return await self.run_task(
            task=task_description,
            robot=robot,
            scene_input={"camera_topic": camera_topic},
        )

    # ------------------------------------------------------------------
    # Task planner
    # ------------------------------------------------------------------
    def _plan_task(
        self,
        task: str,
        robot: str,
        scene_input: dict[str, Any],
        safety_level: str,
    ) -> TaskPlan:
        """Create a capability plan from a task description.

        This is a rule-based planner. In production, this could be
        replaced with an LLM-based planner.
        """
        task_lower = task.lower()
        camera_topic = scene_input.get("camera_topic", "")
        object_name = scene_input.get("object_name", "")

        # Extract object name from task if not provided
        if not object_name:
            for prefix in ["pick up the ", "grasp the ", "grab the ", "get the "]:
                if prefix in task_lower:
                    object_name = task_lower.split(prefix, 1)[1].strip()
                    break

        # Default plan: locate -> skill -> verify
        steps: list[dict[str, Any]] = []

        if "pick up" in task_lower or "grasp" in task_lower or "grab" in task_lower:
            steps = [
                {
                    "capability": "vlm.object_grounding",
                    "inputs": {"query": object_name, "camera_topic": camera_topic},
                    "context": {"robot": robot},
                    "critical": True,
                },
                {
                    "capability": "skill.grasp",
                    "inputs": {"target": {"object": object_name}, "constraints": {}},
                    "context": {"robot": robot},
                    "critical": True,
                },
                {
                    "capability": "critic.success_detection",
                    "inputs": {"task": task, "camera_topic": camera_topic},
                    "context": {"robot": robot},
                    "critical": False,
                },
            ]
        elif "place" in task_lower or "put" in task_lower:
            destination = ""
            for phrase in [" into ", " onto ", " in ", " on ", " at "]:
                if phrase in task_lower:
                    parts = task_lower.split(phrase, 1)
                    if len(parts) == 2:
                        destination = parts[1].strip()
                        break
            steps = [
                {
                    "capability": "skill.pick_and_place",
                    "inputs": {
                        "target": {"source": object_name, "destination": destination},
                    },
                    "context": {"robot": robot},
                    "critical": True,
                },
                {
                    "capability": "critic.success_detection",
                    "inputs": {"task": task, "camera_topic": camera_topic},
                    "context": {"robot": robot},
                    "critical": False,
                },
            ]
        elif "inspect" in task_lower or "check" in task_lower or "what" in task_lower:
            steps = [
                {
                    "capability": "vlm.scene_understanding",
                    "inputs": {"camera_topic": camera_topic, "text": task},
                    "context": {"robot": robot},
                    "critical": True,
                },
            ]
        elif "navigate" in task_lower or "go to" in task_lower or "move to" in task_lower:
            steps = [
                {
                    "capability": "skill.navigate",
                    "inputs": {"target": {"description": task}, "constraints": {}},
                    "context": {"robot": robot},
                    "critical": True,
                },
            ]
        else:
            # Generic fallback: scene understanding + skill delegation
            steps = [
                {
                    "capability": "vlm.scene_understanding",
                    "inputs": {"camera_topic": camera_topic, "text": task},
                    "context": {"robot": robot},
                    "critical": False,
                },
                {
                    "capability": "skill.inspect",
                    "inputs": {"target": {"description": task}, "constraints": {}},
                    "context": {"robot": robot},
                    "critical": True,
                },
            ]

        return TaskPlan(
            task=task,
            steps=steps,
            context={"robot": robot, "safety_level": safety_level},
        )
