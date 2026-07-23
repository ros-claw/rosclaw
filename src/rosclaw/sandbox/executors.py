"""Fail-closed task executor registry for sandbox actions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rosclaw.kernel import ActionEnvelope, ActionExecutionResult, ActionState, EvidenceLevel

SandboxExecutor = Callable[[Any, ActionEnvelope], ActionExecutionResult]


class SandboxTaskExecutorRegistry:
    def __init__(self) -> None:
        self._executors: dict[str, SandboxExecutor] = {}

    def register(self, capability_id: str, executor: SandboxExecutor) -> None:
        normalized = capability_id.strip()
        if not normalized or not callable(executor):
            raise ValueError("A capability id and callable executor are required")
        self._executors[normalized] = executor

    def capabilities(self) -> tuple[str, ...]:
        return tuple(sorted(self._executors))

    def execute(self, sandbox: Any, action: ActionEnvelope) -> ActionExecutionResult:
        executor = self._executors.get(action.capability_id)
        if executor is None:
            return ActionExecutionResult(
                final_state=ActionState.FAILED,
                evidence_level=EvidenceLevel.REQUESTED,
                errors=[
                    {
                        "code": "TASK_EXECUTOR_NOT_REGISTERED",
                        "message": (
                            f"No sandbox executor is registered for {action.capability_id!r}."
                        ),
                    }
                ],
            )
        return executor(sandbox, action)


__all__ = ["SandboxExecutor", "SandboxTaskExecutorRegistry"]
