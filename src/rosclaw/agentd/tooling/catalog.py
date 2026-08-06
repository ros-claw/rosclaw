"""Tool catalog — registration, quarantine, and guarded execution (PR-05).

The catalog holds ToolDescriptorV2 entries plus the executor for each tool.
Execution is guarded: PHYSICAL_ACTION tools raise immediately (they can only
ever flow through the Operator grant path), and quarantined tools fail
honestly instead of fabricating a result.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from rosclaw.agentd.tooling.result import ToolExecutionResult
from rosclaw.contracts.agent.tool import ExecutionClass, ToolDescriptorV2
from rosclaw.contracts.common import ValidationError

ToolOutput = str | ToolExecutionResult
ToolExecutor = Callable[[dict[str, Any]], Awaitable[ToolOutput]]


class ToolNotCallableError(ValidationError):
    """Raised when code tries to execute a tool the model must never call."""


class ToolQuarantinedError(ValidationError):
    """Raised when a tool/source is quarantined (health check failed)."""


class ToolCatalog:
    def __init__(self) -> None:
        self._descriptors: dict[str, ToolDescriptorV2] = {}
        self._executors: dict[str, ToolExecutor] = {}
        self._quarantine: dict[str, str] = {}  # tool_id -> reason

    def register(self, descriptor: ToolDescriptorV2, executor: ToolExecutor | None = None) -> None:
        if "__" in descriptor.tool_id:
            raise ValidationError(
                f"tool {descriptor.tool_id!r}: '__' is reserved for the wire-name mapping"
            )
        if descriptor.tool_id in self._descriptors:
            raise ValidationError(f"tool {descriptor.tool_id!r} already registered")
        self._descriptors[descriptor.tool_id] = descriptor
        if executor is not None:
            self._executors[descriptor.tool_id] = executor

    def replace(self, descriptor: ToolDescriptorV2, executor: ToolExecutor | None = None) -> None:
        """Idempotent re-registration (e.g. MCP reconnect re-discovery)."""
        self._descriptors[descriptor.tool_id] = descriptor
        if executor is not None:
            self._executors[descriptor.tool_id] = executor

    def get(self, tool_id: str) -> ToolDescriptorV2 | None:
        return self._descriptors.get(tool_id)

    def list(self, *, source: str | None = None) -> list[ToolDescriptorV2]:
        items = sorted(self._descriptors.values(), key=lambda d: d.tool_id)
        if source is not None:
            items = [d for d in items if d.source == source]
        return items

    # -- quarantine -----------------------------------------------------------

    def quarantine_tool(self, tool_id: str, reason: str) -> None:
        self._quarantine[tool_id] = reason

    def quarantine_source(self, source: str, reason: str) -> int:
        count = 0
        for d in self._descriptors.values():
            if d.source == source:
                self._quarantine[d.tool_id] = reason
                count += 1
        return count

    def lift_quarantine(self, tool_id: str) -> None:
        self._quarantine.pop(tool_id, None)

    def lift_source_quarantine(self, source: str) -> int:
        doomed = [tid for tid, d in self._descriptors.items() if d.source == source]
        count = 0
        for tid in doomed:
            if tid in self._quarantine:
                del self._quarantine[tid]
                count += 1
        return count

    def quarantine_reason(self, tool_id: str) -> str | None:
        return self._quarantine.get(tool_id)

    # -- execution --------------------------------------------------------------

    def _canonical(self, tool_id: str) -> str:
        if tool_id in self._descriptors:
            return tool_id
        from rosclaw.agentd.tooling.strict_schema import canonical_name

        candidate = canonical_name(tool_id)
        if candidate in self._descriptors:
            return candidate
        return tool_id

    async def execute(self, tool_id: str, arguments: dict[str, Any]) -> ToolOutput:
        descriptor = self._descriptors.get(self._canonical(tool_id))
        if descriptor is None:
            raise ValidationError(f"tool {tool_id!r} not in catalog")
        tool_id = descriptor.tool_id
        if descriptor.execution_class is ExecutionClass.PHYSICAL_ACTION:
            raise ToolNotCallableError(
                f"tool {tool_id!r} is PHYSICAL_ACTION — never directly executable; "
                "it flows only through REQUEST_APPROVAL → Operator grant → rosclawd"
            )
        if not descriptor.model_callable:
            raise ToolNotCallableError(f"tool {tool_id!r} is not model_callable")
        reason = self._quarantine.get(tool_id)
        if reason is not None:
            raise ToolQuarantinedError(f"tool {tool_id!r} quarantined: {reason}")
        executor = self._executors.get(tool_id)
        if executor is None:
            raise ValidationError(f"tool {tool_id!r} has no executor (source offline?)")
        return await asyncio.wait_for(
            executor(arguments), timeout=descriptor.timeout_ms / 1000.0
        )
