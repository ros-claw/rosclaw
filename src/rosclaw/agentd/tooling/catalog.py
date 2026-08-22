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

from rosclaw.agentd.tooling.capability_adapter import (
    capability_from_tool_descriptor,
    tool_descriptor_from_capability,
)
from rosclaw.agentd.tooling.result import ToolExecutionResult
from rosclaw.contracts.agent.capability import (
    CapabilityDescriptorV2,
    EffectClassV1,
    ProjectionExposure,
    ToolProjectionV1,
)
from rosclaw.contracts.agent.tool import ExecutionClass, ToolDescriptorV2
from rosclaw.contracts.agent.tool_result import (
    ToolResultEnvelopeV2,
    ToolResultErrorV1,
    ToolResultStatusV1,
)
from rosclaw.contracts.common import ValidationError

# N5B：executor 只返回 canonical JSON value（dict）；str 为过渡兼容
# （execute_v2 会 json.loads 归一；非 JSON 文本一律 INVALID_CAPABILITY_OUTPUT）。
ToolOutput = str | dict[str, Any] | ToolExecutionResult
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
        # PR-N5A：CapabilityDescriptorV2 为 canonical 存储；_descriptors
        # 是过渡期的 legacy 派生视图（N11 删除）。
        self._capabilities: dict[str, CapabilityDescriptorV2] = {}
        self._projections: dict[str, ToolProjectionV1] = {}
        # PR-N5D：registry 代际——任何可见变化（注册/替换/隔离）递增，
        # snapshot.generation 由此来。
        self._generation = 0

    @property
    def generation(self) -> int:
        return self._generation

    def register(self, descriptor: ToolDescriptorV2, executor: ToolExecutor | None = None) -> None:
        if "__" in descriptor.tool_id:
            raise ValidationError(
                f"tool {descriptor.tool_id!r}: '__' is reserved for the wire-name mapping"
            )
        if descriptor.tool_id in self._descriptors:
            raise ValidationError(f"tool {descriptor.tool_id!r} already registered")
        self._descriptors[descriptor.tool_id] = descriptor
        self._capabilities[descriptor.tool_id] = capability_from_tool_descriptor(descriptor)
        if executor is not None:
            self._executors[descriptor.tool_id] = executor
        self._generation += 1

    def replace(self, descriptor: ToolDescriptorV2, executor: ToolExecutor | None = None) -> None:
        """Idempotent re-registration (e.g. MCP reconnect re-discovery)."""
        self._descriptors[descriptor.tool_id] = descriptor
        self._capabilities[descriptor.tool_id] = capability_from_tool_descriptor(descriptor)
        if executor is not None:
            self._executors[descriptor.tool_id] = executor
        self._generation += 1

    # -- N5A capability/projection surface --------------------------------------

    def register_capability(
        self, capability: CapabilityDescriptorV2, executor: ToolExecutor | None = None
    ) -> None:
        """直接注册 V2 能力（canonical）；同步派生 legacy 视图。"""
        cid = capability.capability_id
        if cid in self._capabilities:
            raise ValidationError(f"capability {cid!r} already registered")
        self._capabilities[cid] = capability
        self._descriptors[cid] = tool_descriptor_from_capability(capability)
        if executor is not None:
            self._executors[cid] = executor
        self._generation += 1

    def capability(self, tool_id: str) -> CapabilityDescriptorV2 | None:
        """canonical 能力视图（审批/并发/Verifier 应读这里）。"""
        return self._capabilities.get(self._canonical(tool_id))

    def list_capabilities(self, *, source: str | None = None) -> list[CapabilityDescriptorV2]:
        items = sorted(self._capabilities.values(), key=lambda c: c.capability_id)
        if source is not None:
            items = [c for c in items if c.source == source]
        return items

    def register_projection(self, projection: ToolProjectionV1) -> None:
        """注册能力→工具投影。PHYSICAL_EFFECT 拒绝 direct（fail closed）。"""
        cap = self._capabilities.get(projection.capability_id)
        if cap is None:
            raise ValidationError(
                f"projection {projection.tool_name!r}: capability "
                f"{projection.capability_id!r} not registered"
            )
        if (
            cap.effect.class_ is EffectClassV1.PHYSICAL_EFFECT
            and projection.exposure is ProjectionExposure.DIRECT
        ):
            raise ValidationError(
                f"projection {projection.tool_name!r}: PHYSICAL_EFFECT capability "
                f"{cap.capability_id!r} is never directly model-exposed — use "
                "propose_only (ActionAdmission) or internal"
            )
        self._projections[projection.tool_name] = projection

    def projection(self, tool_name: str) -> ToolProjectionV1 | None:
        return self._projections.get(tool_name)

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
        self._generation += 1

    def quarantine_source(self, source: str, reason: str) -> int:
        count = 0
        for d in self._descriptors.values():
            if d.source == source:
                self._quarantine[d.tool_id] = reason
                count += 1
        if count:
            self._generation += 1
        return count

    def lift_quarantine(self, tool_id: str) -> None:
        if tool_id in self._quarantine:
            del self._quarantine[tool_id]
            self._generation += 1

    def lift_source_quarantine(self, source: str) -> int:
        doomed = [tid for tid, d in self._descriptors.items() if d.source == source]
        count = 0
        for tid in doomed:
            if tid in self._quarantine:
                del self._quarantine[tid]
                count += 1
        if count:
            self._generation += 1
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
        # PR-N6C：未声明 cooperative cancel 不得墙钟杀死。
        if descriptor.cooperative_cancel:
            return await asyncio.wait_for(
                executor(arguments), timeout=descriptor.timeout_ms / 1000.0
            )
        return await executor(arguments)

    # -- N5B canonical output -----------------------------------------------------

    async def execute_v2(
        self, call_id: str, tool_id: str, arguments: dict[str, Any]
    ) -> ToolResultEnvelopeV2:
        """canonical 输出路径（PR-N5B）：守卫 → 执行 → output_schema
        验证 → ToolResultEnvelopeV2。任何失败都是诚实 envelope，
        不抛出、不让 executor 文本冒充结构化结果。
        """
        import json as _json

        import jsonschema

        def _blocked(code: str, message: str) -> ToolResultEnvelopeV2:
            return ToolResultEnvelopeV2(
                call_id=call_id, capability_id=tool_id,
                status=ToolResultStatusV1.BLOCKED,
                error=ToolResultErrorV1(code=code, message=message),
            )

        from rosclaw.agentd.tooling.recovery import recovery_for

        def _failed(
            code: str, message: str, *, retryable: bool = False,
            recovery: list[str] | None = None,
        ) -> ToolResultEnvelopeV2:
            # PR-N6C：recovery 从注册表投影（同一码同一文）；显式
            # recovery 参数优先（调用点有更精确上下文时）。
            projected = recovery_for(code)
            return ToolResultEnvelopeV2(
                call_id=call_id, capability_id=tool_id,
                status=ToolResultStatusV1.FAILED,
                error=ToolResultErrorV1(
                    code=code, message=message, retryable=retryable,
                    recovery=recovery if recovery is not None
                    else ([projected] if projected else []),
                ),
            )

        descriptor = self._descriptors.get(self._canonical(tool_id))
        if descriptor is None:
            return _blocked("CAPABILITY_UNKNOWN", f"tool {tool_id!r} not in catalog")
        tool_id = descriptor.tool_id
        # 隔离原因优先于 callable 形态——"为什么不能用"先讲诚实原因。
        reason = self._quarantine.get(tool_id)
        if reason is not None:
            return _blocked(
                "CAPABILITY_QUARANTINED", f"tool {tool_id!r} quarantined: {reason}"
            )
        if (
            descriptor.execution_class is ExecutionClass.PHYSICAL_ACTION
            or not descriptor.model_callable
        ):
            return _blocked(
                "TOOL_NOT_CALLABLE",
                f"tool {tool_id!r} is not directly executable (physical or "
                "non-model-callable) — flows only through the approval chain",
            )
        executor = self._executors.get(tool_id)
        if executor is None:
            return _failed(
                "EXECUTOR_OFFLINE", f"tool {tool_id!r} has no executor "
                "(source offline?)"
            )
        if not descriptor.output_schema:
            return _failed(
                "OUTPUT_SCHEMA_MISSING",
                f"tool {tool_id!r} declares no output_schema — refusing to "
                "guess at structure",
                recovery=["为能力补 output_schema（N5B 硬约束）"],
            )
        try:
            # PR-N6C：只有 cooperative_cancel 的 executor 有 deadline；
            # wait_for 的取消传播进 executor（协程取消即停止确认）。
            if descriptor.cooperative_cancel:
                raw = await asyncio.wait_for(
                    executor(arguments), timeout=descriptor.timeout_ms / 1000.0
                )
            else:
                raw = await executor(arguments)
        except TimeoutError:
            return _failed(
                "EXECUTOR_TIMEOUT",
                f"tool {tool_id!r} exceeded {descriptor.timeout_ms}ms",
                retryable=True,
            )
        except Exception as exc:  # noqa: BLE001 — 诚实 FAILED envelope
            return _failed(
                "EXECUTOR_ERROR", f"{type(exc).__name__}: {exc}"[:400],
            )
        # 归一为 canonical value：只接受 dict / ToolExecutionResult /
        # JSON 对象字符串；裸文本不得冒充结构化结果。
        value: Any
        if isinstance(raw, ToolExecutionResult):
            value = {"text": raw.text, "image_mime_types": [
                img.mime_type for img in raw.images
            ]}
        elif isinstance(raw, dict):
            value = raw
        elif isinstance(raw, str):
            try:
                value = _json.loads(raw)
            except ValueError:
                return _failed(
                    "INVALID_CAPABILITY_OUTPUT",
                    f"tool {tool_id!r} returned a bare string masquerading as "
                    "a structured result",
                    recovery=["executor 只返回 canonical JSON value"],
                )
        else:
            return _failed(
                "INVALID_CAPABILITY_OUTPUT",
                f"tool {tool_id!r} returned {type(raw).__name__}, expected "
                "canonical JSON object",
            )
        if not isinstance(value, dict):
            return _failed(
                "INVALID_CAPABILITY_OUTPUT",
                f"tool {tool_id!r} returned non-object JSON "
                f"({type(value).__name__})",
            )
        if "presentationMeta" in value or "presentation_meta" in value:
            return _failed(
                "INVALID_CAPABILITY_OUTPUT",
                f"tool {tool_id!r} executor submitted presentation meta — "
                "presentation_meta is generated only by trusted projections",
            )
        try:
            jsonschema.validate(value, descriptor.output_schema)
        except jsonschema.ValidationError as exc:
            path = ".".join(str(p) for p in exc.absolute_path)
            return _failed(
                "INVALID_CAPABILITY_OUTPUT",
                f"tool {tool_id!r} output failed output_schema: "
                f"{exc.message[:200]}"
                + (f" (at {path})" if path else ""),
            )
        return ToolResultEnvelopeV2(
            call_id=call_id, capability_id=tool_id,
            status=ToolResultStatusV1.SUCCEEDED, value=value,
        )
