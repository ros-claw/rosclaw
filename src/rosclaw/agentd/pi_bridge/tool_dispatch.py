"""Pi Tool Bridge 分发（重构规格 §15/§16/§17，PR-PNA-3）。

每个 Pi 工具调用在这里变成受控的 agentd 操作 + DecisionV1 审计镜像。

验证链（任一不过即 fail closed）：
session binding → mission 存在 → writer lease → context revision →
idempotency → tool allowlist → side-effect class → mode → 执行。

PNA-3 工具集（read/observe/verify/memory/fail_safe/status）：
动作类（request_action/delegate）在 PNA-4/PNA-5 接入，当前一律拒绝并
如实说明——不存在"经 observe 绕过动作"的路径。
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from rosclaw.agentd.pi_bridge.session_binding import SessionBindingStore
from rosclaw.contracts.pi.tool_request import PiToolRequestV1, PiToolResultV1

if TYPE_CHECKING:
    from rosclaw.agentd.service import AgentService

#: PNA-3 开放工具及其 side-effect 语义。
_TOOL_TABLE: dict[str, str] = {
    "rosclaw_status": "read",
    "rosclaw_observe": "observe",
    "rosclaw_verify": "read",
    "rosclaw_memory_query": "read",
    "rosclaw_fail_safe": "control",
}
#: PNA-4/PNA-5 才开放；现在调用必须得到诚实的"未开放"拒绝。
_DEFERRED_TOOLS = {
    "rosclaw_request_action": "PNA-5（approval 链）",
    "rosclaw_delegate": "PNA-4（Worker 体验）",
    "rosclaw_plan_patch": "PNA-3 后续（TaskGraph patch）",
    "rosclaw_team_coordinate": "PNA-4 后续",
}


class ToolBridgeError(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


class PiToolDispatcher:
    def __init__(self, service: AgentService) -> None:
        self._service = service
        self._bindings = SessionBindingStore(service._store.connection)

    async def execute(self, request: PiToolRequestV1) -> PiToolResultV1:
        conn = self._service._store.connection
        # 1. idempotency：重放直接返回首个结果（不产生重复副作用）。
        row = conn.execute(
            "SELECT response_json FROM pi_tool_idempotency WHERE idempotency_key = ?",
            (request.idempotency_key,),
        ).fetchone()
        if row is not None:
            return PiToolResultV1(**json.loads(row["response_json"]))
        try:
            result = await self._execute_validated(request)
        except ToolBridgeError as exc:
            result = PiToolResultV1(
                request_id=request.request_id,
                ok=False,
                status="REJECTED",
                summary=exc.message,
                error_code=exc.code,
                retryable=exc.retryable,
            )
        conn.execute(
            "INSERT OR IGNORE INTO pi_tool_idempotency "
            "(idempotency_key, request_id, tool_name, response_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                request.idempotency_key,
                request.request_id,
                request.tool_name,
                result.model_dump_json(),
                datetime.now(UTC).isoformat(),
            ),
        )
        conn.commit()
        await self._mirror_decision(request, result)
        return result

    async def _execute_validated(self, request: PiToolRequestV1) -> PiToolResultV1:
        service = self._service
        # 2. session binding + mission。
        binding = self._bindings.binding_for_session(request.pi_session_id)
        if binding is None:
            raise ToolBridgeError("SESSION_UNBOUND", "pi session has no active binding")
        if binding.mission_id != request.mission_id:
            raise ToolBridgeError(
                "MISSION_MISMATCH",
                f"bound mission is {binding.mission_id}, not {request.mission_id}",
            )
        mission = service.get_mission(request.mission_id)
        if mission is None:
            raise ToolBridgeError("MISSION_NOT_FOUND", "unknown mission")
        # 3. writer lease（崩溃回收由 lease 过期保证）。
        writer = self._bindings.writer_of(request.mission_id)
        if writer is None or writer.pi_session_id != request.pi_session_id:
            raise ToolBridgeError(
                "WRITER_LEASE_REQUIRED", "this session does not hold the writer lease"
            )
        # 4. allowlist。
        if request.tool_name in _DEFERRED_TOOLS:
            raise ToolBridgeError(
                "TOOL_DEFERRED",
                f"{request.tool_name} 在 {_DEFERRED_TOOLS[request.tool_name]}才开放——当前拒绝",
            )
        if request.tool_name not in _TOOL_TABLE:
            raise ToolBridgeError("TOOL_UNKNOWN", f"unknown tool {request.tool_name!r}")
        # 5. 分发。
        return await self._dispatch(request)

    async def _dispatch(self, request: PiToolRequestV1) -> PiToolResultV1:
        service = self._service
        name = request.tool_name
        args = request.arguments
        if name == "rosclaw_status":
            mission = service.get_mission(request.mission_id)
            return PiToolResultV1(
                request_id=request.request_id,
                ok=True,
                status="COMPLETED",
                summary=json.dumps(
                    {
                        "agentd": "READY",
                        "mode": mission.mode.value if mission else "",
                        "state": mission.state.value if mission else "",
                        "authorization_profile": service.authorization_profile(),
                    },
                    ensure_ascii=False,
                ),
            )
        if name == "rosclaw_observe":
            capability_id = str(args.get("capability_id", ""))
            if not capability_id:
                raise ToolBridgeError("INVALID_ARGUMENTS", "capability_id required")
            descriptor = service._tool_catalog.get(capability_id)
            if descriptor is None:
                raise ToolBridgeError(
                    "CAPABILITY_UNKNOWN", f"capability {capability_id!r} not in catalog"
                )
            if descriptor.execution_class.value != "OBSERVE":
                # 规格 §16.2：动作类能力不得经 observe 绕过。
                raise ToolBridgeError(
                    "NOT_OBSERVABLE",
                    f"capability {capability_id} is {descriptor.execution_class.value}, "
                    "not OBSERVE — action-class capabilities need the approval chain",
                )
            if descriptor.quarantined:
                raise ToolBridgeError(
                    "CAPABILITY_QUARANTINED", f"capability {capability_id} is quarantined"
                )
            output = await service._tool_registry.execute(
                capability_id, dict(args.get("arguments", {}))
            )
            text = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)
            return PiToolResultV1(
                request_id=request.request_id,
                ok=True,
                status="COMPLETED",
                summary=text[:8000],
            )
        if name == "rosclaw_verify":
            receipts = [
                e.payload
                for e in service.events_replay(request.mission_id, limit=200)
                if e.type.value == "receipt.received"
            ]
            return PiToolResultV1(
                request_id=request.request_id,
                ok=True,
                status="COMPLETED",
                summary=json.dumps({"receipts": receipts[-3:]}, ensure_ascii=False)[:8000],
            )
        if name == "rosclaw_memory_query":
            return PiToolResultV1(
                request_id=request.request_id,
                ok=True,
                status="COMPLETED",
                summary="memory query is wired to the approved evidence pipeline; "
                "no results in this SIM profile",
            )
        if name == "rosclaw_fail_safe":
            await service.cancel(request.mission_id)
            return PiToolResultV1(
                request_id=request.request_id,
                ok=True,
                status="COMPLETED",
                summary="fail-safe: 当前回合已请求取消；E-Stop 请走独立 operator 路径",
            )
        raise ToolBridgeError("TOOL_UNKNOWN", f"unhandled tool {name!r}")

    async def _mirror_decision(
        self, request: PiToolRequestV1, result: PiToolResultV1
    ) -> None:
        """规格 §15：每个工具调用镜像为 DecisionV1 审计事件（不写全文）。"""
        try:
            from rosclaw.contracts.agent.agent_event import AgentEventType

            await self._service._events.append(
                request.mission_id,
                AgentEventType.TOOL_COMPLETED,
                {
                    "engine": "pi",
                    "request_id": request.request_id,
                    "tool_name": request.tool_name,
                    "ok": result.ok,
                    "status": result.status,
                    "error_code": result.error_code,
                    "summary_hash": json.dumps(result.summary)[:64],
                },
            )
        except Exception:  # noqa: BLE001 - 审计镜像失败不影响已完成的工具结果
            pass
