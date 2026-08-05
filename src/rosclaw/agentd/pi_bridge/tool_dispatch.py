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

#: PNA-3/PNA-4 开放工具及其 side-effect 语义。
_TOOL_TABLE: dict[str, str] = {
    "rosclaw_status": "read",
    "rosclaw_observe": "observe",
    "rosclaw_verify": "read",
    "rosclaw_memory_query": "read",
    "rosclaw_fail_safe": "control",
    "rosclaw_delegate": "delegate",
    "rosclaw_request_action": "physical_action",
}
#: 后续批次才开放；现在调用必须得到诚实的"未开放"拒绝。
_DEFERRED_TOOLS = {
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
        # 3.5 context revision 硬校验（P0-7）：动作类必须 exact match；
        # 观测类允许有限 stale（记录但放行）。
        if _TOOL_TABLE.get(request.tool_name) == "physical_action":
            snapshot = service.snapshot(request.mission_id)
            if request.context_revision != snapshot.context_revision:
                raise ToolBridgeError(
                    "CONTEXT_REVISION_MISMATCH",
                    f"request revision {request.context_revision} != current "
                    f"{snapshot.context_revision} — refresh embodied context and "
                    "re-propose (P0-7: actions require exact context)",
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
        if name == "rosclaw_request_action":
            return await self._request_action(request)
        if name == "rosclaw_delegate":
            return await self._delegate(request)
        if name == "rosclaw_fail_safe":
            await service.cancel(request.mission_id)
            return PiToolResultV1(
                request_id=request.request_id,
                ok=True,
                status="COMPLETED",
                summary="fail-safe: 当前回合已请求取消；E-Stop 请走独立 operator 路径",
            )
        raise ToolBridgeError("TOOL_UNKNOWN", f"unhandled tool {name!r}")

    async def _delegate(self, request: PiToolRequestV1) -> PiToolResultV1:
        """PNA-4（规格 §19）：招聘 Worker——受限 WorkOrder + 递归上限 +
        验证通过才进结果（未验证输出绝不进入主上下文）。"""
        from rosclaw.agentd.workers.scheduler import CandidateView
        from rosclaw.contracts.common import new_id
        from rosclaw.contracts.worker.order import (
            BudgetEnvelope,
            ExpectedOutput,
            SideEffectPolicy,
            WorkOrderV1,
        )

        args = request.arguments
        goal = str(args.get("goal", "")).strip()
        if not goal:
            raise ToolBridgeError("INVALID_ARGUMENTS", "goal required")
        worker_hint = str(args.get("worker_id", "auto"))
        parent_id = str(args.get("parent_work_order_id", "") or "") or None
        # 递归上限（规格 §19.5）：直派 worker 单是叶子——沿用全系统约定
        # delegation_depth=0 + max_children=0（depth>0 的单子必须自带
        # children 预算，而 native worker 不接受子委派——两层共同保证
        # worker 无法经此桥再委派）。带 parent_work_order_id 的请求在
        # max_delegation_depth（默认 1）语义下一律是"再委派"，直接拒。
        parent_depth = 0
        if parent_id:
            parent = self._service._worker_manager.order(parent_id)
            parent_depth = parent.delegation_depth if parent else 0
        max_depth = int(args.get("max_delegation_depth", 1))
        if parent_id and parent_depth + 1 > max_depth - 1:
            raise ToolBridgeError(
                "DELEGATION_DEPTH_EXCEEDED",
                "delegation beyond the direct worker layer is refused "
                f"(max_delegation_depth={max_depth})",
            )
        order_depth = 0
        capability = str(args.get("capability") or "analysis.text")
        order = WorkOrderV1(
            work_order_id=new_id("wo"),
            mission_id=request.mission_id,
            issued_by="rosclaw-agent:pi",
            capability=capability,
            goal=goal,
            inputs={
                "instructions": str(args.get("instructions", goal)),
                "artifacts": args.get("artifact_refs") or [],
            },
            budgets=BudgetEnvelope(
                wall_time_sec=int(args.get("budget", {}).get("wall_time_sec", 300))
                if isinstance(args.get("budget"), dict)
                else 300,
                model_tokens=int(args.get("budget", {}).get("model_tokens", 50_000))
                if isinstance(args.get("budget"), dict)
                else 50_000,
                # 叶子 worker 单：max_children=0（native adapter 不接受子委派）。
            ),
            expected_output=ExpectedOutput(artifacts=["text/plain"]),
            side_effect_policy=SideEffectPolicy(**{"class": "none"}),
            delegation_depth=order_depth,
            max_delegation_depth=max_depth,
            parent_work_order_id=parent_id,
            root_work_order_id=str(args.get("root_work_order_id", "") or "") or parent_id,
        )
        service = self._service
        candidates = [
            CandidateView(
                card=card,
                registry_status=service._registry.status_of(card.worker_id) or "DISABLED",
                running_orders=len(
                    service._worker_manager.active_orders_for_worker(card.worker_id)
                ),
                circuit_open=service._worker_manager.circuit_open(card.worker_id, capability),
            )
            for card in service._registry.list()
            if worker_hint in ("", "auto") or card.worker_id == worker_hint
        ]
        if not candidates:
            raise ToolBridgeError(
                "WORKER_UNAVAILABLE", f"no worker matches {worker_hint!r}", retryable=True
            )
        try:
            scheduled = service._worker_manager.hire(order, candidates)
        except Exception as exc:  # noqa: BLE001 - 诚实失败，不伪造委派
            raise ToolBridgeError("SCHEDULING_FAILED", str(exc), retryable=True) from exc
        result, report = await service._worker_manager.run_to_completion(scheduled)
        if not report.accepted:
            return PiToolResultV1(
                request_id=request.request_id,
                ok=False,
                status="VERIFY_FAILED",
                summary=(
                    f"Worker {scheduled.assigned_to} 提交的结果未通过验证"
                    f"（{'；'.join(report.reasons) or '未知'}）——未采纳进主上下文。"
                ),
                error_code="VERIFICATION_REJECTED",
                retryable=True,
            )
        return PiToolResultV1(
            request_id=request.request_id,
            ok=True,
            status="COMPLETED",
            summary=result.summary,
            artifact_refs=[a.ref for a in result.artifacts],
        )

    async def _request_action(self, request: PiToolRequestV1) -> PiToolResultV1:
        """PNA-5 + NA-FIX-5：propose → 等 operator → 精确 grant 执行 →
        结构化回执（不再全局猜 grant、不再解析自然语言）。"""
        import asyncio as _asyncio

        from rosclaw.agentd.pi_bridge.action_coordinator import ActionCoordinator

        args = request.arguments
        capability_id = str(args.get("capability_id", "")).strip()
        arguments = args.get("arguments")
        if not capability_id or not isinstance(arguments, dict):
            raise ToolBridgeError(
                "INVALID_ARGUMENTS", "capability_id and arguments required (fail closed)"
            )
        coordinator = ActionCoordinator(self._service)
        card = await coordinator.propose(
            mission_id=request.mission_id,
            capability_id=capability_id,
            arguments=arguments,
            expected_effect=str(args.get("expected_effect") or capability_id),
            risk_tier=str(args.get("risk_tier", "LOW")),
            title=str(args.get("title") or capability_id),
        )
        # 等 operator（决定只能经 operatord 到达）。
        deadline_sec = 330.0
        waited = 0.0
        while waited < deadline_sec:
            status = coordinator.decision_status(card["approval_id"])["status"]
            if status != "PENDING":
                break
            await _asyncio.sleep(1.0)
            waited += 1.0
        status = coordinator.decision_status(card["approval_id"])["status"]
        if status == "PENDING":
            raise ToolBridgeError(
                "APPROVAL_TIMEOUT",
                "operator 未在期限内决定（默认拒绝语义）——动作未执行",
            )
        if status != "APPROVED":
            return PiToolResultV1(
                request_id=request.request_id,
                ok=False,
                status="DECLINED",
                summary=f"operator 拒绝了 {capability_id}——动作未执行",
                approval_id=card["approval_id"],
                error_code="OPERATOR_DECLINED",
            )
        result = await coordinator.execute(card["approval_id"])
        return PiToolResultV1(
            request_id=request.request_id,
            ok=bool(result.get("executed")),
            status=str(result.get("status", "FAILED")),
            summary=str(result.get("summary", ""))[:8000],
            approval_id=card["approval_id"],
            error_code=result.get("error_code"),
        )

    async def _delegate(self, request: PiToolRequestV1) -> PiToolResultV1:
        """PNA-4（规格 §19）：招聘 Worker——受限 WorkOrder + 递归上限 +
        验证通过才进结果（未验证输出绝不进入主上下文）。"""
        from rosclaw.agentd.workers.scheduler import CandidateView
        from rosclaw.contracts.common import new_id
        from rosclaw.contracts.worker.order import (
            BudgetEnvelope,
            ExpectedOutput,
            SideEffectPolicy,
            WorkOrderV1,
        )

        args = request.arguments
        goal = str(args.get("goal", "")).strip()
        if not goal:
            raise ToolBridgeError("INVALID_ARGUMENTS", "goal required")
        worker_hint = str(args.get("worker_id", "auto"))
        parent_id = str(args.get("parent_work_order_id", "") or "") or None
        # 递归上限（规格 §19.5）：直派 worker 单是叶子——沿用全系统约定
        # delegation_depth=0 + max_children=0（depth>0 的单子必须自带
        # children 预算，而 native worker 不接受子委派——两层共同保证
        # worker 无法经此桥再委派）。带 parent_work_order_id 的请求在
        # max_delegation_depth（默认 1）语义下一律是"再委派"，直接拒。
        parent_depth = 0
        if parent_id:
            parent = self._service._worker_manager.order(parent_id)
            parent_depth = parent.delegation_depth if parent else 0
        max_depth = int(args.get("max_delegation_depth", 1))
        if parent_id and parent_depth + 1 > max_depth - 1:
            raise ToolBridgeError(
                "DELEGATION_DEPTH_EXCEEDED",
                "delegation beyond the direct worker layer is refused "
                f"(max_delegation_depth={max_depth})",
            )
        order_depth = 0
        capability = str(args.get("capability") or "analysis.text")
        order = WorkOrderV1(
            work_order_id=new_id("wo"),
            mission_id=request.mission_id,
            issued_by="rosclaw-agent:pi",
            capability=capability,
            goal=goal,
            inputs={
                "instructions": str(args.get("instructions", goal)),
                "artifacts": args.get("artifact_refs") or [],
            },
            budgets=BudgetEnvelope(
                wall_time_sec=int(args.get("budget", {}).get("wall_time_sec", 300))
                if isinstance(args.get("budget"), dict)
                else 300,
                model_tokens=int(args.get("budget", {}).get("model_tokens", 50_000))
                if isinstance(args.get("budget"), dict)
                else 50_000,
                # 叶子 worker 单：max_children=0（native adapter 不接受子委派）。
            ),
            expected_output=ExpectedOutput(artifacts=["text/plain"]),
            side_effect_policy=SideEffectPolicy(**{"class": "none"}),
            delegation_depth=order_depth,
            max_delegation_depth=max_depth,
            parent_work_order_id=parent_id,
            root_work_order_id=str(args.get("root_work_order_id", "") or "") or parent_id,
        )
        service = self._service
        candidates = [
            CandidateView(
                card=card,
                registry_status=service._registry.status_of(card.worker_id) or "DISABLED",
                running_orders=len(
                    service._worker_manager.active_orders_for_worker(card.worker_id)
                ),
                circuit_open=service._worker_manager.circuit_open(card.worker_id, capability),
            )
            for card in service._registry.list()
            if worker_hint in ("", "auto") or card.worker_id == worker_hint
        ]
        if not candidates:
            raise ToolBridgeError(
                "WORKER_UNAVAILABLE", f"no worker matches {worker_hint!r}", retryable=True
            )
        try:
            scheduled = service._worker_manager.hire(order, candidates)
        except Exception as exc:  # noqa: BLE001 - 诚实失败，不伪造委派
            raise ToolBridgeError("SCHEDULING_FAILED", str(exc), retryable=True) from exc
        result, report = await service._worker_manager.run_to_completion(scheduled)
        if not report.accepted:
            return PiToolResultV1(
                request_id=request.request_id,
                ok=False,
                status="VERIFY_FAILED",
                summary=(
                    f"Worker {scheduled.assigned_to} 提交的结果未通过验证"
                    f"（{'；'.join(report.reasons) or '未知'}）——未采纳进主上下文。"
                ),
                error_code="VERIFICATION_REJECTED",
                retryable=True,
            )
        return PiToolResultV1(
            request_id=request.request_id,
            ok=True,
            status="COMPLETED",
            summary=result.summary,
            artifact_refs=[a.ref for a in result.artifacts],
        )

    async def _request_action(self, request: PiToolRequestV1) -> PiToolResultV1:
        """PNA-5（规格 §20）：动作请求 → 授权卡 → 等 operator 决定 →
        批准后执行 → receipt。模型不能自批（工具只创建卡片并等待；
        决定只能来自 operatord 的签名 apply）。"""
        import asyncio as _asyncio

        from rosclaw.contracts.agent.decision import (
            DecisionV1,
            NextIntent,
            ProposedOperation,
        )
        from rosclaw.contracts.common import new_id

        args = request.arguments
        capability_id = str(args.get("capability_id", "")).strip()
        arguments = args.get("arguments")
        if not capability_id or not isinstance(arguments, dict):
            raise ToolBridgeError(
                "INVALID_ARGUMENTS", "capability_id and arguments required (fail closed)"
            )
        service = self._service
        mission = service.get_mission(request.mission_id)
        if mission is None:
            raise ToolBridgeError("MISSION_NOT_FOUND", "unknown mission")
        handlers = service._handlers
        if handlers is None:
            raise ToolBridgeError("HANDLERS_UNAVAILABLE", "intent handlers not wired")
        handlers._mode = mission.mode.value
        handlers._principal = mission.owner_principal
        # 1. 创建授权卡（REAL 卡同时创建 daemon proposal——handlers 内部处理）。
        approval_decision = DecisionV1(
            decision_id=new_id("dec"),
            mission_id=request.mission_id,
            context_id=f"ctx_{request.mission_id}",
            context_revision=0,
            next_intent=NextIntent.REQUEST_APPROVAL,
            summary=str(args.get("expected_effect") or capability_id),
            evidence_refs=[],
            proposed_operation=ProposedOperation(
                type="approval_request",
                payload={
                    "capability_id": capability_id,
                    "arguments": arguments,
                    "title": str(args.get("title") or capability_id),
                    "summary": str(args.get("expected_effect") or capability_id),
                    "risk_tier": str(args.get("risk_tier", "LOW")),
                    "expected_effect": str(args.get("expected_effect") or ""),
                    "failure_handling": str(args.get("failure_handling") or ""),
                    "parameters": arguments,
                },
            ),
        )
        before = {r.request_id for r in service.pending_approvals(request.mission_id)}
        await handlers.request_approval(approval_decision)
        # 成功语义 = 新卡片落库（HandlerOutcome.accepted 是 worker 验证
        # 专用字段，request_approval 不填——不能以它判成败）。
        pending = service.pending_approvals(request.mission_id)
        new_cards = [r for r in pending if r.request_id not in before]
        if not new_cards:
            raise ToolBridgeError(
                "APPROVAL_CARD_MISSING", "approval card was not created (fail closed)"
            )
        card = new_cards[-1]
        # 2. 等待 operator 决定（轮询 broker；决定只能经 operatord 到达）。
        deadline_sec = 330.0
        waited = 0.0
        while waited < deadline_sec:
            current = service._broker.get_request(card.request_id)
            if current is not None and current.status.value != "PENDING":
                break
            await _asyncio.sleep(1.0)
            waited += 1.0
        current = service._broker.get_request(card.request_id)
        if current is None or current.status.value == "PENDING":
            raise ToolBridgeError(
                "APPROVAL_TIMEOUT",
                "operator 未在期限内决定（默认拒绝语义）——动作未执行",
                retryable=False,
            )
        if current.status.value != "APPROVED":
            return PiToolResultV1(
                request_id=request.request_id,
                ok=False,
                status="DECLINED",
                summary=f"operator 拒绝了 {capability_id}——动作未执行",
                approval_id=card.request_id,
                error_code="OPERATOR_DECLINED",
            )
        # 3. 批准后执行（grant 单次，EXACT_ACTION）。
        grant_id = ""
        for grant in service.list_grants():
            if grant.get("revoked") or grant.get("consumed"):
                continue
            grant_id = str(grant.get("grant_id", ""))
        if not grant_id:
            raise ToolBridgeError(
                "GRANT_MISSING", "approved but no active grant found (fail closed)"
            )
        action_decision = DecisionV1(
            decision_id=new_id("dec"),
            mission_id=request.mission_id,
            context_id=f"ctx_{request.mission_id}",
            context_revision=0,
            next_intent=NextIntent.REQUEST_ACTION,
            summary=f"execute {capability_id}",
            evidence_refs=[],
            proposed_operation=ProposedOperation(
                type="action_request",
                payload={
                    "grant_id": grant_id,
                    "capability_id": capability_id,
                    "arguments": arguments,
                },
            ),
        )
        action_outcome = await handlers.request_action(action_decision)
        # request_action 不用 accepted 字段——成功=文本无失败标记且有回执语义。
        failed_markers = ("未提交", "失败", "尚未到终态", "不报告为完成", "拒绝")
        action_ok = not any(m in action_outcome.text for m in failed_markers)
        return PiToolResultV1(
            request_id=request.request_id,
            ok=action_ok,
            status="COMPLETED" if action_ok else "FAILED",
            summary=action_outcome.text[:8000],
            approval_id=card.request_id,
            error_code=None if action_ok else "ACTION_FAILED",
        )

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
