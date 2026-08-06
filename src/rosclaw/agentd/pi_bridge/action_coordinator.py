"""ActionCoordinator（NA-FIX-5，二次审计 P0-5/P0-6）：两阶段动作协议。

```text
pi.action.propose  → approval_id + 不可变卡片（TUI 按此精确展卡）
pi.action.status   → 决策状态轮询（PENDING/APPROVED/DENIED/EXPIRED）
pi.action.execute  → 本卡精确 grant → 执行 → 结构化 ExecutionReceipt
```

红线：
- 绝不取"最后一个全局 active grant"——grant 按 request_id 精确匹配；
- 绝不用自然语言判断成功——结果是结构化字段；
- 多个并发 approval 不串卡（approval_id 全程唯一跟随）；
- 请求级 mode/principal 走 handlers.request_context（P1-5）。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from rosclaw.agentd.pi_bridge.tool_dispatch import ToolBridgeError
from rosclaw.contracts.common import new_id

if TYPE_CHECKING:
    from rosclaw.agentd.service import AgentService


def _decision(intent, mission_id: str, payload: dict, summary: str):
    from rosclaw.contracts.agent.decision import (
        DecisionV1,
        ProposedOperation,
    )

    return DecisionV1(
        decision_id=new_id("dec"),
        mission_id=mission_id,
        context_id=f"ctx_{mission_id}",
        context_revision=0,
        next_intent=intent,
        summary=summary,
        evidence_refs=[],
        proposed_operation=ProposedOperation(type="approval_request", payload=payload),
    )


class ActionCoordinator:
    def __init__(self, service: AgentService) -> None:
        self._service = service

    # -- phase 1: propose --------------------------------------------------------

    async def propose(
        self,
        *,
        mission_id: str,
        capability_id: str,
        arguments: dict[str, Any],
        expected_effect: str,
        risk_tier: str,
        title: str = "",
    ) -> dict[str, Any]:
        """创建授权卡（REAL 卡同时建 daemon proposal），返回精确 approval_id。"""
        from rosclaw.contracts.agent.decision import NextIntent

        service = self._service
        mission = service.get_mission(mission_id)
        if mission is None:
            raise ToolBridgeError("MISSION_NOT_FOUND", "unknown mission")
        handlers = service._handlers
        if handlers is None:
            raise ToolBridgeError("HANDLERS_UNAVAILABLE", "intent handlers not wired")
        decision = _decision(
            NextIntent.REQUEST_APPROVAL,
            mission_id,
            {
                "capability_id": capability_id,
                "arguments": arguments,
                "title": title or capability_id,
                "summary": expected_effect or capability_id,
                "risk_tier": risk_tier,
                "expected_effect": expected_effect,
                "parameters": arguments,
            },
            summary=expected_effect or capability_id,
        )
        with handlers.request_context(
            mode=mission.mode.value, principal=mission.owner_principal
        ):
            await handlers.request_approval(decision)
        pending = service.pending_approvals(mission_id)
        if not pending:
            raise ToolBridgeError(
                "APPROVAL_CARD_MISSING", "approval card was not created (fail closed)"
            )
        card = pending[-1]
        from rosclaw.agentd.operator_socket import display_hash_for

        return {
            "approval_id": card.request_id,
            "display_hash": display_hash_for(card),
            "mode": mission.mode.value,
            "capability_id": capability_id,
            "arguments": arguments,
            "risk_tier": risk_tier,
            "title": card.action_display.title,
            "summary": card.action_display.summary,
            "expires_at": card.expires_at,
        }

    # -- phase 2a: decision status -------------------------------------------------

    def decision_status(self, approval_id: str) -> dict[str, Any]:
        request = self._service._broker.get_request(approval_id)
        if request is None:
            return {"status": "MISSING", "approval_id": approval_id}
        return {"status": request.status.value, "approval_id": approval_id}

    # -- phase 2b: execute（精确 grant，结构化回执） --------------------------------

    async def execute(self, approval_id: str) -> dict[str, Any]:
        from rosclaw.contracts.agent.decision import NextIntent, ProposedOperation

        service = self._service
        request = service._broker.get_request(approval_id)
        if request is None:
            raise ToolBridgeError("APPROVAL_NOT_FOUND", "unknown approval_id")
        if request.status.value == "PENDING":
            raise ToolBridgeError("APPROVAL_PENDING", "operator has not decided yet")
        if request.status.value != "APPROVED":
            return {
                "status": "DECLINED",
                "approval_id": approval_id,
                "executed": False,
                "error_code": "OPERATOR_DECLINED",
            }
        # P0-6：精确 grant——按 request_id 匹配，绝不全局猜。
        row = service._store.connection.execute(
            "SELECT grant_id, consumed, revoked FROM mission_grants WHERE request_id = ?",
            (approval_id,),
        ).fetchone()
        if row is None:
            raise ToolBridgeError(
                "GRANT_MISSING",
                f"no grant bound to approval {approval_id} (fail closed; no global scan)",
            )
        if row["consumed"]:
            raise ToolBridgeError("GRANT_CONSUMED", "grant already consumed (single-use)")
        if row["revoked"]:
            raise ToolBridgeError("GRANT_REVOKED", "grant was revoked")
        grant_id = str(row["grant_id"])
        handlers = service._handlers
        if handlers is None:
            raise ToolBridgeError("HANDLERS_UNAVAILABLE", "intent handlers not wired")
        mission = service.get_mission(request.mission_id)
        action_decision = _decision(
            NextIntent.REQUEST_ACTION,
            request.mission_id,
            {
                "grant_id": grant_id,
                "capability_id": request.daemon_capability_id
                or str(request.action_display.parameters.get("capability_id", "")),
                "arguments": request.action_display.parameters,
            },
            summary=f"execute {approval_id}",
        )
        action_decision = action_decision.model_copy(
            update={
                "proposed_operation": ProposedOperation(
                    type="action_request",
                    payload={
                        "grant_id": grant_id,
                        "capability_id": request.daemon_capability_id or "",
                        "arguments": request.action_display.parameters,
                    },
                )
            }
        )
        with handlers.request_context(
            mode=mission.mode.value if mission else "SIMULATION",
            principal=mission.owner_principal if mission else request.principal,
        ):
            outcome = await handlers.request_action(action_decision)
        # 结构化结果（P0-6）：success 由 handler 的 terminal_receipt/结构化
        # 信号决定，绝不解析自然语言。
        receipt_received = any(
            e.type.value == "receipt.received"
            for e in service.events_replay(request.mission_id, limit=50)
        )
        executed = bool(outcome.terminal_receipt) or receipt_received
        return {
            "status": "COMPLETED" if executed else "FAILED",
            "approval_id": approval_id,
            "grant_id": grant_id,
            "executed": executed,
            "terminal_receipt": bool(outcome.terminal_receipt),
            "summary": outcome.text[:4000],
            "error_code": None if executed else "ACTION_FAILED",
            "completed_at": datetime.now(UTC).isoformat(),
        }
