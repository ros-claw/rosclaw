"""ActionAdmissionService（三审 P0-NA-10/11）：唯一动作准入口。

所有 action 入口——TUI 两阶段 RPC（pi.action.*）与 dispatcher
（pi.tools.execute 的 rosclaw_request_action）——只能经本服务。
不再有绕过验证链的宽松 RPC。

propose 必须携带完整请求上下文（结构化 contract）：

```text
pi_session_id / mission_id / writer lease / context_revision /
body_hash / mode / capability / arguments / idempotency_key
```

execute 时重新读取权威状态，重新验证 lease、revision、body、mode、
grant、TTL——防止 approve 后到 execute 前发生 TOCTOU。

红线：
- stale context、context 缺失、body hash 变化、lease 丢失一律硬拒绝
  （fail closed），不靠提示词约束模型；
- grant 按 request_id 精确匹配，绝不全局扫描；
- 成功只由结构化 terminal_receipt 决定，不解析自然语言；
- ActionTxn 绑定 request_id → session → mission → revision/body_hash
  → approval → grant。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from rosclaw.agentd.pi_bridge.session_binding import SessionBindingStore
from rosclaw.agentd.pi_bridge.tool_dispatch import ToolBridgeError
from rosclaw.contracts.common import new_id

if TYPE_CHECKING:
    from rosclaw.agentd.service import AgentService


@dataclass(frozen=True)
class ActionRequestContext:
    """propose/execute 的完整请求上下文（P0-NA-10 + HOTFIX-1 contract）。

    HOTFIX-1（P0-4A）：context_lease_id 是 agentd 签发的
    ValidatedContextLease——freshness 的权威凭证；无 lease 即
    CONTEXT_NOT_FRESH（不再有"revision 碰巧相同就放行"的 stale 捷径）。
    """

    pi_session_id: str
    mission_id: str
    context_revision: int
    body_hash: str
    mode: str
    idempotency_key: str
    context_lease_id: str


class ActionAdmissionService:
    """唯一 action admission path（dispatcher 与 TUI RPC 共用）。"""

    def __init__(self, service: AgentService) -> None:
        self._service = service
        self._bindings = SessionBindingStore(service._store.connection)

    # -- 验证链 -----------------------------------------------------------------

    def _validate_request_context(self, ctx: ActionRequestContext) -> None:
        """session → mission → writer lease → context lease → revision →
        body → mode 全链硬校验（HOTFIX-1：freshness 以 agentd 签发的
        ValidatedContextLease 为准，不信 TUI 自报的 revision）。"""
        # 1. 全字段非空（HOTFIX-1：空 body_hash/idempotency 不再放行）。
        if not ctx.pi_session_id or not ctx.mission_id:
            raise ToolBridgeError(
                "REQUEST_CONTEXT_REQUIRED", "session/mission required (fail closed)"
            )
        if not ctx.body_hash:
            raise ToolBridgeError(
                "BODY_HASH_REQUIRED",
                "body_hash is required for actions (fail closed; empty ≠ match)",
            )
        if not ctx.idempotency_key:
            raise ToolBridgeError(
                "IDEMPOTENCY_KEY_REQUIRED", "idempotency_key is required (fail closed)"
            )
        if not ctx.mode:
            raise ToolBridgeError("REQUEST_CONTEXT_REQUIRED", "mode required")
        # 2. session binding + mission。
        binding = self._bindings.binding_for_session(ctx.pi_session_id)
        if binding is None:
            raise ToolBridgeError(
                "SESSION_UNBOUND", "pi session has no active binding (fail closed)"
            )
        if binding.mission_id != ctx.mission_id:
            raise ToolBridgeError(
                "MISSION_MISMATCH",
                f"bound mission is {binding.mission_id}, not {ctx.mission_id}",
            )
        mission = self._service.get_mission(ctx.mission_id)
        if mission is None:
            raise ToolBridgeError("MISSION_NOT_FOUND", "unknown mission")
        # 3. writer lease。
        writer = self._bindings.writer_of(ctx.mission_id)
        if writer is None or writer.pi_session_id != ctx.pi_session_id:
            raise ToolBridgeError(
                "WRITER_LEASE_REQUIRED",
                "this session does not hold the writer lease (fail closed)",
            )
        # 4. ValidatedContextLease（HOTFIX-1 核心）：agentd 签发、未过期、
        #    未撤销、session/mission 绑定、revision/body/mode 全匹配。
        from rosclaw.agentd.pi_bridge.context_lease import ContextLeaseStore

        if not ctx.context_lease_id:
            raise ToolBridgeError(
                "CONTEXT_LEASE_REQUIRED",
                "no validated context lease — fetch fresh embodied context "
                "before acting (fail closed)",
            )
        leases = ContextLeaseStore(self._service._store.connection)
        lease = leases.get(ctx.context_lease_id)
        if lease is None or not leases.is_valid(lease):
            raise ToolBridgeError(
                "CONTEXT_NOT_FRESH",
                "context lease expired, revoked, or unknown — refresh embodied "
                "context (fail closed)",
            )
        if (
            lease.pi_session_id != ctx.pi_session_id
            or lease.mission_id != ctx.mission_id
        ):
            raise ToolBridgeError(
                "CONTEXT_LEASE_MISMATCH",
                "context lease belongs to another session/mission (fail closed)",
            )
        snapshot = self._service.snapshot(ctx.mission_id)
        if lease.context_revision != snapshot.context_revision:
            raise ToolBridgeError(
                "CONTEXT_NOT_FRESH",
                f"lease revision {lease.context_revision} != current "
                f"{snapshot.context_revision} — refresh embodied context",
            )
        if ctx.context_revision != lease.context_revision:
            raise ToolBridgeError(
                "CONTEXT_REVISION_MISMATCH",
                "request revision != lease revision — use the context the "
                "lease was issued for",
            )
        current_body_hash = mission.body_binding.effective_body_hash
        if current_body_hash and lease.body_hash != current_body_hash:
            raise ToolBridgeError(
                "BODY_HASH_MISMATCH",
                "body hash changed since context was issued — re-observe and re-propose",
            )
        if ctx.body_hash != lease.body_hash:
            raise ToolBridgeError(
                "BODY_HASH_MISMATCH", "request body hash != lease body hash"
            )
        if mission.mode.value != ctx.mode or lease.mode != ctx.mode:
            raise ToolBridgeError(
                "MODE_MISMATCH",
                f"mission mode is {mission.mode.value}, not {ctx.mode}",
            )

    # -- phase 1: propose --------------------------------------------------------

    async def propose(
        self,
        *,
        request: ActionRequestContext,
        capability_id: str,
        arguments: dict[str, Any],
        expected_effect: str,
        risk_tier: str,
        title: str = "",
    ) -> dict[str, Any]:
        """验证链通过后创建授权卡，返回精确 approval_id（不扫 pending）。"""
        from rosclaw.contracts.agent.decision import (
            DecisionV1,
            NextIntent,
            ProposedOperation,
        )

        if not capability_id.strip() or not isinstance(arguments, dict):
            raise ToolBridgeError(
                "INVALID_ARGUMENTS", "capability_id and arguments required (fail closed)"
            )
        self._validate_request_context(request)
        service = self._service
        mission = service.get_mission(request.mission_id)
        assert mission is not None  # _validate_request_context 已保证
        handlers = service._handlers
        if handlers is None:
            raise ToolBridgeError("HANDLERS_UNAVAILABLE", "intent handlers not wired")
        # 调用方提供精确 request_id——不扫 pending 列表尾、不做 before-after 差集。
        approval_id = new_id("appr")
        decision = DecisionV1(
            decision_id=new_id("dec"),
            mission_id=request.mission_id,
            context_id=f"ctx_{request.mission_id}",
            context_revision=request.context_revision,
            next_intent=NextIntent.REQUEST_APPROVAL,
            summary=expected_effect or capability_id,
            evidence_refs=[],
            proposed_operation=ProposedOperation(
                type="approval_request",
                payload={
                    "request_id": approval_id,
                    "capability_id": capability_id,
                    "arguments": arguments,
                    "title": title or capability_id,
                    "summary": expected_effect or capability_id,
                    "risk_tier": risk_tier,
                    "expected_effect": expected_effect,
                    "parameters": arguments,
                },
            ),
        )
        with handlers.request_context(
            mode=mission.mode.value, principal=mission.owner_principal
        ):
            await handlers.request_approval(decision)
        # 精确验证这张卡被创建——按 ID 查，不扫列表。
        created = service._broker.get_request(approval_id)
        if created is None:
            raise ToolBridgeError(
                "APPROVAL_CARD_MISSING", "approval card was not created (fail closed)"
            )
        from rosclaw.agentd.operator_socket import display_hash_for

        return {
            "approval_id": created.request_id,
            "display_hash": display_hash_for(created),
            "mode": mission.mode.value,
            "capability_id": capability_id,
            "arguments": arguments,
            "risk_tier": risk_tier,
            "title": created.action_display.title,
            "summary": created.action_display.summary,
            "expires_at": created.expires_at,
        }

    # -- phase 2a: decision status -------------------------------------------------

    def decision_status(self, approval_id: str) -> dict[str, Any]:
        request = self._service._broker.get_request(approval_id)
        if request is None:
            return {"status": "MISSING", "approval_id": approval_id}
        return {"status": request.status.value, "approval_id": approval_id}

    # -- phase 2b: execute（TOCTOU 复验 + 精确 grant + 结构化回执） -----------------

    async def execute(
        self, approval_id: str, *, request: ActionRequestContext
    ) -> dict[str, Any]:
        from rosclaw.contracts.agent.decision import (
            DecisionV1,
            NextIntent,
            ProposedOperation,
        )

        service = self._service
        stored = service._broker.get_request(approval_id)
        if stored is None:
            raise ToolBridgeError("APPROVAL_NOT_FOUND", "unknown approval_id")
        if stored.status.value == "PENDING":
            raise ToolBridgeError("APPROVAL_PENDING", "operator has not decided yet")
        if stored.status.value != "APPROVED":
            return {
                "status": "DECLINED",
                "approval_id": approval_id,
                "executed": False,
                "error_code": "OPERATOR_DECLINED",
            }
        # TOCTOU 复验（P0-NA-10 + HOTFIX-1/P0-4B）：approve 之后、execute
        # 之前 lease/revision/body/mode 任一变化都必须拒绝；请求上下文
        # 强制必填——没有"只给 approval_id 就能执行"的绕过路径。
        if stored.context_revision != request.context_revision:
            raise ToolBridgeError(
                "CONTEXT_REVISION_MISMATCH",
                f"card revision {stored.context_revision} != current request "
                f"revision {request.context_revision} — context changed after "
                "approval; re-propose with fresh context",
            )
        self._validate_request_context(request)
        # 卡自身的 revision 与当前权威 snapshot 也必须一致（即使调用方
        # 没带 request context，也不能用过期卡执行）。
        snapshot = service.snapshot(stored.mission_id)
        if stored.context_revision != snapshot.context_revision:
            raise ToolBridgeError(
                "CONTEXT_REVISION_MISMATCH",
                f"card revision {stored.context_revision} is stale (current "
                f"{snapshot.context_revision}) — re-propose with fresh context",
            )
        # P0-6：精确 grant——按 request_id 匹配，绝不全局猜。
        row = service._store.connection.execute(
            "SELECT grant_id, consumed, revoked, expires_at FROM mission_grants "
            "WHERE request_id = ?",
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
        expires_at = str(row["expires_at"] or "")
        if expires_at and expires_at < datetime.now(UTC).isoformat():
            raise ToolBridgeError("GRANT_EXPIRED", "grant expired before execute (TOCTOU)")
        grant_id = str(row["grant_id"])
        handlers = service._handlers
        if handlers is None:
            raise ToolBridgeError("HANDLERS_UNAVAILABLE", "intent handlers not wired")
        mission = service.get_mission(stored.mission_id)
        decision = DecisionV1(
            decision_id=new_id("dec"),
            mission_id=stored.mission_id,
            context_id=f"ctx_{stored.mission_id}",
            context_revision=stored.context_revision,
            next_intent=NextIntent.REQUEST_ACTION,
            summary=f"execute {approval_id}",
            evidence_refs=[],
            proposed_operation=ProposedOperation(
                type="action_request",
                payload={
                    "grant_id": grant_id,
                    "capability_id": stored.daemon_capability_id or "",
                    "arguments": stored.action_display.parameters,
                },
            ),
        )
        with handlers.request_context(
            mode=mission.mode.value if mission else "SIMULATION",
            principal=mission.owner_principal if mission else stored.principal,
        ):
            outcome = await handlers.request_action(decision)
        # 结构化结果（P0-6/P0-NA-13）：success 只由 handler 的
        # terminal_receipt 决定——绝不扫 Mission 事件找"任意 receipt"
        # （旧 receipt 会为新动作背书），也不解析自然语言。
        executed = bool(outcome.terminal_receipt)
        return {
            "status": "COMPLETED" if executed else "FAILED",
            "approval_id": approval_id,
            "grant_id": grant_id,
            "executed": executed,
            "terminal_receipt": bool(outcome.terminal_receipt),
            "evidence_ref": outcome.evidence_ref,
            "summary": outcome.text[:4000],
            "error_code": None if executed else "ACTION_FAILED",
            "completed_at": datetime.now(UTC).isoformat(),
        }
