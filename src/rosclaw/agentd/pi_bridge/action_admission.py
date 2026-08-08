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

    def _validate_session_and_caller(
        self,
        ctx: ActionRequestContext,
        *,
        caller_pid: int | None,
        caller_uid: int | None,
    ) -> None:
        """session/mission/writer/caller 身份校验（propose/execute 共用）。
        不含 context lease——lease 语义在 propose（调用方刚观测）与
        execute（内核重新观测）之间不同（P0-5B：operator 等待 330s，
        不能复用最初 30s lease）。"""
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
        writer = self._bindings.writer_of(ctx.mission_id)
        if writer is None or writer.pi_session_id != ctx.pi_session_id:
            raise ToolBridgeError(
                "WRITER_LEASE_REQUIRED",
                "this session does not hold the writer lease (fail closed)",
            )
        if (
            caller_pid is not None
            and caller_uid is not None
            and (writer.owner_uid != caller_uid or writer.owner_pid != caller_pid)
        ):
            raise ToolBridgeError(
                "CALLER_MISMATCH",
                f"caller pid {caller_pid}/uid {caller_uid} is not the "
                f"writer process (pid {writer.owner_pid}) — fail closed",
            )
        if mission.mode.value != ctx.mode:
            raise ToolBridgeError(
                "MODE_MISMATCH",
                f"mission mode is {mission.mode.value}, not {ctx.mode}",
            )

    def _validate_request_context(
        self,
        ctx: ActionRequestContext,
        *,
        caller_pid: int | None = None,
        caller_uid: int | None = None,
    ) -> None:
        """session → mission → writer lease → caller identity → context
        lease → revision → body → mode 全链硬校验（HOTFIX-1：freshness
        以 agentd 签发的 ValidatedContextLease 为准；P0-5A：SO_PEERCRED
        的 PID/UID 必须与 writer owner 匹配）。"""
        # 1-3. session/mission/writer/caller 身份（共用方法）。
        self._validate_session_and_caller(
            ctx, caller_pid=caller_pid, caller_uid=caller_uid
        )
        binding = self._bindings.binding_for_session(ctx.pi_session_id)
        mission = self._service.get_mission(ctx.mission_id)
        assert binding is not None and mission is not None
        # 4. ValidatedContextLease（HOTFIX-1 核心）：agentd 签发、未过期、
        #    未撤销、session/mission 绑定、revision/body/mode 全匹配。
        #    P0-5A：lease 记录的 caller_uid 必须与当前调用者一致。
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
        # P0-5A：lease 签发给哪个 caller，就只能由哪个 caller 使用
        # （caller_uid=-1 是旧 lease——不允许用于 action）。
        if (
            caller_uid is not None
            and lease.caller_uid >= 0
            and lease.caller_uid != caller_uid
        ):
            raise ToolBridgeError(
                "CALLER_MISMATCH",
                "context lease was issued to a different caller (fail closed)",
            )
        # P0-5B：lease 的 context_hash 必须与当前权威 envelope hash 一致
        # ——内容变化但 revision 未升时 fail closed。
        from rosclaw.agentd.pi_bridge.context import build_embodied_context
        from rosclaw.agentd.pi_bridge.context_lease import context_hash_of

        current_envelope = build_embodied_context(self._service, ctx.mission_id)
        current_hash = context_hash_of(current_envelope)
        if lease.context_hash and lease.context_hash != current_hash:
            raise ToolBridgeError(
                "CONTEXT_HASH_MISMATCH",
                "context content changed without a revision bump — fail closed",
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
        caller_pid: int | None = None,
        caller_uid: int | None = None,
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
        self._validate_request_context(
            request, caller_pid=caller_pid, caller_uid=caller_uid
        )
        service = self._service
        mission = service.get_mission(request.mission_id)
        assert mission is not None  # _validate_request_context 已保证
        # P0-4D：capability/execution class/risk 由 ToolCatalog 权威决定——
        # 模型自报的 risk_tier 只能是提示，不能降低权威 tier。
        # MCP 发现是 lazy 的（此前只在 legacy send_turn 触发）——pi 路径
        # 的 admission 必须自己保证发现完成（幂等）。
        await service._ensure_mcp_discovered()
        descriptor = service._tool_catalog.get(capability_id)
        if descriptor is None:
            raise ToolBridgeError(
                "CAPABILITY_UNKNOWN",
                f"capability {capability_id!r} not in ToolCatalog (fail closed)",
            )
        if descriptor.execution_class.value != "PHYSICAL_ACTION":
            raise ToolBridgeError(
                "NOT_ACTIONABLE",
                f"capability {capability_id} is {descriptor.execution_class.value}, "
                "not PHYSICAL_ACTION — only action-class capabilities may be proposed",
            )
        if service._tool_catalog.quarantine_reason(capability_id) is not None:
            raise ToolBridgeError(
                "CAPABILITY_QUARANTINED", f"capability {capability_id} is quarantined"
            )
        if mission.mode.value not in list(descriptor.supported_modes):
            raise ToolBridgeError(
                "MODE_FORBIDDEN",
                f"capability {capability_id} does not support mode {mission.mode.value}",
            )
        authoritative_risk = descriptor.risk_tier
        _tier_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
        if _tier_rank.get(risk_tier, -1) > _tier_rank.get(authoritative_risk, 0):
            # 模型可以把风险往高报（更保守），但不能往低报。
            authoritative_risk = risk_tier
        # P0-5C：完整 JSON Schema 校验 + 默认值展开——人批准的是展开后
        # 的真实参数（660Hz/0.25s/18%），不是 {}；normalized 对象同一份
        # 用于 hash/展示/grant/executor/receipt。
        from rosclaw.agentd.pi_bridge.action_schema import (
            validate_action_arguments,
        )

        schema = descriptor.input_schema or {}
        try:
            normalized_arguments = validate_action_arguments(schema, arguments)
        except Exception as exc:  # noqa: BLE001 - ValidationError
            raise ToolBridgeError(
                "INVALID_ARGUMENTS",
                f"capability {capability_id} arguments rejected: {exc}",
            ) from exc
        # ExactActionV1：不可变精确动作合约（capability/args/mission/
        # mode/body/context/risk 全一等字段 + intent hash）。
        from rosclaw.contracts.operator.exact_action import (
            build_exact_action,
        )

        exact_action = build_exact_action(
            capability_id=capability_id,
            capability_version=descriptor.version,
            capability_source=descriptor.source,
            normalized_arguments=normalized_arguments,
            authoritative_risk_tier=authoritative_risk,
            side_effect_class=descriptor.side_effect_class.value,
            mission_id=request.mission_id,
            mode=mission.mode.value,
            body_id=mission.body_binding.body_id,
            body_hash=request.body_hash,
            context_revision=request.context_revision,
            context_hash="",
            expected_effect=expected_effect,
            created_at=datetime.now(UTC).isoformat(),
            expires_at=datetime.now(UTC).isoformat(),  # 建卡时由 TTL 覆盖
        )
        # 标题由合约派生——capability 永远在标题里，不允许"危险
        # capability + 无害 title"分离（五审场景 D）。
        derived_title = exact_action.title
        handlers = service._handlers
        if handlers is None:
            raise ToolBridgeError("HANDLERS_UNAVAILABLE", "intent handlers not wired")
        # P0-4C：ActionTxn——idempotency 持久化状态机。同 key 同 hash
        # 返回既有事务（不重复建卡）；同 key 不同 hash 抛 CONFLICT。
        from rosclaw.agentd.pi_bridge.action_txn import (
            ActionTxnStore,
            IdempotencyConflictError,
            request_hash_of,
        )

        txns = ActionTxnStore(service._store.connection)
        # P0-5C：hash 用 normalized arguments（展开默认值后）——
        # 与卡片/executor/receipt 同一对象。
        request_hash = request_hash_of(
            capability_id=capability_id,
            arguments=normalized_arguments,
            mission_id=request.mission_id,
            mode=request.mode,
            context_revision=request.context_revision,
            body_hash=request.body_hash,
        )
        arguments_hash = exact_action.arguments_hash
        try:
            txn = txns.create(
                idempotency_key=request.idempotency_key,
                request_hash=request_hash,
                pi_session_id=request.pi_session_id,
                mission_id=request.mission_id,
                context_lease_id=request.context_lease_id,
                context_revision=request.context_revision,
                body_hash=request.body_hash,
                mode=request.mode,
                capability_id=capability_id,
                arguments_hash=arguments_hash,
                risk_tier=authoritative_risk,
            )
        except IdempotencyConflictError as exc:
            raise ToolBridgeError("IDEMPOTENCY_CONFLICT", str(exc)) from exc
        if txn.approval_id:
            # 幂等重放：事务已建卡——直接返回既有卡（不重复副作用）。
            if txn.state not in ("AWAITING_OPERATOR", "APPROVED", "DISPATCHING",
                                 "RECEIPT_PENDING", "COMPLETED"):
                raise ToolBridgeError(
                    "TXN_STATE_INVALID", f"txn {txn.txn_id} in {txn.state} without approval"
                )
            stored = service._broker.get_request(txn.approval_id)
            if stored is None:
                raise ToolBridgeError("APPROVAL_NOT_FOUND", "txn approval missing")
            from rosclaw.agentd.operator_socket import display_hash_for

            return {
                "approval_id": stored.request_id,
                "display_hash": display_hash_for(stored),
                "mode": mission.mode.value,
                "capability_id": capability_id,
                "arguments": normalized_arguments,
                "risk_tier": authoritative_risk,
                "title": stored.action_display.title,
                "summary": stored.action_display.summary,
                "expires_at": stored.expires_at,
                "txn_id": txn.txn_id,
                "idempotent_replay": True,
            }
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
                    "arguments": normalized_arguments,
                    # P0-5C：title 由合约派生（含 capability），不允许
                    # "危险 capability + 无害 title"分离。
                    "title": derived_title,
                    "summary": expected_effect or capability_id,
                    "risk_tier": authoritative_risk,
                    "expected_effect": expected_effect,
                    "parameters": normalized_arguments,
                    "arguments_hash": arguments_hash,
                    "action_intent_hash": exact_action.action_intent_hash,
                    "exact_action_json": exact_action.model_dump_json(),
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

        display_hash = display_hash_for(created)
        if txn.state == "PROPOSED":
            txn = txns.transition(
                txn.txn_id,
                "AWAITING_OPERATOR",
                approval_id=approval_id,
                display_hash=display_hash,
            )
        return {
            "approval_id": created.request_id,
            "display_hash": display_hash,
            "mode": mission.mode.value,
            "capability_id": capability_id,
            "arguments": normalized_arguments,
            "risk_tier": authoritative_risk,
            "title": created.action_display.title,
            "summary": created.action_display.summary,
            "expires_at": created.expires_at,
            "txn_id": txn.txn_id,
            "action_intent_hash": exact_action.action_intent_hash,
        }

    # -- phase 2a: decision status -------------------------------------------------

    def decision_status(self, approval_id: str) -> dict[str, Any]:
        request = self._service._broker.get_request(approval_id)
        if request is None:
            return {"status": "MISSING", "approval_id": approval_id}
        return {"status": request.status.value, "approval_id": approval_id}

    # -- phase 2b: execute（TOCTOU 复验 + 精确 grant + 结构化回执） -----------------

    async def execute(
        self,
        approval_id: str,
        *,
        request: ActionRequestContext,
        caller_pid: int | None = None,
        caller_uid: int | None = None,
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
            from rosclaw.agentd.pi_bridge.action_txn import ActionTxnStore

            _txns = ActionTxnStore(service._store.connection)
            _txn = _txns.get_by_approval(approval_id)
            if _txn is not None and _txn.state == "AWAITING_OPERATOR":
                _txns.transition(_txn.txn_id, "DECLINED")
            return {
                "status": "DECLINED",
                "approval_id": approval_id,
                "executed": False,
                "error_code": "OPERATOR_DECLINED",
                "txn_id": _txn.txn_id if _txn else "",
            }
        # TOCTOU 复验（P0-NA-10 + P0-5A/5B）：approve 之后、execute 之前
        # lease/revision/body/mode 任一变化都必须拒绝；请求上下文强制
        # 必填——没有"只给 approval_id 就能执行"的绕过路径。
        if stored.context_revision != request.context_revision:
            raise ToolBridgeError(
                "CONTEXT_REVISION_MISMATCH",
                f"card revision {stored.context_revision} != current request "
                f"revision {request.context_revision} — context changed after "
                "approval; re-propose with fresh context",
            )
        # 身份链：session/mission/writer/caller（operator 等待可能远超
        # 30s context lease——不能要求调用方的 lease 仍新鲜；改为内核
        # 此刻重新观测并精确比对原 approval/txn，P0-5B.6）。
        self._validate_session_and_caller(
            request, caller_pid=caller_pid, caller_uid=caller_uid
        )
        # 内核 fresh 重观测：当前 envelope 与卡记录逐项精确比对——模型
        # 不能在批准后修改动作；body/mode/revision 任一变化即拒。
        from rosclaw.agentd.pi_bridge.context import build_embodied_context

        mission_now = service.get_mission(stored.mission_id)
        if mission_now is None:
            raise ToolBridgeError("MISSION_NOT_FOUND", "mission gone after approval")
        fresh = build_embodied_context(service, stored.mission_id)
        if fresh.context_revision != stored.context_revision:
            raise ToolBridgeError(
                "CONTEXT_NOT_FRESH",
                f"context revision moved to {fresh.context_revision} after approval "
                f"(card has {stored.context_revision}) — re-propose required",
            )
        current_body_hash = mission_now.body_binding.effective_body_hash
        if current_body_hash and stored.effective_body_hash and (
            current_body_hash != stored.effective_body_hash
        ):
            raise ToolBridgeError(
                "BODY_HASH_MISMATCH",
                "body changed after approval — re-propose required",
            )
        if mission_now.mode.value != stored.mode:
            raise ToolBridgeError(
                "MODE_MISMATCH",
                f"mode changed to {mission_now.mode.value} after approval — re-propose",
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
        # P0-4C/4D：ActionTxn 驱动状态机 + capability 精确恢复（SIM 卡
        # daemon_capability_id 为空——capability 以 txn/卡参数为准）。
        from rosclaw.agentd.pi_bridge.action_txn import ActionTxnStore

        txns = ActionTxnStore(service._store.connection)
        txn = txns.get_by_approval(approval_id)
        capability_id = (
            (txn.capability_id if txn else "")
            or stored.daemon_capability_id
            or str(stored.action_display.parameters.get("capability_id", ""))
        )
        if not capability_id:
            raise ToolBridgeError(
                "CAPABILITY_MISSING",
                "approval card lost its capability binding (fail closed)",
            )
        if txn is not None:
            # broker 的批准是权威——txn 状态以之为准同步（operatord
            # decide 经 apply_decision 落库，不走 admission）。
            if txn.state == "AWAITING_OPERATOR":
                txn = txns.transition(txn.txn_id, "APPROVED")
            txn = txns.transition(txn.txn_id, "DISPATCHING", grant_id=grant_id)
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
                    "capability_id": capability_id,
                    "arguments": stored.action_display.parameters,
                },
            ),
        )
        with handlers.request_context(
            mode=mission.mode.value if mission else "SIMULATION",
            principal=mission.owner_principal if mission else stored.principal,
        ):
            outcome = await handlers.request_action(decision)
        # 结构化结果（P0-6/P0-NA-13/P0-4F）：success 只由 handler 的
        # terminal_receipt 决定——绝不扫 Mission 事件找"任意 receipt"
        # （旧 receipt 会为新动作背书），也不解析自然语言。
        executed = bool(outcome.terminal_receipt)
        action_id = ""
        receipt_id = ""
        if outcome.evidence_ref and outcome.evidence_ref.startswith("receipt://"):
            receipt_id = outcome.evidence_ref.removeprefix("receipt://")
            action_id = receipt_id
        if txn is not None:
            if executed:
                txn = txns.transition(
                    txn.txn_id, "RECEIPT_PENDING",
                    action_id=action_id, receipt_id=receipt_id,
                )
                txn = txns.transition(txn.txn_id, "COMPLETED")
            else:
                txn = txns.transition(txn.txn_id, "FAILED")
        # ExecutionOutcomeV1（P0-4F）：全 ID 链，不是 bool+文本。
        return {
            "schema_version": "rosclaw.execution_outcome.v1",
            "status": "COMPLETED" if executed else "FAILED",
            "txn_id": txn.txn_id if txn else "",
            "approval_id": approval_id,
            "grant_id": grant_id,
            "action_id": action_id,
            "receipt_id": receipt_id,
            "capability_id": capability_id,
            "executed": executed,
            "terminal": True,
            "terminal_receipt": bool(outcome.terminal_receipt),
            "verified": executed,
            "evidence_ref": outcome.evidence_ref,
            "summary": outcome.text[:4000],
            "error_code": None if executed else "ACTION_FAILED",
            "completed_at": datetime.now(UTC).isoformat(),
        }
