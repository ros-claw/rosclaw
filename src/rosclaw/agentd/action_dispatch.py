"""Action Dispatch（PR-H9 提取）：已批准动作的执行派发。

旧 ServiceIntentHandlers.request_action 的忠实提取——Worker/AgentLoop
无关的纯执行路径：broker grant 验证（单次消费）→ SIM executor /
daemon consent plane / daemon action channel → receipt 事件落账。

语义红线（与旧实现一致）：
- executor 缺失是"派发前拒绝"——不消费 operator 单次授权；
- FIXTURE/SYNTHETIC 回执永远不算完成（fail closed）；
- 非 REAL 证据不可用于证明真实物理执行。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from rosclaw.contracts.agent.decision import DecisionV1
from rosclaw.contracts.common import new_id
from rosclaw.contracts.operator.approval import ActionDisplayV1, ApprovalRequestV2
from rosclaw.operator import GrantDeniedError

if TYPE_CHECKING:
    from rosclaw.agentd.service import AgentService


@dataclass
class HandlerOutcome:
    """Structured result of an action dispatch（与旧 PR-04 合约一致）。"""

    text: str
    accepted: bool | None = None
    terminal_receipt: bool = False
    evidence_ref: str | None = None
    error_code: str | None = None


def _verification_summary(verification: dict) -> str:
    """Render only evidence metrics actually present in a terminal receipt."""
    fields = [
        ("success", "success"),
        ("observer", "observer"),
        ("target_gain_db", "target_gain_db"),
        ("target_prominence_db", "target_prominence_db"),
        ("rms_gain_db", "rms_gain_db"),
        ("observed_rms_dbfs", "observed_rms_dbfs"),
        ("content_recognition_performed", "content_recognition_performed"),
        ("human_hearing_confirmed", "human_hearing_confirmed"),
    ]
    rendered = [f"{label}={verification[key]}" for key, label in fields if key in verification]
    return ", ".join(rendered) if rendered else "no structured verification metrics"


async def _emit(service: AgentService, type_str: str, mission_id: str, payload: dict) -> None:
    import contextlib

    from rosclaw.contracts.agent.agent_event import AgentEventType

    with contextlib.suppress(Exception):
        await service._events.append(
            mission_id, AgentEventType(type_str), payload
        )


async def request_action(
    service: AgentService,
    decision: DecisionV1,
    *,
    mode: str,
    principal: str,
) -> HandlerOutcome:
    """执行已批准的 REQUEST_ACTION（broker verify → 通道派发 → receipt）。"""
    broker = service._broker
    payload = (
        decision.proposed_operation.payload if decision.proposed_operation else None
    ) or {}
    grant_id = payload.get("grant_id")
    if not grant_id:
        return HandlerOutcome(
            text=(
                "请求动作缺少 grant_id。EXACT_ACTION 流程：先 REQUEST_APPROVAL 获得授权，"
                "再在动作请求中引用 grant_id。已拒绝（fail closed）。"
            )
        )
    channel = service._action_channel
    capability = str(payload.get("capability_id", "sim.hold_position"))
    arguments = payload.get("arguments") or {}
    # 六审 §6.2.4：SIM 路径先解析执行通道再验 grant——executor 缺失
    # 是"派发前拒绝"（系统/配置错误），不得消费 operator 的单次授权；
    # 只有真实派发后（含 domain 失败）才消费。
    sim_channel = None
    if channel is None and mode == "SIMULATION":
        executor_identity = str(payload.get("executor_identity") or "")
        executors = service._sim_executors
        if executor_identity:
            sim_channel = executors.get(executor_identity)
            if sim_channel is None:
                return HandlerOutcome(
                    text=(
                        f"SIM 执行通道不可用（executor={executor_identity}，"
                        f"capability={capability}）——动作未派发、grant 未消费"
                        "（fail closed）。"
                    ),
                    error_code="EXECUTOR_FOR_BODY_UNAVAILABLE",
                )
        else:
            sim_channel = next(iter(executors.values())) if len(executors) == 1 else None
    try:
        # 动作意图由 broker 从已批准的卡片重算，不采信模型自报（§11.2）。
        intent = broker.action_intent_for_grant(str(grant_id))
        grant = broker.verify(
            str(grant_id),
            principal=principal,
            body_hash=service._current_body_hash(),
            mode=mode,
            risk_tier=str(payload.get("risk_tier", "LOW")),
            action_intent=intent,
        )
    except GrantDeniedError as exc:
        return HandlerOutcome(text=f"授权校验失败（{exc.reason_code}）：{exc}。动作未提交。")
    # EXACT_ACTION 单次性：verify 成功即消费（批次 B 事件）。
    await _emit(
        service, "grant.consumed", decision.mission_id,
        {"grant_id": str(grant_id), "public_hash": grant.public_hash},
    )
    consent = service._consent_channel
    # ADR-0007 完整路径只在"该授权确实关联了 daemon proposal"时生效
    # （REAL 模式）；否则回落到 SIM action_channel 或诚实无通道。
    proposal_id = None
    action_id = None
    if consent is not None:
        grant_row = broker._conn.execute(
            "SELECT request_id FROM mission_grants WHERE grant_id = ?",
            (grant.grant_id,),
        ).fetchone()
        approval_req = broker.get_request(grant_row["request_id"]) if grant_row else None
        if approval_req is not None:
            extras = approval_req.model_dump(mode="json")
            proposal_id = getattr(approval_req, "daemon_proposal_id", None) or extras.get(
                "daemon_proposal_id"
            )
            action_id = getattr(approval_req, "daemon_action_id", None) or extras.get(
                "daemon_action_id"
            )
    if consent is not None and proposal_id:
        from rosclaw.agentd.consent_channel import ConsentChannelError

        try:
            proposal = await consent.proposal(proposal_id)
        except ConsentChannelError as exc:
            return HandlerOutcome(text=f"读取 daemon proposal 失败（fail closed）：{exc}")
        state = proposal.get("state")
        if state != "TERMINAL":
            return HandlerOutcome(
                text=(
                    f"daemon proposal 尚未到终态（state={state}）。"
                    "若批准时 operator 已裁决，receipt 应在数秒内可用；否则该 proposal "
                    "可能已过期/失效。不报告为完成。"
                )
            )
        if not action_id:
            return HandlerOutcome(
                text="proposal 已终态但无 action_id，无法回读 receipt（fail closed）。"
            )
        try:
            receipt = await consent.action_receipt(action_id)
        except ConsentChannelError as exc:
            return HandlerOutcome(text=f"回执读取失败（fail closed）：{exc}")
        inner = receipt.get("receipt") if isinstance(receipt.get("receipt"), dict) else receipt
        trust = inner.get("trust_level", "UNKNOWN")
        final_state = inner.get("final_state", "UNKNOWN")
        evidence_domain = inner.get("evidence_domain", "UNKNOWN")
        evidence_level = inner.get("evidence_level", "UNKNOWN")
        usable_for_real = inner.get("usable_for_real_execution") is True
        verification = (
            inner.get("verification_result")
            if isinstance(inner.get("verification_result"), dict)
            else {}
        )
        provenance = (inner.get("authorization_decision") or {}).get("provenance") or {}
        if trust == "SYNTHETIC":
            return HandlerOutcome(
                text="回执为 FIXTURE/SYNTHETIC 证据——拒绝当作完成（fail closed）。"
            )
        if inner.get("action_id") not in (None, action_id):
            return HandlerOutcome(
                text="回执 action 与请求不匹配——不报告为我们的动作（fail closed）。"
            )
        verified = final_state == "COMPLETED" and trust not in ("UNKNOWN", "")
        await _emit(
            service, "receipt.received", decision.mission_id,
            {
                "action_id": action_id,
                "final_state": final_state,
                "trust_level": trust,
                "verified": verified,
            },
        )
        return HandlerOutcome(
            text=(
                f"动作已由 rosclawd consent plane 完成：state={final_state}, "
                f"trust_level={trust}, evidence_domain={evidence_domain}, "
                f"evidence_level={evidence_level}, "
                f"usable_for_real_execution={str(usable_for_real).lower()}。"
                f"验证：{_verification_summary(verification)}。"
                f"授权来源：proposal {proposal_id[:20]}…, "
                f"operator={provenance.get('operator_principal')}, "
                f"channel={provenance.get('decision_channel')}。grant 已消费。"
            ),
            terminal_receipt=verified,
            evidence_ref=f"receipt://{action_id}",
        )
    if channel is None:
        # PR-12：无 daemon 时，SIMULATION 可经 SimActionChannel 在 SIM
        # 身体上执行（SIMULATED receipt，永不证明 REAL）。
        if sim_channel is not None:
            from rosclaw.agentd.sim_executor import SimActionError

            try:
                outcome = await sim_channel.execute(
                    capability_id=capability,
                    arguments=arguments,
                    grant_id=grant.grant_id,
                    mode=mode,
                )
            except SimActionError as exc:
                return HandlerOutcome(text=f"SIM 执行失败（fail closed）：{exc}")
            receipt = outcome.receipt
            await _emit(
                service, "receipt.received", decision.mission_id,
                {
                    "receipt_id": outcome.receipt_id,
                    "action_id": outcome.action_id,
                    "final_state": outcome.final_state,
                    "trust_level": "SIMULATED",
                    "verified": True,
                    "evidence_domain": "simulation",
                    "usable_for_real_execution": False,
                },
            )
            effect_note = (
                "物理效果已由独立观测证明。"
                if receipt.get("physical_effect_proven")
                else "没有声学/物理观测——只能确认驱动执行，不能独立证明物理效果（§18.4）。"
            )
            return HandlerOutcome(
                text=(
                    f"动作已由 SIM 执行器完成（{receipt['executor']}）："
                    f"capability={capability}, final_state=COMPLETED, "
                    "evidence_domain=simulation, usable_for_real_execution=false。"
                    f"{effect_note}grant 已消费。"
                ),
                terminal_receipt=True,
                evidence_ref=f"receipt://{outcome.receipt_id}",
            )
        return HandlerOutcome(
            text=(
                f"授权已验证（grant {grant.grant_id[:20]}…，EXACT_ACTION 已消费）。"
                "注意：当前 agentd 未连接 rosclawd 执行通道，SIMULATION 下没有物理动作被派发；"
                "这不是执行回执。"
            )
        )
    from rosclaw.agentd.action_channel import ActionChannelError

    try:
        if mode == "REAL":
            proposal = await channel.request_real_proposal(
                capability_id=capability,
                arguments=arguments,
                grant_id=grant.grant_id,
                grant_public_hash=grant.public_hash,
                principal_id=grant.principal,
                risk_tier=str(payload.get("risk_tier", "LOW")),
                display={
                    "title": str(payload.get("title") or decision.summary or "真实动作请求"),
                    "summary": str(payload.get("summary") or decision.summary or ""),
                    "risk_tier": str(payload.get("risk_tier", "LOW")),
                    "parameters": {
                        "capability_id": capability,
                        "arguments": arguments,
                    },
                    "mission_grant_public_hash": grant.public_hash,
                },
            )
            return HandlerOutcome(
                text=(
                    "REAL 动作已提交到 rosclawd Operator Broker，尚未执行："
                    f"request_id={proposal.request_id[:24]}…, "
                    f"action_id={proposal.action_id[:24]}…, state={proposal.state}。"
                    "需要由受信 Operator 进程独立审阅并确认；Agent 未获得 permit，"
                    "也没有自行授权。"
                )
            )
        outcome = await channel.request_nonreal_action(
            capability_id=capability,
            arguments=arguments,
            grant_id=grant.grant_id,
            execution_mode=mode,
        )
    except ActionChannelError as exc:
        return HandlerOutcome(text=f"动作派发/回执校验失败（fail closed）：{exc}")
    await _emit(
        service, "receipt.received", decision.mission_id,
        {
            "action_id": outcome.action_id,
            "final_state": outcome.state,
            "trust_level": outcome.trust_level,
            "verified": outcome.verified,
        },
    )
    if not outcome.verified:
        return HandlerOutcome(
            text=(
                f"动作已提交但未达验证标准（state={outcome.state}, "
                f"trust={outcome.trust_level}）。提交不等于完成——不报告为成功。"
            )
        )
    return HandlerOutcome(
        text=(
            f"动作已在 {mode} 完成并经回执验证："
            f"action_id={outcome.action_id[:20]}…, trust_level={outcome.trust_level}。"
            "非 REAL 证据不可用于证明真实物理执行；grant 已消费。"
        ),
        terminal_receipt=True,
        evidence_ref=f"receipt://{outcome.action_id}",
    )


async def request_approval(
    service: AgentService,
    decision: DecisionV1,
    *,
    mode: str,
    principal: str,
) -> HandlerOutcome:
    """创建授权卡（旧 ServiceIntentHandlers.request_approval 提取）。"""
    broker = service._broker
    payload = (
        decision.proposed_operation.payload if decision.proposed_operation else None
    ) or {}
    capability_id = payload.get("capability_id")
    arguments = payload.get("arguments")
    if mode == "REAL" and (
        not isinstance(capability_id, str)
        or not capability_id.strip()
        or not isinstance(arguments, dict)
    ):
        return HandlerOutcome(
            text=(
                "REAL 精确动作授权缺少 capability_id 或 arguments；未创建空授权卡。"
                "请从可信能力合约取得完整动作标识与参数后重新请求（fail closed）。"
            )
        )
    approval_ttl_sec = 300.0 if mode == "REAL" else 600.0
    # 六审 §4.2：admission 传入统一 created_at/expires_at 时必须原样
    # 采用（ExactAction/Approval/ActionTxn 同一时间边界）。
    unified_created_at = str(payload.get("created_at") or "")
    unified_expires_at = str(payload.get("expires_at") or "")
    display = ActionDisplayV1(
        title=str(payload.get("title") or decision.summary or "动作请求"),
        summary=str(payload.get("summary") or decision.summary or ""),
        risk_tier=payload.get("risk_tier", "LOW"),
        expected_effect=str(payload.get("expected_effect") or ""),
        failure_handling=str(payload.get("failure_handling") or ""),
        parameters=payload.get("parameters") or arguments or {},
    )
    request = ApprovalRequestV2(
        request_id=str(payload.get("request_id") or new_id("appr")),
        mission_id=decision.mission_id,
        task_id=payload.get("task_id"),
        principal=principal,
        body_id=payload.get("body_id", service._body_id),
        effective_body_hash=service._current_body_hash(),
        mode=mode,
        action_display=display,
        context_id=decision.context_id,
        context_revision=decision.context_revision,
        requested_tier=payload.get("tier", "EXACT_ACTION"),
        created_at=unified_created_at or datetime.now(UTC).isoformat(),
        expires_at=unified_expires_at
        or (datetime.now(UTC) + timedelta(seconds=approval_ttl_sec)).isoformat(),
        exact_action_json=str(payload.get("exact_action_json") or ""),
    )
    broker.create_request(request)
    await _emit(
        service, "approval.requested", decision.mission_id,
        {
            "request_id": request.request_id,
            "risk_tier": display.risk_tier,
            "title": display.title,
            "summary": display.summary,
        },
    )
    # ADR-0007：daemon consent plane 只接受显式 REAL 动作的 proposal。
    consent = service._consent_channel
    proposal_note = ""
    if consent is not None and mode == "REAL":
        from rosclaw.agentd.consent_channel import ConsentChannelError

        try:
            proposal = await consent.create_proposal(
                capability_id=str(capability_id),
                arguments=arguments,
                display=display.model_dump(mode="json"),
                execution_mode=mode,
                ttl_sec=approval_ttl_sec,
                client_reference={
                    "agent_request_id": request.request_id,
                    "mission_id": request.mission_id,
                },
            )
            proposal_id = proposal.get("request_id")
            action_id = proposal.get("action_id")
            broker._conn.execute(
                "UPDATE operator_requests SET request_json = ? WHERE request_id = ?",
                (
                    request.model_copy(
                        update={
                            "daemon_proposal_id": proposal_id,
                            "daemon_action_id": action_id,
                            "daemon_capability_id": str(proposal.get("capability_id") or ""),
                            "daemon_action_intent_hash": str(
                                proposal.get("action_intent_hash") or ""
                            ),
                            "expires_at": str(proposal.get("expires_at") or request.expires_at),
                        }
                    ).model_dump_json(),
                    request.request_id,
                ),
            )
            proposal_note = (
                f"\n物理层 daemon proposal {proposal_id} 已创建（nonce/permit "
                "仅在 operator 侧；批准即由 rosclawd 独立签发 permit 并提交）。"
            )
        except ConsentChannelError as exc:
            proposal_note = (
                f"\n注意：daemon consent plane 不可用（{exc}），"
                "物理层未创建 proposal；动作将不会被派发。"
            )
    return HandlerOutcome(
        text=(
            f"已创建授权请求 {request.request_id}（EXACT_ACTION，{int(approval_ttl_sec // 60)} 分钟有效）：\n"
            f"【{display.title}】{display.summary}\n"
            f"风险等级 {display.risk_tier}；预期效果：{display.expected_effect or '—'}；"
            f"失败处理：{display.failure_handling or '—'}\n"
            f"请确认：chat 中输入 /approve {request.request_id} 或 /deny {request.request_id}，"
            "或在 Console 的 Approvals 页操作。在你确认前我不会推进该动作。"
            f"{proposal_note}"
        )
    )
