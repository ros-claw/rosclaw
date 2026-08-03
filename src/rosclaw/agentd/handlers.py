"""IntentHandlers wiring: AgentLoop → WorkerManager / OperatorBroker.

The handler converts validated decisions into bounded WorkOrders or
approval requests. Authorization semantics (ADR-0006):

- REQUEST_APPROVAL creates a broker approval card and parks the mission in
  WAIT_APPROVAL; the human decides out-of-band (chat command or console);
- REQUEST_ACTION verifies the referenced grant through the broker
  (fail closed) and is honest that SIMULATION has no physical dispatch;
- the agent never sees private permit material — only grant_id + public
  scope hashes.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from rosclaw.agentd.workers import WorkerManager, WorkerRegistry
from rosclaw.agentd.workers.scheduler import CandidateView
from rosclaw.contracts.agent.decision import DecisionV1
from rosclaw.contracts.common import new_id
from rosclaw.contracts.operator.approval import ActionDisplayV1, ApprovalRequestV2
from rosclaw.contracts.worker.order import (
    BudgetEnvelope,
    ExpectedOutput,
    SideEffectPolicy,
    WorkOrderV1,
)
from rosclaw.operator import GrantDeniedError, OperatorBroker
from rosclaw.team.membership import MemberState


class ServiceIntentHandlers:
    def __init__(
        self,
        *,
        registry: WorkerRegistry,
        manager: WorkerManager,
        actor_id: str,
        broker: OperatorBroker | None = None,
        body_id: str = "sim/ur5e",
        body_hash: str = "",
        principal: str = "user:local:1000",
        mode: str = "SIMULATION",
    ) -> None:
        self._registry = registry
        self._manager = manager
        self._actor_id = actor_id
        self._broker = broker
        self._body_id = body_id
        self._body_hash = body_hash
        self._principal = principal
        self._mode = mode

    # ------------------------------------------------------------------
    async def hire_worker(self, decision: DecisionV1) -> str:
        payload = (
            decision.proposed_operation.payload if decision.proposed_operation else None
        ) or {}
        goal = str(payload.get("goal") or decision.summary or "子任务")
        capability = str(payload.get("capability") or "analysis.text")
        instructions = str(payload.get("instructions") or "")
        order = WorkOrderV1(
            work_order_id=new_id("wo"),
            mission_id=decision.mission_id,
            task_id=payload.get("task_id"),
            issued_by=self._actor_id,
            capability=capability,
            goal=goal,
            inputs={
                "instructions": instructions,
                "artifacts": payload.get("artifacts") or [],
            },
            budgets=BudgetEnvelope(
                wall_time_sec=int(payload.get("wall_time_sec", 120)),
                model_tokens=int(payload.get("model_tokens", 50_000)),
            ),
            expected_output=ExpectedOutput(artifacts=["text/plain"]),
            side_effect_policy=SideEffectPolicy(**{"class": "none"}),
        )
        candidates = [
            CandidateView(
                card=card,
                registry_status=self._registry.status_of(card.worker_id) or "DISABLED",
                running_orders=0,
                circuit_open=self._manager.circuit_open(card.worker_id, capability),
            )
            for card in self._registry.list()
        ]
        try:
            scheduled = self._manager.hire(order, candidates)
        except Exception as exc:  # noqa: BLE001 - honest scheduling failure
            return f"无法招聘 Worker（{exc}）。没有伪造委派，任务保持未委派状态。"
        result, report = await self._manager.run_to_completion(scheduled)
        if report.accepted:
            return (
                f"Worker {scheduled.assigned_to} 已完成并通过验证（lease 校验、"
                f"secret 扫描、证据绑定）。结果：\n{result.summary}"
            )
        reasons = "；".join(report.reasons) or "未知原因"
        return (
            f"Worker 提交了结果但未通过 ROSClaw 验证，未采纳（{reasons}）。"
            "我不会把未验证的 Worker 输出当作事实。"
        )

    # ------------------------------------------------------------------
    async def request_approval(self, decision: DecisionV1) -> str:
        if self._broker is None:
            return "授权通道（Operator Broker）尚未启用；已停止推进（fail closed）。"
        payload = (
            decision.proposed_operation.payload if decision.proposed_operation else None
        ) or {}
        capability_id = payload.get("capability_id")
        arguments = payload.get("arguments")
        if self._mode == "REAL" and (
            not isinstance(capability_id, str)
            or not capability_id.strip()
            or not isinstance(arguments, dict)
        ):
            return (
                "REAL 精确动作授权缺少 capability_id 或 arguments；未创建空授权卡。"
                "请从可信能力合约取得完整动作标识与参数后重新请求（fail closed）。"
            )
        approval_ttl_sec = 300.0 if self._mode == "REAL" else 600.0
        display = ActionDisplayV1(
            title=str(payload.get("title") or decision.summary or "动作请求"),
            summary=str(payload.get("summary") or decision.summary or ""),
            risk_tier=payload.get("risk_tier", "LOW"),
            expected_effect=str(payload.get("expected_effect") or ""),
            failure_handling=str(payload.get("failure_handling") or ""),
            parameters=payload.get("parameters") or arguments or {},
        )
        request = ApprovalRequestV2(
            request_id=new_id("appr"),
            mission_id=decision.mission_id,
            task_id=payload.get("task_id"),
            principal=self._principal,
            body_id=payload.get("body_id", self._body_id),
            effective_body_hash=self._body_hash,
            mode=self._mode,
            action_display=display,
            context_id=decision.context_id,
            context_revision=decision.context_revision,
            requested_tier=payload.get("tier", "EXACT_ACTION"),
            created_at=datetime.now(UTC).isoformat(),
            expires_at=(datetime.now(UTC) + timedelta(seconds=approval_ttl_sec)).isoformat(),
        )
        self._broker.create_request(request)
        # ADR-0007：daemon consent plane 只接受显式 REAL 动作的 proposal
        # （PROPOSAL_REAL_ACTION_REQUIRED）。SIMULATION 走认知层 grant +
        # action_channel；REAL 才需要 daemon proposal/permit。
        consent = getattr(self, "_consent_channel", None)
        proposal_note = ""
        if consent is not None and self._mode == "REAL":
            from rosclaw.agentd.consent_channel import ConsentChannelError

            try:
                proposal = await consent.create_proposal(
                    capability_id=str(capability_id),
                    arguments=arguments,
                    display=display.model_dump(mode="json"),
                    execution_mode=self._mode,
                    ttl_sec=approval_ttl_sec,
                )
                proposal_id = proposal.get("request_id")
                action_id = proposal.get("action_id")
                # Persist the linkage on the stored approval request.
                self._broker._conn.execute(
                    "UPDATE operator_requests SET request_json = ? WHERE request_id = ?",
                    (
                        request.model_copy(
                            update={
                                "daemon_proposal_id": proposal_id,
                                "daemon_action_id": action_id,
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
        return (
            f"已创建授权请求 {request.request_id}（EXACT_ACTION，"
            f"{int(approval_ttl_sec // 60)} 分钟有效）：\n"
            f"【{display.title}】{display.summary}\n"
            f"风险等级 {display.risk_tier}；预期效果：{display.expected_effect or '—'}；"
            f"失败处理：{display.failure_handling or '—'}\n"
            f"请确认：chat 中输入 /approve {request.request_id} 或 /deny {request.request_id}，"
            "或在 Console 的 Approvals 页操作。在你确认前我不会推进该动作。"
            f"{proposal_note}"
        )

    # ------------------------------------------------------------------
    async def request_action(self, decision: DecisionV1) -> str:
        if self._broker is None:
            return "物理动作通道未接入；未提交任何动作请求（fail closed）。"
        payload = (
            decision.proposed_operation.payload if decision.proposed_operation else None
        ) or {}
        grant_id = payload.get("grant_id")
        if not grant_id:
            return (
                "请求动作缺少 grant_id。EXACT_ACTION 流程：先 REQUEST_APPROVAL 获得授权，"
                "再在动作请求中引用 grant_id。已拒绝（fail closed）。"
            )
        try:
            # 动作意图由 broker 从已批准的卡片重算，不采信模型自报
            # （精确动作与参数一次性确认，§11.2）。
            intent = self._broker.action_intent_for_grant(str(grant_id))
            grant = self._broker.verify(
                str(grant_id),
                principal=self._principal,
                body_hash=self._body_hash,
                mode=self._mode,
                risk_tier=str(payload.get("risk_tier", "LOW")),
                action_intent=intent,
            )
        except GrantDeniedError as exc:
            return f"授权校验失败（{exc.reason_code}）：{exc}。动作未提交。"
        consent = getattr(self, "_consent_channel", None)
        # ADR-0007 完整路径只在"该授权确实关联了 daemon proposal"时生效
        # （REAL 模式）；否则回落到 SIM action_channel 或诚实无通道。
        proposal_id = None
        action_id = None
        if consent is not None:
            grant_row = self._broker._conn.execute(
                "SELECT request_id FROM mission_grants WHERE grant_id = ?",
                (grant.grant_id,),
            ).fetchone()
            approval_req = self._broker.get_request(grant_row["request_id"]) if grant_row else None
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
                return f"读取 daemon proposal 失败（fail closed）：{exc}"
            state = proposal.get("state")
            if state != "TERMINAL":
                return (
                    f"daemon proposal 尚未到终态（state={state}）。"
                    "若批准时 operator 已裁决，receipt 应在数秒内可用；否则该 proposal "
                    "可能已过期/失效。不报告为完成。"
                )
            if not action_id:
                return "proposal 已终态但无 action_id，无法回读 receipt（fail closed）。"
            try:
                receipt = await consent.action_receipt(action_id)
            except ConsentChannelError as exc:
                return f"回执读取失败（fail closed）：{exc}"
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
                return "回执为 FIXTURE/SYNTHETIC 证据——拒绝当作完成（fail closed）。"
            if inner.get("action_id") not in (None, action_id):
                return "回执 action 与请求不匹配——不报告为我们的动作（fail closed）。"
            return (
                f"动作已由 rosclawd consent plane 完成：state={final_state}, "
                f"trust_level={trust}, evidence_domain={evidence_domain}, "
                f"evidence_level={evidence_level}, "
                f"usable_for_real_execution={str(usable_for_real).lower()}。"
                f"验证：success={verification.get('success')}, "
                f"observer={verification.get('observer')}, "
                f"target_gain_db={verification.get('target_gain_db')}, "
                f"target_prominence_db={verification.get('target_prominence_db')}。"
                f"授权来源：proposal {proposal_id[:20]}…, "
                f"operator={provenance.get('operator_principal')}, "
                f"channel={provenance.get('decision_channel')}。grant 已消费。"
            )
        channel = getattr(self, "_action_channel", None)
        if channel is None:
            return (
                f"授权已验证（grant {grant.grant_id[:20]}…，EXACT_ACTION 已消费）。"
                "注意：当前 agentd 未连接 rosclawd 执行通道，SIMULATION 下没有物理动作被派发；"
                "这不是执行回执。"
            )
        capability = str(payload.get("capability_id", "sim.hold_position"))
        arguments = payload.get("arguments") or {}
        from rosclaw.agentd.action_channel import ActionChannelError

        try:
            if self._mode == "REAL":
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
                return (
                    "REAL 动作已提交到 rosclawd Operator Broker，尚未执行："
                    f"request_id={proposal.request_id[:24]}…, "
                    f"action_id={proposal.action_id[:24]}…, state={proposal.state}。"
                    "需要由受信 Operator 进程独立审阅并确认；Agent 未获得 permit，"
                    "也没有自行授权。"
                )
            outcome = await channel.request_nonreal_action(
                capability_id=capability,
                arguments=arguments,
                grant_id=grant.grant_id,
                execution_mode=self._mode,
            )
        except ActionChannelError as exc:
            return f"动作派发/回执校验失败（fail closed）：{exc}"
        if not outcome.verified:
            return (
                f"动作已提交但未达验证标准（state={outcome.state}, "
                f"trust={outcome.trust_level}）。提交不等于完成——不报告为成功。"
            )
        return (
            f"动作已在 {self._mode} 完成并经回执验证："
            f"action_id={outcome.action_id[:20]}…, trust_level={outcome.trust_level}。"
            "非 REAL 证据不可用于证明真实物理执行；grant 已消费。"
        )

    async def team_coordinate(self, decision: DecisionV1) -> str:
        coordinator = getattr(self, "_team_coordinator", None)
        if coordinator is None:
            return "Team Fabric 尚未启用；未进行团队协调（fail closed）。"
        payload = (
            decision.proposed_operation.payload if decision.proposed_operation else None
        ) or {}
        op = decision.proposed_operation.type if decision.proposed_operation else ""
        if op != "team_task_claim":
            return f"团队操作 {op!r} 暂未实现；未执行（fail closed）。"
        from rosclaw.team.allocator import Bid, TaskAnnouncement

        required = tuple(payload.get("required_capabilities") or ("navigation.local",))
        announcement = TaskAnnouncement(
            task_id=payload.get("task_id") or new_id("ttask"),
            team_id=coordinator._team_id,
            team_epoch=coordinator.epoch(),
            required_capabilities=required,
            deadline_ms=payload.get("deadline_ms"),
            success_criteria=str(payload.get("success_criteria") or decision.summary),
            idempotency_key=payload.get("idempotency_key"),
        )
        # Synthetic local bids from member cards (local_sim honesty: these
        # are declared-capability bids, not measured performance).
        bids = []
        for member in coordinator.membership.members(states=(MemberState.READY,)):
            held = set(member.capabilities)
            fit = len(set(required) & held) / max(len(required), 1)
            bids.append(
                Bid(
                    member_id=member.member_id,
                    eta_ms=float(payload.get("eta_ms", 1000)),
                    energy_cost=100.0,
                    capability_fit=fit,
                    reliability=0.5,  # UNVERIFIED: no track record yet
                    current_load=0.0,
                    comms_quality=1.0,
                )
            )
        try:
            task_id, winner = coordinator.announce_and_award(announcement, bids)
        except Exception as exc:  # noqa: BLE001 - honest allocation failure
            return f"团队任务分配失败（{exc}）。未创建任务（fail closed）。"
        return (
            f"团队任务 {task_id} 已按 contract_net.v1 分配给 {winner}"
            f"（epoch {coordinator.epoch()}，bids 特征向量已入 journal）。"
            "注意：分配是契约建议，执行仍由各机器人本地 Native Agent 与 "
            "rosclawd 独立裁决。"
        )
