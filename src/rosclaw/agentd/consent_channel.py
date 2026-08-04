"""Daemon consent channel (ADR-0007, K5 完整版).

agentd 的认知层授权（ApprovalRequestV2/MissionGrant）与 daemon 的物理层
consent plane（上游 #185 operator proposals）的桥：

- ``create_proposal``：Agent 侧提交 proposal（**永不**附带 nonce/permit）；
- ``decide``：operator 侧裁决（只有 daemon 服务 UID 能列出 nonce 与裁决；
  same-UID 开发环境验证协议，生产必须 UID 分离）；
- ACCEPT 后 daemon 内部签发 permit、以原 Agent UID 提交动作并监督到
  终态 Receipt——agentd 只读回 public 状态与 receipt provenance。
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

from rosclaw.contracts.common import ValidationError, new_id
from rosclaw.daemon.client import DaemonClient, DaemonClientError
from rosclaw.kernel.contracts import (
    ActionEnvelope,
    EvidenceLevel,
    ExecutionMode,
    VerificationPolicy,
)


class ConsentChannelError(ValidationError):
    """Proposal creation/decision/receipt failed (fail closed)."""


class DaemonConsentChannel:
    def __init__(
        self,
        client: DaemonClient,
        *,
        actor_id: str,
        body_id: str,
        body_hash: str,
    ) -> None:
        self._client = client
        self._actor_id = actor_id
        self._body_id = body_id
        self._body_hash = body_hash
        self._session_id: str | None = None

    async def create_proposal(
        self,
        *,
        capability_id: str,
        arguments: dict[str, Any],
        display: dict[str, Any],
        execution_mode: str = "SIMULATION",
        risk_class: str = "low",
        ttl_sec: float = 60.0,
        client_reference: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Agent-side proposal. Returns the public view (no nonce, no permit).

        client_reference（R1/P0-6）：{agent_request_id, mission_id} 绑定，
        daemon 原样带入 challenge/receipt，agentd 精确比对。
        """
        envelope = ActionEnvelope(
            action_id=new_id("act"),
            actor_id=self._actor_id,
            agent_framework="rosclaw-native",
            # 与上游一致：proposal 不引用预建 session（daemon 会因会话
            # 绑定语义拒绝 SESSION_ID_CONFLICT）；动作会话由 daemon 在
            # 接受后自行建立与监督。
            session_id=new_id("sess"),
            body_id=self._body_id,
            body_snapshot_hash=self._body_hash,
            capability_id=capability_id,
            arguments=arguments,
            execution_mode=ExecutionMode(execution_mode),
            risk_class=risk_class,
            deadline_at=datetime.now(UTC) + timedelta(seconds=ttl_sec),
            verification_policy=VerificationPolicy(
                required_evidence=EvidenceLevel.DRIVER_CONFIRMED,
                timeout_sec=ttl_sec,
            ),
        )
        try:
            created = await asyncio.to_thread(
                self._client.create_operator_proposal,
                envelope,
                display=display,
                ttl_sec=ttl_sec,
                client_reference=client_reference,
            )
        except DaemonClientError as exc:
            raise ConsentChannelError(
                f"operator.proposal.create failed: {exc.code}: {exc}"
            ) from exc
        proposal = created.get("proposal") or {}
        if "challenge_nonce" in proposal:
            raise ConsentChannelError(
                "daemon leaked decision challenge to the agent view — refusing"
            )
        return proposal

    async def decide(self, *args, **kwargs):
        """已移除（审计 P0-01）：agentd 不裁决 daemon proposal。

        决定路径在 rosclaw-operatord（enrollment proof + rosclawd ACL）。
        """
        raise ConsentChannelError(
            "agentd no longer decides daemon proposals (P0-01) — "
            "decisions belong to rosclaw-operatord"
        )


    async def proposal(self, request_id: str) -> dict[str, Any]:
        try:
            result = await asyncio.to_thread(self._client.get_operator_proposal, request_id)
        except DaemonClientError as exc:
            raise ConsentChannelError(
                f"operator.proposal.status failed: {exc.code}: {exc}"
            ) from exc
        return result.get("proposal") or {}

    async def action_receipt(self, action_id: str) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(self._client.get_execution_receipt, action_id)
        except DaemonClientError as exc:
            raise ConsentChannelError(f"action.receipt failed: {exc.code}: {exc}") from exc
