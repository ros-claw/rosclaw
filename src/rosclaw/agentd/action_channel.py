"""Daemon action channel (K3 完整版, ADR-0001).

The agentd's ONLY physical channel: the sanctioned northbound
``DaemonClient`` over the authenticated Unix socket. agentd builds an
ActionEnvelope (SIMULATION in P0), submits it as a *request*, waits for a
terminal scheduler state, and reads back the execution receipt. A receipt
is verified against the requested action — a submitted command is never
reported as a completed task (总纲 §12.3).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from rosclaw.contracts.common import ValidationError, new_id
from rosclaw.daemon.client import DaemonClient, DaemonClientError
from rosclaw.kernel.contracts import (
    ActionEnvelope,
    AuthorizationContext,
    EvidenceLevel,
    ExecutionMode,
    VerificationPolicy,
)


class ActionChannelError(ValidationError):
    """Dispatch or receipt verification failed (fail closed)."""


@dataclass(frozen=True)
class ActionOutcome:
    action_id: str
    state: str
    receipt: dict[str, Any]
    trust_level: str
    verified: bool


@dataclass(frozen=True)
class ActionProposalOutcome:
    request_id: str
    action_id: str
    state: str


class DaemonActionChannel:
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
        self._sessions_by_capability: dict[str, str] = {}

    async def _ensure_session(self, capability_id: str) -> str:
        existing = self._sessions_by_capability.get(capability_id)
        if existing is not None:
            return existing
        session_id = new_id("sess")
        await asyncio.to_thread(
            self._client.create_session,
            session_id=session_id,
            actor_id=self._actor_id,
            agent_framework="rosclaw-native",
            body_scope=[self._body_id],
            # daemon 拒绝空 scope 与 `*`：显式能力清单。
            capability_scope=[capability_id],
            ttl_ms=30_000,
        )
        self._sessions_by_capability[capability_id] = session_id
        return session_id

    async def request_nonreal_action(
        self,
        *,
        capability_id: str,
        arguments: dict[str, Any],
        grant_id: str,
        execution_mode: str = "SIMULATION",
        timeout_sec: float = 30.0,
    ) -> ActionOutcome:
        """Submit a SIMULATION or SHADOW action and verify its receipt."""
        mode = ExecutionMode(str(execution_mode).upper())
        if mode not in {ExecutionMode.SIMULATION, ExecutionMode.SHADOW}:
            raise ActionChannelError(f"non-real channel does not accept {mode.value}")
        try:
            session_id = await self._ensure_session(capability_id)
        except DaemonClientError as exc:
            raise ActionChannelError(
                f"daemon session failed (daemon offline?): {exc.code}: {exc}"
            ) from exc
        envelope = ActionEnvelope(
            action_id=new_id("act"),
            actor_id=self._actor_id,
            agent_framework="rosclaw-native",
            session_id=session_id,
            body_id=self._body_id,
            body_snapshot_hash=self._body_hash,
            capability_id=capability_id,
            arguments=arguments,
            execution_mode=mode,
            deadline_at=datetime.now(UTC) + timedelta(seconds=timeout_sec),
            authorization=AuthorizationContext(
                principal_id="user:local:1000",
                approved=True,
                approval_id=grant_id,
                scopes=[mode.value.lower()],
            ),
            verification_policy=VerificationPolicy(
                required_evidence=EvidenceLevel.TASK_VERIFIED,
                timeout_sec=timeout_sec,
            ),
        )
        try:
            submitted = await asyncio.to_thread(self._client.request_action, envelope)
        except DaemonClientError as exc:
            raise ActionChannelError(f"daemon rejected action request: {exc.code}: {exc}") from exc
        action_id = submitted.get("action_id", envelope.action_id)
        try:
            status = await asyncio.to_thread(
                self._client.wait_for_action, action_id, timeout_sec=timeout_sec
            )
        except DaemonClientError as exc:
            raise ActionChannelError(f"action did not finish: {exc.code}: {exc}") from exc
        receipt = await asyncio.to_thread(self._client.get_execution_receipt, action_id)
        return self._verify_outcome(action_id, status, receipt, envelope)

    async def request_sim_action(
        self,
        *,
        capability_id: str,
        arguments: dict[str, Any],
        grant_id: str,
        timeout_sec: float = 30.0,
    ) -> ActionOutcome:
        """Backward-compatible SIMULATION entrypoint."""

        return await self.request_nonreal_action(
            capability_id=capability_id,
            arguments=arguments,
            grant_id=grant_id,
            execution_mode="SIMULATION",
            timeout_sec=timeout_sec,
        )

    async def request_real_proposal(
        self,
        *,
        capability_id: str,
        arguments: dict[str, Any],
        grant_id: str,
        grant_public_hash: str,
        principal_id: str,
        risk_tier: str,
        display: dict[str, Any],
        timeout_sec: float = 60.0,
    ) -> ActionProposalOutcome:
        """Create an exact daemon proposal without deciding or dispatching it."""

        action_id = new_id("act")
        envelope = ActionEnvelope(
            action_id=action_id,
            actor_id=self._actor_id,
            agent_framework="rosclaw-native",
            session_id=action_id,
            body_id=self._body_id,
            body_snapshot_hash=self._body_hash,
            capability_id=capability_id,
            arguments=arguments,
            execution_mode=ExecutionMode.REAL,
            risk_class=risk_tier.lower(),
            deadline_at=datetime.now(UTC) + timedelta(seconds=timeout_sec + 60.0),
            authorization=AuthorizationContext(
                principal_id=principal_id,
                approved=False,
                approval_id=grant_id,
                scopes=[],
                provenance={"mission_grant_public_hash": grant_public_hash},
            ),
            verification_policy=VerificationPolicy(
                required_evidence=EvidenceLevel.TASK_VERIFIED,
                timeout_sec=timeout_sec,
            ),
        )
        try:
            created = await asyncio.to_thread(
                self._client.create_operator_proposal,
                envelope,
                display=display,
                ttl_sec=timeout_sec,
            )
        except DaemonClientError as exc:
            raise ActionChannelError(
                f"daemon rejected operator proposal: {exc.code}: {exc}"
            ) from exc
        proposal = created.get("proposal") or {}
        if (
            created.get("command_dispatched") is not False
            or created.get("permit_exposed") is not False
        ):
            raise ActionChannelError(
                "daemon proposal response violated pending/no-permit invariant"
            )
        if not proposal.get("request_id") or not proposal.get("action_id"):
            raise ActionChannelError("daemon proposal response omitted public identifiers")
        return ActionProposalOutcome(
            request_id=str(proposal.get("request_id", "")),
            action_id=str(proposal.get("action_id", action_id)),
            state=str(proposal.get("state", "UNKNOWN")),
        )

    def _verify_outcome(
        self,
        action_id: str,
        status: dict[str, Any],
        receipt: dict[str, Any],
        envelope: ActionEnvelope,
    ) -> ActionOutcome:
        """A submitted command is not a completed task — check the receipt."""
        state = str(status.get("state", "UNKNOWN"))
        # daemon 返回 {"action_id":..., "receipt": {...}} 信封。
        inner = receipt.get("receipt") if isinstance(receipt.get("receipt"), dict) else receipt
        receipt_action_id = inner.get("action_id")
        if receipt_action_id not in (None, action_id):
            raise ActionChannelError(
                f"receipt action {receipt_action_id!r} "
                f"!= requested {action_id!r} — not reporting as our action"
            )
        trust = str(inner.get("trust_level", "UNKNOWN"))
        if trust == "SYNTHETIC":
            raise ActionChannelError("receipt is FIXTURE/SYNTHETIC — never usable as real evidence")
        expected_trust = (
            "SIMULATED" if envelope.execution_mode is ExecutionMode.SIMULATION else "VERIFIED"
        )
        verified = state in ("FINISHED",) and trust == expected_trust
        return ActionOutcome(
            action_id=action_id,
            state=state,
            receipt=receipt,
            trust_level=trust,
            verified=verified,
        )
