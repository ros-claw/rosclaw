"""Daemon-owned action queue, permit gate, E-Stop, leases, and receipts."""

from __future__ import annotations

import contextlib
import hmac
import logging
import os
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from math import isfinite
from pathlib import Path
from typing import Any

from rosclaw.contracts.operator.decision import (
    DecisionChallengeV1,
    DecisionReceiptV1,
    OperatorDecisionProofV1,
    verify_b64,
)
from rosclaw.daemon.health import SupervisionState
from rosclaw.daemon.identity import DaemonIdentity
from rosclaw.daemon.ledger import (
    DaemonLedger,
    LedgerError,
    LedgerEvent,
    LedgerIntegrityError,
)
from rosclaw.daemon.operator_registry import OperatorRegistry, RegistryError
from rosclaw.daemon.permits import ExecutionPermit, PermitAuthority, action_intent_hash
from rosclaw.daemon.protocol import DAEMON_PROTOCOL_VERSION, PeerCredentials
from rosclaw.daemon.session_manager import (
    AgentSession,
    SessionError,
    SessionManager,
)
from rosclaw.daemon.watchdog import RuntimeWatchdog
from rosclaw.daemon.worker_manager import WorkerError, WorkerManager
from rosclaw.kernel import (
    RECEIPT_SCHEMA_VERSION,
    ActionEnvelope,
    ActionState,
    AuthorizationContext,
    EvidenceLevel,
    ExecutionMode,
    OrphanPolicy,
)
from rosclaw.kernel.contracts import utc_now
from rosclaw.operator import (
    OperatorDecision,
    OperatorProposal,
    OperatorProposalError,
    OperatorProposalStore,
    ProposalState,
)

logger = logging.getLogger("rosclaw.daemon.service")

MIN_OPERATOR_PERMIT_TTL_SEC = 1.0
MAX_OPERATOR_PERMIT_TTL_SEC = 300.0


class ControlPlaneError(RuntimeError):
    """Structured daemon service error."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class _ActionJob:
    action: ActionEnvelope
    peer: PeerCredentials
    state: str = "QUEUED"
    submitted_at: datetime = field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    receipt: dict[str, Any] | None = None
    future: Future[dict[str, Any]] | None = None
    lease_expires_at: datetime | None = None
    lease_expires_monotonic: float = 0.0
    last_lease_renewed_at: datetime | None = None
    session_lost: bool = False
    terminal_override: tuple[ActionState, str, str] | None = None
    stop_requested: bool = False
    stop_receipt: dict[str, Any] | None = None
    stop_completed: threading.Event = field(default_factory=threading.Event)

    def to_dict(self) -> dict[str, Any]:
        receipt = self.receipt if isinstance(self.receipt, dict) else {}
        errors = receipt.get("errors")
        first_error = errors[0] if isinstance(errors, list) and errors else {}
        return {
            "action_id": self.action.action_id,
            "state": self.state,
            "scheduler_state": self.state,
            "final_state": receipt.get("final_state"),
            "error_code": (first_error.get("code") if isinstance(first_error, dict) else None),
            "execution_mode": self.action.execution_mode.value,
            "body_id": self.action.body_id,
            "capability_id": self.action.capability_id,
            "submitted_at": _iso(self.submitted_at),
            "started_at": _iso(self.started_at),
            "finished_at": _iso(self.finished_at),
            "session_id": self.action.session_id,
            "orphan_policy": self.action.orphan_policy.value,
            "action_lease": {
                "ttl_ms": self.action.lease_ttl_ms,
                "renew_interval_ms": self.action.renew_interval_ms,
                "last_renewed_at": _iso(self.last_lease_renewed_at),
                "expires_at": _iso(self.lease_expires_at),
                "active": self.state in {"QUEUED", "RUNNING"} and self.terminal_override is None,
            },
            "receipt": self.receipt,
        }


class DaemonControlPlane:
    """Own the only canonical queue feeding a daemon Runtime ActionGateway."""

    def __init__(
        self,
        *,
        runtime: Any,
        permits: PermitAuthority | None = None,
        ledger: DaemonLedger | None = None,
        sessions: SessionManager | None = None,
        worker_manager: WorkerManager | None = None,
        operator_proposals: OperatorProposalStore | None = None,
        max_workers: int = 4,
        max_queued_actions: int = 64,
        max_retained_actions: int = 1024,
        state_dir: Path | None = None,
    ):
        self.runtime = runtime
        self.ledger = ledger
        if permits is None:
            permits = PermitAuthority(ledger=ledger)
        elif ledger is not None and permits.ledger is not ledger:
            raise ValueError("DaemonControlPlane permit authority must use the same ledger")
        self.permits = permits
        self.sessions = sessions or SessionManager()
        self.operator_proposals = operator_proposals or OperatorProposalStore()
        self._operator_decision_lock = threading.RLock()
        # 二次复核 R2/P0-5：持久化 Ed25519 enrollment registry（空表=全拒，
        # 无首调抢注窗口；重启不丢）。state_dir=None 时纯内存（仅测试）。
        if state_dir is None and ledger is not None:
            ledger_path = getattr(ledger, "path", None)
            if ledger_path:
                state_dir = Path(ledger_path).parent
        self._operator_registry = OperatorRegistry(
            (state_dir / "operator-enrollments.json") if state_dir else None
        )
        # 二次复核 R1：daemon 自己的签名身份（DecisionReceiptV1）。
        self._daemon_identity = DaemonIdentity.load_or_create(state_dir)
        queue_capacity = max(1, max_queued_actions)
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, max_workers),
            thread_name_prefix="rosclawd-action",
        )
        self._queue_slots = threading.BoundedSemaphore(queue_capacity)
        self._max_retained_actions = max(
            queue_capacity,
            max(1, max_retained_actions),
        )
        self._evicted_actions = 0
        self._jobs: dict[str, _ActionJob] = {}
        self._lock = threading.RLock()
        self._started_at = utc_now()
        self._instance_id = f"daemon_{uuid.uuid4().hex}"
        self._supervision_state = SupervisionState.STARTING
        self._running = False
        self._closed = False
        self._hardware_actions_executed = 0
        self._emergency_stop_requests = 0
        self._recovery_required = False
        self._recovery_action_ids: list[str] = []
        self._recovery_real_action_ids: list[str] = []
        self._ledger_write_failed = False
        self._ledger_failure: str | None = None
        self.workers = worker_manager or WorkerManager(
            on_generation_change=self._on_worker_generation_change
        )
        self._watchdog = RuntimeWatchdog(self._watchdog_tick)
        if self.ledger is not None:
            self._restore_jobs_from_ledger()
            self._restore_recovery_from_ledger()
            self._invalidate_previous_generation_operator_proposals()

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("DaemonControlPlane cannot restart after close")
            if self._running:
                return
            self._recover_incomplete_jobs_locked()
            self._evict_terminal_jobs_locked(reserve_slot=False)
            self._running = True
            self._supervision_state = SupervisionState.DISARMED
        self._watchdog.start()
        self.workers.start()

    def get_runtime_status(self, peer: PeerCredentials) -> dict[str, Any]:
        with self._lock:
            counts = {"QUEUED": 0, "RUNNING": 0, "FINISHED": 0, "CANCELLED": 0}
            for job in self._jobs.values():
                counts[job.state] = counts.get(job.state, 0) + 1
            runtime_state = getattr(getattr(self.runtime, "state", None), "name", "UNKNOWN")
            driver_names = list(getattr(self.runtime, "driver_names", ()))
            executors = list(getattr(self.runtime.action_gateway, "registered_executors", ()))
            return {
                "protocol_version": DAEMON_PROTOCOL_VERSION,
                "running": self._running and not self._closed,
                "daemon_pid": __import__("os").getpid(),
                "daemon_uid": __import__("os").geteuid(),
                "client_peer": peer.to_dict(),
                "process_separated": peer.pid != __import__("os").getpid(),
                "privilege_separated": peer.uid != __import__("os").geteuid(),
                "southbound_owner": "rosclawd",
                "daemon_instance_id": self._instance_id,
                "supervision_state": self._supervision_state.value,
                "runtime_state": runtime_state,
                "robot_id": str(getattr(getattr(self.runtime, "config", None), "robot_id", "")),
                "emergency_stop_latched": bool(
                    getattr(self.runtime, "emergency_stop_latched", False)
                ),
                "drivers": driver_names,
                "registered_executors": executors,
                "robot_pack": getattr(self.runtime, "robot_pack_status", None),
                "queue": counts,
                "history": {
                    "retained": len(self._jobs),
                    "capacity": self._max_retained_actions,
                    "evicted": self._evicted_actions,
                },
                "permits": self.permits.status(),
                "sessions": self.sessions.status(),
                "operator_proposals": self.operator_proposals.status(),
                "watchdog": self._watchdog.status(),
                "workers": self.workers.status(),
                "ledger": self._ledger_status_locked(),
                "recovery": {
                    "required": self._recovery_required,
                    "action_ids": list(self._recovery_action_ids),
                    "real_action_ids": list(self._recovery_real_action_ids),
                },
                "hardware_actions_executed": self._hardware_actions_executed,
                "emergency_stop_requests": self._emergency_stop_requests,
                "started_at": _iso(self._started_at),
            }

    def create_session(
        self,
        *,
        session_id: str,
        actor_id: str,
        agent_framework: str,
        body_scope: list[str],
        capability_scope: list[str],
        ttl_ms: int,
        peer: PeerCredentials,
    ) -> dict[str, Any]:
        self._require_running()
        try:
            session = self.sessions.create_session(
                session_id=session_id,
                actor_id=actor_id,
                agent_framework=agent_framework,
                body_scope=body_scope,
                capability_scope=capability_scope,
                ttl_ms=ttl_ms,
                peer=peer,
            )
        except SessionError as exc:
            raise ControlPlaneError(exc.code, exc.message) from exc
        self._append_session_event("SESSION_CREATED", session)
        return {"session": session.to_dict()}

    def create_operator_proposal(
        self,
        action: ActionEnvelope,
        *,
        display: dict[str, Any],
        ttl_sec: float,
        peer: PeerCredentials,
        client_reference: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Create a daemon-owned pending proposal without accepting caller approval claims."""

        self._require_running()
        if self.ledger is None:
            raise ControlPlaneError(
                "PROPOSAL_LEDGER_REQUIRED",
                "Operator proposals require the durable daemon ledger",
            )
        try:
            proposal = self.operator_proposals.create(
                action,
                display=display,
                origin_peer=peer,
                daemon_instance_id=self._instance_id,
                ttl_sec=ttl_sec,
                client_reference=client_reference,
            )
            new_proposal = proposal.audited_transition_count == 0
            if not new_proposal:
                return self._operator_submission_result(proposal)
            session_ttl_ms = min(
                3_600_000,
                max(60_000, int(float(ttl_sec) * 1000) + 10_000, action.lease_ttl_ms),
            )
            session = self.sessions.create_session(
                session_id=proposal.action.session_id,
                actor_id=proposal.action.actor_id,
                agent_framework=proposal.action.agent_framework,
                body_scope=[proposal.action.body_id],
                capability_scope=[proposal.action.capability_id],
                ttl_ms=session_ttl_ms,
                peer=peer,
            )
        except (OperatorProposalError, SessionError) as exc:
            raise ControlPlaneError(exc.code, exc.message) from exc
        self._append_session_event("SESSION_CREATED", session)
        try:
            self.ledger.append(
                "OPERATOR_PROPOSAL_CREATED",
                entity_kind="OPERATOR_PROPOSAL",
                entity_id=proposal.request_id,
                payload={
                    "proposal": proposal.operator_dict(),
                    "action": proposal.action.to_dict(),
                },
            )
            proposal.audited_transition_count = len(proposal.transitions)
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._mark_ledger_failure_locked(exc)
            self.operator_proposals.transition(
                proposal,
                ProposalState.INVALIDATED,
                failure_code="LEDGER_UNAVAILABLE",
                failure_message="Proposal creation could not be recorded durably",
            )
            raise ControlPlaneError(
                "LEDGER_UNAVAILABLE",
                "rosclawd could not durably record the operator proposal",
            ) from exc
        return {
            "proposal": proposal.public_dict(),
            "decision": "APPROVAL_PENDING",
            "command_dispatched": False,
            "permit_exposed": False,
        }

    def get_operator_proposal(
        self,
        request_id: str,
        peer: PeerCredentials,
    ) -> dict[str, Any]:
        """Read a proposal as its Agent owner or as the daemon/operator UID."""

        try:
            proposal = self.operator_proposals.get(request_id)
        except OperatorProposalError as exc:
            raise ControlPlaneError(exc.code, exc.message) from exc
        self._require_proposal_reader(proposal, peer)
        self._audit_expired_operator_proposal(proposal)
        self._synchronize_operator_proposal(proposal)
        return {"proposal": proposal.public_dict(), "permit_exposed": False}

    def cancel_operator_proposal(
        self,
        request_id: str,
        peer: PeerCredentials,
    ) -> dict[str, Any]:
        """Cancel an owned pending proposal without granting decision authority."""

        self._require_running()
        with self._operator_decision_lock:
            try:
                proposal = self.operator_proposals.get(request_id)
            except OperatorProposalError as exc:
                raise ControlPlaneError(exc.code, exc.message) from exc
            self._require_proposal_reader(proposal, peer)
            self._audit_expired_operator_proposal(proposal)
            if proposal.state is ProposalState.CANCELLED:
                return self._operator_submission_result(proposal)
            if proposal.state not in {ProposalState.CREATED, ProposalState.PRESENTED}:
                raise ControlPlaneError(
                    "PROPOSAL_NOT_PENDING",
                    f"Proposal is no longer cancellable ({proposal.state.value})",
                )
            self.operator_proposals.transition(proposal, ProposalState.CANCELLED)
            self._append_operator_event("OPERATOR_PROPOSAL_CANCELLED", proposal)
            with contextlib.suppress(SessionError):
                session = self.sessions.close_session(
                    proposal.action.session_id,
                    proposal.origin_peer,
                    reason="operator_proposal_cancelled",
                )
                self._append_session_event("SESSION_CLOSED", session)
            return self._operator_submission_result(proposal)

    def list_pending_operator_proposals(self, peer: PeerCredentials) -> dict[str, Any]:
        """Return trusted broker views（P0-4.1：管理员或已登记 operator UID）。"""

        self._require_operator_reader(peer, "read pending operator proposals")
        proposals = self.operator_proposals.pending()
        for proposal in self.operator_proposals.all():
            self._audit_expired_operator_proposal(proposal)
        for proposal in proposals:
            if proposal.state is ProposalState.CREATED:
                self.operator_proposals.transition(proposal, ProposalState.PRESENTED)
                self._append_operator_event("OPERATOR_PROPOSAL_PRESENTED", proposal)
        return {
            "schema_version": "rosclaw.operator.pending-list.v1",
            "proposals": [proposal.operator_dict() for proposal in proposals],
            "count": len(proposals),
        }

    def register_operator_enrollment(
        self,
        enrollment_id: str,
        *,
        public_key_pem: str,
        operator_uid: int,
        purpose: str = "operator-decision",
        peer: PeerCredentials,
    ) -> dict[str, Any]:
        """登记 operator Ed25519 公钥（二次复核 R2/P0-5）。

        仅 daemon 服务 UID（管理员，经 `rosclaw operatord register-daemon`）。
        **没有 bootstrap 首调窗口**——空 registry 一样只认管理员。
        registry 持久化：daemon 重启不丢、不重新开放抢注。
        """
        self._require_daemon_uid(peer, "register operator enrollments")
        normalized_id = self._identifier(enrollment_id, "enrollment_id")
        if isinstance(operator_uid, bool) or not isinstance(operator_uid, int) or operator_uid < 0:
            raise ControlPlaneError(
                "INVALID_ARGUMENT", "operator_uid must be a non-negative integer"
            )
        try:
            record = self._operator_registry.register(
                normalized_id,
                public_key_pem=public_key_pem,
                operator_uid=operator_uid,
                purpose=purpose,
            )
        except RegistryError as exc:
            raise ControlPlaneError(exc.code, exc.message) from exc
        self._append_operator_enrollment_event("OPERATOR_ENROLLMENT_REGISTERED", record)
        return {
            "schema_version": "rosclaw.operator.enrollment.v2",
            **record.public_dict(),
            "registered": True,
        }

    def revoke_operator_enrollment(
        self,
        enrollment_id: str,
        *,
        peer: PeerCredentials,
    ) -> dict[str, Any]:
        """吊销 operator enrollment（管理员）——被吊销者立即失去决定权。"""
        self._require_daemon_uid(peer, "revoke operator enrollments")
        normalized_id = self._identifier(enrollment_id, "enrollment_id")
        try:
            record = self._operator_registry.revoke(normalized_id)
        except RegistryError as exc:
            raise ControlPlaneError(exc.code, exc.message) from exc
        self._append_operator_enrollment_event("OPERATOR_ENROLLMENT_REVOKED", record)
        return {"schema_version": "rosclaw.operator.enrollment.v2", **record.public_dict()}

    def list_operator_enrollments(self, *, peer: PeerCredentials) -> dict[str, Any]:
        """列出 enrollments 的公开元数据（管理员；公钥指纹不含私钥材料）。"""
        self._require_daemon_uid(peer, "list operator enrollments")
        return {
            "schema_version": "rosclaw.operator.enrollment-list.v1",
            "enrollments": [r.public_dict() for r in self._operator_registry.list()],
        }

    def _append_operator_enrollment_event(self, kind: str, record: Any) -> None:
        if self.ledger is None:
            return
        try:
            self.ledger.append(
                kind,
                entity_kind="OPERATOR_ENROLLMENT",
                entity_id=record.enrollment_id,
                payload={"enrollment": record.public_dict()},
            )
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._mark_ledger_failure_locked(exc)
            raise ControlPlaneError(
                "LEDGER_UNAVAILABLE", "could not durably record enrollment change"
            ) from exc

    def daemon_identity_dict(self) -> dict[str, Any]:
        """daemon 签名公钥（公开信息，任何 peer 可读；信任锚是 socket 隔离）。"""
        return {
            "schema_version": "rosclaw.daemon.identity.v1",
            "daemon_instance_id": self._instance_id,
            "daemon_key_id": self._daemon_identity.key_id,
            "public_key_pem": self._daemon_identity.public_key_pem,
        }

    def _proposal_display_hash(self, proposal: OperatorProposal) -> str:
        """与 agentd 共用同一公式的展示指纹（P0-6 字段绑定）。"""
        from rosclaw.contracts.operator.decision import compute_display_hash

        display = proposal.display
        return compute_display_hash(
            request_id=proposal.request_id,
            title=str(display.get("title", "")),
            summary=str(display.get("summary", "")),
            risk_tier=str(display.get("risk_tier", "")),
            parameters=dict(display.get("parameters", {})),
            body_hash=proposal.action.body_snapshot_hash,
            expires_at=_iso(proposal.expires_at) or "",
        )

    def _operator_challenge_for(self, proposal: OperatorProposal) -> DecisionChallengeV1:
        return DecisionChallengeV1(
            proposal_id=proposal.request_id,
            challenge_nonce=proposal.challenge_nonce,
            display_hash=self._proposal_display_hash(proposal),
            execution_mode=proposal.action.execution_mode.value,
            capability_id=proposal.action.capability_id,
            canonical_args_hash=proposal.action_intent_hash,
            issued_at=_iso(proposal.created_at) or "",
            expires_at=_iso(proposal.expires_at) or "",
            daemon_instance_id=self._instance_id,
            agent_request_id=proposal.client_reference.get("agent_request_id", ""),
            mission_id=proposal.client_reference.get("mission_id", ""),
        )

    def get_operator_challenge(
        self,
        request_id: str,
        peer: PeerCredentials,
    ) -> dict[str, Any]:
        """operatord 取一次性挑战（P0-3：nonce 与 daemon 存储同源）。"""
        self._require_operator_reader(peer, "read operator challenge")
        try:
            proposal = self.operator_proposals.get(request_id)
        except OperatorProposalError as exc:
            raise ControlPlaneError(exc.code, exc.message) from exc
        self._audit_expired_operator_proposal(proposal)
        if proposal.state not in {ProposalState.CREATED, ProposalState.PRESENTED}:
            raise ControlPlaneError(
                "PROPOSAL_NOT_PENDING",
                f"Proposal is no longer pending ({proposal.state.value})",
            )
        return {"challenge": self._operator_challenge_for(proposal).payload()}

    def _require_operator_reader(self, peer: PeerCredentials, operation: str) -> None:
        """daemon 管理员或已登记且 active 的 operator UID（P0-4.1）。"""
        if peer.uid == os.geteuid():
            return
        if peer.uid in self._operator_registry.active_operator_uids():
            return
        raise ControlPlaneError(
            "PERMISSION_DENIED",
            f"Only the rosclawd service UID or an enrolled operator may {operation}",
        )

    def decide_operator_proposal(
        self,
        request_id: str,
        *,
        decision: str,
        principal_id: str,
        channel: str,
        reason: str,
        peer: PeerCredentials,
        proof: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply a trusted exact decision and submit an accepted proposal atomically.

        二次复核 R1/P0-3/P0-4/P0-6：唯一决策凭证是 Ed25519 签名的
        ``OperatorDecisionProofV1``——proof 内嵌 daemon 签发的同一个
        challenge（nonce 同源）；**没有 daemon-UID 直通**（同 UID 一样
        要 proof，消除同 UID 测试假阳性）；验证成功后的 arm/permit 走
        不暴露 socket 的内部方法。
        """
        normalized_principal = self._identifier(principal_id, "principal_id")
        normalized_channel = self._identifier(channel, "channel")
        normalized_reason = self._reason(reason, "decision reason")
        try:
            normalized_decision = OperatorDecision(str(decision).upper())
        except ValueError as exc:
            raise ControlPlaneError(
                "INVALID_OPERATOR_DECISION", "decision must be ACCEPT or DECLINE"
            ) from exc
        try:
            parsed_proof = OperatorDecisionProofV1.from_dict(proof)
        except (ValueError, KeyError, TypeError) as exc:
            raise ControlPlaneError(
                "INVALID_OPERATOR_PROOF", f"invalid operator proof: {exc}"
            ) from exc
        if parsed_proof.decision != normalized_decision.value:
            raise ControlPlaneError(
                "OPERATOR_PROOF_DECISION_MISMATCH",
                "proof decision does not match the requested decision",
            )
        enrollment = self._operator_registry.active(parsed_proof.enrollment_id)
        if enrollment is None:
            raise ControlPlaneError(
                "PERMISSION_DENIED",
                "unknown or revoked operator enrollment — decisions require an "
                "active enrollment registered by the daemon administrator",
            )
        # 纵深防御：proof 有效且调用方就是登记的 operator UID——其他 UID
        # 即使拿到拷贝的公开元数据也无法决定（T1 负向）。
        if peer.uid != enrollment.operator_uid:
            raise ControlPlaneError(
                "PERMISSION_DENIED",
                "caller UID does not match the enrollment's operator UID",
            )
        if not verify_b64(
            enrollment.public_key_pem,
            parsed_proof.signing_payload(),
            parsed_proof.signature_b64,
        ):
            raise ControlPlaneError(
                "PERMISSION_DENIED", "operator proof signature verification failed"
            )
        with self._operator_decision_lock:
            try:
                proposal = self.operator_proposals.get(request_id)
            except OperatorProposalError as exc:
                raise ControlPlaneError(exc.code, exc.message) from exc
            if proposal.daemon_instance_id != self._instance_id:
                raise ControlPlaneError(
                    "PROPOSAL_DAEMON_GENERATION_MISMATCH",
                    "Proposal belongs to a previous daemon generation",
                )
            if proposal.state in {ProposalState.SUBMITTED, ProposalState.TERMINAL}:
                if normalized_decision is not OperatorDecision.ACCEPT:
                    raise ControlPlaneError(
                        "PROPOSAL_ALREADY_DECIDED", "Accepted proposal cannot be declined"
                    )
                self._synchronize_operator_proposal(proposal)
                return self._operator_submission_result(proposal)
            if proposal.state is ProposalState.DECLINED:
                if normalized_decision is OperatorDecision.DECLINE:
                    return self._operator_submission_result(proposal)
                raise ControlPlaneError(
                    "PROPOSAL_ALREADY_DECIDED", "Declined proposal cannot be accepted"
                )
            if proposal.state not in {ProposalState.CREATED, ProposalState.PRESENTED}:
                raise ControlPlaneError(
                    "PROPOSAL_NOT_PENDING",
                    f"Proposal is no longer pending ({proposal.state.value})",
                )
            # P0-3/P0-6：proof 的 challenge 必须与 daemon 持有的 proposal
            # 逐字段一致（含同一个 challenge_nonce）。
            expected = self._operator_challenge_for(proposal).payload()
            got = parsed_proof.challenge.payload()
            mismatched = sorted(
                key
                for key in expected
                if key != "protocol_version" and str(got.get(key, "")) != str(expected[key])
            )
            if mismatched:
                raise ControlPlaneError(
                    "OPERATOR_PROOF_CHALLENGE_MISMATCH",
                    "proof challenge fields do not match the live proposal: "
                    + ", ".join(mismatched),
                )
            if not hmac.compare_digest(
                parsed_proof.challenge.challenge_nonce, proposal.challenge_nonce
            ):
                raise ControlPlaneError(
                    "PROPOSAL_CHALLENGE_MISMATCH",
                    "Operator challenge does not match the live proposal",
                )
            self._require_decided_within_window(parsed_proof, proposal)
            if action_intent_hash(proposal.action) != proposal.action_intent_hash:
                self._invalidate_operator_proposal(
                    proposal,
                    code="PROPOSAL_MUTATED",
                    message="Stored action changed after proposal creation",
                )
                raise ControlPlaneError("PROPOSAL_MUTATED", "Stored proposal action changed")

            if normalized_decision is OperatorDecision.DECLINE:
                self.operator_proposals.transition(
                    proposal,
                    ProposalState.DECLINED,
                    operator_principal=normalized_principal,
                    decision_channel=normalized_channel,
                    decision_reason=normalized_reason,
                )
                self._append_operator_event("OPERATOR_PROPOSAL_DECLINED", proposal)
                receipt = self._finalize_decision(proposal, parsed_proof, normalized_principal)
                result = self._operator_submission_result(proposal)
                result["decision_receipt"] = receipt.to_dict()
                return result

            try:
                self.sessions.require_action(proposal.action, proposal.origin_peer)
            except SessionError as exc:
                self._invalidate_operator_proposal(
                    proposal,
                    code=exc.code,
                    message=exc.message,
                )
                raise ControlPlaneError(exc.code, exc.message) from exc

            armed_by_decision = False
            try:
                self.operator_proposals.transition(
                    proposal,
                    ProposalState.ACCEPTED,
                    operator_principal=normalized_principal,
                    decision_channel=normalized_channel,
                    decision_reason=normalized_reason,
                )
                self._append_operator_event("OPERATOR_PROPOSAL_ACCEPTED", proposal)
                with self._lock:
                    armed = self._supervision_state is SupervisionState.ARMED
                if not armed:
                    # P0-4：内部 arm——外部 peer 不再冒充 daemon。
                    self._arm_after_operator_decision(
                        f"Operator accepted proposal {proposal.request_id}"
                    )
                    armed_by_decision = True
                issued = self._issue_permit_after_operator_decision(
                    proposal.action,
                    principal_id=normalized_principal,
                    target_peer_uid=proposal.origin_peer.uid,
                    expires_in_sec=min(
                        60.0,
                        max(1.0, (proposal.expires_at - utc_now()).total_seconds()),
                    ),
                    reason=normalized_reason,
                    approval_context={
                        "proposal_request_id": proposal.request_id,
                        "action_intent_hash": proposal.action_intent_hash,
                        "decision_channel": normalized_channel,
                        "operator_principal": normalized_principal,
                        "operator_enrollment_id": parsed_proof.enrollment_id,
                        "human_confirmation_method": parsed_proof.human_confirmation_method,
                        "decided_at": proposal.public_dict()["decided_at"],
                    },
                )
                self.operator_proposals.transition(proposal, ProposalState.PERMIT_ISSUED)
                self._append_operator_event(
                    "OPERATOR_PROPOSAL_PERMIT_ISSUED",
                    proposal,
                    extra={"permit_id": issued["permit"]["permit_id"]},
                )
                authorized = issued.get("authorized_action")
                if not isinstance(authorized, dict):
                    raise ControlPlaneError(
                        "PERMIT_INJECTION_FAILED", "rosclawd produced no authorized action"
                    )
                ticket = self.request_action(
                    ActionEnvelope.from_dict(authorized),
                    proposal.origin_peer,
                )
                self.operator_proposals.transition(proposal, ProposalState.SUBMITTED)
                self._append_operator_event(
                    "OPERATOR_PROPOSAL_SUBMITTED",
                    proposal,
                    extra={"action_id": proposal.action.action_id},
                )
            except Exception as exc:
                dispatch_may_have_started = proposal.state is ProposalState.SUBMITTED
                with contextlib.suppress(Exception):
                    self.permits.revoke_session(
                        proposal.action.session_id,
                        reason="operator_proposal_submission_failed",
                    )
                with contextlib.suppress(Exception):
                    self._invalidate_operator_proposal(
                        proposal,
                        code=str(getattr(exc, "code", "PROPOSAL_SUBMISSION_FAILED")),
                        message=str(getattr(exc, "message", exc)),
                    )
                if armed_by_decision or dispatch_may_have_started:
                    with contextlib.suppress(Exception):
                        self._disarm_after_operator_rollback(
                            f"Rollback after proposal {proposal.request_id} failed"
                        )
                if isinstance(exc, ControlPlaneError):
                    raise
                raise ControlPlaneError(
                    "PROPOSAL_SUBMISSION_FAILED",
                    "Accepted proposal could not be submitted",
                ) from exc
            receipt = self._finalize_decision(proposal, parsed_proof, normalized_principal)
            result = self._operator_submission_result(proposal)
            result["action"] = ticket
            result["decision_receipt"] = receipt.to_dict()
            return result

    def _require_decided_within_window(
        self, proof: OperatorDecisionProofV1, proposal: OperatorProposal
    ) -> None:
        """decided_at 必须落在 [issued_at-60s, expires_at] 窗口内。"""
        try:
            decided = datetime.fromisoformat(proof.decided_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ControlPlaneError(
                "INVALID_OPERATOR_PROOF", f"decided_at is not ISO-8601: {exc}"
            ) from exc
        if decided.tzinfo is None:
            decided = decided.replace(tzinfo=UTC)
        if decided < proposal.created_at - timedelta(seconds=60):
            raise ControlPlaneError(
                "INVALID_OPERATOR_PROOF", "decided_at predates the challenge (clock replay?)"
            )
        if decided > proposal.expires_at:
            raise ControlPlaneError(
                "PROPOSAL_EXPIRED", "decision was made after the proposal expired"
            )

    def _finalize_decision(
        self,
        proposal: OperatorProposal,
        proof: OperatorDecisionProofV1,
        principal: str,
    ) -> DecisionReceiptV1:
        """焚毁 nonce（持久化、跨重启防重放）并签发 DecisionReceiptV1。"""
        try:
            self._operator_registry.burn_nonce(proposal.challenge_nonce)
        except RegistryError as exc:
            raise ControlPlaneError(exc.code, exc.message) from exc
        receipt = DecisionReceiptV1(
            proposal_id=proposal.request_id,
            decision=proof.decision,
            operator_enrollment_id=proof.enrollment_id,
            operator_principal=principal,
            human_confirmation_method=proof.human_confirmation_method,
            challenge_nonce=proposal.challenge_nonce,
            decided_at=proof.decided_at,
            expires_at=_iso(proposal.expires_at) or "",
            daemon_instance_id=self._instance_id,
            daemon_key_id=self._daemon_identity.key_id,
            agent_request_id=proposal.client_reference.get("agent_request_id", ""),
            mission_id=proposal.client_reference.get("mission_id", ""),
            execution_mode=proposal.action.execution_mode.value,
            capability_id=proposal.action.capability_id,
            canonical_args_hash=proposal.action_intent_hash,
            display_hash=self._proposal_display_hash(proposal),
        ).sign(self._daemon_identity.private_key)
        proposal.decision_receipt = receipt.to_dict()
        self._append_operator_event(
            "OPERATOR_DECISION_RECEIPT_ISSUED",
            proposal,
            extra={"receipt_id": receipt.receipt_id, "decision": proof.decision},
        )
        return receipt

    def heartbeat_session(self, session_id: str, peer: PeerCredentials) -> dict[str, Any]:
        self._require_running()
        try:
            session = self.sessions.heartbeat(session_id, peer)
        except SessionError as exc:
            if exc.code == "SESSION_EXPIRED":
                with contextlib.suppress(SessionError):
                    expired = self.sessions.get_session(session_id, peer)
                    self._handle_lost_session(expired)
            raise ControlPlaneError(exc.code, exc.message) from exc
        return {"session": session.to_dict()}

    def close_session(
        self,
        session_id: str,
        peer: PeerCredentials,
        *,
        reason: str = "client_closed",
    ) -> dict[str, Any]:
        self._require_running()
        try:
            session = self.sessions.close_session(session_id, peer, reason=reason)
        except SessionError as exc:
            raise ControlPlaneError(exc.code, exc.message) from exc
        self._handle_lost_session(session)
        return {"session": session.to_dict()}

    def get_session(self, session_id: str, peer: PeerCredentials) -> dict[str, Any]:
        try:
            session = self.sessions.get_session(session_id, peer)
        except SessionError as exc:
            raise ControlPlaneError(exc.code, exc.message) from exc
        return {"session": session.to_dict()}

    def renew_action_lease(
        self,
        action_id: str,
        session_id: str,
        peer: PeerCredentials,
    ) -> dict[str, Any]:
        self._require_running()
        with self._lock:
            job = self._jobs.get(action_id) or self._load_persisted_job(action_id)
            if job is None:
                raise ControlPlaneError("ACTION_NOT_FOUND", f"No action {action_id!r} exists")
            self._require_job_owner(job, peer)
            if job.action.session_id != session_id:
                raise ControlPlaneError(
                    "ACTION_SESSION_MISMATCH",
                    "Action lease belongs to a different Agent Session",
                )
            if job.state not in {"QUEUED", "RUNNING"} or job.terminal_override is not None:
                raise ControlPlaneError("ACTION_NOT_ACTIVE", "Action lease is no longer active")
            session_peer = job.peer if peer.uid == os.geteuid() else peer
            try:
                self.sessions.heartbeat(session_id, session_peer)
            except SessionError as exc:
                raise ControlPlaneError(exc.code, exc.message) from exc
            now = utc_now()
            job.last_lease_renewed_at = now
            job.lease_expires_at = now + timedelta(milliseconds=job.action.lease_ttl_ms)
            job.lease_expires_monotonic = time.monotonic() + job.action.lease_ttl_ms / 1000.0
            lease = job.to_dict()["action_lease"]
        self._append_lease_event(action_id, session_id, "ACTION_LEASE_RENEWED", lease)
        return {"action_id": action_id, "session_id": session_id, "action_lease": lease}

    def arm_runtime(self, reason: str, peer: PeerCredentials) -> dict[str, Any]:
        self._require_daemon_uid(peer, "arm rosclawd")
        return self._arm_core(reason, peer)

    def _arm_after_operator_decision(self, reason: str) -> dict[str, Any]:
        """P0-4：operator 决定后的内部 arm——proof 已在 decide 路径验证；
        本方法不注册到 socket dispatch，外部 peer 无法冒充 daemon 调用。"""
        daemon_self = PeerCredentials(pid=os.getpid(), uid=os.geteuid(), gid=os.getegid())
        return self._arm_core(reason, daemon_self)

    def _arm_core(self, reason: str, peer: PeerCredentials) -> dict[str, Any]:
        normalized = self._reason(reason, "arm reason")
        with self._lock:
            if self._recovery_required:
                raise ControlPlaneError(
                    "RECOVERY_REVIEW_REQUIRED",
                    "Interrupted REAL work must be reviewed before arming",
                )
            if bool(getattr(self.runtime, "emergency_stop_latched", False)):
                raise ControlPlaneError(
                    "EMERGENCY_STOP_LATCHED",
                    "Restart rosclawd and complete preflight before re-arming",
                )
            if not self._running or self._closed:
                raise ControlPlaneError("DAEMON_STOPPING", "rosclawd is not running")
            self._supervision_state = SupervisionState.ARMED
        self._append_supervision_event("RUNTIME_ARMED", normalized, peer)
        return {
            "supervision_state": self._supervision_state.value,
            "reason": normalized,
            "daemon_instance_id": self._instance_id,
        }

    def issue_execution_permit(
        self,
        action: ActionEnvelope,
        *,
        principal_id: str,
        target_peer_uid: int,
        expires_in_sec: float,
        reason: str,
        peer: PeerCredentials,
        approval_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Issue one audited, exact-action REAL permit as the daemon service UID."""

        self._require_daemon_uid(peer, "issue REAL execution permits")
        return self._issue_permit_core(
            action,
            principal_id=principal_id,
            target_peer_uid=target_peer_uid,
            expires_in_sec=expires_in_sec,
            reason=reason,
            operator_peer=peer,
            approval_context=approval_context,
        )

    def _issue_permit_after_operator_decision(
        self,
        action: ActionEnvelope,
        *,
        principal_id: str,
        target_peer_uid: int,
        expires_in_sec: float,
        reason: str,
        approval_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """P0-4：operator 决定后的内部 permit——proof 已在 decide 路径验证；
        本方法不注册到 socket dispatch，外部 peer 无法冒充 daemon 调用。"""
        daemon_self = PeerCredentials(pid=os.getpid(), uid=os.geteuid(), gid=os.getegid())
        context = dict(approval_context or {})
        context["via"] = "operator_decision"
        return self._issue_permit_core(
            action,
            principal_id=principal_id,
            target_peer_uid=target_peer_uid,
            expires_in_sec=expires_in_sec,
            reason=reason,
            operator_peer=daemon_self,
            approval_context=context,
        )

    def _issue_permit_core(
        self,
        action: ActionEnvelope,
        *,
        principal_id: str,
        target_peer_uid: int,
        expires_in_sec: float,
        reason: str,
        operator_peer: PeerCredentials,
        approval_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._require_running()
        normalized_reason = self._reason(reason, "permit reason")
        normalized_principal = self._identifier(principal_id, "principal_id")
        if isinstance(target_peer_uid, bool) or not isinstance(target_peer_uid, int):
            raise ControlPlaneError("INVALID_ARGUMENT", "target_peer_uid must be an integer")
        if target_peer_uid < 0:
            raise ControlPlaneError(
                "INVALID_ARGUMENT",
                "target_peer_uid must be non-negative",
            )
        if isinstance(expires_in_sec, bool) or not isinstance(expires_in_sec, (int, float)):
            raise ControlPlaneError(
                "INVALID_ARGUMENT",
                "expires_in_sec must be numeric",
            )
        permit_ttl = float(expires_in_sec)
        if (
            not isfinite(permit_ttl)
            or not MIN_OPERATOR_PERMIT_TTL_SEC <= permit_ttl <= MAX_OPERATOR_PERMIT_TTL_SEC
        ):
            raise ControlPlaneError(
                "INVALID_ARGUMENT",
                (
                    "expires_in_sec must be finite and between "
                    f"{MIN_OPERATOR_PERMIT_TTL_SEC:g} and "
                    f"{MAX_OPERATOR_PERMIT_TTL_SEC:g} seconds"
                ),
            )
        if action.execution_mode not in {ExecutionMode.REAL, ExecutionMode.SHADOW}:
            raise ControlPlaneError(
                "PERMIT_REAL_ACTION_REQUIRED",
                "Operator permits may be issued only for explicit REAL or SHADOW actions "
                "(FTC-100: SHADOW exercises the permission chain with actuation blocked)",
            )
        if not action.body_snapshot_hash.strip():
            raise ControlPlaneError(
                "INVALID_ACTION",
                "REAL permit proposals require a non-empty body_snapshot_hash",
            )
        if len(action.action_id) > 256 or any(
            ord(character) < 0x20 for character in action.action_id
        ):
            raise ControlPlaneError(
                "INVALID_ACTION",
                "action_id must contain at most 256 characters and no control characters",
            )
        if self.ledger is None:
            raise ControlPlaneError(
                "PERMIT_LEDGER_REQUIRED",
                "Official operator permit issuance requires the durable daemon ledger",
            )

        target_peer = PeerCredentials(pid=0, uid=target_peer_uid, gid=0)
        try:
            session = self.sessions.require_action(action, target_peer)
        except SessionError as exc:
            raise ControlPlaneError(exc.code, exc.message) from exc

        now = utc_now()
        deadline = action.deadline_at
        if deadline is None or now >= deadline:
            raise ControlPlaneError(
                "ACTION_DEADLINE_EXPIRED",
                "Action deadline expired before operator permit issuance",
            )
        expires_at = min(now + timedelta(seconds=permit_ttl), deadline)
        expected_executor = f"{action.capability_id}:{ExecutionMode.REAL.value}"
        with self._lock:
            if self._supervision_state is not SupervisionState.ARMED:
                raise ControlPlaneError(
                    "RUNTIME_DISARMED",
                    "rosclawd must be armed before an operator can issue a REAL permit",
                )
            if self._recovery_required:
                raise ControlPlaneError(
                    "RECOVERY_REVIEW_REQUIRED",
                    "Interrupted REAL work must be reviewed before permit issuance",
                )
            if self._ledger_write_failed:
                raise ControlPlaneError(
                    "LEDGER_UNAVAILABLE",
                    "rosclawd durable ledger failed; no REAL permit can be issued",
                )
            if bool(getattr(self.runtime, "emergency_stop_latched", False)):
                raise ControlPlaneError(
                    "EMERGENCY_STOP_LATCHED",
                    "Restart rosclawd and complete preflight before permit issuance",
                )
            expected_shadow_executor = f"{action.capability_id}:{ExecutionMode.SHADOW.value}"
            registered = set(getattr(self.runtime.action_gateway, "registered_executors", ()))
            required_executor = (
                expected_executor
                if action.execution_mode is ExecutionMode.REAL
                else expected_shadow_executor
            )
            if required_executor not in registered:
                raise ControlPlaneError(
                    "REAL_EXECUTOR_UNAVAILABLE",
                    (
                        f"No daemon-side {action.execution_mode.value} executor is registered "
                        f"for {action.capability_id!r}"
                    ),
                )
            issued_at = _iso(now)
            permit = ExecutionPermit(
                permit_id=f"permit_{uuid.uuid4().hex}",
                principal_id=normalized_principal,
                peer_uid=target_peer_uid,
                body_id=action.body_id,
                body_snapshot_hash=action.body_snapshot_hash,
                capabilities=(action.capability_id,),
                action_intent_hash=action_intent_hash(action),
                expires_at=expires_at,
                max_uses=1,
                session_id=action.session_id,
                authorization_provenance=dict(approval_context or {}),
            )
            approval = {
                "schema_version": "rosclaw.daemon.operator_approval.v1",
                "reason": normalized_reason,
                "operator_peer": operator_peer.to_dict(),
                "target_peer_uid": target_peer_uid,
                "daemon_instance_id": self._instance_id,
                "issued_at": issued_at,
            }
            if approval_context:
                approval["provenance"] = dict(approval_context)
            try:
                self.permits.register(permit, audit_context=approval)
            except Exception as exc:  # noqa: BLE001
                self._mark_ledger_failure_locked(exc)
                raise ControlPlaneError(
                    "LEDGER_UNAVAILABLE",
                    "rosclawd could not durably record the operator permit",
                ) from exc

        authorized_action = action.to_dict()
        authorized_action["authorization"] = AuthorizationContext(
            principal_id=normalized_principal,
            approved=True,
            approval_id=permit.permit_id,
            scopes=[action.capability_id],
        ).to_dict()
        return {
            "permit": permit.to_dict(),
            "authorized_action": authorized_action,
            "operator_approval": approval,
            "session": session.to_dict(),
        }

    def _synchronize_operator_proposal(self, proposal: OperatorProposal) -> None:
        if proposal.state is not ProposalState.SUBMITTED:
            return
        try:
            status = self.get_action_status(proposal.action.action_id, proposal.origin_peer)
        except ControlPlaneError:
            return
        if status.get("state") not in {"FINISHED", "CANCELLED"}:
            return
        self.operator_proposals.transition(proposal, ProposalState.TERMINAL)
        self._append_operator_event(
            "OPERATOR_PROPOSAL_TERMINAL",
            proposal,
            extra={
                "action_id": proposal.action.action_id,
                "action_state": status.get("state"),
                "final_state": status.get("final_state"),
            },
        )

    def _operator_submission_result(self, proposal: OperatorProposal) -> dict[str, Any]:
        result: dict[str, Any] = {
            "proposal": proposal.public_dict(),
            "decision": proposal.state.value,
            "command_dispatched": proposal.state
            in {
                ProposalState.SUBMITTED,
                ProposalState.TERMINAL,
            },
            "permit_injected": proposal.state
            in {
                ProposalState.PERMIT_ISSUED,
                ProposalState.SUBMITTED,
                ProposalState.TERMINAL,
            },
            "permit_exposed": False,
        }
        if proposal.state in {ProposalState.SUBMITTED, ProposalState.TERMINAL}:
            with contextlib.suppress(ControlPlaneError):
                result["action"] = self.get_action_status(
                    proposal.action.action_id,
                    proposal.origin_peer,
                )
        return result

    def _invalidate_operator_proposal(
        self,
        proposal: OperatorProposal,
        *,
        code: str,
        message: str,
    ) -> None:
        self.operator_proposals.transition(
            proposal,
            ProposalState.INVALIDATED,
            failure_code=code,
            failure_message=message[:1024],
        )
        self._append_operator_event("OPERATOR_PROPOSAL_INVALIDATED", proposal)

    def _append_operator_event(
        self,
        event_type: str,
        proposal: OperatorProposal,
        *,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if self.ledger is None:
            raise ControlPlaneError(
                "PROPOSAL_LEDGER_REQUIRED",
                "Operator proposal transitions require the durable daemon ledger",
            )
        payload = {
            "request_id": proposal.request_id,
            "action_id": proposal.action.action_id,
            "action_intent_hash": proposal.action_intent_hash,
            "state": proposal.state.value,
            "operator_principal": proposal.operator_principal,
            "decision_channel": proposal.decision_channel,
            "decision_reason": proposal.decision_reason,
            "decided_at": proposal.public_dict()["decided_at"],
            "daemon_instance_id": proposal.daemon_instance_id,
        }
        if extra:
            payload.update(extra)
        try:
            self.ledger.append(
                event_type,
                entity_kind="OPERATOR_PROPOSAL",
                entity_id=proposal.request_id,
                payload=payload,
            )
            proposal.audited_transition_count = len(proposal.transitions)
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._mark_ledger_failure_locked(exc)
            raise ControlPlaneError(
                "LEDGER_UNAVAILABLE",
                "rosclawd could not durably record the operator proposal transition",
            ) from exc

    def _audit_expired_operator_proposal(self, proposal: OperatorProposal) -> None:
        if proposal.state is ProposalState.EXPIRED and proposal.audited_transition_count < len(
            proposal.transitions
        ):
            self._append_operator_event("OPERATOR_PROPOSAL_EXPIRED", proposal)

    def _invalidate_previous_generation_operator_proposals(self) -> None:
        """Durably close pending consent from every earlier daemon generation."""

        assert self.ledger is not None
        latest: dict[str, tuple[str, dict[str, Any]]] = {}
        for event in self.ledger.events(entity_kind="OPERATOR_PROPOSAL"):
            state = event.payload.get("state")
            if not isinstance(state, str):
                raw_proposal = event.payload.get("proposal")
                state = raw_proposal.get("state") if isinstance(raw_proposal, dict) else None
            if isinstance(state, str):
                latest[event.entity_id] = (state, event.payload)
        invalidatable = {
            ProposalState.CREATED.value,
            ProposalState.PRESENTED.value,
            ProposalState.ACCEPTED.value,
            ProposalState.PERMIT_ISSUED.value,
        }
        for request_id, (state, payload) in latest.items():
            if state not in invalidatable:
                continue
            self.ledger.append(
                "OPERATOR_PROPOSAL_INVALIDATED",
                entity_kind="OPERATOR_PROPOSAL",
                entity_id=request_id,
                payload={
                    "request_id": request_id,
                    "action_id": payload.get("action_id")
                    or (
                        payload.get("proposal", {}).get("action_id")
                        if isinstance(payload.get("proposal"), dict)
                        else None
                    ),
                    "state": ProposalState.INVALIDATED.value,
                    "failure_code": "PROPOSAL_DAEMON_RESTARTED",
                    "failure_message": (
                        "Pending operator decision was invalidated by daemon generation change"
                    ),
                    "daemon_instance_id": self._instance_id,
                },
            )

    @staticmethod
    def _require_proposal_reader(
        proposal: OperatorProposal,
        peer: PeerCredentials,
    ) -> None:
        if peer.uid in {proposal.origin_peer.uid, os.geteuid()}:
            return
        raise ControlPlaneError(
            "PROPOSAL_OWNERSHIP_MISMATCH",
            "Authenticated Unix peer does not own this operator proposal",
        )

    def disarm_runtime(self, reason: str, peer: PeerCredentials) -> dict[str, Any]:
        self._require_daemon_uid(peer, "disarm rosclawd")
        return self._disarm_core(reason, peer)

    def _disarm_after_operator_rollback(self, reason: str) -> dict[str, Any]:
        """P0-4：proposal 提交失败回滚时的内部 disarm（不经 socket）。"""
        daemon_self = PeerCredentials(pid=os.getpid(), uid=os.geteuid(), gid=os.getegid())
        return self._disarm_core(reason, daemon_self)

    def _disarm_core(self, reason: str, peer: PeerCredentials) -> dict[str, Any]:
        normalized = self._reason(reason, "disarm reason")
        stop_receipt = self._request_safety_stop(f"runtime disarmed: {normalized}")
        with self._lock:
            self._supervision_state = SupervisionState.ESTOPPED
        self._append_supervision_event("RUNTIME_DISARMED", normalized, peer)
        return {
            "supervision_state": self._supervision_state.value,
            "reason": normalized,
            "stop_receipt": stop_receipt,
        }

    def get_worker_status(
        self,
        peer: PeerCredentials,
        *,
        worker_id: str | None = None,
    ) -> dict[str, Any]:
        self._require_running()
        try:
            if worker_id is None:
                return self.workers.status()
            return {"worker": self.workers.get_status(worker_id)}
        except WorkerError as exc:
            raise ControlPlaneError(exc.code, exc.message) from exc

    def control_worker(
        self,
        operation: str,
        worker_id: str,
        peer: PeerCredentials,
    ) -> dict[str, Any]:
        self._require_daemon_uid(peer, f"{operation} Adapter workers")
        self._require_running()
        try:
            if operation == "start":
                status = self.workers.start_worker(worker_id)
            elif operation == "stop":
                status = self.workers.stop_worker(worker_id)
            elif operation == "restart":
                status = self.workers.restart_worker(worker_id)
            else:
                raise ControlPlaneError(
                    "INVALID_WORKER_OPERATION",
                    f"Unsupported worker operation {operation!r}",
                )
        except WorkerError as exc:
            raise ControlPlaneError(exc.code, exc.message) from exc
        return {"worker": status}

    def request_action(
        self,
        action: ActionEnvelope,
        peer: PeerCredentials,
    ) -> dict[str, Any]:
        """Idempotently enqueue an action without executing on the socket thread."""

        if len(action.action_id) > 256 or any(
            ord(character) < 0x20 for character in action.action_id
        ):
            raise ControlPlaneError(
                "INVALID_ACTION",
                "action_id must contain at most 256 characters and no control characters",
            )
        try:
            try:
                self.sessions.require_action(action, peer)
            except SessionError as exc:
                if exc.code != "SESSION_NOT_FOUND":
                    raise
                self.sessions.adopt_action(action, peer)
            self.sessions.require_action(action, peer)
        except SessionError as exc:
            raise ControlPlaneError(exc.code, exc.message) from exc
        with self._lock:
            if not self._running or self._closed:
                raise ControlPlaneError("DAEMON_STOPPING", "rosclawd is not accepting actions")
            existing = self._jobs.get(action.action_id) or self._load_persisted_job(
                action.action_id
            )
            if existing is not None:
                self._require_job_owner(existing, peer)
                if existing.action.to_dict() != action.to_dict():
                    raise ControlPlaneError(
                        "ACTION_ID_CONFLICT",
                        (
                            f"Action id {action.action_id!r} is already bound to a "
                            "different immutable request."
                        ),
                    )
                return existing.to_dict()
            if action.deadline_at is None or utc_now() >= action.deadline_at:
                raise ControlPlaneError(
                    "ACTION_DEADLINE_EXPIRED",
                    "Action deadline expired before rosclawd accepted it for dispatch",
                )
            if self._ledger_write_failed:
                raise ControlPlaneError(
                    "LEDGER_UNAVAILABLE",
                    "rosclawd durable ledger failed; no new actions are accepted",
                )
            if action.execution_mode is ExecutionMode.REAL and self._recovery_required:
                raise ControlPlaneError(
                    "RECOVERY_REVIEW_REQUIRED",
                    (
                        "rosclawd recovered an interrupted REAL action. A daemon-UID "
                        "operator must review and acknowledge recovery before new REAL work."
                    ),
                )
            if not self._queue_slots.acquire(blocking=False):
                raise ControlPlaneError("ACTION_QUEUE_FULL", "rosclawd action queue is full")
            try:
                self._evict_terminal_jobs_locked()
                if len(self._jobs) >= self._max_retained_actions:
                    raise ControlPlaneError(
                        "ACTION_HISTORY_FULL",
                        (
                            "rosclawd retained-action capacity is full. Configure a durable "
                            "ledger or archive evidence under operator control."
                        ),
                    )
                now = utc_now()
                job = _ActionJob(
                    action=action,
                    peer=peer,
                    last_lease_renewed_at=now,
                    lease_expires_at=now + timedelta(milliseconds=action.lease_ttl_ms),
                    lease_expires_monotonic=time.monotonic() + action.lease_ttl_ms / 1000.0,
                )
                if self.ledger is not None:
                    try:
                        self.ledger.append(
                            "ACTION_SUBMITTED",
                            entity_kind="ACTION",
                            entity_id=action.action_id,
                            payload={
                                "action": action.to_dict(),
                                "peer": peer.to_dict(),
                                "submitted_at": _iso(job.submitted_at),
                                "action_lease": job.to_dict()["action_lease"],
                            },
                        )
                    except Exception as exc:  # noqa: BLE001
                        self._mark_ledger_failure_locked(exc)
                        raise ControlPlaneError(
                            "LEDGER_UNAVAILABLE",
                            "rosclawd could not durably record the action submission",
                        ) from exc
                self._jobs[action.action_id] = job
            except Exception:
                self._jobs.pop(action.action_id, None)
                self._queue_slots.release()
                raise
            try:
                job.future = self._executor.submit(self._run_action, job)
            except Exception:  # noqa: BLE001
                logger.exception("rosclawd could not schedule action %s", action.action_id)
                receipt = self.runtime.action_gateway.reject(
                    action,
                    code="DAEMON_SCHEDULING_FAILED",
                    message=(
                        "rosclawd could not schedule the action; no executor dispatch occurred."
                    ),
                    state=ActionState.FAILED,
                )
                job.state = "FINISHED"
                job.finished_at = utc_now()
                job.receipt = receipt.to_dict()
                try:
                    self._persist_terminal_job(job)
                except Exception as exc:  # noqa: BLE001
                    self._mark_ledger_failure_locked(exc)
                self._queue_slots.release()
                return job.to_dict()
            job.future.add_done_callback(lambda _future: self._queue_slots.release())
            self._append_lease_event(
                action.action_id,
                action.session_id,
                "ACTION_LEASE_CREATED",
                job.to_dict()["action_lease"],
            )
            return job.to_dict()

    def get_action_status(
        self,
        action_id: str,
        peer: PeerCredentials | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(action_id) or self._load_persisted_job(action_id)
            if job is None:
                raise ControlPlaneError(
                    "ACTION_NOT_FOUND",
                    f"No rosclawd action exists with id {action_id!r}",
                )
            self._require_job_owner(job, peer)
            return job.to_dict()

    def get_execution_receipt(
        self,
        action_id: str,
        peer: PeerCredentials | None = None,
    ) -> dict[str, Any]:
        status = self.get_action_status(action_id, peer)
        receipt = status.get("receipt")
        if not isinstance(receipt, dict):
            raise ControlPlaneError(
                "ACTION_NOT_FINISHED",
                f"Action {action_id!r} has not produced an ExecutionReceipt",
            )
        return {"action_id": action_id, "receipt": receipt}

    def cancel_action(
        self,
        action_id: str,
        peer: PeerCredentials | None = None,
    ) -> dict[str, Any]:
        """Cancel only work that has not started; never claim an active robot stopped."""

        with self._lock:
            job = self._jobs.get(action_id) or self._load_persisted_job(action_id)
            if job is None:
                raise ControlPlaneError(
                    "ACTION_NOT_FOUND",
                    f"No rosclawd action exists with id {action_id!r}",
                )
            self._require_job_owner(job, peer)
            if job.state in {"FINISHED", "CANCELLED"}:
                return {
                    "action_id": action_id,
                    "cancelled": job.state == "CANCELLED",
                    "state": job.state,
                    "message": "Action is already terminal.",
                }
            future = job.future
            if job.state == "QUEUED" and future is not None and future.cancel():
                receipt = self.runtime.action_gateway.reject(
                    job.action,
                    code="ACTION_CANCELLED_BEFORE_DISPATCH",
                    message="Action was cancelled while queued; no executor was dispatched.",
                    state=ActionState.CANCELLED,
                )
                job.state = "CANCELLED"
                job.finished_at = utc_now()
                job.receipt = receipt.to_dict()
                try:
                    self._persist_terminal_job(job)
                except Exception as exc:  # noqa: BLE001
                    self._mark_ledger_failure_locked(exc)
                    raise ControlPlaneError(
                        "LEDGER_UNAVAILABLE",
                        (
                            "The queued action was cancelled locally, but rosclawd could "
                            "not durably record the terminal receipt."
                        ),
                    ) from exc
                return {
                    "action_id": action_id,
                    "cancelled": True,
                    "state": job.state,
                    "receipt": job.receipt,
                }
            return {
                "action_id": action_id,
                "cancelled": False,
                "state": job.state,
                "code": "ACTIVE_ACTION_REQUIRES_EMERGENCY_STOP",
                "message": (
                    "The action has started. Cancellation is not reported as a physical stop; "
                    "use emergency_stop when motion may be active."
                ),
            }

    def emergency_stop(
        self,
        reason: str,
        *,
        source: str,
        timeout_sec: float,
        peer: PeerCredentials,
    ) -> dict[str, Any]:
        """Call the Runtime stop path directly, independent of the action queue."""

        with self._lock:
            self._emergency_stop_requests += 1
            self._supervision_state = SupervisionState.ESTOPPED
        authenticated_source = f"rosclawd.peer.uid{peer.uid}.pid{peer.pid}/{source[:128]}"
        receipt = self.runtime.request_emergency_stop(
            reason,
            source=authenticated_source,
            timeout_sec=timeout_sec,
        )
        if isinstance(receipt, dict):
            payload = dict(receipt)
        elif hasattr(receipt, "to_dict"):
            payload = receipt.to_dict()
        else:
            raise ControlPlaneError(
                "INVALID_ESTOP_RECEIPT",
                "Runtime returned an invalid emergency-stop receipt",
            )
        payload["authenticated_peer"] = peer.to_dict()
        payload["requested_source"] = source
        return payload

    def acknowledge_recovery(
        self,
        reason: str,
        peer: PeerCredentials,
    ) -> dict[str, Any]:
        """Persist daemon-UID review of interrupted REAL action evidence."""

        if peer.uid != os.geteuid():
            raise ControlPlaneError(
                "PERMISSION_DENIED",
                "Only the rosclawd service UID may acknowledge restart recovery",
            )
        if not isinstance(reason, str):
            raise ControlPlaneError(
                "INVALID_ARGUMENT",
                "recovery acknowledgement reason must be a string",
            )
        normalized_reason = reason.strip()
        if not normalized_reason or len(normalized_reason) > 1024:
            raise ControlPlaneError(
                "INVALID_ARGUMENT",
                "recovery acknowledgement reason must contain 1 to 1024 characters",
            )
        with self._lock:
            if not self._recovery_required:
                return {
                    "acknowledged": False,
                    "recovery_required": False,
                    "message": "No restart recovery review is pending.",
                }
            if self.ledger is None:
                raise ControlPlaneError(
                    "RECOVERY_LEDGER_REQUIRED",
                    "Restart recovery cannot be acknowledged without a durable ledger",
                )
            action_ids = list(self._recovery_action_ids)
            real_action_ids = list(self._recovery_real_action_ids)
            acknowledged_at = _iso(utc_now())
            try:
                self.ledger.append(
                    "RECOVERY_ACKNOWLEDGED",
                    entity_kind="RECOVERY",
                    entity_id="rosclawd",
                    payload={
                        "action_ids": action_ids,
                        "real_action_ids": real_action_ids,
                        "reason": normalized_reason,
                        "acknowledged_at": acknowledged_at,
                        "peer": peer.to_dict(),
                    },
                )
            except Exception as exc:  # noqa: BLE001
                self._mark_ledger_failure_locked(exc)
                raise ControlPlaneError(
                    "LEDGER_UNAVAILABLE",
                    "rosclawd could not durably record the recovery acknowledgement",
                ) from exc
            self._recovery_required = False
            self._recovery_action_ids = []
            self._recovery_real_action_ids = []
            return {
                "acknowledged": True,
                "recovery_required": False,
                "action_ids": action_ids,
                "real_action_ids": real_action_ids,
                "reason": normalized_reason,
                "acknowledged_at": acknowledged_at,
                "authenticated_peer": peer.to_dict(),
                "emergency_stop_latched": bool(
                    getattr(self.runtime, "emergency_stop_latched", False)
                ),
            }

    def close(self) -> None:
        """Latch E-Stop, reject new work, cancel queued jobs, and stop Runtime."""

        with self._lock:
            if self._closed:
                return
            self._running = False
            self._closed = True
            self._supervision_state = SupervisionState.STOPPING
            queued = [
                job
                for job in self._jobs.values()
                if job.state == "QUEUED" and job.future is not None
            ]
        self._watchdog.stop()
        for job in queued:
            try:
                self.cancel_action(job.action.action_id)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "rosclawd could not durably cancel queued action %s during shutdown",
                    job.action.action_id,
                )

        with contextlib.suppress(Exception):
            self.runtime.request_emergency_stop(
                "rosclawd shutdown",
                source="rosclawd.shutdown",
                timeout_sec=1.0,
            )
        self.workers.close()
        self._executor.shutdown(wait=True, cancel_futures=True)
        with contextlib.suppress(Exception):
            self.runtime.stop()

    def _run_action(self, job: _ActionJob) -> dict[str, Any]:
        with self._lock:
            if job.terminal_override is not None:
                return self._finish_overridden_job_locked(job)
            now = utc_now()
            if job.action.deadline_at is None or now >= job.action.deadline_at:
                job.terminal_override = (
                    ActionState.TIMED_OUT,
                    "ACTION_DEADLINE_EXPIRED",
                    "Action expired while queued; no executor dispatch occurred.",
                )
                return self._finish_overridden_job_locked(job)
            if time.monotonic() >= job.lease_expires_monotonic:
                job.terminal_override = (
                    ActionState.TIMED_OUT,
                    "ACTION_LEASE_EXPIRED",
                    "Action lease expired while queued; no executor dispatch occurred.",
                )
                return self._finish_overridden_job_locked(job)
            job.state = "RUNNING"
            job.started_at = utc_now()
            if self.ledger is not None:
                try:
                    self.ledger.append(
                        "ACTION_STARTED",
                        entity_kind="ACTION",
                        entity_id=job.action.action_id,
                        payload={"started_at": _iso(job.started_at)},
                    )
                except Exception as exc:  # noqa: BLE001
                    self._mark_ledger_failure_locked(exc)
                    receipt = self.runtime.action_gateway.reject(
                        job.action,
                        code="DAEMON_LEDGER_WRITE_FAILED",
                        message=(
                            "rosclawd could not durably record action start; no executor "
                            "dispatch occurred and no physical outcome is claimed."
                        ),
                        state=ActionState.FAILED,
                    )
                    job.state = "FINISHED"
                    job.finished_at = utc_now()
                    job.receipt = receipt.to_dict()
                    return job.receipt

        action = job.action
        try:
            if action.execution_mode is ExecutionMode.REAL:
                try:
                    decision = self.permits.authorize(action, job.peer)
                except LedgerError as exc:
                    with self._lock:
                        self._mark_ledger_failure_locked(exc)
                    raise
                if not decision.allowed:
                    receipt = self.runtime.action_gateway.reject(
                        action,
                        code=decision.code,
                        message=decision.message,
                        state=ActionState.BLOCKED,
                    )
                else:
                    with self._lock:
                        armed = self._supervision_state is SupervisionState.ARMED
                    if not armed:
                        receipt = self.runtime.action_gateway.reject(
                            action,
                            code="RUNTIME_DISARMED",
                            message=(
                                "rosclawd must be explicitly armed by its service UID "
                                "before REAL executor dispatch"
                            ),
                            state=ActionState.BLOCKED,
                        )
                    else:
                        payload = action.to_dict()
                        payload["authorization"] = decision.authorization.to_dict()
                        receipt = self.runtime.submit_action(ActionEnvelope.from_dict(payload))
            else:
                receipt = self.runtime.submit_action(action)
            receipt_payload = receipt if isinstance(receipt, dict) else receipt.to_dict()
        except Exception:  # noqa: BLE001
            logger.exception(
                "rosclawd action %s failed before producing a receipt",
                action.action_id,
            )
            receipt = self.runtime.action_gateway.reject(
                action,
                code="DAEMON_ACTION_FAILED",
                message=(
                    "rosclawd failed before a valid execution receipt was produced; "
                    "no successful physical outcome is claimed."
                ),
                state=ActionState.FAILED,
            )
            receipt_payload = receipt.to_dict()
        with self._lock:
            overridden = job.terminal_override is not None
        if overridden:
            self._stop_overridden_action(job)
        with self._lock:
            if job.terminal_override is not None:
                receipt_payload = self._apply_terminal_override(job, receipt_payload)
            job.state = "FINISHED"
            job.finished_at = utc_now()
            job.receipt = receipt_payload
            try:
                self._persist_terminal_job(job)
            except Exception as exc:  # noqa: BLE001
                self._mark_ledger_failure_locked(exc)
                if action.execution_mode is ExecutionMode.REAL:
                    self._merge_recovery_requirement_locked(
                        [action.action_id],
                        [action.action_id],
                    )
                    self._emergency_stop_requests += 1
                    try:
                        self.runtime.request_emergency_stop(
                            "rosclawd could not persist a REAL terminal receipt",
                            source="rosclawd.ledger_failure",
                            timeout_sec=1.0,
                        )
                    except Exception:  # noqa: BLE001
                        logger.exception("rosclawd ledger-failure E-Stop request failed")
                discard = getattr(self.runtime.action_gateway, "discard_receipt", None)
                if callable(discard):
                    discard(action.action_id)
                failure = self.runtime.action_gateway.reject(
                    action,
                    code="DAEMON_LEDGER_TERMINAL_WRITE_FAILED",
                    message=(
                        "The executor returned, but rosclawd could not durably record the "
                        "terminal receipt. The physical outcome is not trusted."
                    ),
                    state=ActionState.FAILED,
                )
                job.receipt = failure.to_dict()
                receipt_payload = job.receipt
            if (
                action.execution_mode is ExecutionMode.REAL
                and receipt_payload.get("dispatch_result", {}).get("accepted") is True
            ):
                self._hardware_actions_executed += 1
        return receipt_payload

    def _watchdog_tick(self) -> None:
        expired_sessions = self.sessions.expire_sessions()
        for session in expired_sessions:
            self._handle_lost_session(session)

        now_monotonic = time.monotonic()
        now = datetime.now(UTC)
        queued: list[_ActionJob] = []
        running: list[_ActionJob] = []
        with self._lock:
            if not self._running or self._closed:
                return
            for job in self._jobs.values():
                if job.state not in {"QUEUED", "RUNNING"} or job.terminal_override is not None:
                    continue
                override: tuple[ActionState, str, str] | None = None
                if job.action.deadline_at is not None and now >= job.action.deadline_at:
                    override = (
                        ActionState.TIMED_OUT,
                        "ACTION_DEADLINE_EXPIRED",
                        "Action exceeded its immutable deadline; safety stop requested.",
                    )
                elif now_monotonic >= job.lease_expires_monotonic and not (
                    job.session_lost
                    and job.action.orphan_policy is OrphanPolicy.CONTINUE_UNTIL_DEADLINE
                ):
                    override = (
                        ActionState.TIMED_OUT,
                        "ACTION_LEASE_EXPIRED",
                        "Action lease was not renewed; safety stop requested.",
                    )
                if override is None:
                    continue
                job.terminal_override = override
                (queued if job.state == "QUEUED" else running).append(job)
        for job in queued:
            self._terminalize_queued_override(job)
        for job in running:
            self._stop_overridden_action(job)

    def _handle_lost_session(self, session: AgentSession) -> None:
        try:
            revoked = self.permits.revoke_session(
                session.session_id,
                reason=f"session_{session.state.value.lower()}",
            )
        except Exception as exc:  # noqa: BLE001
            # Session safety is in-memory and must continue even if audit I/O fails.
            revoked = -1
            with self._lock:
                self._mark_ledger_failure_locked(exc)
        queued: list[_ActionJob] = []
        running: list[_ActionJob] = []
        with self._lock:
            for job in self._jobs.values():
                if job.action.session_id != session.session_id or job.state not in {
                    "QUEUED",
                    "RUNNING",
                }:
                    continue
                job.session_lost = True
                if (
                    job.state == "RUNNING"
                    and job.action.orphan_policy is OrphanPolicy.CONTINUE_UNTIL_DEADLINE
                ):
                    continue
                if job.terminal_override is None:
                    job.terminal_override = (
                        ActionState.ORPHANED,
                        "AGENT_SESSION_LOST",
                        (
                            "Owning Agent Session was lost; orphan policy "
                            f"{job.action.orphan_policy.value} was applied."
                        ),
                    )
                (queued if job.state == "QUEUED" else running).append(job)
        for job in queued:
            self._terminalize_queued_override(job)
        for job in running:
            self._stop_overridden_action(job)
        self._append_session_event(
            "SESSION_LOST" if session.state.value == "LOST" else "SESSION_CLOSED",
            session,
            extra={"revoked_permits": revoked},
        )

    def _terminalize_queued_override(self, job: _ActionJob) -> None:
        with self._lock:
            future = job.future
            if job.state != "QUEUED" or future is None or not future.cancel():
                return
            self._finish_overridden_job_locked(job)

    def _finish_overridden_job_locked(self, job: _ActionJob) -> dict[str, Any]:
        assert job.terminal_override is not None
        state, code, message = job.terminal_override
        receipt = self.runtime.action_gateway.reject(
            job.action,
            code=code,
            message=message,
            state=state,
        )
        job.state = "FINISHED"
        job.finished_at = utc_now()
        job.receipt = receipt.to_dict()
        try:
            self._persist_terminal_job(job)
        except Exception as exc:  # noqa: BLE001
            self._mark_ledger_failure_locked(exc)
        return job.receipt

    def _stop_overridden_action(self, job: _ActionJob) -> None:
        with self._lock:
            if job.stop_requested:
                leader = False
            else:
                job.stop_requested = True
                leader = True
        if not leader:
            job.stop_completed.wait(timeout=2.0)
            return
        try:
            receipt = self._request_safety_stop(
                f"action {job.action.action_id} terminated by rosclawd watchdog"
            )
            with self._lock:
                job.stop_receipt = receipt
        finally:
            job.stop_completed.set()

    def _request_safety_stop(self, reason: str) -> dict[str, Any]:
        with self._lock:
            self._emergency_stop_requests += 1
            self._supervision_state = SupervisionState.ESTOPPED
        try:
            receipt = self.runtime.request_emergency_stop(
                reason,
                source="rosclawd.watchdog",
                timeout_sec=1.0,
            )
            if isinstance(receipt, dict):
                return dict(receipt)
            if hasattr(receipt, "to_dict"):
                return receipt.to_dict()
            return {"final_status": "FAILED", "error": "invalid stop receipt"}
        except Exception as exc:  # noqa: BLE001
            logger.exception("rosclawd watchdog safety stop failed")
            return {
                "final_status": "FAILED",
                "error": f"{type(exc).__name__}: {exc}"[:512],
            }

    def _on_worker_generation_change(
        self,
        worker_id: str,
        old_connection_id: str | None,
        new_connection_id: str,
    ) -> None:
        if old_connection_id is None:
            return
        try:
            self.permits.revoke_all(reason=f"worker_generation_changed:{worker_id}")
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._mark_ledger_failure_locked(exc)
        self._request_safety_stop(
            f"Adapter worker {worker_id} changed connection generation "
            f"from {old_connection_id} to {new_connection_id}"
        )

    @staticmethod
    def _apply_terminal_override(job: _ActionJob, receipt: dict[str, Any]) -> dict[str, Any]:
        assert job.terminal_override is not None
        state, code, message = job.terminal_override
        payload = dict(receipt)
        errors = list(payload.get("errors", []))
        errors.append({"code": code, "message": message})
        transitions = list(payload.get("transitions", []))
        transitions.append({"state": state.value, "at": _iso(utc_now()), "reason": code})
        payload.update(
            {
                "final_state": state.value,
                "errors": errors,
                "transitions": transitions,
                "finished_at": _iso(utc_now()),
                "usable_for_real_execution": False,
                "safety_stop": job.stop_receipt
                or {
                    "final_status": "UNKNOWN",
                    "error": "safety-stop request did not complete before receipt finalization",
                },
            }
        )
        return payload

    def _append_session_event(
        self,
        event_type: str,
        session: AgentSession,
        *,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if self.ledger is None:
            return
        payload = {"session": session.to_dict(), **(extra or {})}
        try:
            self.ledger.append(
                event_type,
                entity_kind="SESSION",
                entity_id=session.session_id,
                payload=payload,
            )
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._mark_ledger_failure_locked(exc)

    def _append_lease_event(
        self,
        action_id: str,
        session_id: str,
        event_type: str,
        lease: dict[str, Any],
    ) -> None:
        if self.ledger is None:
            return
        try:
            self.ledger.append(
                event_type,
                entity_kind="ACTION_LEASE",
                entity_id=action_id,
                payload={"session_id": session_id, "lease": lease},
            )
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._mark_ledger_failure_locked(exc)

    def _append_supervision_event(
        self,
        event_type: str,
        reason: str,
        peer: PeerCredentials,
    ) -> None:
        if self.ledger is None:
            return
        try:
            self.ledger.append(
                event_type,
                entity_kind="SUPERVISION",
                entity_id=self._instance_id,
                payload={"reason": reason, "peer": peer.to_dict(), "at": _iso(utc_now())},
            )
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._mark_ledger_failure_locked(exc)

    def _require_running(self) -> None:
        with self._lock:
            if not self._running or self._closed:
                raise ControlPlaneError("DAEMON_STOPPING", "rosclawd is not accepting work")

    @staticmethod
    def _require_daemon_uid(peer: PeerCredentials, operation: str) -> None:
        if peer.uid != os.geteuid():
            raise ControlPlaneError(
                "PERMISSION_DENIED",
                f"Only the rosclawd service UID may {operation}",
            )

    @staticmethod
    def _reason(value: str, field: str) -> str:
        if not isinstance(value, str) or not value.strip() or len(value) > 1024:
            raise ControlPlaneError(
                "INVALID_ARGUMENT",
                f"{field} must contain 1 to 1024 characters",
            )
        return value.strip()

    @staticmethod
    def _identifier(value: str, field: str) -> str:
        if (
            not isinstance(value, str)
            or not value.strip()
            or len(value) > 256
            or any(ord(character) < 0x20 for character in value)
        ):
            raise ControlPlaneError(
                "INVALID_ARGUMENT",
                f"{field} must contain 1 to 256 printable characters",
            )
        return value.strip()

    def _mark_ledger_failure_locked(self, error: Exception) -> None:
        self._ledger_write_failed = True
        self._ledger_failure = f"{type(error).__name__}: {error}"[:512]
        logger.error("rosclawd durable ledger failed: %s", self._ledger_failure)

    def _ledger_status_locked(self) -> dict[str, Any] | None:
        if self.ledger is None:
            return None
        try:
            status = self.ledger.status()
        except Exception as exc:  # noqa: BLE001
            self._mark_ledger_failure_locked(exc)
            status = {
                "schema_version": "rosclaw.daemon.ledger.v1",
                "path": str(self.ledger.path),
                "anchor_path": str(self.ledger.anchor_path),
                "key_path": str(self.ledger.key_path),
                "integrity_verified": False,
            }
        status["write_failed"] = self._ledger_write_failed
        status["failure"] = self._ledger_failure
        return status

    def _persist_terminal_job(self, job: _ActionJob) -> None:
        if self.ledger is None:
            return
        if job.receipt is None or job.finished_at is None:
            raise LedgerIntegrityError("terminal daemon action is missing receipt metadata")
        self.ledger.append(
            "ACTION_TERMINAL",
            entity_kind="ACTION",
            entity_id=job.action.action_id,
            payload={
                "scheduler_state": job.state,
                "finished_at": _iso(job.finished_at),
                "receipt": job.receipt,
            },
        )

    def _restore_jobs_from_ledger(self) -> None:
        assert self.ledger is not None
        grouped: dict[str, list[LedgerEvent]] = {}
        for event in self.ledger.events(entity_kind="ACTION"):
            grouped.setdefault(event.entity_id, []).append(event)
        for action_id, events in grouped.items():
            self._jobs[action_id] = self._decode_persisted_job(events)

    def _load_persisted_job(self, action_id: str) -> _ActionJob | None:
        if self.ledger is None:
            return None
        try:
            events = self.ledger.events(entity_kind="ACTION", entity_id=action_id)
        except Exception as exc:  # noqa: BLE001
            self._mark_ledger_failure_locked(exc)
            raise ControlPlaneError(
                "LEDGER_UNAVAILABLE",
                "rosclawd could not verify its durable action history",
            ) from exc
        return self._decode_persisted_job(events) if events else None

    @staticmethod
    def _decode_persisted_job(events: list[LedgerEvent]) -> _ActionJob:
        job: _ActionJob | None = None
        entity_id = events[0].entity_id
        for event in events:
            if event.entity_id != entity_id:
                raise LedgerIntegrityError("persisted action event scope is inconsistent")
            if event.event_type == "ACTION_SUBMITTED":
                if job is not None:
                    raise LedgerIntegrityError("persisted action has duplicate submission events")
                raw_action = event.payload.get("action")
                raw_peer = event.payload.get("peer")
                if not isinstance(raw_action, dict) or not isinstance(raw_peer, dict):
                    raise LedgerIntegrityError("persisted action submission is invalid")
                try:
                    action = ActionEnvelope.from_dict(raw_action)
                    peer = PeerCredentials(
                        pid=_persisted_int(raw_peer.get("pid"), "peer.pid"),
                        uid=_persisted_int(raw_peer.get("uid"), "peer.uid"),
                        gid=_persisted_int(raw_peer.get("gid"), "peer.gid"),
                    )
                    submitted_at = _parse_persisted_time(
                        event.payload.get("submitted_at"),
                        "submitted_at",
                    )
                    raw_lease = event.payload.get("action_lease")
                    if raw_lease is None:
                        last_renewed_at = submitted_at
                        lease_expires_at = submitted_at + timedelta(
                            milliseconds=action.lease_ttl_ms
                        )
                    else:
                        if not isinstance(raw_lease, dict):
                            raise ValueError("action_lease must be an object")
                        last_renewed_at = _parse_persisted_time(
                            raw_lease.get("last_renewed_at"),
                            "action_lease.last_renewed_at",
                        )
                        lease_expires_at = _parse_persisted_time(
                            raw_lease.get("expires_at"),
                            "action_lease.expires_at",
                        )
                        if (
                            raw_lease.get("ttl_ms") != action.lease_ttl_ms
                            or raw_lease.get("renew_interval_ms") != action.renew_interval_ms
                            or lease_expires_at <= last_renewed_at
                        ):
                            raise ValueError("action_lease does not match action contract")
                except (KeyError, TypeError, ValueError) as exc:
                    raise LedgerIntegrityError("persisted action submission is invalid") from exc
                if action.action_id != event.entity_id:
                    raise LedgerIntegrityError(
                        "persisted action id does not match its ledger entity"
                    )
                job = _ActionJob(
                    action=action,
                    peer=peer,
                    submitted_at=submitted_at,
                    last_lease_renewed_at=last_renewed_at,
                    lease_expires_at=lease_expires_at,
                )
                continue
            if job is None:
                raise LedgerIntegrityError("persisted action transition precedes submission")
            if event.event_type == "ACTION_STARTED":
                if job.state != "QUEUED":
                    raise LedgerIntegrityError("persisted action has an invalid start transition")
                job.state = "RUNNING"
                job.started_at = _parse_persisted_time(
                    event.payload.get("started_at"),
                    "started_at",
                )
                continue
            if event.event_type == "ACTION_TERMINAL":
                scheduler_state = event.payload.get("scheduler_state")
                receipt = event.payload.get("receipt")
                if (
                    job.state not in {"QUEUED", "RUNNING"}
                    or scheduler_state not in {"FINISHED", "CANCELLED"}
                    or not isinstance(receipt, dict)
                ):
                    raise LedgerIntegrityError("persisted action terminal event is invalid")
                _validate_persisted_receipt(job.action, receipt, str(scheduler_state))
                job.state = str(scheduler_state)
                job.finished_at = _parse_persisted_time(
                    event.payload.get("finished_at"),
                    "finished_at",
                )
                job.receipt = receipt
                continue
            raise LedgerIntegrityError(f"unsupported action ledger event: {event.event_type!r}")
        if job is None:
            raise LedgerIntegrityError("persisted action has no submission event")
        return job

    def _restore_recovery_from_ledger(self) -> None:
        assert self.ledger is not None
        for event in self.ledger.events(entity_kind="RECOVERY", entity_id="rosclawd"):
            if event.event_type == "RECOVERY_REQUIRED":
                try:
                    action_ids = _persisted_string_list(
                        event.payload.get("action_ids"),
                        "action_ids",
                    )
                    real_action_ids = _persisted_string_list(
                        event.payload.get("real_action_ids"),
                        "real_action_ids",
                    )
                    _parse_persisted_time(event.payload.get("required_at"), "required_at")
                except (TypeError, ValueError) as exc:
                    raise LedgerIntegrityError("persisted recovery requirement is invalid") from exc
                if (
                    event.payload.get("reason") != "interrupted_real_action_outcome_unknown"
                    or not set(real_action_ids).issubset(action_ids)
                    or any(action_id not in self._jobs for action_id in action_ids)
                    or any(
                        self._jobs[action_id].action.execution_mode is not ExecutionMode.REAL
                        for action_id in real_action_ids
                    )
                ):
                    raise LedgerIntegrityError("persisted recovery requirement is invalid")
                if self._recovery_required and (
                    not set(self._recovery_action_ids).issubset(action_ids)
                    or not set(self._recovery_real_action_ids).issubset(real_action_ids)
                ):
                    raise LedgerIntegrityError(
                        "persisted recovery requirement discards pending review"
                    )
                self._merge_recovery_requirement_locked(action_ids, real_action_ids)
                continue
            if event.event_type == "RECOVERY_ACKNOWLEDGED":
                if not self._recovery_required:
                    raise LedgerIntegrityError(
                        "persisted recovery acknowledgement has no pending requirement"
                    )
                try:
                    acknowledged_action_ids = _persisted_string_list(
                        event.payload.get("action_ids"),
                        "action_ids",
                    )
                    acknowledged_real_action_ids = _persisted_string_list(
                        event.payload.get("real_action_ids"),
                        "real_action_ids",
                    )
                    _parse_persisted_time(
                        event.payload.get("acknowledged_at"),
                        "acknowledged_at",
                    )
                    reason = event.payload.get("reason")
                    peer = event.payload.get("peer")
                    if not isinstance(peer, dict):
                        raise ValueError("peer must be an object")
                    persisted_peer = PeerCredentials(
                        pid=_persisted_int(peer.get("pid"), "peer.pid"),
                        uid=_persisted_int(peer.get("uid"), "peer.uid"),
                        gid=_persisted_int(peer.get("gid"), "peer.gid"),
                    )
                except (TypeError, ValueError) as exc:
                    raise LedgerIntegrityError(
                        "persisted recovery acknowledgement is invalid"
                    ) from exc
                if (
                    acknowledged_action_ids != self._recovery_action_ids
                    or acknowledged_real_action_ids != self._recovery_real_action_ids
                    or not isinstance(reason, str)
                    or not reason.strip()
                    or len(reason) > 1024
                    or persisted_peer.uid != os.geteuid()
                ):
                    raise LedgerIntegrityError("persisted recovery acknowledgement is invalid")
                self._recovery_required = False
                self._recovery_action_ids = []
                self._recovery_real_action_ids = []
                continue
            raise LedgerIntegrityError(f"unsupported recovery ledger event: {event.event_type!r}")

    def _recover_incomplete_jobs_locked(self) -> None:
        if self.ledger is None:
            return
        incomplete = [job for job in self._jobs.values() if job.state in {"QUEUED", "RUNNING"}]
        unknown_real = [
            job
            for job in incomplete
            if job.state == "RUNNING" and job.action.execution_mode is ExecutionMode.REAL
        ]
        if unknown_real:
            newly_unknown = sorted(job.action.action_id for job in unknown_real)
            action_ids = sorted(set(self._recovery_action_ids).union(newly_unknown))
            real_action_ids = sorted(set(self._recovery_real_action_ids).union(newly_unknown))
            if (
                not self._recovery_required
                or action_ids != self._recovery_action_ids
                or real_action_ids != self._recovery_real_action_ids
            ):
                self.ledger.append(
                    "RECOVERY_REQUIRED",
                    entity_kind="RECOVERY",
                    entity_id="rosclawd",
                    payload={
                        "action_ids": action_ids,
                        "real_action_ids": real_action_ids,
                        "required_at": _iso(utc_now()),
                        "reason": "interrupted_real_action_outcome_unknown",
                    },
                )
            self._merge_recovery_requirement_locked(action_ids, real_action_ids)
        if self._recovery_required:
            try:
                self._emergency_stop_requests += 1
                self.runtime.request_emergency_stop(
                    "rosclawd restart has pending interrupted REAL action recovery",
                    source="rosclawd.restart_recovery",
                    timeout_sec=1.0,
                )
            except Exception:  # noqa: BLE001
                logger.exception("rosclawd restart recovery E-Stop request failed")

        for job in incomplete:
            was_running = job.state == "RUNNING"
            if was_running and job.action.execution_mode is ExecutionMode.REAL:
                code = "DAEMON_RESTART_OUTCOME_UNKNOWN"
                message = (
                    "rosclawd restarted after REAL dispatch began. The physical outcome is "
                    "unknown; E-Stop was requested and operator review is required."
                )
                final_state = ActionState.FAILED
                scheduler_state = "FINISHED"
            elif was_running:
                code = "DAEMON_RESTART_INTERRUPTED"
                message = "rosclawd restarted before the action produced a terminal receipt."
                final_state = ActionState.FAILED
                scheduler_state = "FINISHED"
            else:
                code = "DAEMON_RESTART_CANCELLED_BEFORE_DISPATCH"
                message = (
                    "rosclawd restarted while the action was queued; no executor dispatch "
                    "is claimed."
                )
                final_state = ActionState.CANCELLED
                scheduler_state = "CANCELLED"
            receipt = self.runtime.action_gateway.reject(
                job.action,
                code=code,
                message=message,
                state=final_state,
            )
            job.state = scheduler_state
            job.finished_at = utc_now()
            job.receipt = receipt.to_dict()
            self._persist_terminal_job(job)

    def _merge_recovery_requirement_locked(
        self,
        action_ids: list[str],
        real_action_ids: list[str],
    ) -> None:
        self._recovery_required = True
        self._recovery_action_ids = sorted(set(self._recovery_action_ids).union(action_ids))
        self._recovery_real_action_ids = sorted(
            set(self._recovery_real_action_ids).union(real_action_ids)
        )

    def _evict_terminal_jobs_locked(self, *, reserve_slot: bool = True) -> None:
        """Bound memory while durable history retains action-ID replay protection."""

        target = max(0, self._max_retained_actions - (1 if reserve_slot else 0))
        if len(self._jobs) <= target:
            return
        for action_id, job in list(self._jobs.items()):
            if len(self._jobs) <= target:
                break
            if job.state not in {"FINISHED", "CANCELLED"}:
                continue
            if job.action.execution_mode is ExecutionMode.REAL and self.ledger is None:
                continue
            self._jobs.pop(action_id, None)
            discard = getattr(self.runtime.action_gateway, "discard_receipt", None)
            if callable(discard):
                discard(action_id)
            self._evicted_actions += 1

    @staticmethod
    def _require_job_owner(
        job: _ActionJob,
        peer: PeerCredentials | None,
    ) -> None:
        if peer is None or peer.uid in {job.peer.uid, os.geteuid()}:
            return
        raise ControlPlaneError(
            "ACTION_OWNERSHIP_MISMATCH",
            "The authenticated Unix peer does not own this action.",
        )


def _iso(value: datetime | None) -> str | None:
    return value.isoformat().replace("+00:00", "Z") if value is not None else None


def _parse_persisted_time(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"persisted {field} must be an ISO timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"persisted {field} must be timezone-aware")
    return parsed.astimezone(UTC)


def _persisted_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"persisted {field} must be a non-negative integer")
    return value


def _persisted_string_list(value: Any, field: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and bool(item) for item in value)
        or value != sorted(set(value))
    ):
        raise ValueError(f"persisted {field} must be a sorted unique non-empty string list")
    return list(value)


def _validate_persisted_receipt(
    action: ActionEnvelope,
    receipt: dict[str, Any],
    scheduler_state: str,
) -> None:
    expected = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "action_id": action.action_id,
        "mode": action.execution_mode.value,
        "execution_mode": action.execution_mode.value,
        "body_id": action.body_id,
        "body_snapshot_hash": action.body_snapshot_hash,
        "capability_id": action.capability_id,
    }
    if any(receipt.get(field) != value for field, value in expected.items()):
        raise LedgerIntegrityError("persisted receipt does not match its immutable action")
    try:
        raw_final_state = receipt.get("final_state")
        raw_evidence_level = receipt.get("evidence_level")
        if not isinstance(raw_final_state, str) or not isinstance(raw_evidence_level, str):
            raise ValueError("receipt enum fields must be strings")
        final_state = ActionState(raw_final_state)
        evidence_level = EvidenceLevel(raw_evidence_level)
        started_at = _parse_persisted_time(receipt.get("started_at"), "receipt.started_at")
        finished_at = _parse_persisted_time(receipt.get("finished_at"), "receipt.finished_at")
    except (TypeError, ValueError) as exc:
        raise LedgerIntegrityError("persisted receipt has invalid typed fields") from exc
    if final_state not in {
        ActionState.COMPLETED,
        ActionState.BLOCKED,
        ActionState.FAILED,
        ActionState.CANCELLED,
        ActionState.TIMED_OUT,
        ActionState.DEGRADED,
    }:
        raise LedgerIntegrityError("persisted receipt is not terminal")
    if scheduler_state == "CANCELLED" and final_state is not ActionState.CANCELLED:
        raise LedgerIntegrityError("persisted cancellation receipt is inconsistent")
    if finished_at < started_at:
        raise LedgerIntegrityError("persisted receipt timestamps are inconsistent")
    if not isinstance(receipt.get("dispatch_result"), dict) or not isinstance(
        receipt.get("errors"), list
    ):
        raise LedgerIntegrityError("persisted receipt evidence fields are invalid")

    verified = action.execution_mode is not ExecutionMode.FIXTURE and evidence_level in {
        EvidenceLevel.PHYSICALLY_OBSERVED,
        EvidenceLevel.TASK_VERIFIED,
    }
    if action.execution_mode is ExecutionMode.FIXTURE:
        trust_level = "SYNTHETIC"
    elif action.execution_mode is ExecutionMode.REPLAY:
        trust_level = "RECORDED"
    elif action.execution_mode is ExecutionMode.SIMULATION:
        trust_level = "SIMULATED"
    elif verified:
        trust_level = "VERIFIED"
    else:
        trust_level = "UNVERIFIED"
    if (
        receipt.get("verified") is not verified
        or receipt.get("trust_level") != trust_level
        or receipt.get("usable_for_real_execution")
        is not (
            action.execution_mode is ExecutionMode.REAL
            and final_state is ActionState.COMPLETED
            and verified
        )
    ):
        raise LedgerIntegrityError("persisted receipt trust fields are inconsistent")


__all__ = ["ControlPlaneError", "DaemonControlPlane", "SupervisionState"]
