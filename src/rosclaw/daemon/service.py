"""Daemon-owned action queue, permit gate, E-Stop, leases, and receipts."""

from __future__ import annotations

import contextlib
import logging
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from rosclaw.daemon.ledger import (
    DaemonLedger,
    LedgerError,
    LedgerEvent,
    LedgerIntegrityError,
)
from rosclaw.daemon.permits import PermitAuthority
from rosclaw.daemon.protocol import DAEMON_PROTOCOL_VERSION, PeerCredentials
from rosclaw.kernel import (
    RECEIPT_SCHEMA_VERSION,
    ActionEnvelope,
    ActionState,
    EvidenceLevel,
    ExecutionMode,
)
from rosclaw.kernel.contracts import utc_now

logger = logging.getLogger("rosclaw.daemon.service")


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
        max_workers: int = 4,
        max_queued_actions: int = 64,
        max_retained_actions: int = 1024,
    ):
        self.runtime = runtime
        self.ledger = ledger
        if permits is None:
            permits = PermitAuthority(ledger=ledger)
        elif ledger is not None and permits.ledger is not ledger:
            raise ValueError("DaemonControlPlane permit authority must use the same ledger")
        self.permits = permits
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
        self._running = False
        self._closed = False
        self._hardware_actions_executed = 0
        self._emergency_stop_requests = 0
        self._recovery_required = False
        self._recovery_action_ids: list[str] = []
        self._recovery_real_action_ids: list[str] = []
        self._ledger_write_failed = False
        self._ledger_failure: str | None = None
        if self.ledger is not None:
            self._restore_jobs_from_ledger()
            self._restore_recovery_from_ledger()

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("DaemonControlPlane cannot restart after close")
            if self._running:
                return
            self._recover_incomplete_jobs_locked()
            self._evict_terminal_jobs_locked(reserve_slot=False)
            self._running = True

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
                job = _ActionJob(action=action, peer=peer)
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
            queued = [
                job
                for job in self._jobs.values()
                if job.state == "QUEUED" and job.future is not None
            ]
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
        self._executor.shutdown(wait=True, cancel_futures=True)
        with contextlib.suppress(Exception):
            self.runtime.stop()

    def _run_action(self, job: _ActionJob) -> dict[str, Any]:
        with self._lock:
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
        is not (action.execution_mode is ExecutionMode.REAL and verified)
    ):
        raise LedgerIntegrityError("persisted receipt trust fields are inconsistent")


__all__ = ["ControlPlaneError", "DaemonControlPlane"]
