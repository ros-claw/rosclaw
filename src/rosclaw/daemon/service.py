"""Daemon-owned action queue, permit gate, E-Stop, leases, and receipts."""

from __future__ import annotations

import contextlib
import logging
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from rosclaw.daemon.permits import PermitAuthority
from rosclaw.daemon.protocol import DAEMON_PROTOCOL_VERSION, PeerCredentials
from rosclaw.kernel import ActionEnvelope, ActionState, ExecutionMode
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
        max_workers: int = 4,
        max_queued_actions: int = 64,
        max_retained_actions: int = 1024,
    ):
        self.runtime = runtime
        self.permits = permits or PermitAuthority()
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

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("DaemonControlPlane cannot restart after close")
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
                "queue": counts,
                "history": {
                    "retained": len(self._jobs),
                    "capacity": self._max_retained_actions,
                    "evicted": self._evicted_actions,
                },
                "permits": self.permits.status(),
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

        with self._lock:
            if not self._running or self._closed:
                raise ControlPlaneError("DAEMON_STOPPING", "rosclawd is not accepting actions")
            existing = self._jobs.get(action.action_id)
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
            if not self._queue_slots.acquire(blocking=False):
                raise ControlPlaneError("ACTION_QUEUE_FULL", "rosclawd action queue is full")
            try:
                self._evict_terminal_non_real_jobs_locked()
                if len(self._jobs) >= self._max_retained_actions:
                    raise ControlPlaneError(
                        "ACTION_HISTORY_FULL",
                        (
                            "rosclawd retained-action capacity is full. REAL action "
                            "records are never evicted in-process; archive evidence and "
                            "restart the daemon under operator control."
                        ),
                    )
                job = _ActionJob(action=action, peer=peer)
                self._jobs[action.action_id] = job
                job.future = self._executor.submit(self._run_action, job)
                job.future.add_done_callback(lambda _future: self._queue_slots.release())
            except Exception:
                self._jobs.pop(action.action_id, None)
                self._queue_slots.release()
                raise
            return job.to_dict()

    def get_action_status(
        self,
        action_id: str,
        peer: PeerCredentials | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(action_id)
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
            job = self._jobs.get(action_id)
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
            self.cancel_action(job.action.action_id)

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

        action = job.action
        try:
            if action.execution_mode is ExecutionMode.REAL:
                decision = self.permits.authorize(action, job.peer)
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
            if (
                action.execution_mode is ExecutionMode.REAL
                and receipt_payload.get("dispatch_result", {}).get("accepted") is True
            ):
                self._hardware_actions_executed += 1
        return receipt_payload

    def _evict_terminal_non_real_jobs_locked(self) -> None:
        """Bound memory without making a REAL action ID executable twice."""

        if len(self._jobs) < self._max_retained_actions:
            return
        for action_id, job in list(self._jobs.items()):
            if len(self._jobs) < self._max_retained_actions:
                break
            if (
                job.state not in {"FINISHED", "CANCELLED"}
                or job.action.execution_mode is ExecutionMode.REAL
            ):
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


__all__ = ["ControlPlaneError", "DaemonControlPlane"]
