"""Fail-closed canonical state machine for simulated G1 kick execution."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from rosclaw.robot_pack.g1.safety_policy import G1KickPermit, G1KickSafetyPolicy
from rosclaw.simforge.tasks.g1_goalforge.concepts import (
    ShotParameters,
    hash_json,
)


class KickExecutionState(StrEnum):
    IDLE = "IDLE"
    PREPARE = "PREPARE"
    STANDING_READY = "STANDING_READY"
    COM_SHIFT = "COM_SHIFT"
    KICK_TRIGGERED = "KICK_TRIGGERED"
    BALL_CONTACTED = "BALL_CONTACTED"
    RECOVERY = "RECOVERY"
    STABLE = "STABLE"
    TASK_VERIFIED = "TASK_VERIFIED"
    SAFE_STOP = "SAFE_STOP"
    FAILED = "FAILED"


@dataclass(frozen=True)
class ExecutorEvent:
    sequence: int
    state: KickExecutionState
    event: str
    monotonic_time: float
    details: tuple[tuple[str, str | float | bool], ...] = ()


@dataclass(frozen=True)
class G1KickExecutionReceipt:
    session_id: str
    permit_id: str
    action_id: str
    body_hash: str
    policy_hash: str
    terminal_state: KickExecutionState
    kick_trigger_count: int
    stale_feedback_observed: bool
    duplicate_rejected: bool
    task_physically_verified: bool
    events: tuple[ExecutorEvent, ...]
    evidence_domain: str = "SHADOW"
    schema_version: str = "rosclaw.g1.kick_execution_receipt.v1"

    @property
    def receipt_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["terminal_state"] = self.terminal_state.value
        value["events"] = [
            {
                **asdict(event),
                "state": event.state.value,
                "details": dict(event.details),
            }
            for event in self.events
        ]
        return value


class G1KickSimulationExecutor:
    """A deterministic executor; transport I/O is injected at the boundary."""

    def __init__(
        self,
        *,
        body_hash: str,
        journal_path: Path,
        source_checkout: Path,
        stale_after_sec: float = 0.10,
    ) -> None:
        journal = journal_path.expanduser().resolve()
        checkout = source_checkout.expanduser().resolve()
        if journal == checkout or checkout in journal.parents:
            raise ValueError("G1 executor journal must be outside source checkout")
        if not 0.02 <= stale_after_sec <= 1.0:
            raise ValueError("G1 executor stale threshold must be in [0.02, 1.0]")
        self.body_hash = body_hash
        self.journal_path = journal
        self.stale_after_sec = stale_after_sec
        self.safety = G1KickSafetyPolicy()
        self.state = KickExecutionState.IDLE
        self._permit: G1KickPermit | None = None
        self._parameters: ShotParameters | None = None
        self._events: list[ExecutorEvent] = []
        self._trigger_count = 0
        self._last_feedback: float | None = None
        self._stale_observed = False
        self._duplicate_rejected = False
        self._physically_verified = False
        self._used_actions = self._load_used_actions()

    def prepare(
        self,
        *,
        permit: G1KickPermit,
        parameters: ShotParameters,
        now: float,
    ) -> None:
        if self.state is not KickExecutionState.IDLE:
            raise RuntimeError("executor is not idle")
        if permit.action_id in self._used_actions:
            self._duplicate_rejected = True
            self._record("duplicate_action_rejected", now)
            self.state = KickExecutionState.SAFE_STOP
            raise ValueError("duplicate or replayed G1 action id")
        valid, errors = self.safety.validate(
            permit=permit,
            parameters=parameters,
            expected_body_hash=self.body_hash,
            now_monotonic=now,
        )
        if not valid:
            self.state = KickExecutionState.SAFE_STOP
            self._record("permit_rejected", now, errors=",".join(errors))
            raise PermissionError("G1 kick permit rejected: " + ",".join(errors))
        self._permit = permit
        self._parameters = parameters
        self.state = KickExecutionState.PREPARE
        self._record("prepared", now)

    def feedback(self, *, now: float, stable: bool, dds_connected: bool = True) -> None:
        if not dds_connected:
            self._stale_observed = True
            self._safe_fault("dds_lost", now)
            return
        self._last_feedback = now
        if self.state is KickExecutionState.PREPARE and stable:
            self.state = KickExecutionState.STANDING_READY
            self._record("standing_ready", now)
        elif self.state is KickExecutionState.RECOVERY and stable:
            self.state = KickExecutionState.STABLE
            self._record("post_kick_stable", now)

    def begin_com_shift(self, *, now: float) -> None:
        self._require_state(KickExecutionState.STANDING_READY)
        self._require_fresh(now)
        self._require_lease(now)
        self.state = KickExecutionState.COM_SHIFT
        self._record("com_shift", now)

    def trigger_kick(self, *, now: float) -> None:
        self._require_state(KickExecutionState.COM_SHIFT)
        self._require_fresh(now)
        self._require_lease(now)
        self._trigger_count += 1
        if self._trigger_count != 1:
            self.state = KickExecutionState.SAFE_STOP
            raise RuntimeError("kick trigger may be emitted only once")
        self.state = KickExecutionState.KICK_TRIGGERED
        self._persist_action()
        self._record("kick_triggered", now)

    def ball_contact(self, *, now: float) -> None:
        self._require_state(KickExecutionState.KICK_TRIGGERED)
        self.state = KickExecutionState.BALL_CONTACTED
        self._record("ball_contacted", now)

    def begin_recovery(self, *, now: float) -> None:
        if self.state not in {
            KickExecutionState.KICK_TRIGGERED,
            KickExecutionState.BALL_CONTACTED,
        }:
            raise RuntimeError("recovery requires a triggered physical kick")
        self.state = KickExecutionState.RECOVERY
        self._record("recovery", now)

    def cancel(self, *, now: float) -> None:
        if self.state in {
            KickExecutionState.PREPARE,
            KickExecutionState.STANDING_READY,
            KickExecutionState.COM_SHIFT,
        }:
            self.state = KickExecutionState.SAFE_STOP
            self._record("cancelled_before_trigger", now)
            return
        if self.state in {
            KickExecutionState.KICK_TRIGGERED,
            KickExecutionState.BALL_CONTACTED,
            KickExecutionState.RECOVERY,
        }:
            self.state = KickExecutionState.RECOVERY
            self._record("cancel_deferred_until_physical_recovery", now)
            return
        self._record("cancel_noop", now)

    def fault(self, *, fault: str, now: float) -> None:
        supported = {
            "agent-kill",
            "worker-crash",
            "dds-loss",
            "state-stale",
            "imu-stale",
            "policy-timeout",
        }
        if fault not in supported:
            raise ValueError(f"unsupported G1 executor fault: {fault}")
        if fault in {"state-stale", "imu-stale", "dds-loss"}:
            self._stale_observed = True
        self._safe_fault(fault, now)

    def verify_task(
        self,
        *,
        now: float,
        independent_physical_verification: bool,
    ) -> None:
        self._require_state(KickExecutionState.STABLE)
        self._require_fresh(now)
        if not independent_physical_verification or self._stale_observed:
            self.state = KickExecutionState.FAILED
            self._record("task_verification_rejected", now)
            raise RuntimeError("unknown/stale physical state cannot be TASK_VERIFIED")
        self._physically_verified = True
        self.state = KickExecutionState.TASK_VERIFIED
        self._record("task_verified", now)

    def receipt(self) -> G1KickExecutionReceipt:
        if self._permit is None or self._parameters is None:
            raise RuntimeError("executor has no prepared action")
        return G1KickExecutionReceipt(
            session_id=self._permit.session_id,
            permit_id=self._permit.permit_id,
            action_id=self._permit.action_id,
            body_hash=self.body_hash,
            policy_hash=self._parameters.policy_hash,
            terminal_state=self.state,
            kick_trigger_count=self._trigger_count,
            stale_feedback_observed=self._stale_observed,
            duplicate_rejected=self._duplicate_rejected,
            task_physically_verified=self._physically_verified,
            events=tuple(self._events),
        )

    def _safe_fault(self, fault: str, now: float) -> None:
        if self._trigger_count:
            self.state = KickExecutionState.RECOVERY
            self._record(f"{fault}_recovery_required", now)
        else:
            self.state = KickExecutionState.SAFE_STOP
            self._record(f"{fault}_safe_stop", now)

    def _require_fresh(self, now: float) -> None:
        if self._last_feedback is None or now - self._last_feedback > self.stale_after_sec:
            self._stale_observed = True
            self.state = KickExecutionState.SAFE_STOP
            self._record("feedback_stale", now)
            raise RuntimeError("G1 LowState/IMU feedback is stale")

    def _require_lease(self, now: float) -> None:
        if self._permit is None or now >= self._permit.lease_deadline_monotonic:
            self.state = KickExecutionState.SAFE_STOP
            self._record("lease_expired", now)
            raise RuntimeError("G1 action lease expired")

    def _require_state(self, required: KickExecutionState) -> None:
        if self.state is not required:
            raise RuntimeError(f"expected {required.value}, got {self.state.value}")

    def _record(
        self,
        event: str,
        now: float,
        **details: str | float | bool,
    ) -> None:
        self._events.append(
            ExecutorEvent(
                sequence=len(self._events),
                state=self.state,
                event=event,
                monotonic_time=now,
                details=tuple(sorted(details.items())),
            )
        )

    def _load_used_actions(self) -> set[str]:
        if not self.journal_path.exists():
            return set()
        value = json.loads(self.journal_path.read_text(encoding="utf-8"))
        if value.get("schema_version") != "rosclaw.g1.action_journal.v1":
            raise ValueError("unsupported G1 action journal")
        actions = value.get("used_action_ids")
        if not isinstance(actions, list) or any(not isinstance(item, str) for item in actions):
            raise ValueError("G1 action journal is malformed")
        return set(actions)

    def _persist_action(self) -> None:
        if self._permit is None:
            raise RuntimeError("cannot persist an absent G1 permit")
        self._used_actions.add(self._permit.action_id)
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.journal_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "schema_version": "rosclaw.g1.action_journal.v1",
                    "used_action_ids": sorted(self._used_actions),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.journal_path)


__all__ = [
    "ExecutorEvent",
    "G1KickExecutionReceipt",
    "G1KickSimulationExecutor",
    "KickExecutionState",
]
