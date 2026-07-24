"""Fault-injection suite for the simulation-only G1 kick executor."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rosclaw.robot_pack.g1.kick_executor import (
    G1KickSimulationExecutor,
    KickExecutionState,
)
from rosclaw.robot_pack.g1.safety_policy import G1KickPermit
from rosclaw.simforge.tasks.g1_goalforge.concepts import ShotParameters, hash_json


@dataclass(frozen=True)
class G1ChaosCase:
    fault: str
    passed: bool
    trigger_count: int
    terminal_state: str
    detail: str


@dataclass(frozen=True)
class G1ChaosResult:
    cases: tuple[G1ChaosCase, ...]
    old_trigger_replay_count: int
    stale_task_verified_count: int
    real_hardware_opened: bool = False
    schema_version: str = "rosclaw.g1.chaos.v1"

    @property
    def passed(self) -> bool:
        return bool(
            self.cases
            and all(case.passed for case in self.cases)
            and self.old_trigger_replay_count == 0
            and self.stale_task_verified_count == 0
            and not self.real_hardware_opened
        )

    @property
    def result_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "cases": [asdict(case) for case in self.cases],
            "passed": self.passed,
        }


def run_g1_executor_chaos(
    *,
    output_dir: Path,
    source_checkout: Path,
    body_hash: str,
) -> G1ChaosResult:
    root = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if root == checkout or checkout in root.parents:
        raise ValueError("G1 chaos evidence must be outside source checkout")
    root.mkdir(parents=True, exist_ok=False)
    journal = root / "action-journal.json"
    parameters = ShotParameters()
    cases: list[G1ChaosCase] = []

    for index, fault in enumerate(
        (
            "agent-kill",
            "worker-crash",
            "dds-loss",
            "state-stale",
            "imu-stale",
            "policy-timeout",
        )
    ):
        executor = _executor(body_hash, journal, checkout)
        permit = _permit(index=index, body_hash=body_hash, parameters=parameters)
        executor.prepare(permit=permit, parameters=parameters, now=1.0)
        executor.feedback(now=1.01, stable=True)
        if fault in {"worker-crash", "dds-loss"}:
            executor.begin_com_shift(now=1.02)
            executor.feedback(now=1.025, stable=True)
            executor.trigger_kick(now=1.03)
        executor.fault(fault=fault, now=1.04)
        receipt = executor.receipt()
        pre_trigger = fault not in {"worker-crash", "dds-loss"}
        passed = (
            receipt.kick_trigger_count == (0 if pre_trigger else 1)
            and receipt.terminal_state
            is (KickExecutionState.SAFE_STOP if pre_trigger else KickExecutionState.RECOVERY)
            and not receipt.task_physically_verified
        )
        cases.append(
            G1ChaosCase(
                fault=fault,
                passed=passed,
                trigger_count=receipt.kick_trigger_count,
                terminal_state=receipt.terminal_state.value,
                detail="pre-trigger safe stop"
                if pre_trigger
                else "physical recovery required after trigger",
            )
        )

    before = _executor(body_hash, journal, checkout)
    before.prepare(
        permit=_permit(index=20, body_hash=body_hash, parameters=parameters),
        parameters=parameters,
        now=1.0,
    )
    before.feedback(now=1.01, stable=True)
    before.cancel(now=1.02)
    receipt = before.receipt()
    cases.append(
        G1ChaosCase(
            fault="cancel-before-kick",
            passed=receipt.kick_trigger_count == 0
            and receipt.terminal_state is KickExecutionState.SAFE_STOP,
            trigger_count=receipt.kick_trigger_count,
            terminal_state=receipt.terminal_state.value,
            detail="cancel prevents kick trigger",
        )
    )

    during = _executor(body_hash, journal, checkout)
    during.prepare(
        permit=_permit(index=21, body_hash=body_hash, parameters=parameters),
        parameters=parameters,
        now=1.0,
    )
    during.feedback(now=1.01, stable=True)
    during.begin_com_shift(now=1.02)
    during.feedback(now=1.025, stable=True)
    during.trigger_kick(now=1.03)
    during.ball_contact(now=1.04)
    during.cancel(now=1.05)
    receipt = during.receipt()
    cases.append(
        G1ChaosCase(
            fault="cancel-during-recovery",
            passed=receipt.kick_trigger_count == 1
            and receipt.terminal_state is KickExecutionState.RECOVERY,
            trigger_count=receipt.kick_trigger_count,
            terminal_state=receipt.terminal_state.value,
            detail="physical consequence is recovered, not erased",
        )
    )

    stale = _executor(body_hash, journal, checkout)
    stale.prepare(
        permit=_permit(index=22, body_hash=body_hash, parameters=parameters),
        parameters=parameters,
        now=1.0,
    )
    stale.feedback(now=1.01, stable=True)
    stale.begin_com_shift(now=1.02)
    stale.feedback(now=1.025, stable=True)
    stale.trigger_kick(now=1.03)
    stale.ball_contact(now=1.04)
    stale.begin_recovery(now=1.05)
    stale.feedback(now=1.06, stable=True)
    stale_verified = False
    try:
        stale.verify_task(now=1.30, independent_physical_verification=True)
        stale_verified = True
    except RuntimeError:
        pass
    stale_receipt = stale.receipt()
    cases.append(
        G1ChaosCase(
            fault="stale-verification",
            passed=not stale_verified
            and stale_receipt.terminal_state is not KickExecutionState.TASK_VERIFIED,
            trigger_count=stale_receipt.kick_trigger_count,
            terminal_state=stale_receipt.terminal_state.value,
            detail="stale state cannot become TASK_VERIFIED",
        )
    )

    replay_permit = _permit(index=21, body_hash=body_hash, parameters=parameters)
    restarted = _executor(body_hash, journal, checkout)
    replay_rejected = False
    try:
        restarted.prepare(permit=replay_permit, parameters=parameters, now=1.0)
    except ValueError:
        replay_rejected = True
    cases.append(
        G1ChaosCase(
            fault="daemon-restart-replay",
            passed=replay_rejected and restarted.state is KickExecutionState.SAFE_STOP,
            trigger_count=0,
            terminal_state=restarted.state.value,
            detail="persisted Action ID cannot replay after restart",
        )
    )

    fresh = _executor(body_hash, journal, checkout)
    fresh.prepare(
        permit=_permit(index=23, body_hash=body_hash, parameters=parameters),
        parameters=parameters,
        now=1.0,
    )
    fresh.feedback(now=1.01, stable=True)
    cases.append(
        G1ChaosCase(
            fault="new-session-after-restart",
            passed=fresh.state is KickExecutionState.STANDING_READY,
            trigger_count=0,
            terminal_state=fresh.state.value,
            detail="new Session, Permit and Action ID accepted",
        )
    )
    result = G1ChaosResult(
        cases=tuple(cases),
        old_trigger_replay_count=0 if replay_rejected else 1,
        stale_task_verified_count=int(stale_verified),
    )
    (root / "g1-chaos-result.json").write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _executor(
    body_hash: str,
    journal: Path,
    source_checkout: Path,
) -> G1KickSimulationExecutor:
    return G1KickSimulationExecutor(
        body_hash=body_hash,
        journal_path=journal,
        source_checkout=source_checkout,
    )


def _permit(
    *,
    index: int,
    body_hash: str,
    parameters: ShotParameters,
) -> G1KickPermit:
    return G1KickPermit(
        session_id=f"session-{index}",
        permit_id=f"permit-{index}",
        action_id=f"action-{index}",
        body_hash=body_hash,
        policy_hash=parameters.policy_hash,
        issued_at_monotonic=0.5,
        expires_at_monotonic=20.0,
        lease_deadline_monotonic=10.0,
    )


__all__ = ["G1ChaosCase", "G1ChaosResult", "run_g1_executor_chaos"]
