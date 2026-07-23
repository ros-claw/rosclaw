"""Independent checks for G1 DDS execution receipts."""

from __future__ import annotations

from dataclasses import dataclass

from rosclaw.robot_pack.g1.kick_executor import (
    G1KickExecutionReceipt,
    KickExecutionState,
)


@dataclass(frozen=True)
class G1ExecutionVerification:
    valid: bool
    no_trigger_replay: bool
    feedback_fresh: bool
    task_verified: bool
    errors: tuple[str, ...]


def verify_execution_receipt(
    receipt: G1KickExecutionReceipt,
) -> G1ExecutionVerification:
    errors: list[str] = []
    no_replay = receipt.kick_trigger_count <= 1
    if not no_replay:
        errors.append("kick_trigger_replayed")
    fresh = not receipt.stale_feedback_observed
    task_verified = (
        receipt.terminal_state is KickExecutionState.TASK_VERIFIED
        and receipt.task_physically_verified
    )
    if receipt.terminal_state is KickExecutionState.TASK_VERIFIED and not fresh:
        errors.append("stale_state_marked_verified")
    if task_verified and receipt.kick_trigger_count != 1:
        errors.append("verified_without_exactly_one_trigger")
    return G1ExecutionVerification(
        valid=not errors,
        no_trigger_replay=no_replay,
        feedback_fresh=fresh,
        task_verified=task_verified,
        errors=tuple(errors),
    )


__all__ = ["G1ExecutionVerification", "verify_execution_receipt"]
