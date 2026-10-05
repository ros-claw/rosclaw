"""A scheduler FINISHED state includes failed tasks and is never success evidence."""

from types import SimpleNamespace

import pytest

from rosclaw.agentd.action_channel import ActionChannelError, DaemonActionChannel
from rosclaw.kernel.contracts import ExecutionMode


def outcome(patch=None):
    envelope = SimpleNamespace(
        execution_mode=ExecutionMode.SIMULATION,
        body_id="base", body_snapshot_hash="bound", capability_id="coverage.execute",
    )
    receipt = {
        "action_id": "action", "final_state": "COMPLETED", "verified": True,
        "trust_level": "SIMULATED", "evidence_level": "TASK_VERIFIED",
        "body_id": "base", "body_snapshot_hash": "bound",
        "capability_id": "coverage.execute", "execution_mode": "SIMULATION",
    }
    receipt.update(patch or {})
    channel = DaemonActionChannel(None, actor_id="agent", body_id="base", body_hash="bound")
    return channel._verify_outcome("action", {"state": "FINISHED"}, {"receipt": receipt}, envelope)


def test_canonical_completed_receipt_uses_action_state():
    result = outcome()
    assert result.verified
    assert result.state == "COMPLETED"


@pytest.mark.parametrize("patch", [
    {"final_state": "FAILED", "verified": False, "evidence_level": "REQUESTED"},
    {"final_state": "DEGRADED"}, {"final_state": "CANCELLED"},
    {"verified": False}, {"verified": "true"},
    {"evidence_level": "COMMAND_DISPATCHED"},
    {"body_id": "other"}, {"body_snapshot_hash": "other"},
    {"capability_id": "other"}, {"execution_mode": "REAL"},
])
def test_finished_scheduler_does_not_verify_failed_or_mismatched_receipt(patch):
    assert not outcome(patch).verified


@pytest.mark.parametrize("action", [None, "other"])
def test_receipt_requires_exact_action_binding(action):
    with pytest.raises(ActionChannelError):
        outcome({"action_id": action})
