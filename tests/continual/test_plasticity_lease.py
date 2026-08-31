from __future__ import annotations

import pytest

from rosclaw.continual import (
    AgentPolicyBinding,
    AgentUpdateMode,
    PlasticityLease,
    audit_plasticity_lease,
)

_A = "sha256:" + "a" * 64
_B = "sha256:" + "b" * 64
_C = "sha256:" + "c" * 64


def _lease() -> PlasticityLease:
    return PlasticityLease(
        lease_id="three-player-cycle-7",
        bindings=(
            AgentPolicyBinding("passer", _A, AgentUpdateMode.FROZEN),
            AgentPolicyBinding("finisher", _B, AgentUpdateMode.PLASTIC),
            AgentPolicyBinding("goalkeeper", _C, AgentUpdateMode.FROZEN),
        ),
        dataset_manifest_hash=_A,
        scenario_contract_hash=_B,
        maximum_optimizer_steps=500,
    )


def test_lease_requires_exactly_one_plastic_agent() -> None:
    with pytest.raises(ValueError, match="plasticity lease"):
        PlasticityLease(
            lease_id="bad",
            bindings=(
                AgentPolicyBinding("passer", _A, AgentUpdateMode.PLASTIC),
                AgentPolicyBinding("finisher", _B, AgentUpdateMode.PLASTIC),
            ),
            dataset_manifest_hash=_A,
            scenario_contract_hash=_B,
            maximum_optimizer_steps=1,
        )


def test_only_focal_agent_may_change_inside_lease() -> None:
    lease = _lease()
    before = {"passer": _A, "finisher": _B, "goalkeeper": _C}
    after = {**before, "finisher": _C}
    audit = audit_plasticity_lease(
        lease=lease,
        optimizer_steps=400,
        before_policy_hashes=before,
        after_policy_hashes=after,
    )
    assert audit.passed
    assert audit.changed_agent_ids == ("finisher",)


def test_frozen_partner_drift_and_step_overrun_fail_closed() -> None:
    lease = _lease()
    before = {"passer": _A, "finisher": _B, "goalkeeper": _C}
    after = {**before, "passer": _B}
    audit = audit_plasticity_lease(
        lease=lease,
        optimizer_steps=501,
        before_policy_hashes=before,
        after_policy_hashes=after,
    )
    assert not audit.passed
    assert audit.reasons == (
        "OPTIMIZER_STEP_BUDGET_EXCEEDED",
        "FROZEN_AGENT_CHANGED:passer",
    )
