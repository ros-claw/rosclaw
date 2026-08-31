from __future__ import annotations

from dataclasses import replace

import pytest

from rosclaw.continual.individual_scope import (
    FrozenPartner,
    FrozenPartnerSet,
    IndividualGrowthScope,
    IndividualPromotionEvidence,
)
from tests.continual.helpers import digest, policy


def _scope(agent_id: str = "agent.goalkeeper") -> IndividualGrowthScope:
    champion, _ = policy(0)
    return IndividualGrowthScope(
        agent_id=agent_id,
        body_hash=champion.body_hash,
        body_state_hash=digest(f"body-state:{agent_id}"),
        foundation_policy_hash=digest("foundation"),
        personal_adapter_hash=digest(f"adapter:{agent_id}"),
        role_policy_hash=digest(f"role:{agent_id}"),
        residual_policy_hash=digest(f"residual:{agent_id}"),
        capability_profile_hash=digest(f"capability:{agent_id}"),
        career_lineage_hash=digest(f"career:{agent_id}"),
        personal_memory_namespace=f"memory.{agent_id}",
        failure_memory_namespace=f"failure.{agent_id}",
        parent_policy=champion,
        champion_policy=champion,
    )


def _partners(scope: IndividualGrowthScope) -> FrozenPartnerSet:
    return FrozenPartnerSet(
        focal_agent_id=scope.agent_id,
        partners=(
            FrozenPartner(
                agent_id="agent.finisher",
                champion_policy_hash=digest("finisher"),
                body_hash=digest("finisher-body"),
                foundation_policy_hash=scope.foundation_policy_hash,
                capability_profile_hash=digest("finisher-capability"),
            ),
        ),
        numerical_contract_hash=digest("numerical"),
        scenario_contract_hash=digest("scenario"),
    )


def test_individual_candidate_is_bound_to_focal_agent_and_frozen_partners() -> None:
    scope = _scope()
    candidate, _ = policy(1, parent=scope.champion_policy)
    partners = _partners(scope)

    staged = scope.stage_candidate(candidate, partners=partners)

    assert staged.candidate_policy == candidate
    assert staged.candidate_partner_snapshot_hash == partners.snapshot_hash
    with pytest.raises(ValueError, match="another focal agent"):
        scope.stage_candidate(
            candidate,
            partners=replace(partners, focal_agent_id="agent.playmaker"),
        )


def test_promotion_updates_only_the_individual_scope() -> None:
    scope = _scope()
    teammate = _scope("agent.playmaker")
    candidate, _ = policy(1, parent=scope.champion_policy)
    staged = scope.stage_candidate(candidate, partners=_partners(scope))
    evidence = IndividualPromotionEvidence(
        agent_id=scope.agent_id,
        parent_policy_hash=scope.champion_policy.version_hash,
        candidate_policy_hash=candidate.version_hash,
        frozen_partner_snapshot_hash=staged.candidate_partner_snapshot_hash or "",
        matched_seed_commitment_hash=digest("seeds"),
        gate_report_hash=digest("gate"),
        personal_memory_namespace=scope.personal_memory_namespace,
        retention_passed=True,
        safety_passed=True,
        team_compatibility_passed=True,
    )

    promoted = staged.promote_candidate(evidence)

    assert promoted.generation == 1
    assert promoted.champion_policy == candidate
    assert promoted.candidate_policy is None
    assert teammate.generation == 0
    assert teammate.champion_policy.version_hash != promoted.champion_policy.version_hash


def test_failed_or_cross_agent_evidence_cannot_promote() -> None:
    scope = _scope()
    candidate, _ = policy(1, parent=scope.champion_policy)
    staged = scope.stage_candidate(candidate, partners=_partners(scope))
    evidence = IndividualPromotionEvidence(
        agent_id=scope.agent_id,
        parent_policy_hash=scope.champion_policy.version_hash,
        candidate_policy_hash=candidate.version_hash,
        frozen_partner_snapshot_hash=staged.candidate_partner_snapshot_hash or "",
        matched_seed_commitment_hash=digest("seeds"),
        gate_report_hash=digest("gate"),
        personal_memory_namespace=scope.personal_memory_namespace,
        retention_passed=True,
        safety_passed=False,
        team_compatibility_passed=True,
    )

    with pytest.raises(ValueError, match="failed"):
        staged.promote_candidate(evidence)
    with pytest.raises(ValueError, match="another agent"):
        staged.promote_candidate(replace(evidence, agent_id="agent.finisher"))
