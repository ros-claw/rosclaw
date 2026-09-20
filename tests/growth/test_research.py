from dataclasses import replace

import pytest

from rosclaw.dream.contracts import DreamBudget
from rosclaw.dream.control import DreamBudgetUsage
from rosclaw.feedback.contracts import canonical_hash
from rosclaw.growth.research import (
    ExperimentFamily,
    FamilyExperiment,
    ResearchBudget,
    ResearchCampaign,
    ResearchHypothesis,
    ResearchObservation,
    ResearchRoute,
    detect_plateau,
    research_budget_available,
    route_research,
)


def h(value):
    return canonical_hash({"value": value})


def campaign(task="fixture.skill"):
    return ResearchCampaign(
        task,
        ResearchHypothesis(
            "authority",
            "interface limits skill",
            "broader interface improves paired score",
            h("conditions"),
            h("evaluation"),
        ),
        ExperimentFamily("family", h("mechanism")),
        ResearchBudget(DreamBudget(0.0, 64, 16, 3600.0, 0.0, 0.0), 3),
        h("train"),
        h("dev"),
        h("retention"),
        h("sealed"),
    )


def experiment(i, gain=0.0):
    return FamilyExperiment(
        h(i), h("mechanism"), h("evaluation"), None if gain is None else h(f"bank-{i}"), gain
    )


def test_unknown_is_not_failure_or_permission():
    result = route_research(campaign(), ResearchObservation())
    assert result.route is ResearchRoute.NEED_EVIDENCE
    assert not result.training_authorized and not result.promotion_authorized


@pytest.mark.parametrize(
    "field",
    [
        "train_snapshot_hash",
        "development_snapshot_hash",
        "retention_snapshot_hash",
        "sealed_commitment",
    ],
)
def test_planning_can_record_absent_bank_but_cannot_reach_team_gate(field):
    pending = replace(campaign(), **{field: None})
    assert not pending.banks_bound
    observation = ResearchObservation(True, True, True, True, True, True, True, (h("receipt"),))
    decision = route_research(pending, observation)
    assert decision.route is ResearchRoute.NEED_EVIDENCE
    assert not decision.training_authorized and not decision.promotion_authorized


def test_completely_unmaterialized_campaign_does_not_invent_hashes():
    pending = replace(
        campaign(),
        train_snapshot_hash=None,
        development_snapshot_hash=None,
        retention_snapshot_hash=None,
        sealed_commitment=None,
    )
    assert not pending.banks_bound
    assert route_research(pending, ResearchObservation()).route is ResearchRoute.NEED_EVIDENCE
    assert pending.campaign_hash != campaign().campaign_hash


@pytest.mark.parametrize(
    "values,expected",
    [
        ({"oracle_pass": False}, ResearchRoute.NEED_EVIDENCE),
        (
            {"oracle_pass": False, "oracle_assay_pass": True},
            ResearchRoute.ACTION_SPACE_OR_ENVIRONMENT,
        ),
        ({"oracle_pass": True, "imitation_pass": False}, ResearchRoute.REPRESENTATION_OR_DAGGER),
        (
            {"oracle_pass": True, "imitation_pass": True, "closed_loop_pass": False},
            ResearchRoute.CREDIT_OR_ON_POLICY,
        ),
        (
            {
                "oracle_pass": True,
                "imitation_pass": True,
                "closed_loop_pass": True,
                "development_pass": True,
                "blind_pass": False,
            },
            ResearchRoute.COVERAGE_OR_CURRICULUM,
        ),
        (
            {"individual_skill_pass": True, "retention_pass": False},
            ResearchRoute.STABILITY_PLASTICITY,
        ),
        ({"individual_skill_pass": True}, ResearchRoute.NEED_EVIDENCE),
    ],
)
def test_stage_order_and_non_promoting_routes(values, expected):
    result = route_research(
        campaign(), ResearchObservation(**values, evidence_hashes=(h("receipt"),))
    )
    assert result.route is expected
    assert not result.training_authorized and not result.promotion_authorized


def test_team_integration_requires_all_individual_gates():
    observation = ResearchObservation(True, True, True, True, True, True, True, (h("receipt"),))
    for task in ("grasping", "receiving", "navigation"):
        assert route_research(campaign(task), observation).route is ResearchRoute.TEAM_INTEGRATION
    assert (
        route_research(campaign(), replace(observation, blind_pass=None)).route
        is ResearchRoute.NEED_EVIDENCE
    )


@pytest.mark.parametrize("oracle_pass", [None, False, True])
def test_failed_assay_routes_to_search_before_interpreting_capability(oracle_pass):
    result = route_research(
        campaign(),
        ResearchObservation(
            oracle_pass=oracle_pass,
            oracle_assay_pass=False,
            evidence_hashes=(h("authenticated-controls"),),
        ),
    )
    assert result.route is ResearchRoute.SEARCH_OR_ASSAY
    assert result.reason == "oracle_assay_controls_failed"
    assert not result.training_authorized and not result.promotion_authorized


def test_unknown_assay_does_not_establish_a_capacity_failure():
    result = route_research(
        campaign(), ResearchObservation(oracle_pass=False, evidence_hashes=(h("exam"),))
    )
    assert result.route is ResearchRoute.NEED_EVIDENCE
    assert result.reason == "oracle_failure_requires_assay_controls"


def test_assay_does_not_hide_retention_failure_or_latched_plateau():
    observation = ResearchObservation(
        oracle_pass=False,
        oracle_assay_pass=False,
        retention_pass=False,
        evidence_hashes=(h("controls"), h("retention")),
    )
    assert route_research(campaign(), observation).route is ResearchRoute.STABILITY_PLASTICITY
    assert (
        route_research(campaign(), observation, tuple(experiment(i) for i in range(3))).route
        is ResearchRoute.STOP_FAMILY
    )


@pytest.mark.parametrize("value", [0, 1, "passed", float("nan")])
def test_assay_requires_typed_upstream_evidence(value):
    with pytest.raises(ValueError):
        ResearchObservation(oracle_assay_pass=value, evidence_hashes=(h("controls"),))


def test_assay_judgment_without_evidence_is_rejected():
    with pytest.raises(ValueError):
        ResearchObservation(oracle_assay_pass=True)


def test_plateau_is_latched_and_family_rename_does_not_erase_it():
    history = tuple(experiment(i) for i in range(3)) + (experiment(3, 1.0),)
    changed = replace(campaign(), family=ExperimentFamily("renamed", h("mechanism")))
    plateau = detect_plateau(changed, history)
    assert plateau.stopped and len(plateau.experiment_hashes) == 3
    assert (
        route_research(changed, ResearchObservation(), history).route is ResearchRoute.STOP_FAMILY
    )


def test_unknown_blind_result_breaks_consecutive_streak():
    history = (experiment(0), experiment(1, None), experiment(2), experiment(3))
    assert not detect_plateau(campaign(), history).stopped


def test_duplicates_reused_banks_and_changed_exam_rejected():
    one = experiment(0)
    for history in (
        (one, one),
        (one, replace(experiment(1), blind_bank_commitment=one.blind_bank_commitment)),
        (replace(one, evaluation_contract_hash=h("changed-exam")),),
    ):
        with pytest.raises(ValueError):
            detect_plateau(campaign(), history)


@pytest.mark.parametrize("gain", [True, float("nan"), float("inf"), "0"])
def test_invalid_improvement_rejected(gain):
    with pytest.raises(ValueError):
        experiment(0, gain)


def test_campaign_binds_exam_banks_and_budget():
    original = campaign()
    with pytest.raises(ValueError):
        replace(original, sealed_commitment=original.train_snapshot_hash)
    with pytest.raises(ValueError):
        ResearchObservation(oracle_pass=False)
    with pytest.raises(ValueError):
        ResearchObservation(oracle_pass=1, evidence_hashes=(h("receipt"),))
    assert (
        replace(original, sealed_commitment=h("new-sealed")).campaign_hash != original.campaign_hash
    )
    with pytest.raises(ValueError):
        ResearchBudget(original.budget.execution, True)


def test_budget_does_not_forget_failed_or_reserved_work():
    values = {
        "usage": DreamBudgetUsage(cpu_rollouts=60, candidates=14),
        "requested": DreamBudgetUsage(cpu_rollouts=4, candidates=2),
        "experiments_reserved": 2,
        "elapsed_wall_seconds": 3500.0,
        "requested_wall_seconds": 100.0,
    }
    assert research_budget_available(campaign(), **values)
    for change in (
        {"experiments_reserved": 3},
        {"requested_wall_seconds": 101.0},
        {"requested": DreamBudgetUsage(cpu_rollouts=5)},
        {"requested": DreamBudgetUsage(gpu_seconds=1.0)},
    ):
        assert not research_budget_available(campaign(), **{**values, **change})
    with pytest.raises(ValueError):
        research_budget_available(campaign(), **{**values, "elapsed_wall_seconds": float("nan")})
