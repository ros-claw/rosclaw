from __future__ import annotations

import hashlib

import pytest

from rosclaw.growth import (
    ActiveSamplingPolicy,
    ActiveSamplingStatus,
    ApplicabilityEvidence,
    ApplicabilityGate,
    CandidateEvidenceGate,
    CandidateEvidenceStatus,
    CandidateExecutionEvidence,
    EvidenceLevel,
    ExecutedSafeSupport,
    MetricDirection,
    ModeDecision,
    ModeGate,
    ModeSelectionContext,
    MutationBudget,
    NumericBindingTolerance,
    ParameterDimension,
    RobustnessEvidence,
    RobustnessGate,
    RobustnessProfile,
    RobustnessStatus,
    SupportAxis,
    SupportPoint,
    SupportTopologyContract,
    TrustRegion,
)


def _hash(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _sample(
    *,
    domain: str,
    dimension_ids: tuple[str, str],
) -> tuple[ActiveSamplingStatus, tuple[str, ...], dict[str, float]]:
    first, second = dimension_ids
    anchor = {first: 1.0, second: 1.0}
    decision = ActiveSamplingPolicy().propose(
        dimensions=(
            ParameterDimension(first, 0.5, 1.5),
            ParameterDimension(second, 0.5, 1.5),
        ),
        supports=(
            ExecutedSafeSupport(
                first,
                0.7,
                1.3,
                (0.7, 0.9, 1.0, 1.2, 1.3),
                _hash(domain + first),
            ),
            ExecutedSafeSupport(
                second,
                0.7,
                1.3,
                (0.7, 0.99, 1.0, 1.01, 1.3),
                _hash(domain + second),
            ),
        ),
        mutation_budget=MutationBudget(
            allowed_dimension_ids=dimension_ids,
            maximum_changed_dimensions=1,
            maximum_total_normalized_delta=0.2,
        ),
        trust_region=TrustRegion(
            anchor=anchor,
            maximum_absolute_delta={first: 0.3, second: 0.3},
            anchor_artifact_hash=_hash(domain + "anchor"),
        ),
    )
    return (
        decision.status,
        decision.changed_dimension_ids,
        dict(decision.candidate_parameters or {}),
    )


def test_active_sampling_contract_is_reused_across_navigation_and_manipulation() -> None:
    navigation = _sample(
        domain="navigation",
        dimension_ids=("linear_gain", "angular_gain"),
    )
    manipulation = _sample(
        domain="manipulation",
        dimension_ids=("force_gain", "closure_speed"),
    )

    assert navigation[0] is ActiveSamplingStatus.PROPOSED
    assert navigation[1] == ("angular_gain",)
    assert navigation[2] == {"angular_gain": pytest.approx(0.845), "linear_gain": 1.0}
    assert manipulation[0] is ActiveSamplingStatus.PROPOSED
    assert manipulation[1] == ("closure_speed",)


def test_active_sampling_is_proposal_only_and_requires_replay() -> None:
    decision = ActiveSamplingPolicy().propose(
        dimensions=(ParameterDimension("gain", 0.0, 2.0),),
        supports=(ExecutedSafeSupport("gain", 0.7, 1.3, (0.7, 1.0, 1.3), _hash("support")),),
        mutation_budget=MutationBudget(("gain",), 1, 0.2),
        trust_region=TrustRegion(
            anchor={"gain": 1.0},
            maximum_absolute_delta={"gain": 0.3},
            anchor_artifact_hash=_hash("anchor"),
        ),
    )

    assert decision.status is ActiveSamplingStatus.PROPOSED
    assert decision.to_dict()["sim_replay_required"] is True
    assert decision.to_dict()["activation_allowed"] is False
    assert decision.to_dict()["hardware_authorized"] is False


def test_active_sampling_never_extrapolates_outside_attested_safe_interval() -> None:
    decision = ActiveSamplingPolicy().propose(
        dimensions=(ParameterDimension("gain", 0.0, 2.0),),
        supports=(ExecutedSafeSupport("gain", 0.9, 1.1, (0.9, 1.0, 1.1), _hash("support")),),
        mutation_budget=MutationBudget(("gain",), 1, 0.2),
        trust_region=TrustRegion(
            anchor={"gain": 1.0},
            maximum_absolute_delta={"gain": 0.8},
            anchor_artifact_hash=_hash("anchor"),
        ),
    )

    assert decision.candidate_parameters is not None
    assert 0.9 <= decision.candidate_parameters["gain"] <= 1.1
    with pytest.raises(ValueError, match="inside the safe interval"):
        ExecutedSafeSupport("gain", 0.9, 1.1, (0.8, 1.0, 1.1), _hash("invalid"))


def test_mutation_budget_and_trust_region_fail_closed() -> None:
    dimensions = (
        ParameterDimension("gain", 0.0, 2.0),
        ParameterDimension("offset", -1.0, 1.0),
    )
    budget = MutationBudget(("gain",), 1, 0.1)
    candidate = {"gain": 1.4, "offset": 0.2}

    assert budget.violations(
        anchor={"gain": 1.0, "offset": 0.0},
        candidate=candidate,
        dimensions=dimensions,
    ) == (
        "disallowed_dimension",
        "changed_dimension_budget_exceeded",
        "normalized_delta_budget_exceeded",
    )
    region = TrustRegion(
        anchor={"gain": 1.0, "offset": 0.0},
        maximum_absolute_delta={"gain": 0.1, "offset": 0.0},
        anchor_artifact_hash=_hash("anchor"),
    )
    assert region.violations(candidate) == ("trust_region_exceeded",)
    assert region.violations({"gain": float("nan"), "offset": 0.0}) == ("invalid_parameter_vector",)


def test_numeric_binding_tolerance_handles_json_round_trip_without_shape_drift() -> None:
    tolerance = NumericBindingTolerance(absolute=1e-8, relative=1e-8)

    assert tolerance.equivalent(
        {"gain": 0.3, "offset": 0.1},
        {"gain": 0.30000000000000004, "offset": 0.100000001},
    )
    assert tolerance.mismatched_dimensions(
        {"gain": 0.3, "offset": 0.1},
        {"gain": 0.31, "offset": 0.1},
    ) == ("gain",)
    assert tolerance.mismatched_dimensions({"gain": 0.3}, {"other": 0.3}) == (
        "parameter_shape_mismatch",
    )


def test_support_topology_requires_full_interaction_grid() -> None:
    contract = SupportTopologyContract(
        axes=(
            SupportAxis("speed", ("slow", "fast")),
            SupportAxis("load", ("low", "high")),
        )
    )
    partial = tuple(
        point for point in contract.required_points if "speed=fast" not in point.point_id
    )

    rejected = contract.evaluate(partial)
    passed = contract.evaluate(contract.required_points)

    assert rejected.complete is False
    assert rejected.required_point_count == 4
    assert rejected.observed_point_count == 2
    assert len(rejected.missing_point_ids) == 2
    assert passed.complete is True
    assert passed.hardware_authorized is False


def test_candidate_predictions_cannot_substitute_for_executed_evidence() -> None:
    candidate_hash = _hash("candidate")
    prediction = CandidateExecutionEvidence(
        candidate_artifact_hash=candidate_hash,
        evidence_hash=_hash("prediction"),
        evidence_level=EvidenceLevel.WORLD_MODEL,
        physics_executed=False,
        strict_replay=False,
        independently_verified=False,
    )
    gate = CandidateEvidenceGate(minimum_executed_receipts=1)

    decision = gate.evaluate(candidate_artifact_hash=candidate_hash, evidence=(prediction,))

    assert decision.status is CandidateEvidenceStatus.NEEDS_EXECUTION
    assert decision.reasons == ("insufficient_executed_evidence",)
    assert decision.activation_allowed is False


def test_candidate_execution_gate_binds_exact_artifact_and_strict_replay() -> None:
    candidate_hash = _hash("candidate")
    executed = CandidateExecutionEvidence(
        candidate_artifact_hash=candidate_hash,
        evidence_hash=_hash("execution"),
        evidence_level=EvidenceLevel.PHYSICS_REPLAY,
        physics_executed=True,
        strict_replay=True,
        independently_verified=True,
        execution_receipt_hash=_hash("receipt"),
    )
    gate = CandidateEvidenceGate()

    passed = gate.evaluate(candidate_artifact_hash=candidate_hash, evidence=(executed,))
    mismatched = gate.evaluate(
        candidate_artifact_hash=_hash("other-candidate"),
        evidence=(executed,),
    )

    assert passed.status is CandidateEvidenceStatus.PASSED
    assert passed.accepted_evidence_hashes == (_hash("execution"),)
    assert mismatched.status is CandidateEvidenceStatus.REJECTED
    assert "candidate_hash_mismatch" in mismatched.reasons


def test_duplicate_execution_evidence_cannot_satisfy_receipt_count() -> None:
    candidate_hash = _hash("candidate")
    executed = CandidateExecutionEvidence(
        candidate_artifact_hash=candidate_hash,
        evidence_hash=_hash("execution"),
        evidence_level=EvidenceLevel.PHYSICS_REPLAY,
        physics_executed=True,
        strict_replay=True,
        independently_verified=True,
        execution_receipt_hash=_hash("receipt"),
    )

    decision = CandidateEvidenceGate(minimum_executed_receipts=2).evaluate(
        candidate_artifact_hash=candidate_hash,
        evidence=(executed, executed),
    )

    assert decision.status is CandidateEvidenceStatus.REJECTED
    assert decision.reasons == ("duplicate_evidence", "insufficient_executed_evidence")


def _robustness(
    label: str,
    values: tuple[float, ...],
    *,
    direction: MetricDirection = MetricDirection.MAXIMIZE,
    safety_violation_count: int = 0,
) -> RobustnessEvidence:
    return RobustnessEvidence(
        artifact_hash=_hash(label + "artifact"),
        metric_id="task.quality",
        direction=direction,
        values=values,
        evidence_hash=_hash(label + "evidence"),
        safety_violation_count=safety_violation_count,
    )


def test_robustness_gate_rejects_lucky_mean_with_bad_tail() -> None:
    gate = RobustnessGate(
        RobustnessProfile(
            minimum_samples=10,
            tail_fraction=0.2,
            maximum_worst_regression=0.05,
            maximum_cvar_regression=0.05,
        )
    )
    parent = _robustness("parent", (0.8,) * 10)
    lucky_candidate = _robustness("candidate", (1.2,) * 9 + (0.1,))

    decision = gate.evaluate(parent=parent, candidate=lucky_candidate)

    assert sum(lucky_candidate.values) / len(lucky_candidate.values) > 0.8
    assert decision.status is RobustnessStatus.REJECTED
    assert decision.reasons == (
        "candidate_worst_regression",
        "candidate_cvar_regression",
    )


def test_robustness_gate_handles_minimize_metrics_and_safety() -> None:
    gate = RobustnessGate(RobustnessProfile(minimum_samples=3, tail_fraction=1 / 3))
    parent = _robustness(
        "parent",
        (1.0, 1.1, 0.9),
        direction=MetricDirection.MINIMIZE,
    )
    candidate = _robustness(
        "candidate",
        (0.8, 0.9, 0.7),
        direction=MetricDirection.MINIMIZE,
    )
    unsafe = _robustness(
        "unsafe",
        (0.8, 0.9, 0.7),
        direction=MetricDirection.MINIMIZE,
        safety_violation_count=1,
    )

    assert gate.evaluate(parent=parent, candidate=candidate).status is RobustnessStatus.PASSED
    unsafe_decision = gate.evaluate(parent=parent, candidate=unsafe)
    assert unsafe_decision.status is RobustnessStatus.REJECTED
    assert unsafe_decision.reasons == ("candidate_safety_violation",)


def test_applicability_gate_falls_back_to_parent_outside_support() -> None:
    evidence = ApplicabilityEvidence(
        candidate_artifact_hash=_hash("candidate"),
        parent_artifact_hash=_hash("parent"),
        context_hash=_hash("context"),
        evidence_hash=_hash("applicability"),
        in_distribution=False,
        confidence=0.4,
        support_distance=0.8,
    )

    decision = ApplicabilityGate(
        minimum_confidence=0.8,
        maximum_support_distance=0.2,
    ).evaluate(evidence)

    assert decision.used_candidate is False
    assert decision.selected_artifact_hash == evidence.parent_artifact_hash
    assert decision.reasons == (
        "out_of_distribution",
        "low_applicability_confidence",
        "support_distance_exceeded",
    )
    assert decision.activation_allowed is False


class _FixtureModeGate:
    def evaluate(self, context: ModeSelectionContext) -> ModeDecision:
        return ModeDecision(
            context_hash=context.context_hash,
            selected_mode_id=context.available_mode_ids[0],
            evidence_hash=_hash("mode-evidence"),
        )


def test_mode_gate_is_an_adapter_protocol_without_core_domain_logic() -> None:
    gate = _FixtureModeGate()
    context = ModeSelectionContext(
        context_hash=_hash("context"),
        feature_schema_hash=_hash("features"),
        available_mode_ids=("nominal", "recovery"),
    )

    assert isinstance(gate, ModeGate)
    decision = gate.evaluate(context)
    assert decision.selected_mode_id == "nominal"
    assert decision.hardware_authorized is False


def test_support_point_rejects_duplicate_axes() -> None:
    with pytest.raises(ValueError, match="axes must be non-empty and unique"):
        SupportPoint((("speed", "slow"), ("speed", "fast")))


def test_numeric_contracts_reject_boolean_and_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="finite number"):
        ParameterDimension("gain", True, 1.0)
    with pytest.raises(ValueError, match="finite number"):
        RobustnessEvidence(
            artifact_hash=_hash("artifact"),
            metric_id="task.quality",
            direction=MetricDirection.MAXIMIZE,
            values=(0.5, float("nan")),
            evidence_hash=_hash("evidence"),
        )


@pytest.mark.parametrize(
    ("domain", "dimension_ids"),
    (
        ("navigation", ("linear_gain", "angular_gain")),
        ("manipulation", ("force_gain", "closure_speed")),
    ),
)
def test_cross_domain_growth_evidence_loop_requires_execution_and_robustness(
    domain: str,
    dimension_ids: tuple[str, str],
) -> None:
    sampling_status, _, candidate_parameters = _sample(
        domain=domain,
        dimension_ids=dimension_ids,
    )
    candidate_hash = _hash(repr(sorted(candidate_parameters.items())))
    parent_hash = _hash(domain + "parent")
    prediction = CandidateExecutionEvidence(
        candidate_artifact_hash=candidate_hash,
        evidence_hash=_hash(domain + "prediction"),
        evidence_level=EvidenceLevel.WORLD_MODEL,
        physics_executed=False,
        strict_replay=False,
        independently_verified=False,
    )
    execution = CandidateExecutionEvidence(
        candidate_artifact_hash=candidate_hash,
        evidence_hash=_hash(domain + "execution"),
        evidence_level=EvidenceLevel.PHYSICS_REPLAY,
        physics_executed=True,
        strict_replay=True,
        independently_verified=True,
        execution_receipt_hash=_hash(domain + "receipt"),
    )
    evidence_gate = CandidateEvidenceGate()

    prediction_only = evidence_gate.evaluate(
        candidate_artifact_hash=candidate_hash,
        evidence=(prediction,),
    )
    executed = evidence_gate.evaluate(
        candidate_artifact_hash=candidate_hash,
        evidence=(prediction, execution),
    )
    robust = RobustnessGate(
        RobustnessProfile(minimum_samples=4, tail_fraction=0.25)
    ).evaluate(
        parent=RobustnessEvidence(
            artifact_hash=parent_hash,
            metric_id="task.quality",
            direction=MetricDirection.MAXIMIZE,
            values=(0.70, 0.72, 0.74, 0.76),
            evidence_hash=_hash(domain + "parent-robustness"),
        ),
        candidate=RobustnessEvidence(
            artifact_hash=candidate_hash,
            metric_id="task.quality",
            direction=MetricDirection.MAXIMIZE,
            values=(0.75, 0.77, 0.79, 0.81),
            evidence_hash=_hash(domain + "candidate-robustness"),
        ),
    )
    applicable = ApplicabilityGate(
        minimum_confidence=0.8,
        maximum_support_distance=0.2,
    ).evaluate(
        ApplicabilityEvidence(
            candidate_artifact_hash=candidate_hash,
            parent_artifact_hash=parent_hash,
            context_hash=_hash(domain + "context"),
            evidence_hash=_hash(domain + "applicability"),
            in_distribution=True,
            confidence=0.9,
            support_distance=0.1,
        )
    )

    assert sampling_status is ActiveSamplingStatus.PROPOSED
    assert prediction_only.status is CandidateEvidenceStatus.NEEDS_EXECUTION
    assert executed.status is CandidateEvidenceStatus.PASSED
    assert robust.status is RobustnessStatus.PASSED
    assert applicable.used_candidate is True
    assert applicable.selected_artifact_hash == candidate_hash
    # These contracts produce evidence and a selection, not an activation grant.
    assert executed.activation_allowed is False
    assert applicable.activation_allowed is False
