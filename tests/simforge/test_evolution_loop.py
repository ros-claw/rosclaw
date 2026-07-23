from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import replace
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from rosclaw.simforge.candidates import (
    CandidateCompiler,
    CandidateGenerator,
    ParameterBound,
    SearchAlgorithm,
    SearchCandidateGenerator,
    TemplateCandidateGenerator,
)
from rosclaw.simforge.differential import (
    DisagreementStatus,
    SimulatorLabels,
    compare_simulators,
)
from rosclaw.simforge.distribution import ScenarioSampler
from rosclaw.simforge.evaluation import EpisodeOutcome, EvaluationBundle, PairedEpisode
from rosclaw.simforge.evolution import EvolutionRun, EvolutionState
from rosclaw.simforge.falsification import CounterexampleStore, Falsifier
from rosclaw.simforge.holdout import HiddenHoldoutService, create_holdout_signing_key
from rosclaw.simforge.models import Partition, ScenarioDistributionSpec
from rosclaw.simforge.promotion_v3 import GateDecision, GateV3Policy, StatisticalGateV3


def _compiler() -> CandidateCompiler:
    return CandidateCompiler(
        parent_policy={
            "/controller/velocity_factor": 1.0,
            "/controller/damping": 0.2,
            "/trajectory/minimum_clearance_m": 0.02,
            "/trajectory/waypoint_policy": "direct",
        },
        allowed_bounds={
            "/controller/velocity_factor": ParameterBound(0.4, 1.0),
            "/controller/damping": ParameterBound(0.1, 1.0),
            "/trajectory/minimum_clearance_m": ParameterBound(0.02, 0.08),
            "/trajectory/waypoint_policy": ParameterBound(choices=("direct", "obstacle_avoidance")),
        },
    )


def test_template_and_search_generate_only_immutable_whitelisted_patches() -> None:
    compiler = _compiler()
    template = TemplateCandidateGenerator.generate(
        compiler, failure_signature_id="MIDPATH_COLLISION:obstacle"
    )
    assert template.human_involvement.fully_autonomous
    assert template.to_dict()["constraints"]["gateway_bypass"] is False
    assert template.to_dict()["rollback"]["parent_policy_hash"] == compiler.parent_policy_hash

    with pytest.raises(ValueError, match="not whitelisted"):
        compiler.compile(
            {"/arbitrary/source.py": "malicious"},
            failure_signature_id="MIDPATH_COLLISION",
            generator=CandidateGenerator(type="model", algorithm="proposal"),
        )

    for algorithm in SearchAlgorithm:
        search = SearchCandidateGenerator(compiler, seed=73)

        def objective(patch: object) -> float:
            changes = {item.path: item.new for item in patch.changes}  # type: ignore[attr-defined]
            return -abs(float(changes["/trajectory/minimum_clearance_m"]) - 0.055)

        winner, trace = search.optimize(
            failure_signature_id="MIDPATH_COLLISION",
            algorithm=algorithm,
            objective=objective,
            budget=12,
        )
        assert winner.generator.algorithm == algorithm.value
        assert len(trace) == 12
        assert all(change.path in compiler.allowed_bounds for change in winner.changes)

    categorical = CandidateCompiler(
        parent_policy={"/mode": "direct"},
        allowed_bounds={"/mode": ParameterBound(choices=("direct", "avoid"))},
    )
    categorical_winner, _ = SearchCandidateGenerator(categorical, seed=1).optimize(
        failure_signature_id="MIDPATH_COLLISION",
        algorithm=SearchAlgorithm.RANDOM,
        objective=lambda _patch: 1.0,
        budget=2,
    )
    assert categorical_winner.changes[0].new == "avoid"


def _bundle(
    partition: Partition, candidate_hash: str, *, unsafe_candidate: bool = False
) -> EvaluationBundle:
    pairs = []
    for index in range(200):
        baseline_success = index >= 40
        candidate_unsafe = unsafe_candidate and index == 0
        pairs.append(
            PairedEpisode(
                pair_id=f"{partition.value}-{index}",
                scenario_commitment="sha256:"
                + hashlib.sha256(f"scenario-{index}".encode()).hexdigest(),
                seed_commitment="sha256:" + hashlib.sha256(f"seed-{index}".encode()).hexdigest(),
                baseline=EpisodeOutcome(
                    success=baseline_success,
                    collision=False,
                    unsafe_allow=False,
                    false_block=not baseline_success,
                    robustness=0.01 if baseline_success else -0.01,
                ),
                candidate=EpisodeOutcome(
                    success=not candidate_unsafe,
                    collision=candidate_unsafe,
                    unsafe_allow=candidate_unsafe,
                    false_block=False,
                    robustness=-0.02 if candidate_unsafe else 0.03,
                ),
                physics_executed=True,
                independently_verified=True,
                strict_replay=True,
                artifact_hash_valid=True,
                data_quality_valid=True,
            )
        )
    return EvaluationBundle.from_pairs(
        task_id="shield_reach_v1",
        candidate_hash=candidate_hash,
        partition=partition,
        pairs=pairs,
        bootstrap_seed=99,
    )


def test_gate_v3_requires_all_fourteen_checks_and_rejects_safety_regression() -> None:
    candidate_hash = "sha256:" + "a" * 64
    validation = _bundle(Partition.VALIDATION, candidate_hash)
    holdout = _bundle(Partition.HOLDOUT, candidate_hash)
    gate = StatisticalGateV3()

    missing = gate.evaluate(
        validation=validation,
        holdout=None,
        stress_worlds=None,
        stress_complete=None,
        counterexample_regression_passed=None,
        critical_backend_disagreements=None,
    )
    assert missing.decision is GateDecision.NEED_MORE_EVIDENCE

    passed = gate.evaluate(
        validation=validation,
        holdout=holdout,
        stress_worlds=1000,
        stress_complete=True,
        counterexample_regression_passed=True,
        critical_backend_disagreements=0,
    )
    assert passed.decision is GateDecision.SIM_CHAMPION
    assert [check.gate for check in passed.checks] == [f"G{index}" for index in range(1, 15)]
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/simforge/schema/evaluation_bundle.schema.json"
    )
    Draft202012Validator(json.loads(schema_path.read_text())).validate(validation.aggregate_dict())

    undersized = gate.evaluate(
        validation=replace(validation, paired_episodes=199),
        holdout=holdout,
        stress_worlds=999,
        stress_complete=True,
        counterexample_regression_passed=True,
        critical_backend_disagreements=0,
    )
    assert undersized.decision is GateDecision.NEED_MORE_EVIDENCE
    assert next(check for check in undersized.checks if check.gate == "G3").missing

    rejected = gate.evaluate(
        validation=_bundle(Partition.VALIDATION, candidate_hash, unsafe_candidate=True),
        holdout=holdout,
        stress_worlds=1000,
        stress_complete=True,
        counterexample_regression_passed=True,
        critical_backend_disagreements=0,
    )
    assert rejected.decision is GateDecision.REJECTED
    assert not next(check for check in rejected.checks if check.gate == "G6").passed


def test_pair_attestation_rejects_reused_scenario_seed_under_new_pair_id() -> None:
    outcome = EpisodeOutcome(
        success=True,
        collision=False,
        unsafe_allow=False,
        false_block=False,
        robustness=0.1,
    )
    pairs = [
        PairedEpisode(
            pair_id=f"pair-{index}",
            scenario_commitment="sha256:" + "1" * 64,
            seed_commitment="sha256:" + "2" * 64,
            baseline=outcome,
            candidate=outcome,
            physics_executed=True,
            independently_verified=True,
            strict_replay=True,
            artifact_hash_valid=True,
            data_quality_valid=True,
        )
        for index in range(2)
    ]
    bundle = EvaluationBundle.from_pairs(
        task_id="pair-reuse",
        candidate_hash="sha256:" + "a" * 64,
        partition=Partition.VALIDATION,
        pairs=pairs,
    )
    assert bundle.attestation.scenario_seed_paired is False


def test_hidden_holdout_worker_returns_signed_aggregate_only(tmp_path) -> None:
    compiler = CandidateCompiler(
        parent_policy={"/shield/risk_threshold": 0.8},
        allowed_bounds={"/shield/risk_threshold": ParameterBound(0.1, 0.9)},
    )
    candidate = compiler.compile(
        {"/shield/risk_threshold": 0.5},
        failure_signature_id="UNSAFE_ALLOW",
        generator=CandidateGenerator(type="search", algorithm="random"),
    )
    private_bundle = tmp_path / "holdout-private.json"
    cases = [
        {
            "case_id": f"hidden-{index}",
            "scenario_commitment": "sha256:" + hashlib.sha256(f"s-{index}".encode()).hexdigest(),
            "seed_commitment": "sha256:" + hashlib.sha256(f"k-{index}".encode()).hexdigest(),
            "risk": index / 20,
            "should_allow": index / 20 <= 0.5,
        }
        for index in range(20)
    ]
    private_bundle.write_text(
        json.dumps(
            {
                "task_id": "hidden_threshold_test",
                "runner": "threshold_shield_test_v1",
                "threshold_path": "/shield/risk_threshold",
                "baseline_threshold": 0.8,
                "seed_ledger_manifest_hash": "sha256:" + "b" * 64,
                "cases": cases,
            }
        )
    )
    os.chmod(private_bundle, 0o600)
    signing_key = tmp_path / "holdout.key"
    public_key = create_holdout_signing_key(signing_key)
    service = HiddenHoldoutService(
        private_bundle_path=private_bundle,
        signing_key_path=signing_key,
        source_checkout=tmp_path / "source-checkout",
        timeout_sec=30,
    )
    result = service.evaluate(candidate)

    assert result.verify(expected_public_key=public_key)
    assert result.paired_episodes == 20
    assert result.metrics.candidate_unsafe_allow_rate == 0
    assert "cases" not in result.to_dict()
    assert '"seed":' not in json.dumps(result.to_dict(), separators=(",", ":"))
    assert not replace(result, candidate_hash="sha256:" + "c" * 64).verify()


def test_falsification_minimization_external_store_and_state_machine(tmp_path) -> None:
    distribution = ScenarioDistributionSpec.from_dict(
        {"variables": {"x": {"distribution": "uniform", "min": 0.0, "max": 1.0}}}
    )
    falsifier = Falsifier(ScenarioSampler(distribution))
    failures = falsifier.search(
        task_id="boundary_task",
        candidate_hash="sha256:" + "d" * 64,
        seed=5,
        budget=9,
        evaluator=lambda scenario: (
            0.5 - float(dict(scenario.values)["x"]),
            "BOUNDARY_FAILURE",
            "sha256:" + "e" * 64,
        ),
    )
    assert failures and failures[0].robustness < 0
    minimized = falsifier.minimize(
        failures[0],
        nominal={"x": 0.0},
        still_fails=lambda values: float(values["x"]) > 0.5,
    )
    assert 0.5 < minimized["x"] <= 1.0
    store = CounterexampleStore(root=tmp_path / "evidence", source_checkout=tmp_path / "src")
    artifact = store.append(failures[0])
    assert artifact == store.append(failures[0])
    with pytest.raises(ValueError, match="non-finite robustness"):
        falsifier.search(
            task_id="boundary_task",
            candidate_hash="sha256:" + "d" * 64,
            seed=5,
            budget=1,
            evaluator=lambda _scenario: (math.nan, "BAD_DATA", "sha256:" + "e" * 64),
        )

    run = EvolutionRun(run_id="run-1", task_id="boundary_task")
    for state in (
        EvolutionState.FAILURE_CLUSTERED,
        EvolutionState.DIAGNOSED,
        EvolutionState.CANDIDATES_GENERATED,
    ):
        run.transition(state, reason="automatic phase transition")
    with pytest.raises(RuntimeError, match="invalid evolution transition"):
        run.transition(EvolutionState.HOLDOUT_EVAL, reason="skip forbidden")


def test_cross_backend_compares_labels_not_exact_qpos() -> None:
    baseline = SimulatorLabels(
        safe=True,
        collision=False,
        success=True,
        stopped=True,
        task_verified=True,
        clearance_category="safe",
        final_error_m=0.005,
    )
    numeric = replace(baseline, final_error_m=0.006)
    event = compare_simulators(
        scenario_id="scenario-1",
        baseline_backend="mujoco_cpu",
        comparison_backend="mjwarp",
        baseline=baseline,
        comparison=numeric,
    )
    assert event.status is DisagreementStatus.NUMERIC_VARIATION
    assert not event.promotion_blocked

    critical = compare_simulators(
        scenario_id="scenario-1",
        baseline_backend="mujoco_cpu",
        comparison_backend="isaac_newton",
        baseline=baseline,
        comparison=replace(baseline, safe=False, collision=True),
    )
    assert critical.status is DisagreementStatus.CRITICAL_DISAGREEMENT
    assert critical.promotion_blocked

    with pytest.raises(ValueError, match="must be finite"):
        replace(baseline, final_error_m=math.nan)

    with pytest.raises(ValueError, match="unsafe allow bound"):
        GateV3Policy(max_unsafe_allow_rate=2.0)
