from __future__ import annotations

import math

import pytest

from rosclaw.simforge.budget import (
    BudgetAction,
    BudgetScope,
    DataBudgetManager,
    DataBudgetSpec,
)
from rosclaw.simforge.distribution import ScenarioSampler
from rosclaw.simforge.models import (
    Partition,
    SamplingStrategy,
    ScenarioDistributionSpec,
    SimForgeTaskSpec,
)
from rosclaw.simforge.monitors import (
    RobustnessAggregator,
    SafetyPredicateMonitor,
    TemporalPredicateMonitor,
)
from rosclaw.simforge.seed_ledger import SeedLedger


def _distribution() -> ScenarioDistributionSpec:
    return ScenarioDistributionSpec.from_dict(
        {
            "variables": {
                "target_x": {"distribution": "uniform", "min": -0.55, "max": 0.2},
                "target_y": {"distribution": "uniform", "min": 0.25, "max": 0.75},
                "target_z": {"distribution": "uniform", "min": 0.25, "max": 0.8},
                "obstacle_x": {"distribution": "uniform", "min": -0.5, "max": 0.1},
                "obstacle_height": {"values": [0.1, 0.2, 0.35]},
                "control_latency_ms": {"distribution": "choice", "values": [0, 25, 80]},
            },
            "constraints": [
                "finite_values",
                "target_reachable",
                {
                    "name": "target_not_inside_obstacle",
                    "parameters": {"half_x": 0.04, "half_y": 0.04},
                },
            ],
        }
    )


@pytest.mark.parametrize("strategy", list(SamplingStrategy))
def test_level_a_sampling_is_deterministic_and_constrained(strategy: SamplingStrategy) -> None:
    sampler = ScenarioSampler(_distribution())
    first = sampler.sample(count=32, seed=41, partition=Partition.DEVELOPMENT, strategy=strategy)
    second = sampler.sample(count=32, seed=41, partition=Partition.DEVELOPMENT, strategy=strategy)

    assert first == second
    assert len({sample.scenario_id for sample in first}) == 32
    assert all(sample.partition is Partition.DEVELOPMENT for sample in first)
    assert all(sample.distribution_hash == _distribution().digest for sample in first)


def test_unknown_or_expression_constraint_fails_closed() -> None:
    spec = ScenarioDistributionSpec.from_dict(
        {
            "variables": {"x": {"distribution": "uniform", "min": 0, "max": 1}},
            "constraints": ["__import__('os').system('false')"],
        }
    )
    with pytest.raises(ValueError, match="unregistered"):
        ScenarioSampler(spec)


def test_partition_visibility_and_holdout_seed_commitment() -> None:
    assert Partition.DISCOVERY.candidate_may_view_cases
    assert Partition.COUNTEREXAMPLE_REGRESSION.candidate_may_view_cases
    assert not Partition.VALIDATION.candidate_may_view_cases
    assert not Partition.HOLDOUT.candidate_may_view_cases

    ledger = SeedLedger(task_id="shield_reach_v1", secret=b"phase-two-test-secret")
    development = ledger.allocate(Partition.DEVELOPMENT, 20)
    validation = ledger.allocate(Partition.VALIDATION, 20)
    holdout = ledger.allocate(Partition.HOLDOUT, 200)
    ledger.assert_disjoint()
    manifest = ledger.public_manifest()

    assert len({item.seed for item in development + validation + holdout}) == 240
    assert "seed" in manifest["partitions"]["development"][0]
    assert "seed" not in manifest["partitions"]["validation"][0]
    assert "seed" not in manifest["partitions"]["holdout"][0]
    assert manifest["partitions"]["holdout"][0]["commitment"].startswith("sha256:")

    hidden_sample = ScenarioSampler(_distribution()).sample(
        count=1,
        seed=7,
        partition=Partition.HOLDOUT,
    )[0]
    assert "seed" not in hidden_sample.to_dict()
    assert "seed_commitment" in hidden_sample.to_dict()


def test_task_spec_rejects_non_whitelisted_shape() -> None:
    task = SimForgeTaskSpec.from_dict(
        {
            "task_id": "shield_reach_v1",
            "suite_id": "core_v1",
            "body": {
                "body_id": "sim_ur5e",
                "required_capabilities": ["arm.follow_trajectory"],
            },
            "backends": {
                "discovery": ["mujoco_cpu", "mjwarp"],
                "evaluation": ["mujoco_cpu"],
                "differential": ["mjwarp"],
            },
            "scenario_distribution": {"ref": "scenario_distribution.yaml"},
            "success_spec": {"eventually": {"condition": "goal", "within_sec": 5}},
            "safety_spec": {"always": ["no_collision"]},
            "candidate_space": {
                "type": "parameter_patch",
                "allowed_paths": ["/trajectory/clearance"],
            },
            "evidence_requirements": {"minimum_seeds": 20},
        }
    )
    assert task.evidence_requirements.holdout_required
    assert task.candidate_allowed_paths == ("/trajectory/clearance",)

    value = {**task.__dict__, "candidate_allowed_paths": ("relative",)}
    with pytest.raises(ValueError, match="JSON pointers"):
        SimForgeTaskSpec(**value)

    malformed = task.__dict__.copy()
    malformed["evidence_requirements"] = {
        "physics_executed": "false",
        "strict_replay": True,
        "artifact_hashes": True,
        "minimum_seeds": 20,
        "holdout_required": True,
    }
    with pytest.raises(ValueError, match="physics_executed must be boolean"):
        SimForgeTaskSpec.from_dict(
            {
                "schema_version": "rosclaw.simforge.task.v1",
                "task_id": "strict_bool",
                "suite_id": "core_v1",
                "body": {
                    "body_id": "sim",
                    "required_capabilities": ["arm.follow_trajectory"],
                },
                "backends": {
                    "discovery": ["mujoco_cpu"],
                    "evaluation": ["mujoco_cpu"],
                    "differential": [],
                },
                "scenario_distribution": {"ref": "scenario_distribution.yaml"},
                "success_spec": {"eventually": True},
                "safety_spec": {"always": True},
                "candidate_space": {"allowed_paths": ["/shield/threshold"]},
                "evidence_requirements": malformed["evidence_requirements"],
            }
        )


def test_continuous_and_temporal_robustness() -> None:
    monitor = SafetyPredicateMonitor(
        required_clearance_m=0.02,
        velocity_limit=2.0,
        force_limit=50.0,
        deadline_sec=5.0,
        stop_distance_limit_m=0.25,
    )
    rho = monitor.observe(
        {
            "minimum_clearance_m": 0.035,
            "joint_limit_margin_rad": 0.03,
            "peak_velocity": 1.4,
            "peak_force_n": 30,
            "elapsed_sec": 4.0,
            "actual_stop_distance_m": 0.1,
        }
    )
    assert rho == pytest.approx(0.015)
    assert monitor.robustness == rho
    assert SafetyPredicateMonitor().observe({"joint_limit_margin_rad": 0.1}) == -math.inf

    temporal = TemporalPredicateMonitor(horizon_sec=5)
    temporal.observe_always(timestamp_sec=0, margin=0.1)
    temporal.observe_always(timestamp_sec=2, margin=0.01)
    temporal.observe_eventually(timestamp_sec=4, margin=0.2)
    assert temporal.satisfied
    assert temporal.always_robustness == 0.01

    values = [-0.5, -0.1, 0.2, 0.8, 1.0]
    assert RobustnessAggregator.minimum(values) == -0.5
    assert RobustnessAggregator.quantile(values, 0.5) == 0.2
    assert RobustnessAggregator.lower_tail_cvar(values, 0.4) == pytest.approx(-0.3)
    assert RobustnessAggregator.minimum([math.nan]) == -math.inf


def test_data_budget_attacks_fail_closed_without_serializing_full_record() -> None:
    manager = DataBudgetManager(
        DataBudgetSpec(
            max_event_record_bytes=128,
            max_trace_record_bytes=128,
            max_nested_depth=3,
            max_string_length=32,
            max_collection_items=50,
            max_episode_semantic_bytes=256,
            max_episode_raw_bytes=256,
            max_run_bytes=512,
            max_workspace_bytes=1024,
        )
    )
    recursive: dict[str, object] = {}
    recursive["recovery_hint"] = recursive

    assert (
        manager.inspect_record(recursive, scope=BudgetScope.EVENT).action is BudgetAction.SUMMARIZE
    )
    assert (
        manager.inspect_record({"provider": "x" * 1000}, scope=BudgetScope.TRACE).action
        is BudgetAction.SUMMARIZE
    )
    assert (
        manager.inspect_record(list(range(100)), scope=BudgetScope.EPISODE_RAW).action
        is BudgetAction.STOP_RECORDING
    )
    assert (
        manager.inspect_record({"rho": math.nan}, scope=BudgetScope.EVENT).action
        is BudgetAction.SUMMARIZE
    )
    assert (
        manager.account_external_bytes(scope=BudgetScope.RUN, size=1024).action
        is BudgetAction.ABORT_FAIL_CLOSED
    )
    with pytest.raises(ValueError, match="non-negative integer"):
        manager.account_external_bytes(scope=BudgetScope.RUN, size=True)
    non_string_key = manager.inspect_record({1: "value"}, scope=BudgetScope.EVENT)
    assert not non_string_key.accepted

    accepted = manager.inspect_record({"event": "ok"}, scope=BudgetScope.EVENT)
    assert accepted.accepted
    manager.commit(accepted, scope=BudgetScope.EVENT)
    assert manager.usage()["event"] == accepted.estimated_bytes
