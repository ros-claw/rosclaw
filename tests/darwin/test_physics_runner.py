"""Physics Darwin produces paired evidence accepted by Promotion Gate V2."""

import pytest

from rosclaw.auto.promotion.gate import PromotionGate
from rosclaw.darwin.physics_runner import PairedTrajectoryCase, PhysicsDarwinRunner
from rosclaw.sandbox.backends import ScenarioSpec

HOME = [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
COLLISION = [
    3.4426358094526863,
    -0.7680767522686045,
    2.253070730803216,
    2.480201653011009,
    -5.099721659051599,
    5.976851207161098,
]


def _case(seed=42, *, jitter=0.002):
    return PairedTrajectoryCase(
        scenario=ScenarioSpec(
            scenario_id=f"shield-{seed}",
            robot_id="ur5e",
            world_id="tabletop",
            body_snapshot_hash="resolved-by-runner",
            model_hash="resolved-by-runner",
            seed=seed,
            metadata={"initial_qpos_jitter_rad": jitter},
        ),
        baseline_trajectory=[COLLISION, HOME],
        candidate_trajectory=[HOME],
    )


def test_paired_physics_candidate_can_reach_sim_champion_gate(tmp_path):
    cases = [_case(seed) for seed in (42, 43)]
    result = PhysicsDarwinRunner().run(cases, artifact_root=tmp_path)

    assert result.baseline_metrics["success_rate"] == 0.0
    assert result.candidate_metrics["success_rate"] == 1.0
    assert result.candidate_metrics["collision_rate"] == 0.0
    assert result.candidate_metrics["replay_success_rate"] == 1.0
    assert all(pair["baseline"] and pair["candidate"] for pair in result.per_seed.values())

    gate = PromotionGate().evaluate(
        result.baseline_metrics,
        result.candidate_metrics,
        current_level="baseline",
        per_seed=result.per_seed,
        sandbox_risk_score=result.candidate_metrics["collision_rate"],
        simulation_receipts=result.simulation_receipts,
        regression_results=result.regression_results,
    )
    assert gate.passed is True
    assert gate.decision == "promote_to_sim"


@pytest.mark.parametrize("count", [0, 1, 101])
def test_physics_darwin_bounds_case_count(tmp_path, count):
    cases = [_case(seed) for seed in range(count)]

    with pytest.raises(ValueError, match="PHYSICS_DARWIN_CASE_COUNT_OUT_OF_RANGE"):
        PhysicsDarwinRunner().run(cases, artifact_root=tmp_path)


@pytest.mark.parametrize("seed", [True, -1, 2**63, "42"])
def test_physics_darwin_rejects_invalid_seed(tmp_path, seed):
    with pytest.raises(ValueError, match="PHYSICS_DARWIN_INVALID_SEED"):
        PhysicsDarwinRunner().run([_case(seed), _case(43)], artifact_root=tmp_path)


@pytest.mark.parametrize("jitter", [False, 0.0, -0.1, 0.100001, float("inf"), float("nan")])
def test_physics_darwin_bounds_randomization(tmp_path, jitter):
    with pytest.raises(ValueError, match="PHYSICS_DARWIN_REQUIRES_SEED_RANDOMIZATION"):
        PhysicsDarwinRunner().run([_case(42, jitter=jitter), _case(43)], artifact_root=tmp_path)
