"""Physics Darwin produces paired evidence accepted by Promotion Gate V2."""

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


def test_paired_physics_candidate_can_reach_sim_champion_gate(tmp_path):
    cases = [
        PairedTrajectoryCase(
            scenario=ScenarioSpec(
                scenario_id=f"shield-{seed}",
                robot_id="ur5e",
                world_id="tabletop",
                body_snapshot_hash="sha256:ur5e-body",
                model_hash="resolved-by-runner",
                seed=seed,
            ),
            baseline_trajectory=[COLLISION, HOME],
            candidate_trajectory=[HOME],
        )
        for seed in (42, 43)
    ]
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
        simulation_receipts=result.simulation_receipts,
        regression_results=result.regression_results,
    )
    assert gate.passed is True
    assert gate.decision == "promote_to_sim"
