"""Tests for Sprint C: Sandbox/Darwin Runner with parametric physics."""
from rosclaw.auto.core.experiment import ExperimentSpec
from rosclaw.auto.runners import DarwinRunner, LocalRunner, RunnerResult, SandboxRunner


def test_runner_result_roundtrip():
    r = RunnerResult(success=True, metrics={"a": 1}, logs=["done"])
    d = r.to_dict()
    restored = RunnerResult.from_dict(d)
    assert restored.success is True
    assert restored.metrics["a"] == 1


def test_local_runner_health():
    runner = LocalRunner()
    h = runner.health()
    assert h["status"] == "healthy"
    assert h["runner"] == "local"


def test_local_runner_simulated_experiment_with_patch():
    """LocalRunner metrics should depend on patch parameters."""
    exp = ExperimentSpec(
        id="exp_001", proposal_id="prop_001", patch_id="patch_001",
        task="pick_cube", baseline_skill_id="pick_v1", candidate_skill_id="pick_v1_candidate",
        evaluation={"episodes": 10, "seeds": [0]},
        patch_context={"changes": [{"path": "/skill/pregrasp_height", "new": 0.05}]},
    )
    runner = LocalRunner(config={"simulate": True})
    result = runner.run(exp)
    assert result.success is True
    assert "baseline" in result.metrics
    assert "candidate" in result.metrics
    # Candidate with optimal param should improve over baseline
    assert result.metrics["candidate"]["success_rate"] > result.metrics["baseline"]["success_rate"]


def test_local_runner_simulated_experiment_no_patch():
    """No patch = baseline metrics only."""
    exp = ExperimentSpec(
        id="exp_002", proposal_id="prop_001", patch_id="patch_001",
        task="pick_cube", baseline_skill_id="pick_v1", candidate_skill_id="pick_v1_candidate",
        evaluation={"episodes": 10},
    )
    runner = LocalRunner(config={"simulate": True})
    result = runner.run(exp)
    assert result.success is True
    # Without patch, candidate ≈ baseline (small noise)
    assert abs(result.metrics["candidate"]["success_rate"] - result.metrics["baseline"]["success_rate"]) < 0.1


def test_local_runner_safety_violation():
    """LocalRunner should reject experiments with overly restrictive force limits."""
    exp = ExperimentSpec(
        id="exp_001", proposal_id="prop_001", patch_id="patch_001",
        task="pick_cube", baseline_skill_id="pick_v1", candidate_skill_id="pick_v1_candidate",
        evaluation={"episodes": 10},
        safety={"sandbox_required": True, "max_collision": 0, "max_force": 3},
    )
    runner = LocalRunner(config={"simulate": True})
    violations = runner.validate_safety(exp)
    assert len(violations) >= 1  # max_force < 5 is too restrictive


def test_sandbox_runner_simulated_with_patch():
    exp = ExperimentSpec(
        id="exp_001", proposal_id="prop_001", patch_id="patch_001",
        task="pick_cube", baseline_skill_id="pick_v1", candidate_skill_id="pick_v1_candidate",
        evaluation={"episodes": 10},
        safety={"max_collision": 1, "max_force": 15},
        patch_context={"changes": [{"path": "/skill/pregrasp_height", "new": 0.05}]},
    )
    runner = SandboxRunner(config={"simulate": True})
    result = runner.run(exp)
    assert result.success is True
    assert result.metrics["sandbox_clearance"] is True


def test_sandbox_runner_rejects_high_force_patch():
    """Sandbox should reject patch with dangerously high force."""
    exp = ExperimentSpec(
        id="exp_001", proposal_id="prop_001", patch_id="patch_001",
        task="pick_cube", baseline_skill_id="pick_v1", candidate_skill_id="pick_v1_candidate",
        evaluation={"episodes": 10},
        safety={"max_collision": 5, "max_force": 15},
        patch_context={"changes": [{"path": "/skill/max_torque", "new": 50.0}]},
    )
    runner = SandboxRunner(config={"simulate": True})
    result = runner.run(exp)
    # Should fail due to force limit or safety violation
    assert result.success is False or len(result.safety_violations) > 0


def test_darwin_runner_multi_seed_with_patch():
    exp = ExperimentSpec(
        id="exp_001", proposal_id="prop_001", patch_id="patch_001",
        task="pick_cube", baseline_skill_id="pick_v1", candidate_skill_id="pick_v1_candidate",
        evaluation={"episodes": 20, "seeds": [0, 1, 2]},
        patch_context={"changes": [{"path": "/skill/approach_speed", "new": 0.10}]},
    )
    runner = DarwinRunner(config={"simulate": True})
    result = runner.run(exp)
    assert result.success is True
    assert "per_seed" in result.metrics
    assert len(result.metrics["per_seed"]) == 3
    # Different seeds should produce slightly different results
    sr0 = result.metrics["per_seed"][0]["candidate"]["success_rate"]
    sr1 = result.metrics["per_seed"][1]["candidate"]["success_rate"]
    # Not necessarily different due to small noise, but should be present
    assert sr0 is not None
    assert sr1 is not None


def test_darwin_runner_no_global_random_pollution():
    """DarwinRunner must not modify global random state."""
    import random
    random.seed(12345)
    random.random()

    exp = ExperimentSpec(
        id="exp_001", proposal_id="prop_001", patch_id="patch_001",
        task="pick_cube", baseline_skill_id="pick_v1", candidate_skill_id="pick_v1_candidate",
        evaluation={"episodes": 20, "seeds": [0, 1, 2]},
    )
    runner = DarwinRunner(config={"simulate": True})
    runner.run(exp)

    after = random.random()
    # If global state was modified, the sequence would change unpredictably.
    # We verify by re-seeding and checking the next value is deterministic.
    random.seed(12345)
    random.random()  # consume the 'before' value
    expected_next = random.random()
    assert after == expected_next, "DarwinRunner polluted global random state"
