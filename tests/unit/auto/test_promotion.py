"""Tests for Sprint D: Champion/DeadEnd/Promotion Gate."""

from rosclaw.auto.core.champion import Champion
from rosclaw.auto.promotion import ChampionStore, LineageTracker, PromotionGate, RollbackManager
from rosclaw.auto.storage.local_store import LocalStore
from rosclaw.sandbox.backends import ReplayReport
from rosclaw.sandbox.evidence import SimulationEvidenceVerification


def _simulation_receipt(seed: int, variant: str) -> dict:
    is_safe = variant == "candidate"
    return {
        "schema_version": "rosclaw.simulation_receipt.v1",
        "evidence_domain": "SIMULATION",
        "physics_executed": True,
        "scenario_id": f"pair-{seed}",
        "pair_id": f"pair-{seed}",
        "evaluation_variant": variant,
        "seed": seed,
        "body_snapshot_hash": "sha256:body",
        "is_safe": is_safe,
        "collision_pairs": [] if is_safe else [["robot", "table"]],
        "model_hash": "sha256:model",
        "world_asset_hash": "sha256:world",
        "backend": {"name": "mujoco_cpu", "fingerprint": "sha256:backend"},
        "request": {
            "scenario": {
                "schema_version": "rosclaw.scenario.v1",
                "scenario_id": f"pair-{seed}",
                "robot_id": "ur5e",
                "world_id": "tabletop",
                "body_snapshot_hash": "sha256:body",
                "model_hash": "sha256:model",
                "seed": seed,
                "metadata": {"initial_qpos_jitter_rad": 0.002},
            },
            "trajectory": [[0.0 if variant == "baseline" else 0.1]],
            "control_dt_sec": None,
            "max_joint_delta_rad": 0.005,
            "max_joint_velocity_radps": 3.15,
            "max_final_tracking_error_rad": 0.25,
            "settle_steps": 100,
            "max_steps": 250_000,
        },
        "randomization": {
            "seed": seed,
            "seed_applied": True,
            "initial_qpos_jitter_rad": 0.002,
            "initial_state_hash": f"sha256:state-{seed}",
            "parameter_hash": "sha256:parameters",
        },
    }


def _verified(_receipt: dict) -> SimulationEvidenceVerification:
    replay = ReplayReport(True, True, True, True, 0.0, "strict_replay_verified")
    return SimulationEvidenceVerification(True, replay)


def test_promotion_gate_passes():
    gate = PromotionGate(
        {"min_success_improvement": 0.05, "max_collision_increase": 0.0},
        receipt_verifier=_verified,
    )
    baseline = {"success_rate": 0.0, "collision_rate": 1.0}
    candidate = {"success_rate": 1.0, "collision_rate": 0.0}
    per_seed = {
        0: {
            "baseline": {"success_rate": 0.0, "collision_rate": 1.0},
            "candidate": {"success_rate": 1.0, "collision_rate": 0.0},
        },
        1: {
            "baseline": {"success_rate": 0.0, "collision_rate": 1.0},
            "candidate": {"success_rate": 1.0, "collision_rate": 0.0},
        },
    }
    receipts = [
        _simulation_receipt(seed, variant)
        for seed in (0, 1)
        for variant in ("baseline", "candidate")
    ]
    regression = {
        "passed": True,
        "critical_regressions": [],
        "suite": "physics_counterexample_v1",
        "episodes": 4,
    }
    result = gate.evaluate(
        baseline,
        candidate,
        current_level="baseline",
        per_seed=per_seed,
        sandbox_risk_score=0.1,
        simulation_receipts=receipts,
        regression_results=regression,
    )
    assert result.passed is True
    assert result.decision == "promote_to_sim"
    assert result.next_level == "sim"

    missing_risk = gate.evaluate(
        baseline,
        candidate,
        current_level="baseline",
        per_seed=per_seed,
        simulation_receipts=receipts,
        regression_results=regression,
    )
    assert missing_risk.passed is False
    assert (
        next(check for check in missing_risk.checks if check["name"] == "sandbox_clearance")[
            "missing_evidence"
        ]
        is True
    )


def test_promotion_gate_rejects_worse():
    gate = PromotionGate()
    baseline = {"success_rate": 0.60, "collision_rate": 0.05}
    candidate = {"success_rate": 0.45, "collision_rate": 0.15}
    result = gate.evaluate(baseline, candidate)
    assert result.passed is False
    assert result.decision == "reject"


def test_promotion_gate_needs_physics_evidence():
    gate = PromotionGate()
    baseline = {"success_rate": 0.40, "collision_rate": 0.10}
    candidate = {"success_rate": 0.42, "collision_rate": 0.10}  # small improvement
    result = gate.evaluate(baseline, candidate)
    assert result.passed is False
    assert result.decision == "need_more_evidence"


def test_promotion_gate_fails_closed_when_receipt_verifier_raises():
    def broken_verifier(_receipt):
        raise RuntimeError("verifier unavailable")

    gate = PromotionGate(receipt_verifier=broken_verifier)
    result = gate.evaluate(
        {"success_rate": 0.0, "collision_rate": 1.0},
        {"success_rate": 1.0, "collision_rate": 0.0},
        simulation_receipts=[_simulation_receipt(0, "baseline")],
    )
    assert result.passed is False
    assert result.decision == "need_more_evidence"
    physics = next(check for check in result.checks if check["name"] == "physics_evidence")
    assert "verifier_error:RuntimeError" in physics["detail"]


def test_promotion_gate_rejects_unpaired_safety_thresholds():
    gate = PromotionGate(receipt_verifier=_verified)
    receipts = [
        _simulation_receipt(seed, variant)
        for seed in (0, 1)
        for variant in ("baseline", "candidate")
    ]
    for receipt in receipts:
        if receipt["evaluation_variant"] == "candidate":
            receipt["request"]["max_joint_velocity_radps"] = 1000.0
    per_seed = {
        seed: {
            "baseline": {"success_rate": 0.0, "collision_rate": 1.0},
            "candidate": {"success_rate": 1.0, "collision_rate": 0.0},
        }
        for seed in (0, 1)
    }
    result = gate.evaluate(
        {"success_rate": 0.0, "collision_rate": 1.0},
        {"success_rate": 1.0, "collision_rate": 0.0},
        per_seed=per_seed,
        sandbox_risk_score=0.0,
        simulation_receipts=receipts,
        regression_results={
            "passed": True,
            "critical_regressions": [],
            "suite": "physics_counterexample_v1",
            "episodes": 4,
        },
    )
    paired = next(check for check in result.checks if check["name"] == "paired_seed_evidence")
    assert result.passed is False
    assert paired["passed"] is False


def test_simulation_evidence_cannot_promote_beyond_sim():
    gate = PromotionGate(receipt_verifier=_verified)
    result = gate.evaluate(
        {"success_rate": 0.0, "collision_rate": 1.0},
        {"success_rate": 1.0, "collision_rate": 0.0},
        current_level="sim",
    )
    assert result.passed is False
    assert result.decision == "need_hardware_evidence"


def test_champion_store_save_and_get():
    store = LocalStore("./.rosclaw_auto_test_promo")
    cs = ChampionStore(store)
    champ = Champion(
        id="champ_001",
        skill_id="pick_v1.5",
        task_id="pick_cube",
        level="sim",
        metrics={"success_rate": 0.76},
    )
    cs.save_champion(champ)
    retrieved = cs.get_champion("pick_cube", "sim")
    assert retrieved is not None
    assert retrieved.skill_id == "pick_v1.5"


def test_champion_store_best_champion():
    store = LocalStore("./.rosclaw_auto_test_promo2")
    cs = ChampionStore(store)
    cs.save_champion(
        Champion(id="c1", skill_id="pick_v1", task_id="pick_cube", level="baseline", metrics={})
    )
    cs.save_champion(
        Champion(id="c2", skill_id="pick_v1.5", task_id="pick_cube", level="sim", metrics={})
    )
    cs.save_champion(
        Champion(id="c3", skill_id="pick_v1.8", task_id="pick_cube", level="real", metrics={})
    )
    best = cs.get_best_champion("pick_cube")
    assert best is not None
    assert best.level == "real"


def test_rollback_manager():
    store = LocalStore("./.rosclaw_auto_test_promo3")
    cs = ChampionStore(store)
    rb = RollbackManager(store)

    cs.save_champion(
        Champion(id="c1", skill_id="pick_v1", task_id="pick_cube", level="baseline", metrics={})
    )
    cs.save_champion(
        Champion(id="c2", skill_id="pick_v1.5", task_id="pick_cube", level="sim", metrics={})
    )

    target = rb.rollback("pick_cube")
    assert target is not None
    assert target.level == "baseline"


def test_lineage_tracker():
    store = LocalStore("./.rosclaw_auto_test_promo4")
    lt = LineageTracker(store)

    lt.record("pick_v1.1", "pick_v1.0", "patch_001", "exp_001", "improved", {"sr": 0.5})
    lt.record("pick_v1.2", "pick_v1.1", "patch_002", "exp_002", "champion", {"sr": 0.7})

    chain = lt.get_lineage("pick_v1.2")
    assert len(chain) == 3  # v1.0 (root), v1.1, v1.2
    assert chain[-1].skill_id == "pick_v1.2"
    assert chain[-1].result == "champion"


def test_lineage_tree_rendering():
    store = LocalStore("./.rosclaw_auto_test_promo5")
    lt = LineageTracker(store)
    lt.record("pick_v1.1", "pick_v1.0", "p1", "e1", "improved", {})
    lt.record("pick_v1.2", "pick_v1.1", "p2", "e2", "champion", {})
    tree = lt.render_tree("pick_v1.0")
    assert "pick_v1.0" in tree
    assert "champion" in tree
