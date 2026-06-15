"""Tests for Sprint D: Champion/DeadEnd/Promotion Gate."""
from rosclaw.auto.core.champion import Champion
from rosclaw.auto.promotion import ChampionStore, LineageTracker, PromotionGate, RollbackManager
from rosclaw.auto.storage.local_store import LocalStore


def test_promotion_gate_passes():
    gate = PromotionGate({"min_success_improvement": 0.05, "max_collision_increase": 0.0})
    baseline = {"success_rate": 0.40, "collision_rate": 0.10}
    candidate = {"success_rate": 0.55, "collision_rate": 0.08}
    per_seed = {
        0: {"candidate": {"success_rate": 0.54}},
        1: {"candidate": {"success_rate": 0.56}},
    }
    result = gate.evaluate(baseline, candidate, current_level="baseline",
                           per_seed=per_seed, sandbox_risk_score=0.1)
    assert result.passed is True
    assert result.decision == "promote_to_sim"
    assert result.next_level == "sim"


def test_promotion_gate_rejects_worse():
    gate = PromotionGate()
    baseline = {"success_rate": 0.60, "collision_rate": 0.05}
    candidate = {"success_rate": 0.45, "collision_rate": 0.15}
    result = gate.evaluate(baseline, candidate)
    assert result.passed is False
    assert result.decision == "reject"


def test_promotion_gate_need_more_data():
    gate = PromotionGate()
    baseline = {"success_rate": 0.40, "collision_rate": 0.10}
    candidate = {"success_rate": 0.42, "collision_rate": 0.10}  # small improvement
    result = gate.evaluate(baseline, candidate)
    assert result.passed is False
    assert result.decision == "need_more_data"


def test_champion_store_save_and_get():
    store = LocalStore("./.rosclaw_auto_test_promo")
    cs = ChampionStore(store)
    champ = Champion(id="champ_001", skill_id="pick_v1.5", task_id="pick_cube",
                     level="sim", metrics={"success_rate": 0.76})
    cs.save_champion(champ)
    retrieved = cs.get_champion("pick_cube", "sim")
    assert retrieved is not None
    assert retrieved.skill_id == "pick_v1.5"


def test_champion_store_best_champion():
    store = LocalStore("./.rosclaw_auto_test_promo2")
    cs = ChampionStore(store)
    cs.save_champion(Champion(id="c1", skill_id="pick_v1", task_id="pick_cube", level="baseline", metrics={}))
    cs.save_champion(Champion(id="c2", skill_id="pick_v1.5", task_id="pick_cube", level="sim", metrics={}))
    cs.save_champion(Champion(id="c3", skill_id="pick_v1.8", task_id="pick_cube", level="real", metrics={}))
    best = cs.get_best_champion("pick_cube")
    assert best is not None
    assert best.level == "real"


def test_rollback_manager():
    store = LocalStore("./.rosclaw_auto_test_promo3")
    cs = ChampionStore(store)
    rb = RollbackManager(store)

    cs.save_champion(Champion(id="c1", skill_id="pick_v1", task_id="pick_cube", level="baseline", metrics={}))
    cs.save_champion(Champion(id="c2", skill_id="pick_v1.5", task_id="pick_cube", level="sim", metrics={}))

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
