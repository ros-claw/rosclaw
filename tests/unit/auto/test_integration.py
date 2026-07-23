"""Integration tests for Sprint B/C/D end-to-end workflow."""

import shutil

from rosclaw.auto.config import AutoConfig
from rosclaw.auto.engine.auto_engine import AutoEngine
from rosclaw.auto.events.publishers import AutoPublisher
from rosclaw.auto.events.schemas import PraxisFailedEvent
from rosclaw.auto.events.subscribers import AutoSubscriber


class FakeBus:
    def __init__(self):
        self.subscriptions = {}
        self.published = []

    def subscribe(self, topic, handler):
        self.subscriptions.setdefault(topic, []).append(handler)

    def unsubscribe(self, topic, handler):
        if topic in self.subscriptions:
            self.subscriptions[topic] = [h for h in self.subscriptions[topic] if h != handler]

    def publish(self, event):
        self.published.append(event)


def test_end_to_end_failure_to_proposal():
    """Simulate: PraxisFailedEvent -> FailureCase -> Proposal."""
    store_path = "./.rosclaw_auto_test_e2e_v2"
    shutil.rmtree(store_path, ignore_errors=True)
    engine = AutoEngine(config=AutoConfig(local_store_path=store_path))
    bus = FakeBus()
    sub = AutoSubscriber(engine=engine, event_bus=bus)
    AutoPublisher(event_bus=bus)
    sub.subscribe_all()

    # Simulate 3 failures to cross threshold
    for i in range(3):
        evt = PraxisFailedEvent(
            event_id=f"evt_{i}",
            task_id="pick_cube",
            skill_id="pick_v1",
            failure_mode="missed_grasp",
            severity="medium",
            evidence={"search_space": {"pregrasp_height": [0.02, 0.08]}},
        )
        bus.subscriptions["rosclaw.practice.failed"][0](evt)

    failures = engine.list_failures("pick_cube")
    assert len(failures) == 3
    proposals = engine.list_proposals("pick_cube")
    assert len(proposals) >= 1  # threshold = 3 triggered proposal


def test_end_to_end_experiment_and_evaluation():
    """Full loop: experiment -> runner -> evaluation -> decision."""
    engine = AutoEngine(
        config=AutoConfig(local_store_path="./.rosclaw_auto_test_e2e2", runner_backend="mock")
    )
    engine.create_task("pick_cube", "panda", "pick_v1")

    # Create and run experiment
    exp = engine.create_experiment(
        "prop_001", "patch_001", "pick_cube", "pick_v1", "pick_v1_candidate", episodes=10
    )
    exp.safety = {"sandbox_required": False, "max_collision": 0, "max_force": 15}
    engine.store.save("experiments", exp.id, exp.to_dict())
    raw = engine.run_experiment(exp, runner="local")
    assert raw["success"] is True
    assert "baseline" in raw["metrics"]
    assert "candidate" in raw["metrics"]

    # Evaluate
    b = raw["metrics"]["baseline"]
    c = raw["metrics"]["candidate"]
    per_seed = raw["metrics"].get("per_seed")
    eval_res = engine.create_evaluation(exp.id, b, c, per_seed)
    assert eval_res.decision in ["promote_to_sim", "reject", "need_more_evidence"]


def test_end_to_end_champion_and_lineage(store_test_champion):
    """Test champion promotion + lineage tracking."""
    engine = AutoEngine(config=AutoConfig(local_store_path="./.rosclaw_auto_test_e2e3"))
    task = engine.create_task("pick_cube", "panda", "pick_v1")

    champ = store_test_champion(
        engine,
        "pick_v1.5",
        task.id,
        "sim",
        {"success_rate": 0.76},
        "pick_v1",
        "patch_001",
        "exp_001",
    )
    assert champ.level == "sim"

    lineage = engine.get_lineage("pick_v1.5")
    assert len(lineage) >= 1
    assert lineage[-1].skill_id == "pick_v1.5"

    tree = engine.render_lineage_tree("pick_v1")
    assert "pick_v1.5" in tree


def test_end_to_end_rollback(store_test_champion):
    """Test rollback from sim to baseline."""
    engine = AutoEngine(config=AutoConfig(local_store_path="./.rosclaw_auto_test_e2e4"))
    task = engine.create_task("pick_cube", "panda", "pick_v1")

    engine.promote_champion("pick_v1", task.id, "baseline", {}, "", "", "")
    store_test_champion(
        engine, "pick_v1.5", task.id, "sim", {"success_rate": 0.76}, "pick_v1", "p1", "e1"
    )

    rolled = engine.rollback_skill(task.id)
    assert rolled is not None
    assert rolled.level == "baseline"
