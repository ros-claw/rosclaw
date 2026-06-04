"""Integration tests for AutoEngine."""
import tempfile
import shutil
from rosclaw.auto.engine import AutoEngine
from rosclaw.auto.config import AutoConfig
from rosclaw.auto.storage import LocalStore


def test_full_workflow():
    with tempfile.TemporaryDirectory() as tmp:
        config = AutoConfig(local_store_path=tmp)
        engine = AutoEngine(config, LocalStore(tmp))

        # 1. Create task
        task = engine.create_task("pick_cube_auto", "panda", "pick_v1", env="maniskill")
        assert task.id.startswith("task_")

        # 2. Simulate failure
        fc = engine.create_failure_case("evt_001", task.id, "pick_v1",
                                        phase="grasp", failure_mode="missed_grasp")
        assert fc.failure_mode == "missed_grasp"

        # 3. Create proposal
        prop = engine.create_proposal(fc.id, task.name, task.target_skill_id,
                                      "increase pregrasp height", {"z": [0.02, 0.08]})
        assert prop.id.startswith("prop_")

        # 4. Create patch & experiment
        patch = engine.create_patch(prop.id, task.target_skill_id,
                                    [{"path": "/z", "old": 0.02, "new": 0.05}])
        exp = engine.create_experiment(prop.id, patch.id, task.name,
                                       "pick_v1", "pick_v1_candidate")
        assert exp.id.startswith("exp_")

        # 5. Evaluate (promote case)
        ev = engine.create_evaluation(exp.id,
                                      {"success_rate": 0.42, "collision_rate": 0.12},
                                      {"success_rate": 0.68, "collision_rate": 0.03})
        assert ev.decision.startswith("promote")

        # 6. Promote champion
        champ = engine.promote_champion("pick_v2", task.id, "sim",
                                        ev.candidate_metrics, "pick_v1", patch.id, exp.id)
        assert champ.level == "sim"

        # 7. Register dead-end
        de = engine.register_deadend(task.id, "scale > 0.8", "collision spikes")
        assert de.id.startswith("de_")

        # 8. Generate report
        report = engine.generate_report(task.id)
        assert report.proposals_created >= 1
        assert report.champions_promoted >= 1
        assert report.deadends_registered >= 1

        # 9. Run dry-run evolution
        report2 = engine.run(task.id, rounds=2, dry_run=True)
        assert report2.proposals_created >= 1
