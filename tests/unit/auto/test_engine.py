"""Integration tests for AutoEngine."""
import tempfile

import pytest

from rosclaw.auto.config import AutoConfig
from rosclaw.auto.engine import AutoEngine
from rosclaw.auto.storage import LocalStore
from rosclaw.core.event_bus import EventBus
from rosclaw.sense.config import SenseConfig
from rosclaw.sense.runtime import SenseRuntime


@pytest.fixture
def sense_runtime_kick_not_ready():
    bus = EventBus()
    cfg = SenseConfig(
        robot_id="g1_lab_01",
        collector="mock",
        update_hz=0.0,
        extra={"scenario": "kick_not_ready"},
    )
    runtime = SenseRuntime(cfg, event_bus=bus, robot_id="g1_lab_01")
    runtime.initialize()
    runtime.tick()
    yield runtime
    runtime.stop()


@pytest.fixture
def sense_runtime_normal():
    bus = EventBus()
    cfg = SenseConfig(
        robot_id="g1_lab_01",
        collector="mock",
        update_hz=0.0,
        extra={"scenario": "normal"},
    )
    runtime = SenseRuntime(cfg, event_bus=bus, robot_id="g1_lab_01")
    runtime.initialize()
    runtime.tick()
    yield runtime
    runtime.stop()


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


class TestAutoEngineBodySense:
    def test_create_failure_case_with_body_condition(self, sense_runtime_kick_not_ready):
        with tempfile.TemporaryDirectory() as tmp:
            config = AutoConfig(local_store_path=tmp)
            engine = AutoEngine(config, sense_runtime=sense_runtime_kick_not_ready)

            fc = engine.create_failure_case(
                "evt_sense_001", "task_001", "kick_ball",
                phase="swing", failure_mode="overheat",
            )
            assert fc.evidence.get("body_condition_failure") is True
            assert "body_sense_snapshot" in fc.evidence
            assert fc.evidence["body_sense_snapshot"]["overall_status"] == "not_ready"

    def test_create_failure_case_ready_state_not_flagged(self, sense_runtime_normal):
        with tempfile.TemporaryDirectory() as tmp:
            config = AutoConfig(local_store_path=tmp)
            engine = AutoEngine(config, sense_runtime=sense_runtime_normal)

            fc = engine.create_failure_case(
                "evt_sense_002", "task_002", "observe_scene",
                phase="perceive", failure_mode="target_lost",
            )
            assert fc.evidence.get("body_condition_failure") is False
            assert "body_sense_snapshot" in fc.evidence

    def test_create_failure_case_without_sense_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = AutoConfig(local_store_path=tmp)
            engine = AutoEngine(config)

            fc = engine.create_failure_case(
                "evt_sense_003", "task_003", "pick_v1",
                phase="grasp", failure_mode="missed_grasp",
                evidence={"custom": True},
            )
            assert "body_condition_failure" not in fc.evidence
            assert fc.evidence.get("custom") is True
