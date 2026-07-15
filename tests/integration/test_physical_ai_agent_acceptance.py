"""No-hardware acceptance loop for the physical-AI agent infrastructure."""

import json
import time

from rosclaw.auto.promotion.gate import PromotionGate
from rosclaw.core.runtime import Runtime, RuntimeConfig


def test_failure_evolves_through_how_auto_darwin_and_skill_registry(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path / ".rosclaw"))
    runtime = Runtime(
        RuntimeConfig(
            robot_id="ur5e",
            enable_firewall=True,
            enable_memory=True,
            enable_practice=True,
            enable_how=True,
            enable_auto=True,
            timeline_output_dir=str(tmp_path / "practice"),
        )
    )
    runtime.initialize()
    runtime.start()

    try:
        action = {
            "request_id": "acceptance_failure_001",
            "task_id": "reach_fixture",
            "skill_name": "reach",
            "instruction": "Reject an unsafe joint target and learn from it",
            "capability": "skill.reach",
            "trajectory": [
                [0.0, -1.57, 1.57, 0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        }
        result = runtime.execute(action)
        assert result["status"] in {"blocked", "error"}

        deadline = time.monotonic() + 2.0
        topics = []
        while time.monotonic() < deadline:
            topics = [event.topic for event in runtime.event_bus.get_history(limit=200)]
            if "rosclaw.how.recovery_hint.generated" in topics:
                break
            time.sleep(0.05)
        assert "firewall.action_blocked" in topics
        assert "rosclaw.how.recovery_hint.generated" in topics
        assert "praxis.failed" in topics
        episode_id = next(
            event.payload["episode_id"]
            for event in runtime.event_bus.get_history(topic="praxis.failed")
        )
        metadata_path = (
            tmp_path / ".rosclaw" / "artifacts" / "episodes" / episode_id / "metadata.json"
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert metadata["status"] == "BLOCKED"
        assert "firewall.action_blocked" in metadata["received_events"]
        assert "praxis.failed" in metadata["received_events"]

        memory = runtime.memory
        assert memory is not None
        assert memory.explain_last_failure() is not None

        auto_plugin = runtime.auto
        assert auto_plugin is not None and auto_plugin.engine is not None
        engine = auto_plugin.engine
        failures = engine.list_failures("reach_fixture")
        assert failures

        proposals = engine.list_proposals("reach_fixture")
        assert proposals
        proposal = proposals[0]
        patch = engine.create_patch(
            proposal.id,
            "reach",
            changes=[{"path": "/skill/pregrasp_height", "old": 0.02, "new": 0.05}],
            rollback_plan={"restore": 0.02},
        )
        experiment = engine.create_experiment(
            proposal.id,
            patch.id,
            "reach_fixture",
            "reach",
            "reach_candidate",
            episodes=6,
            seeds=[0, 1, 2],
        )
        experiment.safety = {"sandbox_required": True, "max_collision": 2, "max_force": 15}
        engine.store.save("experiments", experiment.id, experiment.to_dict())

        sandbox_result = engine.run_experiment(experiment, runner="sandbox")
        assert sandbox_result["success"] is True
        assert sandbox_result["metrics"]["sandbox_clearance"] is True

        darwin_result = engine.run_experiment(experiment, runner="darwin")
        assert darwin_result["success"] is True
        assert set(darwin_result["metrics"]["per_seed"]) == {0, 1, 2}

        engine.promotion_gate = PromotionGate(
            {"min_success_improvement": 0.0, "max_collision_increase": 0.05}
        )
        evaluation = engine.create_evaluation(
            experiment.id,
            darwin_result["metrics"]["baseline"],
            darwin_result["metrics"]["candidate"],
            darwin_result["metrics"]["per_seed"],
            sandbox_risk_score=0.0,
        )
        assert evaluation.decision.startswith("promote")

        champion = engine.promote_champion(
            "reach_candidate",
            "reach_fixture",
            "sim",
            darwin_result["metrics"]["candidate"],
            parent_skill="reach",
            patch_id=patch.id,
            experiment_id=experiment.id,
        )
        assert champion.level == "sim"

        registry = runtime.skill_manager.registry
        registered = registry.get("reach_candidate")
        assert registered is not None
        assert registered.skill_type == "learned"
        assert registered.metadata["level"] == "sim"
    finally:
        runtime.stop()
