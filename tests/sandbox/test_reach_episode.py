"""System tests for the real MuJoCo UR5e reach path."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse

from rosclaw.core.runtime import Runtime, RuntimeConfig
from rosclaw.kernel import ActionEnvelope, ActionState, EvidenceLevel, ExecutionMode


def _runtime(tmp_path: Path, *, robot_id: str = "sim_ur5e") -> Runtime:
    runtime = Runtime(
        RuntimeConfig(
            robot_id=robot_id,
            default_eurdf_robot="ur5e",
            enable_event_persistence=False,
            enable_tracing=True,
            trace_home=str(tmp_path / "trace"),
            enable_firewall=False,
            enable_memory=False,
            enable_practice=False,
            enable_skill_manager=False,
            enable_knowledge=False,
            enable_how=False,
            enable_auto=False,
            enable_provider=False,
            enable_sense=False,
            sandbox_world_id="tabletop",
            sandbox_artifact_root=str(tmp_path / "artifacts"),
        )
    )
    runtime.initialize()
    return runtime


def _action(
    action_id: str,
    *,
    target: list[float] | None = None,
    robot_id: str = "sim_ur5e",
) -> ActionEnvelope:
    arguments: dict[str, object] = {"task": "reach", "seed": 7, "max_steps": 1200}
    if target is not None:
        arguments["target"] = target
    return ActionEnvelope(
        action_id=action_id,
        actor_id="pytest",
        agent_framework="pytest",
        session_id="sandbox-system-test",
        body_id=robot_id,
        body_snapshot_hash="sha256:test-ur5e",
        capability_id="sandbox.reach",
        arguments=arguments,
        execution_mode=ExecutionMode.SIMULATION,
    )


def _file_from_uri(uri: str) -> Path:
    parsed = urlparse(uri)
    assert parsed.scheme == "file"
    return Path(parsed.path)


def test_legal_reach_advances_physics_and_writes_real_artifacts(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    try:
        receipt = runtime.submit_action(_action("reach-success"))
    finally:
        runtime.stop()

    assert receipt.final_state is ActionState.COMPLETED
    assert receipt.evidence_level is EvidenceLevel.TASK_VERIFIED
    assert receipt.verified is True
    assert receipt.simulation_result is not None
    assert receipt.simulation_result["has_physics"] is True
    assert receipt.simulation_result["steps"] > 0
    assert receipt.simulation_result["final_time"] > 0.0
    assert receipt.verification_result is not None
    assert receipt.verification_result["success"] is True
    assert receipt.verification_result["final_error_m"] < 0.02
    transition_states = [transition.state for transition in receipt.transitions]
    assert ActionState.POLICY_VALIDATED in transition_states
    assert ActionState.SIMULATION_VALIDATED in transition_states
    assert ActionState.EFFECT_OBSERVED in transition_states
    assert ActionState.TASK_VERIFIED in transition_states

    artifact_paths = [_file_from_uri(uri) for uri in receipt.artifacts]
    assert artifact_paths
    assert all(path.is_file() for path in artifact_paths)
    receipt_path = next(path for path in artifact_paths if path.name == "receipt.json")
    persisted = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert persisted["action_id"] == "reach-success"
    assert persisted["final_state"] == "COMPLETED"


def test_out_of_bounds_target_is_blocked_before_simulation(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    try:
        receipt = runtime.submit_action(_action("reach-oob", target=[2.0, 0.0, 0.5]))
    finally:
        runtime.stop()

    assert receipt.final_state is ActionState.BLOCKED
    assert receipt.verified is False
    assert receipt.policy_decision["validation_type"] == "StaticPolicyValidation"
    assert receipt.policy_decision["reason"] == "target_outside_workspace"
    assert receipt.simulation_result is None


def test_target_inside_table_is_blocked_as_collision_goal(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    try:
        receipt = runtime.submit_action(_action("reach-collision", target=[-0.25, 0.5, 0.18]))
    finally:
        runtime.stop()

    assert receipt.final_state is ActionState.BLOCKED
    assert receipt.policy_decision["reason"] == "target_intersects_table"
    assert "collision_target" in receipt.policy_decision["violations"]


def test_missing_robot_model_fails_instead_of_using_stub(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, robot_id="missing_robot_model")
    try:
        receipt = runtime.submit_action(_action("reach-missing", robot_id="missing_robot_model"))
    finally:
        runtime.stop()

    assert receipt.final_state is ActionState.FAILED
    assert receipt.verified is False
    assert receipt.errors[0]["code"] == "PHYSICS_UNAVAILABLE"


def test_reach_is_reproducible_and_not_a_fixed_result(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    try:
        first = runtime.submit_action(_action("reach-repeat-a"))
        second = runtime.submit_action(_action("reach-repeat-b"))
        different = runtime.submit_action(_action("reach-different", target=[-0.13, 0.62, 0.46]))
    finally:
        runtime.stop()

    assert first.verification_result == second.verification_result
    assert first.simulation_result is not None
    assert second.simulation_result is not None
    assert first.simulation_result["final_qpos"] == second.simulation_result["final_qpos"]
    assert different.verification_result != first.verification_result
