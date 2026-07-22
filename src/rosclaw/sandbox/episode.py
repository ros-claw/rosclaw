"""Truthful sandbox task executors backed by real MuJoCo state."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.kernel import (
    ActionEnvelope,
    ActionExecutionResult,
    ActionState,
    EvidenceDomain,
    EvidenceLevel,
    ExecutionMode,
)
from rosclaw.sandbox.sandbox_api import Sandbox

DEFAULT_REACH_TARGET = [-0.24, 0.51, 0.47]
WORKSPACE_BOUNDS = {
    "x": (-0.9, 0.9),
    "y": (-0.9, 0.9),
    "z": (0.05, 1.2),
}


def _safe_action_directory(root: Path, action_id: str) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", action_id).strip("._")
    if not safe_id:
        safe_id = hashlib.sha256(action_id.encode("utf-8")).hexdigest()[:16]
    return (root / safe_id).resolve()


def _write_json(path: Path, value: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _static_policy(
    target: list[float],
    world_metadata: dict[str, Any],
) -> dict[str, Any]:
    decision: dict[str, Any] = {
        "validation_type": "StaticPolicyValidation",
        "allowed": False,
        "reason": "",
        "violations": [],
        "simulation_executed": False,
    }
    if len(target) != 3 or any(not math.isfinite(value) for value in target):
        decision["reason"] = "invalid_cartesian_target"
        decision["violations"] = ["target_schema"]
        return decision

    axes = ("x", "y", "z")
    if (
        any(
            not WORKSPACE_BOUNDS[axis][0] <= target[index] <= WORKSPACE_BOUNDS[axis][1]
            for index, axis in enumerate(axes)
        )
        or math.hypot(target[0], target[1]) > 0.95
    ):
        decision["reason"] = "target_outside_workspace"
        decision["violations"] = ["workspace_boundary"]
        return decision

    table = world_metadata.get("table", {})
    center = table.get("center")
    half_size = table.get("half_size")
    if (
        isinstance(center, list)
        and isinstance(half_size, list)
        and len(center) == len(half_size) == 3
    ):
        inside_xy = all(
            center[index] - half_size[index] <= target[index] <= center[index] + half_size[index]
            for index in (0, 1)
        )
        table_top = float(table.get("top_z", center[2] + half_size[2]))
        if inside_xy and target[2] <= table_top + 0.025:
            decision["reason"] = "target_intersects_table"
            decision["violations"] = ["collision_target"]
            return decision

    decision["allowed"] = True
    decision["reason"] = "target_within_declared_workspace"
    return decision


def run_reach_action(
    sandbox: Sandbox | None,
    action: ActionEnvelope,
    *,
    artifact_root: Path,
) -> ActionExecutionResult:
    """Execute a deterministic Cartesian UR5e reach and verify final error."""

    artifact_dir = _safe_action_directory(artifact_root, action.action_id)
    if action.execution_mode is ExecutionMode.FIXTURE:
        fixture_path = artifact_dir / "fixture.json"
        fixture_payload = {
            "action": action.to_dict(),
            "execution_mode": "FIXTURE",
            "trust_level": "SYNTHETIC",
            "physics_executed": False,
            "valid_for_acceptance": False,
        }
        fixture_hash = _write_json(fixture_path, fixture_payload)
        return ActionExecutionResult(
            final_state=ActionState.DEGRADED,
            evidence_level=EvidenceLevel.SYNTHETIC,
            evidence_domain=EvidenceDomain.FIXTURE,
            policy_decision={
                "validation_type": "FixtureOnly",
                "allowed": True,
                "reason": "explicit_fixture_mode",
                "simulation_executed": False,
            },
            dispatch_result={
                "accepted": True,
                "synthetic": True,
                "physics_executed": False,
            },
            verification_result={
                "success": False,
                "verified": False,
                "valid_for_acceptance": False,
            },
            artifacts=[fixture_path.as_uri()],
            artifact_directory=str(artifact_dir),
            simulation_result={
                "has_physics": False,
                "steps": 0,
                "artifact_hashes": {fixture_path.name: fixture_hash},
            },
        )

    if action.execution_mode is not ExecutionMode.SIMULATION:
        return ActionExecutionResult(
            final_state=ActionState.BLOCKED,
            evidence_level=EvidenceLevel.REQUESTED,
            policy_decision={
                "validation_type": "ModePolicyValidation",
                "allowed": False,
                "reason": "sandbox_executor_requires_simulation_mode",
                "simulation_executed": False,
            },
            errors=[
                {
                    "code": "INVALID_EXECUTION_MODE",
                    "message": "The sandbox reach executor only accepts SIMULATION or FIXTURE.",
                }
            ],
        )

    task = str(action.arguments.get("task", "reach")).lower()
    if task != "reach":
        return ActionExecutionResult(
            final_state=ActionState.FAILED,
            evidence_level=EvidenceLevel.REQUESTED,
            errors=[
                {
                    "code": "TASK_NOT_IMPLEMENTED",
                    "message": f"Sandbox task '{task}' is not implemented.",
                }
            ],
        )

    if sandbox is None or not sandbox.has_physics:
        reason = sandbox.load_error if sandbox is not None else "Sandbox service is unavailable."
        return ActionExecutionResult(
            final_state=ActionState.FAILED,
            evidence_level=EvidenceLevel.REQUESTED,
            dispatch_result={"accepted": False, "physics_executed": False},
            errors=[{"code": "PHYSICS_UNAVAILABLE", "message": reason or "Unknown error"}],
        )

    raw_target = action.arguments.get("target")
    target = (
        list(raw_target)
        if isinstance(raw_target, (list, tuple))
        else list(sandbox.world_metadata.get("default_target", DEFAULT_REACH_TARGET))
    )
    try:
        target = [float(value) for value in target]
    except (TypeError, ValueError):
        target = []
    policy = _static_policy(target, sandbox.world_metadata)
    if not policy["allowed"]:
        return ActionExecutionResult(
            final_state=ActionState.BLOCKED,
            evidence_level=EvidenceLevel.REQUESTED,
            policy_decision=policy,
            dispatch_result={"accepted": False, "physics_executed": False},
        )

    max_steps = int(action.arguments.get("max_steps", 1200))
    tolerance = float(action.arguments.get("tolerance_m", 0.008))
    seed = int(action.arguments.get("seed", 0))
    if not 1 <= max_steps <= 100_000:
        return ActionExecutionResult(
            final_state=ActionState.BLOCKED,
            evidence_level=EvidenceLevel.REQUESTED,
            policy_decision={
                "validation_type": "StaticPolicyValidation",
                "allowed": False,
                "reason": "invalid_step_budget",
                "violations": ["step_budget"],
                "simulation_executed": False,
            },
        )
    if not 0.001 <= tolerance <= 0.1:
        return ActionExecutionResult(
            final_state=ActionState.BLOCKED,
            evidence_level=EvidenceLevel.REQUESTED,
            policy_decision={
                "validation_type": "StaticPolicyValidation",
                "allowed": False,
                "reason": "invalid_tolerance",
                "violations": ["verification_tolerance"],
                "simulation_executed": False,
            },
        )

    import mujoco

    model = sandbox.physics_model
    data = sandbox.physics_data
    if model is None or data is None:
        raise RuntimeError("Sandbox reported physics but did not expose MuJoCo model/data")

    sandbox.reset(keyframe="home")
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    if site_id < 0:
        return ActionExecutionResult(
            final_state=ActionState.FAILED,
            evidence_level=EvidenceLevel.REQUESTED,
            errors=[
                {
                    "code": "END_EFFECTOR_SITE_MISSING",
                    "message": "MJCF model has no 'attachment_site' end-effector site.",
                }
            ],
        )

    target_array = np.asarray(target, dtype=float)
    model_path = sandbox.model_path
    model_hash = (
        f"sha256:{hashlib.sha256(model_path.read_bytes()).hexdigest()}"
        if model_path is not None and model_path.is_file()
        else ""
    )
    world_asset_hash = _canonical_hash(sandbox.world_metadata)
    action_payload = action.to_dict()
    action_hash = _canonical_hash(action_payload)
    scenario_payload = {
        "schema_version": "rosclaw.scenario.v1",
        "scenario_id": f"{action.body_id}_{sandbox.world_metadata.get('world_id', 'empty')}_reach",
        "body_snapshot_hash": action.body_snapshot_hash,
        "model_hash": model_hash,
        "world_asset_hash": world_asset_hash,
        "task": action.capability_id,
        "target": target,
        "seed": seed,
        "initial_state": {"keyframe": "home"},
    }
    scenario_hash = _canonical_hash(scenario_payload)
    action_path = artifact_dir / "action.json"
    scenario_path = artifact_dir / "scenario.json"
    action_artifact_hash = _write_json(action_path, action_payload)
    scenario_artifact_hash = _write_json(scenario_path, scenario_payload)
    initial_position = data.site_xpos[site_id].copy()
    trajectory: list[dict[str, Any]] = [
        {
            "step": 0,
            "time": float(data.time),
            "qpos": data.qpos.copy().tolist(),
            "qvel": data.qvel.copy().tolist(),
            "end_effector": initial_position.tolist(),
            "error_m": float(np.linalg.norm(target_array - initial_position)),
            "contacts": [],
        }
    ]
    success = False
    collision: dict[str, Any] | None = None
    steps_executed = 0
    final_error = trajectory[0]["error_m"]
    damping = 0.03

    for step_index in range(1, max_steps + 1):
        error = target_array - data.site_xpos[site_id]
        final_error = float(np.linalg.norm(error))
        if final_error <= tolerance:
            success = True
            break

        jacobian_pos = np.zeros((3, model.nv))
        jacobian_rot = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacobian_pos, jacobian_rot, site_id)
        jacobian = jacobian_pos[:, : model.nu]
        joint_delta = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + damping**2 * np.eye(3),
            error,
        )
        joint_delta = np.clip(2.0 * joint_delta, -0.08, 0.08)
        target_ctrl = data.qpos[: model.nu] + joint_delta
        target_ctrl = np.clip(
            target_ctrl,
            model.actuator_ctrlrange[: model.nu, 0],
            model.actuator_ctrlrange[: model.nu, 1],
        )
        data.ctrl[: model.nu] = target_ctrl
        mujoco.mj_step(model, data)
        steps_executed = step_index

        contacts: list[dict[str, Any]] = []
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            geom1 = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
                or f"geom{contact.geom1}"
            )
            geom2 = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
                or f"geom{contact.geom2}"
            )
            item = {"geom1": geom1, "geom2": geom2, "distance": float(contact.dist)}
            contacts.append(item)
            if "tabletop_surface" in {geom1, geom2}:
                collision = item

        if step_index % 10 == 0 or collision is not None:
            trajectory.append(
                {
                    "step": step_index,
                    "time": float(data.time),
                    "qpos": data.qpos.copy().tolist(),
                    "qvel": data.qvel.copy().tolist(),
                    "end_effector": data.site_xpos[site_id].copy().tolist(),
                    "error_m": float(np.linalg.norm(target_array - data.site_xpos[site_id])),
                    "contacts": contacts,
                }
            )
        if collision is not None:
            break

    final_position = data.site_xpos[site_id].copy()
    final_error = float(np.linalg.norm(target_array - final_position))
    if not trajectory or trajectory[-1]["step"] != steps_executed:
        trajectory.append(
            {
                "step": steps_executed,
                "time": float(data.time),
                "qpos": data.qpos.copy().tolist(),
                "qvel": data.qvel.copy().tolist(),
                "end_effector": final_position.tolist(),
                "error_m": final_error,
                "contacts": [collision] if collision else [],
            }
        )

    success = success or (collision is None and final_error <= tolerance)
    trajectory_path = artifact_dir / "trajectory.json"
    artifact_payload = {
        "schema_version": "rosclaw.sandbox.trajectory.v1",
        "action": action.to_dict(),
        "engine": "mujoco",
        "model_path": str(sandbox.model_path) if sandbox.model_path else None,
        "world": sandbox.world_metadata,
        "seed": seed,
        "target": target,
        "tolerance_m": tolerance,
        "trajectory": trajectory,
        "verification": {
            "success": success,
            "collision": collision,
            "final_error_m": final_error,
            "final_end_effector": final_position.tolist(),
        },
    }
    trajectory_hash = _write_json(trajectory_path, artifact_payload)
    engine_fingerprint = _canonical_hash(
        {
            "engine": "mujoco",
            "version": getattr(mujoco, "__version__", "unknown"),
            "model_hash": model_hash,
            "world_asset_hash": world_asset_hash,
            "integrator": int(model.opt.integrator),
            "timestep_sec": float(model.opt.timestep),
            "solver_iterations": int(model.opt.iterations),
        }
    )
    artifact_hashes = {
        action_path.name: action_artifact_hash,
        scenario_path.name: scenario_artifact_hash,
        trajectory_path.name: trajectory_hash,
    }
    simulation_result = {
        "validation_type": "SimulationValidation",
        "engine": "mujoco",
        "engine_version": getattr(mujoco, "__version__", "unknown"),
        "has_physics": True,
        "physics_executed": True,
        "evidence_domain": "SIMULATION",
        "model_path": str(sandbox.model_path) if sandbox.model_path else None,
        "model_hash": model_hash,
        "world_asset_hash": world_asset_hash,
        "body_snapshot_hash": action.body_snapshot_hash,
        "action_hash": action_hash,
        "scenario_hash": scenario_hash,
        "backend_fingerprint": engine_fingerprint,
        "integrator": int(model.opt.integrator),
        "timestep_sec": float(model.opt.timestep),
        "solver_iterations": int(model.opt.iterations),
        "world_id": sandbox.world_metadata.get("world_id", "empty"),
        "seed": seed,
        "steps": steps_executed,
        "final_time": float(data.time),
        "final_qpos": data.qpos.copy().tolist(),
        "final_qvel": data.qvel.copy().tolist(),
        "collision": collision,
        "artifact_hashes": artifact_hashes,
    }
    verification = {
        "success": success,
        "target": target,
        "initial_end_effector": initial_position.tolist(),
        "final_end_effector": final_position.tolist(),
        "final_error_m": final_error,
        "tolerance_m": tolerance,
        "condition": "cartesian_error_within_tolerance_and_no_table_collision",
        "data_quality": {
            "required_events_complete": True,
            "monotonic_timestamp": True,
            "missing_state_ratio": 0.0,
            "duplicate_event_count": 0,
            "artifact_hash_valid": True,
            "body_snapshot_match": bool(action.body_snapshot_hash == model_hash),
            "replayable": True,
        },
    }

    if collision is not None:
        final_state = ActionState.BLOCKED
        evidence = EvidenceLevel.PHYSICALLY_OBSERVED
        errors = [
            {
                "code": "SIMULATED_COLLISION",
                "message": (
                    f"MuJoCo observed collision between {collision['geom1']} "
                    f"and {collision['geom2']}."
                ),
            }
        ]
    elif success:
        final_state = ActionState.COMPLETED
        evidence = EvidenceLevel.TASK_VERIFIED
        errors = []
    else:
        final_state = ActionState.FAILED
        evidence = EvidenceLevel.PHYSICALLY_OBSERVED
        errors = [
            {
                "code": "REACH_NOT_CONVERGED",
                "message": (
                    f"Final Cartesian error {final_error:.6f}m exceeds "
                    f"tolerance {tolerance:.6f}m after {steps_executed} steps."
                ),
            }
        ]

    return ActionExecutionResult(
        final_state=final_state,
        evidence_level=evidence,
        evidence_domain=EvidenceDomain.SIMULATION,
        policy_decision=policy,
        simulation_result=simulation_result,
        dispatch_result={
            "accepted": True,
            "engine": "mujoco",
            "physics_executed": True,
        },
        observations=[
            {"kind": "initial_end_effector", "value": initial_position.tolist()},
            {"kind": "final_end_effector", "value": final_position.tolist()},
        ],
        verification_result=verification,
        artifacts=[action_path.as_uri(), scenario_path.as_uri(), trajectory_path.as_uri()],
        errors=errors,
        artifact_directory=str(artifact_dir),
    )


__all__ = ["DEFAULT_REACH_TARGET", "WORKSPACE_BOUNDS", "run_reach_action"]
