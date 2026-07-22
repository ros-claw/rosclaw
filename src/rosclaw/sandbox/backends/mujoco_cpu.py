"""Low-latency, deterministic CPU MuJoCo trajectory backend."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np

from rosclaw.sandbox.backends.base import (
    BackendCapabilities,
    CompiledScenario,
    ReplayReport,
    RolloutRequest,
    ScenarioSpec,
    TrajectorySimulationReceipt,
)
from rosclaw.sandbox.backends.fingerprints import (
    canonical_hash,
    file_hash,
    mujoco_fingerprint,
)
from rosclaw.sandbox.sandbox_api import Sandbox


def _atomic_json(path: Path, value: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_path(reference: str) -> Path | None:
    parsed = urlparse(reference)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path)).resolve()
    if parsed.scheme:
        return None
    return Path(reference).expanduser().resolve()


class MujocoCpuBackend:
    """Execute every interpolated waypoint and inspect physical state."""

    name = "mujoco_cpu"

    def __init__(self, sandbox: Sandbox):
        self._sandbox = sandbox

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name=self.name,
            physics=True,
            replay=True,
            contacts=True,
            actuator_forces=True,
            supported_tasks=("sandbox.follow_trajectory",),
        )

    def _environment(self) -> tuple[Any, Any, str, str, str]:
        if not self._sandbox.has_physics:
            raise RuntimeError(self._sandbox.load_error or "PHYSICS_UNAVAILABLE")
        model = self._sandbox.physics_model
        data = self._sandbox.physics_data
        if model is None or data is None:
            raise RuntimeError("PHYSICS_MODEL_DATA_UNAVAILABLE")
        model_hash = file_hash(self._sandbox.model_path)
        world_asset_hash = canonical_hash(self._sandbox.world_metadata)
        fingerprint = mujoco_fingerprint(model, model_hash, world_asset_hash)
        return model, data, model_hash, world_asset_hash, fingerprint

    def compile(self, scenario: ScenarioSpec) -> CompiledScenario:
        _, _, model_hash, world_asset_hash, fingerprint = self._environment()
        if not scenario.body_snapshot_hash:
            raise ValueError("BODY_SNAPSHOT_REQUIRED")
        if not scenario.model_hash:
            raise ValueError("MODEL_HASH_REQUIRED")
        if scenario.model_hash != model_hash:
            raise ValueError("SCENARIO_MODEL_HASH_MISMATCH")
        return CompiledScenario(
            spec=scenario,
            backend_fingerprint=fingerprint,
            world_asset_hash=world_asset_hash,
        )

    def rollout(self, request: RolloutRequest) -> TrajectorySimulationReceipt:
        import mujoco

        model, data, model_hash, world_asset_hash, fingerprint = self._environment()
        scenario_hash = canonical_hash(asdict(request.scenario))
        request_payload = {
            "scenario": asdict(request.scenario),
            "trajectory": request.trajectory,
            "control_dt_sec": request.control_dt_sec,
            "max_joint_delta_rad": request.max_joint_delta_rad,
            "max_joint_velocity_radps": request.max_joint_velocity_radps,
            "max_final_tracking_error_rad": request.max_final_tracking_error_rad,
            "settle_steps": request.settle_steps,
        }
        action_hash = canonical_hash(request_payload)
        backend = {
            "name": self.name,
            "version": getattr(mujoco, "__version__", "unknown"),
            "fingerprint": fingerprint,
            "integrator": int(model.opt.integrator),
            "timestep_sec": float(model.opt.timestep),
            "solver_iterations": int(model.opt.iterations),
        }

        static_error = self._validate_request(request, model)
        if static_error is not None:
            return TrajectorySimulationReceipt(
                scenario_id=request.scenario.scenario_id,
                backend=backend,
                seed=request.scenario.seed,
                body_snapshot_hash=request.scenario.body_snapshot_hash,
                model_hash=model_hash,
                world_asset_hash=world_asset_hash,
                action_hash=action_hash,
                scenario_hash=scenario_hash,
                is_safe=False,
                physics_executed=False,
                reason=static_error,
                violations=[static_error],
                request=request_payload,
            )

        compiled = self.compile(request.scenario)
        if compiled.backend_fingerprint != fingerprint:
            raise RuntimeError("BACKEND_FINGERPRINT_CHANGED_DURING_COMPILE")

        self._sandbox.reset(keyframe="home")
        mujoco.mj_forward(model, data)
        nu = int(model.nu)
        previous_command = data.qpos[:nu].copy()
        previous_velocity = data.qvel[:nu].copy()
        max_velocity = 0.0
        max_acceleration = 0.0
        max_tracking_error = 0.0
        peak_contact_force = 0.0
        minimum_contact_distance = math.inf
        collision_pairs: set[tuple[str, str]] = set()
        unstable_steps = 0
        deadline_missed = 0
        nan_detected = False
        step_count = 0
        samples: list[dict[str, Any]] = []
        control_dt = request.control_dt_sec or float(model.opt.timestep)
        started = time.perf_counter()

        def observe(command: np.ndarray, step_wall_sec: float) -> bool:
            nonlocal max_velocity, max_acceleration, max_tracking_error
            nonlocal peak_contact_force, minimum_contact_distance, unstable_steps
            nonlocal deadline_missed, nan_detected, previous_velocity
            qpos = data.qpos[:nu].copy()
            qvel = data.qvel[:nu].copy()
            nan_detected = nan_detected or bool(
                not np.isfinite(qpos).all() or not np.isfinite(qvel).all()
            )
            max_velocity = max(max_velocity, float(np.max(np.abs(qvel), initial=0.0)))
            acceleration = np.abs(qvel - previous_velocity) / max(control_dt, 1e-9)
            max_acceleration = max(max_acceleration, float(np.max(acceleration, initial=0.0)))
            previous_velocity = qvel
            max_tracking_error = max(
                max_tracking_error,
                float(np.max(np.abs(qpos - command), initial=0.0)),
            )
            if step_wall_sec > control_dt:
                deadline_missed += 1
            if np.max(np.abs(qpos), initial=0.0) > 1e6:
                unstable_steps += 1

            collision = False
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
                minimum_contact_distance = min(minimum_contact_distance, float(contact.dist))
                contact_force = np.zeros(6)
                mujoco.mj_contactForce(model, data, contact_index, contact_force)
                peak_contact_force = max(
                    peak_contact_force, float(np.linalg.norm(contact_force[:3]))
                )
                if float(contact.dist) <= -1e-6:
                    ordered_pair = sorted((str(geom1), str(geom2)))
                    collision_pairs.add((ordered_pair[0], ordered_pair[1]))
                    collision = True
            return collision

        collision_detected = observe(previous_command, 0.0)
        for waypoint in request.trajectory:
            target = np.asarray(waypoint, dtype=float)
            interpolation_steps = max(
                1,
                int(
                    math.ceil(
                        float(np.max(np.abs(target - previous_command), initial=0.0))
                        / request.max_joint_delta_rad
                    )
                ),
            )
            segment_start = previous_command.copy()
            for interpolation_index in range(1, interpolation_steps + 1):
                command = segment_start + (target - segment_start) * (
                    interpolation_index / interpolation_steps
                )
                data.ctrl[:nu] = command
                step_started = time.perf_counter()
                mujoco.mj_step(model, data)
                step_wall = time.perf_counter() - step_started
                step_count += 1
                collision_detected = observe(command, step_wall) or collision_detected
                if step_count % 10 == 0 or collision_detected:
                    samples.append(
                        {
                            "step": step_count,
                            "time": float(data.time),
                            "qpos": data.qpos[:nu].copy().tolist(),
                            "qvel": data.qvel[:nu].copy().tolist(),
                            "command": command.tolist(),
                            "contacts": int(data.ncon),
                        }
                    )
                if collision_detected or nan_detected or unstable_steps:
                    break
            previous_command = target
            if collision_detected or nan_detected or unstable_steps:
                break
            for _ in range(request.settle_steps):
                data.ctrl[:nu] = target
                step_started = time.perf_counter()
                mujoco.mj_step(model, data)
                step_count += 1
                collision_detected = (
                    observe(target, time.perf_counter() - step_started) or collision_detected
                )
                if collision_detected or nan_detected or unstable_steps:
                    break
            if collision_detected or nan_detected or unstable_steps:
                break

        final_qpos = data.qpos[:nu].copy()
        final_tracking_error = float(np.max(np.abs(final_qpos - previous_command), initial=0.0))
        velocity_violation = max_velocity > request.max_joint_velocity_radps + 1e-9
        tracking_violation = final_tracking_error > request.max_final_tracking_error_rad
        violations: list[str] = []
        if collision_detected:
            violations.append("COLLISION_DETECTED")
        if velocity_violation:
            violations.append("JOINT_VELOCITY_LIMIT")
        if tracking_violation:
            violations.append("TRACKING_ERROR")
        if nan_detected:
            violations.append("NAN_DETECTED")
        if unstable_steps:
            violations.append("PHYSICS_UNSTABLE")
        is_safe = not violations
        reason = "physics_trajectory_safe" if is_safe else violations[0].lower()
        metrics = {
            "steps": step_count,
            "final_time": float(data.time),
            "wall_time_sec": time.perf_counter() - started,
            "minimum_clearance_m": (
                None if math.isinf(minimum_contact_distance) else minimum_contact_distance
            ),
            "maximum_joint_velocity": max_velocity,
            "maximum_joint_acceleration": max_acceleration,
            "maximum_tracking_error": max_tracking_error,
            "final_tracking_error": final_tracking_error,
            "peak_contact_force_n": peak_contact_force,
            "collision_count": len(collision_pairs),
            "unstable_steps": unstable_steps,
            "nan_detected": nan_detected,
            "deadline_missed": deadline_missed,
        }
        receipt = TrajectorySimulationReceipt(
            scenario_id=request.scenario.scenario_id,
            backend=backend,
            seed=request.scenario.seed,
            body_snapshot_hash=request.scenario.body_snapshot_hash,
            model_hash=model_hash,
            world_asset_hash=world_asset_hash,
            action_hash=action_hash,
            scenario_hash=scenario_hash,
            is_safe=is_safe,
            physics_executed=True,
            reason=reason,
            metrics=metrics,
            violations=violations,
            collision_pairs=[list(pair) for pair in sorted(collision_pairs)],
            final_qpos=final_qpos.tolist(),
            request=request_payload,
        )
        self._persist_artifacts(receipt, samples, request.artifact_dir)
        return receipt

    @staticmethod
    def _validate_request(request: RolloutRequest, model: Any) -> str | None:
        if not request.trajectory:
            return "EMPTY_TRAJECTORY"
        if not math.isfinite(request.max_joint_delta_rad) or not (
            0.0001 <= request.max_joint_delta_rad <= 0.5
        ):
            return "INVALID_INTERPOLATION_DELTA"
        if not math.isfinite(request.max_joint_velocity_radps) or (
            request.max_joint_velocity_radps <= 0
        ):
            return "INVALID_VELOCITY_LIMIT"
        for waypoint in request.trajectory:
            if len(waypoint) != int(model.nu):
                return "ACTION_DIMENSION_MISMATCH"
            if any(not math.isfinite(float(value)) for value in waypoint):
                return "NON_FINITE_WAYPOINT"
            for index, value in enumerate(waypoint):
                lower, upper = model.actuator_ctrlrange[index]
                if not float(lower) <= float(value) <= float(upper):
                    return f"JOINT_{index}_LIMIT"
        return None

    @staticmethod
    def _persist_artifacts(
        receipt: TrajectorySimulationReceipt,
        samples: list[dict[str, Any]],
        artifact_dir: Path | None,
    ) -> None:
        if artifact_dir is None:
            return
        root = artifact_dir.expanduser().resolve()
        request_path = root / "trajectory_request.json"
        states_path = root / "trajectory_states.json"
        request_hash = _atomic_json(request_path, receipt.request)
        states_hash = _atomic_json(
            states_path,
            {"schema_version": "rosclaw.trajectory_states.v1", "states": samples},
        )
        receipt.artifacts.extend([request_path.as_uri(), states_path.as_uri()])
        receipt.artifact_hashes.update(
            {request_path.name: request_hash, states_path.name: states_hash}
        )
        receipt_path = root / "simulation_receipt.json"
        _atomic_json(receipt_path, receipt.to_dict())
        receipt.artifacts.append(receipt_path.as_uri())

    def replay(
        self, receipt: TrajectorySimulationReceipt | dict[str, Any], *, strict: bool = True
    ) -> ReplayReport:
        source = receipt.to_dict() if isinstance(receipt, TrajectorySimulationReceipt) else receipt
        mismatches: list[str] = []
        try:
            model, _, model_hash, world_hash, fingerprint = self._environment()
            del model
        except RuntimeError as exc:
            return ReplayReport(False, False, False, False, None, str(exc), (str(exc),))

        backend = source.get("backend") or {}
        if source.get("model_hash") != model_hash:
            mismatches.append("model_hash")
        if source.get("world_asset_hash") != world_hash:
            mismatches.append("world_asset_hash")
        if backend.get("fingerprint") != fingerprint:
            mismatches.append("backend_fingerprint")

        hashes_verified = True
        artifacts = {
            path.name: path
            for reference in source.get("artifacts", [])
            if isinstance(reference, str) and (path := _artifact_path(reference)) is not None
        }
        for name, expected in (source.get("artifact_hashes") or {}).items():
            path = artifacts.get(str(name))
            if path is None or not path.is_file():
                hashes_verified = False
                mismatches.append(f"artifact_missing:{name}")
                continue
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != str(expected).removeprefix("sha256:"):
                hashes_verified = False
                mismatches.append(f"artifact_hash:{name}")

        request_data = source.get("request") or {}
        scenario_data = request_data.get("scenario") or {}
        try:
            scenario = ScenarioSpec(**scenario_data)
            replay_request = RolloutRequest(
                scenario=scenario,
                trajectory=request_data["trajectory"],
                control_dt_sec=request_data.get("control_dt_sec"),
                max_joint_delta_rad=float(request_data.get("max_joint_delta_rad", 0.005)),
                max_joint_velocity_radps=float(request_data.get("max_joint_velocity_radps", 3.15)),
                max_final_tracking_error_rad=float(
                    request_data.get("max_final_tracking_error_rad", 0.25)
                ),
                settle_steps=int(request_data.get("settle_steps", 100)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            mismatches.append("request_contract")
            return ReplayReport(
                False,
                not any(name in mismatches for name in ("model_hash", "world_asset_hash")),
                hashes_verified,
                False,
                None,
                f"REPLAY_REQUEST_INVALID: {exc}",
                tuple(mismatches),
            )

        if strict and mismatches:
            return ReplayReport(
                False,
                False,
                hashes_verified,
                False,
                None,
                "REPLAY_ENVIRONMENT_MISMATCH",
                tuple(mismatches),
            )

        replayed = self.rollout(replay_request)
        original_qpos = np.asarray(source.get("final_qpos") or [], dtype=float)
        replay_qpos = np.asarray(replayed.final_qpos, dtype=float)
        qpos_error = (
            float(np.max(np.abs(original_qpos - replay_qpos), initial=0.0))
            if original_qpos.shape == replay_qpos.shape and original_qpos.size
            else None
        )
        deterministic_label = bool(source.get("is_safe")) is replayed.is_safe
        if not deterministic_label:
            mismatches.append("safety_label")
        if qpos_error is None or qpos_error > 1e-9:
            mismatches.append("final_qpos")
        verified = bool(hashes_verified and not mismatches)
        return ReplayReport(
            verified=verified,
            environment_match=not any(
                name in mismatches
                for name in ("model_hash", "world_asset_hash", "backend_fingerprint")
            ),
            hashes_verified=hashes_verified,
            deterministic_label=deterministic_label,
            final_qpos_max_abs_error=qpos_error,
            reason="strict_replay_verified" if verified else "REPLAY_MISMATCH",
            mismatches=tuple(mismatches),
        )

    def close(self) -> None:
        return None


__all__ = ["MujocoCpuBackend"]
