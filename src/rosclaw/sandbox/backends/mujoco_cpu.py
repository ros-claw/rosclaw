"""Low-latency, deterministic CPU MuJoCo trajectory backend."""

from __future__ import annotations

import hashlib
import json
import math
import re
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

_SHA256_RE = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")
_REQUIRED_REPLAY_ARTIFACTS = frozenset({"trajectory_request.json", "trajectory_states.json"})
_MAX_REPLAY_ARTIFACT_BYTES = 64 * 1024 * 1024
_MAX_SCENARIO_METADATA_BYTES = 64 * 1024
_MAX_SIMULATION_STEPS = 250_000


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
        if parsed.netloc not in {"", "localhost"}:
            return None
        return Path(unquote(parsed.path)).resolve()
    if parsed.scheme:
        return None
    return Path(reference).expanduser().resolve()


def _model_layout_error(model: Any) -> str | None:
    """Reject actuator layouts this joint-position backend cannot represent."""

    import mujoco

    if not (int(model.nq) == int(model.nv) == int(model.nu)) or int(model.nu) <= 0:
        return "UNSUPPORTED_ACTUATOR_LAYOUT"
    for actuator_index in range(int(model.nu)):
        if int(model.actuator_trntype[actuator_index]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            return "UNSUPPORTED_ACTUATOR_LAYOUT"
        joint_id = int(model.actuator_trnid[actuator_index][0])
        if joint_id < 0:
            return "UNSUPPORTED_ACTUATOR_LAYOUT"
        if (
            int(model.jnt_qposadr[joint_id]) != actuator_index
            or int(model.jnt_dofadr[joint_id]) != actuator_index
        ):
            return "UNSUPPORTED_ACTUATOR_LAYOUT"
    return None


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
        layout_error = _model_layout_error(model)
        if layout_error is not None:
            raise RuntimeError(layout_error)
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
        scenario_payload = asdict(request.scenario)
        request_payload = {
            "scenario": scenario_payload,
            "trajectory": request.trajectory,
            "control_dt_sec": request.control_dt_sec,
            "max_joint_delta_rad": request.max_joint_delta_rad,
            "max_joint_velocity_radps": request.max_joint_velocity_radps,
            "max_final_tracking_error_rad": request.max_final_tracking_error_rad,
            "settle_steps": request.settle_steps,
            "max_steps": request.max_steps,
        }
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
            try:
                scenario_hash = canonical_hash(scenario_payload)
                action_hash = canonical_hash(request_payload)
                persisted_request = request_payload
            except (TypeError, ValueError):
                scenario_hash = canonical_hash({"invalid_scenario": static_error})
                action_hash = canonical_hash({"invalid_request": static_error})
                persisted_request = {"validation_error": static_error}
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
                request=persisted_request,
            )

        scenario_hash = canonical_hash(scenario_payload)
        action_hash = canonical_hash(request_payload)

        compiled = self.compile(request.scenario)
        if compiled.backend_fingerprint != fingerprint:
            raise RuntimeError("BACKEND_FINGERPRINT_CHANGED_DURING_COMPILE")

        self._sandbox.reset(keyframe="home")
        jitter_rad = float(request.scenario.metadata.get("initial_qpos_jitter_rad", 0.0))
        rng = np.random.default_rng(request.scenario.seed)
        initial_offset = (
            rng.uniform(-jitter_rad, jitter_rad, size=int(model.nu))
            if jitter_rad > 0.0
            else np.zeros(int(model.nu), dtype=float)
        )
        if jitter_rad > 0.0:
            lower = np.asarray(model.actuator_ctrlrange[: int(model.nu), 0], dtype=float)
            upper = np.asarray(model.actuator_ctrlrange[: int(model.nu), 1], dtype=float)
            perturbed = np.clip(data.qpos[: int(model.nu)] + initial_offset, lower, upper)
            initial_offset = perturbed - data.qpos[: int(model.nu)]
            data.qpos[: int(model.nu)] = perturbed
            data.ctrl[: int(model.nu)] = perturbed
        mujoco.mj_forward(model, data)
        nu = int(model.nu)
        previous_command = data.qpos[:nu].copy()
        initial_qpos = previous_command.copy()
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
        step_budget_exhausted = False
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

        def record_sample(command: np.ndarray) -> None:
            if samples and samples[-1]["step"] == step_count:
                return
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

        collision_detected = observe(previous_command, 0.0)
        record_sample(previous_command)
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
                if step_count >= request.max_steps:
                    step_budget_exhausted = True
                    break
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
                    record_sample(command)
                if collision_detected or nan_detected or unstable_steps or step_budget_exhausted:
                    break
            previous_command = target
            if collision_detected or nan_detected or unstable_steps or step_budget_exhausted:
                break
            for _ in range(request.settle_steps):
                if step_count >= request.max_steps:
                    step_budget_exhausted = True
                    break
                data.ctrl[:nu] = target
                step_started = time.perf_counter()
                mujoco.mj_step(model, data)
                step_count += 1
                collision_detected = (
                    observe(target, time.perf_counter() - step_started) or collision_detected
                )
                if step_count % 10 == 0 or collision_detected:
                    record_sample(target)
                if collision_detected or nan_detected or unstable_steps or step_budget_exhausted:
                    break
            if collision_detected or nan_detected or unstable_steps or step_budget_exhausted:
                break

        record_sample(previous_command)

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
        if step_budget_exhausted:
            violations.append("SIMULATION_STEP_BUDGET_EXCEEDED")
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
            "step_budget_exhausted": step_budget_exhausted,
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
            randomization={
                "method": "uniform_initial_qpos_v1",
                "seed": request.scenario.seed,
                "seed_applied": jitter_rad > 0.0,
                "initial_qpos_jitter_rad": jitter_rad,
                "parameter_hash": canonical_hash(
                    {"method": "uniform_initial_qpos_v1", "jitter_rad": jitter_rad}
                ),
                "offset_hash": canonical_hash(initial_offset.tolist()),
                "initial_state_hash": canonical_hash(initial_qpos.tolist()),
            },
        )
        self._persist_artifacts(receipt, samples, request.artifact_dir)
        return receipt

    @staticmethod
    def _validate_request(request: RolloutRequest, model: Any) -> str | None:
        if not isinstance(request.trajectory, list):
            return "INVALID_TRAJECTORY"
        if not request.trajectory:
            return "EMPTY_TRAJECTORY"
        if len(request.trajectory) > 10_000:
            return "TRAJECTORY_WAYPOINT_LIMIT"
        if (
            isinstance(request.scenario.seed, bool)
            or not isinstance(request.scenario.seed, int)
            or not 0 <= request.scenario.seed < 2**63
        ):
            return "INVALID_SEED"
        if not isinstance(request.scenario.metadata, dict):
            return "INVALID_SCENARIO_METADATA"
        try:
            encoded_metadata = json.dumps(
                request.scenario.metadata,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError):
            return "INVALID_SCENARIO_METADATA"
        if len(encoded_metadata) > _MAX_SCENARIO_METADATA_BYTES:
            return "SCENARIO_METADATA_LIMIT"
        jitter = request.scenario.metadata.get("initial_qpos_jitter_rad", 0.0)
        if isinstance(jitter, bool) or not isinstance(jitter, (int, float)):
            return "INVALID_INITIAL_STATE_JITTER"
        if not math.isfinite(float(jitter)) or not 0.0 <= float(jitter) <= 0.1:
            return "INVALID_INITIAL_STATE_JITTER"
        if request.control_dt_sec is not None and (
            isinstance(request.control_dt_sec, bool)
            or not isinstance(request.control_dt_sec, (int, float))
            or not math.isfinite(float(request.control_dt_sec))
            or not 0.0001 <= float(request.control_dt_sec) <= 1.0
        ):
            return "INVALID_CONTROL_DT"
        if request.control_dt_sec is not None and not math.isclose(
            float(request.control_dt_sec),
            float(model.opt.timestep),
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            return "UNSUPPORTED_CONTROL_DT"
        if (
            isinstance(request.max_joint_delta_rad, bool)
            or not isinstance(request.max_joint_delta_rad, (int, float))
            or not math.isfinite(float(request.max_joint_delta_rad))
            or not 0.0001 <= float(request.max_joint_delta_rad) <= 0.5
        ):
            return "INVALID_INTERPOLATION_DELTA"
        if (
            isinstance(request.max_joint_velocity_radps, bool)
            or not isinstance(request.max_joint_velocity_radps, (int, float))
            or not math.isfinite(float(request.max_joint_velocity_radps))
            or float(request.max_joint_velocity_radps) <= 0
        ):
            return "INVALID_VELOCITY_LIMIT"
        if (
            isinstance(request.max_final_tracking_error_rad, bool)
            or not isinstance(request.max_final_tracking_error_rad, (int, float))
            or not math.isfinite(float(request.max_final_tracking_error_rad))
            or not 0.0 <= float(request.max_final_tracking_error_rad) <= math.pi
        ):
            return "INVALID_TRACKING_ERROR_LIMIT"
        if isinstance(request.settle_steps, bool) or not isinstance(request.settle_steps, int):
            return "INVALID_SETTLE_STEPS"
        if not 0 <= request.settle_steps <= 100_000:
            return "INVALID_SETTLE_STEPS"
        if (
            isinstance(request.max_steps, bool)
            or not isinstance(request.max_steps, int)
            or not 1 <= request.max_steps <= _MAX_SIMULATION_STEPS
        ):
            return "INVALID_STEP_BUDGET"
        for waypoint in request.trajectory:
            if not isinstance(waypoint, (list, tuple)):
                return "INVALID_WAYPOINT"
            if len(waypoint) != int(model.nu):
                return "ACTION_DIMENSION_MISMATCH"
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in waypoint
            ):
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

    @staticmethod
    def _persist_replay_result(receipt: TrajectorySimulationReceipt) -> None:
        receipt_path = next(
            (
                path
                for reference in receipt.artifacts
                if (path := _artifact_path(reference)) is not None
                and path.name == "simulation_receipt.json"
            ),
            None,
        )
        if receipt_path is not None:
            _atomic_json(receipt_path, receipt.to_dict())

    def replay(
        self, receipt: TrajectorySimulationReceipt | dict[str, Any], *, strict: bool = True
    ) -> ReplayReport:
        receipt_object = receipt if isinstance(receipt, TrajectorySimulationReceipt) else None
        source = receipt.to_dict() if receipt_object is not None else receipt
        mismatches: list[str] = []

        def finish(report: ReplayReport) -> ReplayReport:
            if receipt_object is not None:
                receipt_object.record_replay(report)
                self._persist_replay_result(receipt_object)
            return report

        if not isinstance(source, dict):
            return finish(
                ReplayReport(
                    False,
                    False,
                    False,
                    False,
                    None,
                    "REPLAY_RECEIPT_INVALID",
                    ("receipt_contract",),
                )
            )

        try:
            model, _, model_hash, world_hash, fingerprint = self._environment()
        except RuntimeError as exc:
            return finish(ReplayReport(False, False, False, False, None, str(exc), (str(exc),)))

        if source.get("schema_version") != "rosclaw.simulation_receipt.v1":
            mismatches.append("schema_version")
        if source.get("evidence_domain") != "SIMULATION":
            mismatches.append("evidence_domain")
        if source.get("physics_executed") is not True:
            mismatches.append("physics_executed")

        backend_value = source.get("backend")
        backend = backend_value if isinstance(backend_value, dict) else {}
        if not isinstance(backend_value, dict):
            mismatches.append("backend_contract")
        if source.get("model_hash") != model_hash:
            mismatches.append("model_hash")
        if source.get("world_asset_hash") != world_hash:
            mismatches.append("world_asset_hash")
        if backend.get("fingerprint") != fingerprint:
            mismatches.append("backend_fingerprint")

        hashes = source.get("artifact_hashes")
        hashes_verified = isinstance(hashes, dict) and bool(hashes)
        artifacts: dict[str, Path] = {}
        duplicate_artifacts: set[str] = set()
        artifact_references = source.get("artifacts")
        if not isinstance(artifact_references, list):
            artifact_references = []
            mismatches.append("artifact_contract")
        for reference in artifact_references:
            if not isinstance(reference, str):
                continue
            path = _artifact_path(reference)
            if path is None:
                continue
            if path.name in artifacts and artifacts[path.name] != path:
                duplicate_artifacts.add(path.name)
            artifacts[path.name] = path
        for name in sorted(duplicate_artifacts):
            hashes_verified = False
            mismatches.append(f"artifact_ambiguous:{name}")
        if not isinstance(hashes, dict) or not _REQUIRED_REPLAY_ARTIFACTS.issubset(hashes):
            hashes_verified = False
            mismatches.append("artifact_set")
            hashes = {}
        for name, expected in hashes.items():
            if Path(str(name)).name != str(name) or not _SHA256_RE.fullmatch(str(expected)):
                hashes_verified = False
                mismatches.append(f"artifact_contract:{name}")
                continue
            path = artifacts.get(str(name))
            if path is None or not path.is_file():
                hashes_verified = False
                mismatches.append(f"artifact_missing:{name}")
                continue
            try:
                if path.stat().st_size > _MAX_REPLAY_ARTIFACT_BYTES:
                    raise ValueError("artifact too large")
                actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            except (OSError, ValueError):
                hashes_verified = False
                mismatches.append(f"artifact_unreadable:{name}")
                continue
            if actual_hash != str(expected).removeprefix("sha256:"):
                hashes_verified = False
                mismatches.append(f"artifact_hash:{name}")

        request_value = source.get("request")
        request_data = request_value if isinstance(request_value, dict) else {}
        scenario_value = request_data.get("scenario")
        scenario_data = scenario_value if isinstance(scenario_value, dict) else {}
        if not isinstance(request_value, dict) or not isinstance(scenario_value, dict):
            mismatches.append("request_contract")
        else:
            field_pairs = (
                ("scenario_id", source.get("scenario_id"), scenario_data.get("scenario_id")),
                ("seed", source.get("seed"), scenario_data.get("seed")),
                (
                    "body_snapshot_hash",
                    source.get("body_snapshot_hash"),
                    scenario_data.get("body_snapshot_hash"),
                ),
                ("model_hash", source.get("model_hash"), scenario_data.get("model_hash")),
            )
            for name, receipt_value, request_value in field_pairs:
                if receipt_value != request_value:
                    mismatches.append(name)
            try:
                if source.get("action_hash") != canonical_hash(request_data):
                    mismatches.append("action_hash")
                if source.get("scenario_hash") != canonical_hash(scenario_data):
                    mismatches.append("scenario_hash")
            except (TypeError, ValueError):
                mismatches.append("request_contract")

        def load_artifact_json(path: Path | None, mismatch: str) -> Any:
            if path is None or not path.is_file():
                return None
            try:
                if path.stat().st_size > _MAX_REPLAY_ARTIFACT_BYTES:
                    raise ValueError("artifact too large")
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                mismatches.append(mismatch)
                return None

        request_artifact = artifacts.get("trajectory_request.json")
        persisted_request = load_artifact_json(request_artifact, "request_artifact")
        if persisted_request is not None and persisted_request != request_data:
            mismatches.append("request_artifact")
        states_artifact = artifacts.get("trajectory_states.json")
        persisted_states = load_artifact_json(states_artifact, "states_artifact")
        if persisted_states is not None and not (
            isinstance(persisted_states, dict)
            and persisted_states.get("schema_version") == "rosclaw.trajectory_states.v1"
            and isinstance(persisted_states.get("states"), list)
            and persisted_states["states"]
        ):
            mismatches.append("states_artifact")

        try:
            scenario = ScenarioSpec(**scenario_data)
            replay_request = RolloutRequest(
                scenario=scenario,
                trajectory=request_data["trajectory"],
                control_dt_sec=request_data.get("control_dt_sec"),
                max_joint_delta_rad=request_data.get("max_joint_delta_rad", 0.005),
                max_joint_velocity_radps=request_data.get("max_joint_velocity_radps", 3.15),
                max_final_tracking_error_rad=request_data.get("max_final_tracking_error_rad", 0.25),
                settle_steps=request_data.get("settle_steps", 100),
                max_steps=request_data.get("max_steps", _MAX_SIMULATION_STEPS),
            )
        except (KeyError, TypeError, ValueError) as exc:
            mismatches.append("request_contract")
            return finish(
                ReplayReport(
                    False,
                    not any(name in mismatches for name in ("model_hash", "world_asset_hash")),
                    hashes_verified,
                    False,
                    None,
                    f"REPLAY_REQUEST_INVALID: {exc}",
                    tuple(dict.fromkeys(mismatches)),
                )
            )

        if strict and mismatches:
            return finish(
                ReplayReport(
                    False,
                    not any(
                        name in mismatches
                        for name in ("model_hash", "world_asset_hash", "backend_fingerprint")
                    ),
                    hashes_verified,
                    False,
                    None,
                    "REPLAY_CONTRACT_MISMATCH",
                    tuple(dict.fromkeys(mismatches)),
                )
            )

        try:
            replayed = self.rollout(replay_request)
        except (RuntimeError, TypeError, ValueError) as exc:
            mismatches.append("replay_execution")
            return finish(
                ReplayReport(
                    False,
                    False,
                    hashes_verified,
                    False,
                    None,
                    f"REPLAY_EXECUTION_FAILED: {exc}",
                    tuple(dict.fromkeys(mismatches)),
                )
            )
        if replayed.physics_executed is not True:
            mismatches.append("physics_executed")
        try:
            original_qpos = np.asarray(source.get("final_qpos") or [], dtype=float)
            if original_qpos.ndim != 1 or not np.isfinite(original_qpos).all():
                raise ValueError("invalid final qpos")
        except (TypeError, ValueError):
            original_qpos = np.asarray([], dtype=float)
            mismatches.append("final_qpos")
        replay_qpos = np.asarray(replayed.final_qpos, dtype=float)
        qpos_error = (
            float(np.max(np.abs(original_qpos - replay_qpos), initial=0.0))
            if original_qpos.shape == replay_qpos.shape and original_qpos.size
            else None
        )
        deterministic_label = (
            isinstance(source.get("is_safe"), bool) and source.get("is_safe") == replayed.is_safe
        )
        if not deterministic_label:
            mismatches.append("safety_label")
        if qpos_error is None or qpos_error > 1e-9:
            mismatches.append("final_qpos")
        if source.get("collision_pairs") != replayed.collision_pairs:
            mismatches.append("collision_pairs")
        if source.get("violations") != replayed.violations:
            mismatches.append("violations")
        original_metrics = source.get("metrics")
        replayed_metrics = replayed.metrics
        deterministic_metrics = (
            "steps",
            "final_time",
            "minimum_clearance_m",
            "maximum_joint_velocity",
            "maximum_joint_acceleration",
            "maximum_tracking_error",
            "final_tracking_error",
            "peak_contact_force_n",
            "collision_count",
            "unstable_steps",
            "nan_detected",
            "step_budget_exhausted",
        )
        if not isinstance(original_metrics, dict):
            mismatches.append("metrics")
        else:
            for name in deterministic_metrics:
                expected = original_metrics.get(name)
                actual_value = replayed_metrics.get(name)
                if isinstance(expected, bool) or isinstance(actual_value, bool):
                    equal = expected is actual_value
                elif isinstance(expected, (int, float)) and isinstance(actual_value, (int, float)):
                    equal = (
                        math.isfinite(float(expected))
                        and math.isfinite(float(actual_value))
                        and math.isclose(
                            float(expected), float(actual_value), rel_tol=1e-9, abs_tol=1e-9
                        )
                    )
                else:
                    equal = expected == actual_value
                if not equal:
                    mismatches.append(f"metrics:{name}")
        randomization_value = source.get("randomization")
        original_randomization = (
            randomization_value if isinstance(randomization_value, dict) else {}
        )
        if not isinstance(randomization_value, dict):
            mismatches.append("randomization")
        for name in ("method", "seed", "parameter_hash", "offset_hash", "initial_state_hash"):
            if original_randomization.get(name) != replayed.randomization.get(name):
                mismatches.append(f"randomization:{name}")
        verified = bool(hashes_verified and not mismatches)
        return finish(
            ReplayReport(
                verified=verified,
                environment_match=not any(
                    name in mismatches
                    for name in ("model_hash", "world_asset_hash", "backend_fingerprint")
                ),
                hashes_verified=hashes_verified,
                deterministic_label=deterministic_label,
                final_qpos_max_abs_error=qpos_error,
                reason="strict_replay_verified" if verified else "REPLAY_MISMATCH",
                mismatches=tuple(dict.fromkeys(mismatches)),
            )
        )

    def close(self) -> None:
        return None


__all__ = ["MujocoCpuBackend"]
