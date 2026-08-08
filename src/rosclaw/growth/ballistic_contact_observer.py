"""Auditable SIM_ONLY observer for G1 foot-ball contact dynamics.

The observer learns two deliberately narrow mappings from strict MuJoCo
replays: contact state to ball launch velocity, and contact state to the
probability that the resulting shot clears the development skill threshold.
It is diagnostic evidence, never a motor command or an online hot swap.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.feedback.contracts import canonical_hash
from rosclaw.growth.ballistic_skill_memory import (
    ballistic_skill_experiment_context_hash,
)

_MAX_TRAJECTORY_BYTES = 2 * 1024 * 1024 * 1024
_MIN_SAMPLES = 12
_MIN_CLASS_COUNT = 4
_MIN_DISTINCT_SEEDS = 6
_MAX_LOO_BRIER = 0.30
_MAX_LOO_LAUNCH_RMSE_MPS = 2.0
_MAX_SKILL_ERROR_M = 0.75
_MIN_SKILL_CROSSING_HEIGHT_M = 0.65

G1_BALLISTIC_CONTACT_FEATURE_NAMES = (
    "foot_velocity_x_mps",
    "foot_velocity_y_mps",
    "foot_velocity_z_mps",
    "contact_normal_x",
    "contact_normal_y",
    "contact_normal_z",
    "peak_contact_force_kn",
    "contact_height_relative_ball_center_m",
)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 71 and value.startswith("sha256:") and all(
        item in "0123456789abcdef" for item in value[7:]
    )


@dataclass(frozen=True)
class G1BallisticContactSample:
    planner_seed: int
    features: tuple[float, ...]
    launch_velocity_xyz_mps: tuple[float, float, float]
    qualified_skill_outcome: bool
    goal_plane_target_error_m: float | None
    goal_crossing_height_m: float | None
    evidence_path: str
    evidence_hash: str
    trajectory_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class G1BallisticContactPrediction:
    launch_velocity_xyz_mps: tuple[float, float, float]
    qualified_skill_probability: float


@dataclass(frozen=True)
class G1BallisticContactObserver:
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    launch_coefficients: tuple[tuple[float, float, float], ...]
    qualification_coefficients: tuple[float, ...]
    samples: tuple[G1BallisticContactSample, ...]
    leave_one_out_brier_score: float
    leave_one_out_launch_rmse_mps: float
    leave_one_seed_out_brier_score: float
    leave_one_seed_out_launch_rmse_mps: float
    distinct_planner_seed_count: int
    positive_count: int
    negative_count: int
    ridge_regularization: float
    body_hash: str
    implementation_hash: str
    experiment_context_hash: str
    source_evidence_hashes: tuple[str, ...]
    training_ready: bool
    failure_codes: tuple[str, ...]
    observer_hash: str
    schema_version: str = "rosclaw.growth.g1_ballistic_contact_observer.v2"

    def predict(self, features: tuple[float, ...]) -> G1BallisticContactPrediction:
        raw = np.asarray(features, dtype=np.float64)
        if raw.shape != (len(G1_BALLISTIC_CONTACT_FEATURE_NAMES),) or not np.all(
            np.isfinite(raw)
        ):
            raise ValueError("ballistic contact observer features are invalid")
        standardized = (raw - np.asarray(self.feature_mean)) / np.asarray(
            self.feature_scale
        )
        design = np.concatenate((np.ones(1), standardized))
        launch = design @ np.asarray(self.launch_coefficients)
        probability = float(
            np.clip(design @ np.asarray(self.qualification_coefficients), 0.0, 1.0)
        )
        if not np.all(np.isfinite(launch)) or not math.isfinite(probability):
            raise ValueError("ballistic contact observer prediction is non-finite")
        return G1BallisticContactPrediction(
            launch_velocity_xyz_mps=tuple(float(item) for item in launch),  # type: ignore[arg-type]
            qualified_skill_probability=probability,
        )

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": self.schema_version,
            "feature_names": list(G1_BALLISTIC_CONTACT_FEATURE_NAMES),
            "feature_mean": list(self.feature_mean),
            "feature_scale": list(self.feature_scale),
            "launch_coefficients": [list(row) for row in self.launch_coefficients],
            "qualification_coefficients": list(self.qualification_coefficients),
            "samples": [sample.to_dict() for sample in self.samples],
            "sample_count": len(self.samples),
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "leave_one_out_brier_score": self.leave_one_out_brier_score,
            "leave_one_out_launch_rmse_mps": self.leave_one_out_launch_rmse_mps,
            "leave_one_seed_out_brier_score": self.leave_one_seed_out_brier_score,
            "leave_one_seed_out_launch_rmse_mps": (
                self.leave_one_seed_out_launch_rmse_mps
            ),
            "distinct_planner_seed_count": self.distinct_planner_seed_count,
            "maximum_leave_one_out_brier_score": _MAX_LOO_BRIER,
            "maximum_leave_one_out_launch_rmse_mps": _MAX_LOO_LAUNCH_RMSE_MPS,
            "ridge_regularization": self.ridge_regularization,
            "body_hash": self.body_hash,
            "implementation_hash": self.implementation_hash,
            "experiment_context_hash": self.experiment_context_hash,
            "source_evidence_hashes": list(self.source_evidence_hashes),
            "training_ready": self.training_ready,
            "failure_codes": list(self.failure_codes),
            "evidence_domain": "SIM_ONLY_DEVELOPMENT",
            "model_role": "POST_CONTACT_DIAGNOSTIC_OBSERVER",
            "sealed_generalization_evidence": False,
            "direct_torque_output": False,
            "online_hot_swap_allowed": False,
            "promotion_authorized": False,
            "hardware_authorized": False,
        }
        if include_hash:
            value["observer_hash"] = self.observer_hash
        return value


def derive_g1_ballistic_contact_observer(
    *,
    evidence_paths: tuple[Path, ...],
    output_path: Path,
    source_checkout: Path,
    ridge_regularization: float = 0.20,
) -> G1BallisticContactObserver:
    if len(evidence_paths) < _MIN_SAMPLES:
        raise ValueError(f"ballistic contact observer requires at least {_MIN_SAMPLES} samples")
    if not 1e-6 <= ridge_regularization <= 100.0:
        raise ValueError("ballistic contact observer ridge regularization is invalid")
    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("ballistic contact observer output must be outside the source checkout")
    if output.exists():
        raise FileExistsError("ballistic contact observer output already exists")

    samples: list[G1BallisticContactSample] = []
    body_hashes: set[str] = set()
    implementation_hashes: set[str] = set()
    context_hashes: set[str] = set()
    source_hashes: set[str] = set()
    for path_value in evidence_paths:
        path = path_value.expanduser().resolve()
        evidence = json.loads(path.read_text(encoding="utf-8"))
        if evidence.get("strict_replay") is not True:
            raise ValueError("ballistic contact observer requires strict replay evidence")
        if dict(evidence.get("claims", {})).get(
            "contact_dynamics_observed_from_physics"
        ) is not True:
            raise ValueError("ballistic contact observer requires physics contact evidence")
        trajectory = Path(str(evidence.get("trajectory_path", ""))).resolve()
        if (
            not trajectory.is_file()
            or not 1 <= trajectory.stat().st_size <= _MAX_TRAJECTORY_BYTES
            or evidence.get("trajectory_hash") != _file_hash(trajectory)
        ):
            raise ValueError("ballistic contact observer trajectory binding is invalid")
        _verify_contact_trace(evidence, trajectory)
        evidence_hash = _file_hash(path)
        if evidence_hash in source_hashes:
            raise ValueError("ballistic contact observer evidence must be unique")
        source_hashes.add(evidence_hash)
        body_hashes.add(str(evidence.get("body_hash", "")))
        implementation_hashes.add(str(evidence.get("implementation_hash", "")))
        flow = dict(evidence.get("flow_config", {}))
        sonic = dict(evidence.get("sonic_runup_config", {}))
        context_hashes.add(
            ballistic_skill_experiment_context_hash(
                flow_config=flow,
                sonic_runup_config=sonic,
                runup_config=dict(evidence.get("runup_config", {})),
                goal_spec=dict(evidence.get("goal_spec", {})),
                approach_strike_candidate_hash=evidence.get(
                    "approach_strike_candidate_hash"
                ),
            )
        )
        samples.append(_sample_from_evidence(evidence, path, evidence_hash))

    if len(body_hashes) != 1 or not _is_sha256(next(iter(body_hashes), "")):
        raise ValueError("ballistic contact observer Body hashes disagree")
    if len(implementation_hashes) != 1 or not _is_sha256(
        next(iter(implementation_hashes), "")
    ):
        raise ValueError("ballistic contact observer implementation hashes disagree")
    if len(context_hashes) != 1:
        raise ValueError("ballistic contact observer experiment contexts disagree")

    x = np.asarray([sample.features for sample in samples], dtype=np.float64)
    launch = np.asarray(
        [sample.launch_velocity_xyz_mps for sample in samples], dtype=np.float64
    )
    qualified = np.asarray(
        [sample.qualified_skill_outcome for sample in samples], dtype=np.float64
    )
    feature_mean, feature_scale, launch_coefficients = _fit_ridge(
        x, launch, ridge_regularization
    )
    _, _, qualification_coefficients = _fit_ridge(
        x, qualified[:, None], ridge_regularization
    )
    loo_launch: list[np.ndarray] = []
    loo_probability: list[float] = []
    for index in range(len(samples)):
        keep = np.arange(len(samples)) != index
        mean, scale, launch_fit = _fit_ridge(x[keep], launch[keep], ridge_regularization)
        _, _, qualification_fit = _fit_ridge(
            x[keep], qualified[keep, None], ridge_regularization
        )
        design = np.concatenate((np.ones(1), (x[index] - mean) / scale))
        loo_launch.append(design @ launch_fit)
        loo_probability.append(
            float(np.clip(design @ qualification_fit[:, 0], 0.0, 1.0))
        )
    launch_rmse = float(np.sqrt(np.mean(np.square(np.asarray(loo_launch) - launch))))
    brier = float(np.mean(np.square(np.asarray(loo_probability) - qualified)))
    planner_seeds = np.asarray([sample.planner_seed for sample in samples])
    distinct_seeds = tuple(sorted(int(item) for item in np.unique(planner_seeds)))
    seed_launch = np.zeros_like(launch)
    seed_probability = np.zeros_like(qualified)
    for seed in distinct_seeds:
        keep = planner_seeds != seed
        mean, scale, launch_fit = _fit_ridge(x[keep], launch[keep], ridge_regularization)
        _, _, qualification_fit = _fit_ridge(
            x[keep], qualified[keep, None], ridge_regularization
        )
        design = np.column_stack((np.ones(np.sum(~keep)), (x[~keep] - mean) / scale))
        seed_launch[~keep] = design @ launch_fit
        seed_probability[~keep] = np.clip(design @ qualification_fit[:, 0], 0.0, 1.0)
    seed_launch_rmse = float(np.sqrt(np.mean(np.square(seed_launch - launch))))
    seed_brier = float(np.mean(np.square(seed_probability - qualified)))
    positive_count = int(np.sum(qualified))
    negative_count = len(samples) - positive_count
    failures: list[str] = []
    if positive_count < _MIN_CLASS_COUNT:
        failures.append("INSUFFICIENT_QUALIFIED_CONTACTS")
    if negative_count < _MIN_CLASS_COUNT:
        failures.append("INSUFFICIENT_REJECTED_CONTACTS")
    if len(distinct_seeds) < _MIN_DISTINCT_SEEDS:
        failures.append("INSUFFICIENT_DISTINCT_PLANNER_SEEDS")
    if brier > _MAX_LOO_BRIER:
        failures.append("LEAVE_ONE_OUT_QUALIFICATION_ERROR_TOO_HIGH")
    if launch_rmse > _MAX_LOO_LAUNCH_RMSE_MPS:
        failures.append("LEAVE_ONE_OUT_LAUNCH_ERROR_TOO_HIGH")
    if seed_brier > _MAX_LOO_BRIER:
        failures.append("LEAVE_ONE_SEED_OUT_QUALIFICATION_ERROR_TOO_HIGH")
    if seed_launch_rmse > _MAX_LOO_LAUNCH_RMSE_MPS:
        failures.append("LEAVE_ONE_SEED_OUT_LAUNCH_ERROR_TOO_HIGH")
    training_ready = not failures

    unsigned = {
        "schema_version": "rosclaw.growth.g1_ballistic_contact_observer.v2",
        "feature_names": list(G1_BALLISTIC_CONTACT_FEATURE_NAMES),
        "feature_mean": feature_mean.tolist(),
        "feature_scale": feature_scale.tolist(),
        "launch_coefficients": launch_coefficients.tolist(),
        "qualification_coefficients": qualification_coefficients[:, 0].tolist(),
        "samples": [sample.to_dict() for sample in samples],
        "sample_count": len(samples),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "leave_one_out_brier_score": brier,
        "leave_one_out_launch_rmse_mps": launch_rmse,
        "leave_one_seed_out_brier_score": seed_brier,
        "leave_one_seed_out_launch_rmse_mps": seed_launch_rmse,
        "distinct_planner_seed_count": len(distinct_seeds),
        "maximum_leave_one_out_brier_score": _MAX_LOO_BRIER,
        "maximum_leave_one_out_launch_rmse_mps": _MAX_LOO_LAUNCH_RMSE_MPS,
        "ridge_regularization": ridge_regularization,
        "body_hash": next(iter(body_hashes)),
        "implementation_hash": next(iter(implementation_hashes)),
        "experiment_context_hash": next(iter(context_hashes)),
        "source_evidence_hashes": [sample.evidence_hash for sample in samples],
        "training_ready": training_ready,
        "failure_codes": failures,
        "evidence_domain": "SIM_ONLY_DEVELOPMENT",
        "model_role": "POST_CONTACT_DIAGNOSTIC_OBSERVER",
        "sealed_generalization_evidence": False,
        "direct_torque_output": False,
        "online_hot_swap_allowed": False,
        "promotion_authorized": False,
        "hardware_authorized": False,
    }
    observer = _observer_from_dict(unsigned, canonical_hash(unsigned))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(observer.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return observer


def load_g1_ballistic_contact_observer(path: Path) -> G1BallisticContactObserver:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    claimed = str(value.pop("observer_hash", ""))
    if claimed != canonical_hash(value):
        raise ValueError("ballistic contact observer hash mismatch")
    return _observer_from_dict(value, claimed)


def _fit_ridge(
    features: np.ndarray, targets: np.ndarray, regularization: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    design = np.column_stack((np.ones(len(features)), (features - mean) / scale))
    penalty = np.eye(design.shape[1]) * regularization
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
    return mean, scale, coefficients


def _sample_from_evidence(
    evidence: dict[str, Any], path: Path, evidence_hash: str
) -> G1BallisticContactSample:
    result = dict(evidence.get("result", {}))
    foot_velocity = _finite_vector(result.get("kick_contact_foot_velocity_xyz_mps"), 3)
    normal = _finite_vector(result.get("kick_contact_normal_xyz"), 3)
    launch = _finite_vector(result.get("ball_launch_velocity_xyz_mps"), 3)
    force = _finite_number(result.get("kick_contact_peak_force_n"))
    height = _finite_number(result.get("kick_contact_height_relative_ball_center_m"))
    if not 0.98 <= float(np.linalg.norm(normal)) <= 1.02:
        raise ValueError("ballistic contact observer normal is not unit length")
    if not 0.0 < force <= 10_000.0 or not -0.25 <= height <= 0.25:
        raise ValueError("ballistic contact observer contact envelope is invalid")
    crossing = result.get("goal_crossing_xyz_m")
    error_value = result.get("goal_plane_target_error_m")
    error = (
        float(error_value)
        if isinstance(error_value, (int, float))
        and not isinstance(error_value, bool)
        and math.isfinite(float(error_value))
        else None
    )
    crossing_height = (
        float(crossing[2])
        if isinstance(crossing, list)
        and len(crossing) == 3
        and isinstance(crossing[2], (int, float))
        and not isinstance(crossing[2], bool)
        and math.isfinite(float(crossing[2]))
        else None
    )
    hard_safe = bool(
        result.get("finite_state") is True
        and result.get("post_kick_fall") is False
        and result.get("joint_limit_violation") is False
        and result.get("torque_limit_violation") is False
    )
    qualified = bool(
        hard_safe
        and result.get("perceptual_continuity_passed") is True
        and result.get("goal_crossed") is True
        and error is not None
        and error <= _MAX_SKILL_ERROR_M
        and crossing_height is not None
        and crossing_height >= _MIN_SKILL_CROSSING_HEIGHT_M
    )
    seed_value = dict(evidence.get("sonic_runup_config", {})).get("planner_seed")
    if not isinstance(seed_value, int) or isinstance(seed_value, bool) or seed_value < 0:
        raise ValueError("ballistic contact observer planner seed is invalid")
    features = (*foot_velocity, *normal, force / 1000.0, height)
    return G1BallisticContactSample(
        planner_seed=seed_value,
        features=tuple(float(item) for item in features),
        launch_velocity_xyz_mps=tuple(float(item) for item in launch),  # type: ignore[arg-type]
        qualified_skill_outcome=qualified,
        goal_plane_target_error_m=error,
        goal_crossing_height_m=crossing_height,
        evidence_path=str(path),
        evidence_hash=evidence_hash,
        trajectory_hash=str(evidence["trajectory_hash"]),
    )


def _verify_contact_trace(evidence: dict[str, Any], trajectory: Path) -> None:
    with np.load(trajectory, allow_pickle=False) as archive:
        required = {
            "right_foot_linear_velocity",
            "ball_contact_force_peak_n",
            "ball_contact_normal",
            "ball_contact_force_world",
        }
        if not required.issubset(archive.files):
            raise ValueError("ballistic contact observer trajectory lacks contact arrays")
        velocity = np.asarray(archive["right_foot_linear_velocity"], dtype=np.float64)
        peak = np.asarray(archive["ball_contact_force_peak_n"], dtype=np.float64)
        normal = np.asarray(archive["ball_contact_normal"], dtype=np.float64)
        force = np.asarray(archive["ball_contact_force_world"], dtype=np.float64)
    if (
        velocity.ndim != 2
        or velocity.shape[1:] != (3,)
        or peak.shape != (len(velocity),)
        or normal.shape != velocity.shape
        or force.shape != velocity.shape
        or not all(np.all(np.isfinite(item)) for item in (velocity, peak, normal, force))
    ):
        raise ValueError("ballistic contact observer contact arrays are invalid")
    measured_peak = _finite_number(
        dict(evidence.get("result", {})).get("kick_contact_peak_force_n")
    )
    if not math.isclose(float(np.max(peak)), measured_peak, rel_tol=1e-7, abs_tol=1e-6):
        raise ValueError("ballistic contact observer peak force binding is invalid")


def _finite_vector(value: Any, size: int) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError("ballistic contact observer vector is invalid")
    result = tuple(_finite_number(item) for item in value)
    if any(abs(item) > 100.0 for item in result):
        raise ValueError("ballistic contact observer vector is outside its envelope")
    return result


def _finite_number(value: Any) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError("ballistic contact observer value must be finite")
    return float(value)


def _observer_from_dict(
    value: dict[str, Any], observer_hash: str
) -> G1BallisticContactObserver:
    if (
        value.get("schema_version")
        != "rosclaw.growth.g1_ballistic_contact_observer.v2"
        or tuple(value.get("feature_names", ())) != G1_BALLISTIC_CONTACT_FEATURE_NAMES
        or value.get("evidence_domain") != "SIM_ONLY_DEVELOPMENT"
        or value.get("model_role") != "POST_CONTACT_DIAGNOSTIC_OBSERVER"
        or value.get("sealed_generalization_evidence") is not False
        or value.get("direct_torque_output") is not False
        or value.get("online_hot_swap_allowed") is not False
        or value.get("promotion_authorized") is not False
        or value.get("hardware_authorized") is not False
    ):
        raise ValueError("ballistic contact observer safety boundary is invalid")
    mean = tuple(float(item) for item in value["feature_mean"])
    scale = tuple(float(item) for item in value["feature_scale"])
    launch_coefficients = tuple(
        tuple(float(item) for item in row) for row in value["launch_coefficients"]
    )
    qualification_coefficients = tuple(
        float(item) for item in value["qualification_coefficients"]
    )
    samples = tuple(_sample_from_dict(item) for item in value["samples"])
    failures_value = value["failure_codes"]
    ready_value = value["training_ready"]
    if not isinstance(failures_value, list) or not isinstance(ready_value, bool):
        raise ValueError("ballistic contact observer decision types are invalid")
    failures = tuple(str(item) for item in failures_value)
    source_hashes = tuple(str(item) for item in value["source_evidence_hashes"])
    hashes = (
        str(value["body_hash"]),
        str(value["implementation_hash"]),
        str(value["experiment_context_hash"]),
        observer_hash,
        *source_hashes,
    )
    positive_count = int(value["positive_count"])
    negative_count = int(value["negative_count"])
    brier = float(value["leave_one_out_brier_score"])
    launch_rmse = float(value["leave_one_out_launch_rmse_mps"])
    seed_brier = float(value["leave_one_seed_out_brier_score"])
    seed_launch_rmse = float(value["leave_one_seed_out_launch_rmse_mps"])
    distinct_seed_count = int(value["distinct_planner_seed_count"])
    ridge = float(value["ridge_regularization"])
    expected_failures: list[str] = []
    if positive_count < _MIN_CLASS_COUNT:
        expected_failures.append("INSUFFICIENT_QUALIFIED_CONTACTS")
    if negative_count < _MIN_CLASS_COUNT:
        expected_failures.append("INSUFFICIENT_REJECTED_CONTACTS")
    if distinct_seed_count < _MIN_DISTINCT_SEEDS:
        expected_failures.append("INSUFFICIENT_DISTINCT_PLANNER_SEEDS")
    if brier > _MAX_LOO_BRIER:
        expected_failures.append("LEAVE_ONE_OUT_QUALIFICATION_ERROR_TOO_HIGH")
    if launch_rmse > _MAX_LOO_LAUNCH_RMSE_MPS:
        expected_failures.append("LEAVE_ONE_OUT_LAUNCH_ERROR_TOO_HIGH")
    if seed_brier > _MAX_LOO_BRIER:
        expected_failures.append("LEAVE_ONE_SEED_OUT_QUALIFICATION_ERROR_TOO_HIGH")
    if seed_launch_rmse > _MAX_LOO_LAUNCH_RMSE_MPS:
        expected_failures.append("LEAVE_ONE_SEED_OUT_LAUNCH_ERROR_TOO_HIGH")
    if (
        len(mean) != len(G1_BALLISTIC_CONTACT_FEATURE_NAMES)
        or len(scale) != len(mean)
        or len(launch_coefficients) != len(mean) + 1
        or any(len(row) != 3 for row in launch_coefficients)
        or len(qualification_coefficients) != len(mean) + 1
        or len(samples) != int(value["sample_count"])
        or len(samples) < _MIN_SAMPLES
        or len(source_hashes) != len(samples)
        or len(set(source_hashes)) != len(source_hashes)
        or source_hashes != tuple(item.evidence_hash for item in samples)
        or not all(_is_sha256(item.trajectory_hash) for item in samples)
        or positive_count != sum(item.qualified_skill_outcome for item in samples)
        or negative_count != len(samples) - positive_count
        or distinct_seed_count != len({item.planner_seed for item in samples})
        or not all(_is_sha256(item) for item in hashes)
        or not all(math.isfinite(item) for item in (*mean, *scale))
        or any(not 1e-6 <= item <= 100.0 for item in scale)
        or not all(math.isfinite(item) for row in launch_coefficients for item in row)
        or any(abs(item) > 10_000.0 for row in launch_coefficients for item in row)
        or not all(math.isfinite(item) for item in qualification_coefficients)
        or any(abs(item) > 10_000.0 for item in qualification_coefficients)
        or not 0.0 <= brier <= 1.0
        or not 0.0 <= seed_brier <= 1.0
        or not math.isfinite(launch_rmse)
        or launch_rmse < 0.0
        or not math.isfinite(seed_launch_rmse)
        or seed_launch_rmse < 0.0
        or not 1e-6 <= ridge <= 100.0
        or value.get("maximum_leave_one_out_brier_score") != _MAX_LOO_BRIER
        or value.get("maximum_leave_one_out_launch_rmse_mps")
        != _MAX_LOO_LAUNCH_RMSE_MPS
        or failures != tuple(expected_failures)
        or ready_value == bool(failures)
    ):
        raise ValueError("ballistic contact observer model geometry is invalid")
    return G1BallisticContactObserver(
        feature_mean=mean,
        feature_scale=scale,
        launch_coefficients=launch_coefficients,  # type: ignore[arg-type]
        qualification_coefficients=qualification_coefficients,
        samples=samples,
        leave_one_out_brier_score=brier,
        leave_one_out_launch_rmse_mps=launch_rmse,
        leave_one_seed_out_brier_score=seed_brier,
        leave_one_seed_out_launch_rmse_mps=seed_launch_rmse,
        distinct_planner_seed_count=distinct_seed_count,
        positive_count=positive_count,
        negative_count=negative_count,
        ridge_regularization=ridge,
        body_hash=str(value["body_hash"]),
        implementation_hash=str(value["implementation_hash"]),
        experiment_context_hash=str(value["experiment_context_hash"]),
        source_evidence_hashes=source_hashes,
        training_ready=ready_value,
        failure_codes=failures,
        observer_hash=observer_hash,
    )


def _sample_from_dict(value: dict[str, Any]) -> G1BallisticContactSample:
    launch = tuple(float(item) for item in value["launch_velocity_xyz_mps"])
    features = tuple(float(item) for item in value["features"])
    qualified = value["qualified_skill_outcome"]
    error = _optional_non_negative_number(value["goal_plane_target_error_m"])
    crossing_height = _optional_non_negative_number(value["goal_crossing_height_m"])
    if (
        len(launch) != 3
        or len(features) != len(G1_BALLISTIC_CONTACT_FEATURE_NAMES)
        or not isinstance(qualified, bool)
        or not isinstance(value["planner_seed"], int)
        or isinstance(value["planner_seed"], bool)
        or value["planner_seed"] < 0
        or not _is_sha256(str(value["evidence_hash"]))
        or not _is_sha256(str(value["trajectory_hash"]))
        or not all(math.isfinite(item) for item in (*launch, *features))
        or any(abs(item) > 100.0 for item in launch)
        or any(abs(item) > 100.0 for item in features[:6])
        or not 0.98 <= float(np.linalg.norm(features[3:6])) <= 1.02
        or not 0.0 < features[6] <= 10.0
        or not -0.25 <= features[7] <= 0.25
    ):
        raise ValueError("ballistic contact observer sample is invalid")
    return G1BallisticContactSample(
        planner_seed=int(value["planner_seed"]),
        features=features,
        launch_velocity_xyz_mps=launch,  # type: ignore[arg-type]
        qualified_skill_outcome=qualified,
        goal_plane_target_error_m=error,
        goal_crossing_height_m=crossing_height,
        evidence_path=str(value["evidence_path"]),
        evidence_hash=str(value["evidence_hash"]),
        trajectory_hash=str(value["trajectory_hash"]),
    )


def _optional_non_negative_number(value: Any) -> float | None:
    if value is None:
        return None
    result = _finite_number(value)
    if result < 0.0:
        raise ValueError("ballistic contact observer metric cannot be negative")
    return result


__all__ = [
    "G1_BALLISTIC_CONTACT_FEATURE_NAMES",
    "G1BallisticContactObserver",
    "G1BallisticContactPrediction",
    "G1BallisticContactSample",
    "derive_g1_ballistic_contact_observer",
    "load_g1_ballistic_contact_observer",
]
