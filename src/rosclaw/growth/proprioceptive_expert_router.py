"""Replay-bound proprioceptive routing across strike-phase experts.

The router is deliberately small and inspectable: robust feature scaling,
one centroid per expert, an uncertainty margin, and an out-of-distribution
fallback.  It can select a SIM_ONLY motion expert; it cannot emit torque,
authorize hardware, or provide promotion truth.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.feedback.contracts import canonical_hash

G1_STRIKE_HANDOFF_FEATURE_NAMES = (
    "abs_pelvis_yaw_rad",
    "abs_pelvis_roll_rad",
    "abs_pelvis_pitch_rad",
    "pelvis_x_m",
    "pelvis_y_m",
    "joint_velocity_rms_rad_s",
)
_MISS_PENALTY_M = 2.0
_UNSAFE_PENALTY_M = 4.0
_MAX_TRAJECTORY_BYTES = 2 * 1024 * 1024 * 1024


@dataclass(frozen=True)
class G1StrikeHandoffFeatures:
    abs_pelvis_yaw_rad: float
    abs_pelvis_roll_rad: float
    abs_pelvis_pitch_rad: float
    pelvis_x_m: float
    pelvis_y_m: float
    joint_velocity_rms_rad_s: float
    schema_version: str = "rosclaw.growth.g1_strike_handoff_features.v1"

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in self.vector):
            raise ValueError("strike handoff features must be finite")
        if any(value < 0.0 for value in self.vector[:3]) or self.vector[-1] < 0.0:
            raise ValueError("absolute orientation and RMS features must be non-negative")

    @property
    def vector(self) -> tuple[float, ...]:
        return (
            self.abs_pelvis_yaw_rad,
            self.abs_pelvis_roll_rad,
            self.abs_pelvis_pitch_rad,
            self.pelvis_x_m,
            self.pelvis_y_m,
            self.joint_velocity_rms_rad_s,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            **dict(zip(G1_STRIKE_HANDOFF_FEATURE_NAMES, self.vector, strict=True)),
        }


@dataclass(frozen=True)
class G1ExpertSelection:
    phase_start_frame: int
    used_fallback: bool
    nearest_distance: float
    distance_margin: float


@dataclass(frozen=True)
class G1ProprioceptiveExpertRouter:
    expert_phases: tuple[int, ...]
    fallback_phase: int
    baseline_phase: int
    feature_location: tuple[float, ...]
    feature_scale: tuple[float, ...]
    centroids: tuple[tuple[float, ...], ...]
    confidence_margin: float
    maximum_centroid_distance: float
    development_seeds: tuple[int, ...]
    source_evidence_hashes: tuple[str, ...]
    body_hash: str
    source_implementation_hash: str
    experiment_context_hash: str
    cross_validation_baseline_mean_error_m: float
    cross_validation_selected_mean_error_m: float
    cross_validation_baseline_precision_hits: int
    cross_validation_selected_precision_hits: int
    cross_validation_baseline_unsafe_episodes: int
    cross_validation_selected_unsafe_episodes: int
    accepted: bool
    failure_codes: tuple[str, ...]
    router_hash: str
    schema_version: str = "rosclaw.growth.g1_proprioceptive_expert_router.v1"

    def select(self, features: G1StrikeHandoffFeatures) -> G1ExpertSelection:
        value = np.asarray(features.vector, dtype=np.float64)
        location = np.asarray(self.feature_location, dtype=np.float64)
        scale = np.asarray(self.feature_scale, dtype=np.float64)
        normalized = (value - location) / scale
        distances = np.asarray(
            [np.linalg.norm(normalized - np.asarray(item)) for item in self.centroids],
            dtype=np.float64,
        )
        order = np.argsort(distances)
        nearest = float(distances[order[0]])
        margin = float(distances[order[1]] - nearest)
        fallback = bool(
            nearest > self.maximum_centroid_distance or margin < self.confidence_margin
        )
        phase = self.fallback_phase if fallback else self.expert_phases[int(order[0])]
        return G1ExpertSelection(
            phase_start_frame=phase,
            used_fallback=fallback,
            nearest_distance=nearest,
            distance_margin=margin,
        )

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": self.schema_version,
            "expert_phases": list(self.expert_phases),
            "fallback_phase": self.fallback_phase,
            "baseline_phase": self.baseline_phase,
            "feature_names": list(G1_STRIKE_HANDOFF_FEATURE_NAMES),
            "feature_location": list(self.feature_location),
            "feature_scale": list(self.feature_scale),
            "centroids": [list(item) for item in self.centroids],
            "confidence_margin": self.confidence_margin,
            "maximum_centroid_distance": self.maximum_centroid_distance,
            "development_seeds": list(self.development_seeds),
            "source_evidence_hashes": list(self.source_evidence_hashes),
            "body_hash": self.body_hash,
            "source_implementation_hash": self.source_implementation_hash,
            "experiment_context_hash": self.experiment_context_hash,
            "miss_penalty_m": _MISS_PENALTY_M,
            "unsafe_penalty_m": _UNSAFE_PENALTY_M,
            "cross_validation_baseline_mean_error_m": (
                self.cross_validation_baseline_mean_error_m
            ),
            "cross_validation_selected_mean_error_m": (
                self.cross_validation_selected_mean_error_m
            ),
            "cross_validation_baseline_precision_hits": (
                self.cross_validation_baseline_precision_hits
            ),
            "cross_validation_selected_precision_hits": (
                self.cross_validation_selected_precision_hits
            ),
            "cross_validation_baseline_unsafe_episodes": (
                self.cross_validation_baseline_unsafe_episodes
            ),
            "cross_validation_selected_unsafe_episodes": (
                self.cross_validation_selected_unsafe_episodes
            ),
            "accepted": self.accepted,
            "failure_codes": list(self.failure_codes),
            "evidence_domain": "SIM_ONLY_DEVELOPMENT",
            "sealed_generalization_evidence": False,
            "promotion_truth_allowed": False,
            "activation_authorized": False,
            "hardware_authorized": False,
        }
        if include_hash:
            value["router_hash"] = self.router_hash
        return value


@dataclass(frozen=True)
class _Probe:
    seed: int
    phase: int
    features: G1StrikeHandoffFeatures
    error_m: float
    precision_radius_m: float
    safe: bool
    objective: float
    evidence_hash: str


def strike_handoff_features(
    pelvis_pose_xyz_wxyz: np.ndarray, joint_velocity: np.ndarray
) -> G1StrikeHandoffFeatures:
    pose = np.asarray(pelvis_pose_xyz_wxyz, dtype=np.float64)
    velocity = np.asarray(joint_velocity, dtype=np.float64)
    if pose.shape != (7,) or velocity.shape != (29,):
        raise ValueError("strike handoff state requires pose[7] and joint_velocity[29]")
    if not np.all(np.isfinite(pose)) or not np.all(np.isfinite(velocity)):
        raise ValueError("strike handoff state must be finite")
    w, x, y, z = (float(item) for item in pose[3:7])
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-12:
        raise ValueError("strike handoff quaternion norm is zero")
    w, x, y, z = (item / norm for item in (w, x, y, z))
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return G1StrikeHandoffFeatures(
        abs_pelvis_yaw_rad=abs(yaw),
        abs_pelvis_roll_rad=abs(roll),
        abs_pelvis_pitch_rad=abs(pitch),
        pelvis_x_m=float(pose[0]),
        pelvis_y_m=float(pose[1]),
        joint_velocity_rms_rad_s=float(np.sqrt(np.mean(np.square(velocity)))),
    )


def derive_g1_proprioceptive_expert_router(
    *,
    evidence_paths: tuple[Path, ...],
    output_path: Path,
    source_checkout: Path,
    expert_phases: tuple[int, ...] = (190, 205, 214),
    fallback_phase: int = 190,
    baseline_phase: int = 214,
    confidence_margin: float = 0.05,
    maximum_centroid_distance: float = 2.5,
    minimum_mean_improvement_m: float = 0.05,
) -> G1ProprioceptiveExpertRouter:
    """Fit and leave-one-seed-out gate a three-expert development router."""

    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("expert-router evidence must be outside the source checkout")
    if output.exists():
        raise FileExistsError("expert-router output already exists")
    phases = tuple(expert_phases)
    if len(phases) != 3 or len(set(phases)) != 3 or any(not 100 <= item <= 240 for item in phases):
        raise ValueError("expert router requires three unique phases in [100, 240]")
    if fallback_phase not in phases or baseline_phase not in phases:
        raise ValueError("fallback and baseline phases must be registered experts")
    if not 0.0 <= confidence_margin <= 1.0:
        raise ValueError("expert-router confidence margin must be in [0, 1]")
    if not 0.5 <= maximum_centroid_distance <= 10.0:
        raise ValueError("expert-router maximum centroid distance must be in [0.5, 10]")
    if len(evidence_paths) < 36:
        raise ValueError("expert router requires at least 12 seeds x 3 paired experts")

    probes: dict[int, dict[int, _Probe]] = {}
    body_hashes: set[str] = set()
    implementation_hashes: set[str] = set()
    context_hashes: set[str] = set()
    for raw_path in evidence_paths:
        path = raw_path.expanduser().resolve()
        evidence = json.loads(path.read_text(encoding="utf-8"))
        if evidence.get("strict_replay") is not True:
            raise ValueError("expert router requires strict replay evidence")
        trajectory = Path(str(evidence.get("trajectory_path", ""))).resolve()
        if not trajectory.is_file() or evidence.get("trajectory_hash") != _file_hash(trajectory):
            raise ValueError("expert-router trajectory binding is invalid")
        if not 1 <= trajectory.stat().st_size <= _MAX_TRAJECTORY_BYTES:
            raise ValueError("expert-router trajectory is empty or too large")
        body_hashes.add(str(evidence.get("body_hash", "")))
        implementation_hashes.add(str(evidence.get("implementation_hash", "")))
        flow = dict(evidence.get("flow_config", {}))
        sonic = dict(evidence.get("sonic_runup_config", {}))
        result = dict(evidence.get("result", {}))
        phase = int(flow.get("kick_phase_start_frame", -1))
        if phase not in phases or int(result.get("selected_kick_phase_start_frame", -1)) != phase:
            raise ValueError("expert-router evidence executed an unexpected phase")
        if float(flow.get("contextual_phase_yaw_threshold_rad", 0.0)) != 0.0:
            raise ValueError("expert-router probes must disable the legacy router")
        seed = int(sonic.pop("planner_seed", -1))
        if seed < 0 or phase in probes.setdefault(seed, {}):
            raise ValueError("expert-router evidence has a duplicate seed/phase")
        flow.pop("kick_phase_start_frame", None)
        flow.pop("contextual_phase_yaw_threshold_rad", None)
        flow.pop("contextual_high_yaw_kick_phase_start_frame", None)
        flow.pop("contextual_phase_calibration_hash", None)
        flow.pop("proprioceptive_router_hash", None)
        context_hashes.add(
            canonical_hash(
                {
                    "flow_config": flow,
                    "sonic_runup_config": sonic,
                    "runup_config": evidence.get("runup_config"),
                    "goal_spec": evidence.get("goal_spec"),
                }
            )
        )
        features = _trajectory_features(trajectory)
        safe = bool(
            result.get("finite_state") is True
            and result.get("post_kick_fall") is False
            and result.get("joint_limit_violation") is False
            and result.get("torque_limit_violation") is False
        )
        crossed = result.get("goal_crossed") is True
        raw_error = result.get("goal_plane_target_error_m")
        error = (
            float(raw_error)
            if crossed and isinstance(raw_error, (int, float)) and math.isfinite(float(raw_error))
            else _MISS_PENALTY_M
        )
        delay = result.get("handoff_to_contact_sec")
        delay_value = float(delay) if isinstance(delay, (int, float)) else 5.0
        saturation = int(result.get("actuator_saturation_steps", 0))
        evidence_hash = _file_hash(path)
        probes[seed][phase] = _Probe(
            seed=seed,
            phase=phase,
            features=features,
            error_m=error,
            precision_radius_m=float(result["precision_radius_m"]),
            safe=safe,
            objective=error + (0.0 if safe else _UNSAFE_PENALTY_M) + 0.03 * delay_value + 0.0002 * saturation,
            evidence_hash=evidence_hash,
        )

    if len(body_hashes) != 1 or len(implementation_hashes) != 1 or len(context_hashes) != 1:
        raise ValueError("expert-router Body, implementation, or experiment contexts disagree")
    if len(probes) < 12 or any(set(items) != set(phases) for items in probes.values()):
        raise ValueError("expert router requires all three paired experts per seed")
    seeds = tuple(sorted(probes))
    for seed in seeds:
        vectors = [np.asarray(probes[seed][phase].features.vector) for phase in phases]
        if any(not np.allclose(vectors[0], value, atol=1e-9, rtol=0.0) for value in vectors[1:]):
            raise ValueError(f"expert-router handoff features differ across phases for seed {seed}")

    features = np.asarray([probes[seed][phases[0]].features.vector for seed in seeds])
    objectives = np.asarray([[probes[seed][phase].objective for phase in phases] for seed in seeds])
    labels = np.argmin(objectives, axis=1)
    if any(int(np.count_nonzero(labels == index)) < 2 for index in range(len(phases))):
        raise ValueError("expert router requires at least two development winners per phase")

    selected_indices: list[int] = []
    for held_out in range(len(seeds)):
        train = np.asarray([index for index in range(len(seeds)) if index != held_out])
        location, scale, centroids = _fit_centroids(features[train], labels[train], len(phases))
        selected_indices.append(
            _select_index(
                features[held_out],
                location=location,
                scale=scale,
                centroids=centroids,
                fallback_index=phases.index(fallback_phase),
                confidence_margin=confidence_margin,
                maximum_centroid_distance=maximum_centroid_distance,
            )[0]
        )
    baseline_probes = [probes[seed][baseline_phase] for seed in seeds]
    selected_probes = [probes[seed][phases[index]] for seed, index in zip(seeds, selected_indices, strict=True)]
    baseline_mean = sum(item.error_m for item in baseline_probes) / len(seeds)
    selected_mean = sum(item.error_m for item in selected_probes) / len(seeds)
    baseline_hits = sum(item.safe and item.error_m <= item.precision_radius_m for item in baseline_probes)
    selected_hits = sum(item.safe and item.error_m <= item.precision_radius_m for item in selected_probes)
    baseline_unsafe = sum(not item.safe for item in baseline_probes)
    selected_unsafe = sum(not item.safe for item in selected_probes)
    failure_codes: list[str] = []
    if selected_mean > baseline_mean - minimum_mean_improvement_m:
        failure_codes.append("INSUFFICIENT_CROSS_VALIDATION_IMPROVEMENT")
    if selected_hits < baseline_hits:
        failure_codes.append("CROSS_VALIDATION_PRECISION_REGRESSION")
    if selected_unsafe:
        failure_codes.append("CROSS_VALIDATION_UNSAFE_SELECTION")
    accepted = not failure_codes
    location, scale, centroids = _fit_centroids(features, labels, len(phases))
    source_hashes = tuple(
        probes[seed][phase].evidence_hash for seed in seeds for phase in phases
    )
    unsigned = {
        "schema_version": "rosclaw.growth.g1_proprioceptive_expert_router.v1",
        "expert_phases": list(phases),
        "fallback_phase": fallback_phase,
        "baseline_phase": baseline_phase,
        "feature_names": list(G1_STRIKE_HANDOFF_FEATURE_NAMES),
        "feature_location": location.tolist(),
        "feature_scale": scale.tolist(),
        "centroids": centroids.tolist(),
        "confidence_margin": confidence_margin,
        "maximum_centroid_distance": maximum_centroid_distance,
        "development_seeds": list(seeds),
        "source_evidence_hashes": list(source_hashes),
        "body_hash": next(iter(body_hashes)),
        "source_implementation_hash": next(iter(implementation_hashes)),
        "experiment_context_hash": next(iter(context_hashes)),
        "miss_penalty_m": _MISS_PENALTY_M,
        "unsafe_penalty_m": _UNSAFE_PENALTY_M,
        "cross_validation_baseline_mean_error_m": baseline_mean,
        "cross_validation_selected_mean_error_m": selected_mean,
        "cross_validation_baseline_precision_hits": baseline_hits,
        "cross_validation_selected_precision_hits": selected_hits,
        "cross_validation_baseline_unsafe_episodes": baseline_unsafe,
        "cross_validation_selected_unsafe_episodes": selected_unsafe,
        "accepted": accepted,
        "failure_codes": failure_codes,
        "evidence_domain": "SIM_ONLY_DEVELOPMENT",
        "sealed_generalization_evidence": False,
        "promotion_truth_allowed": False,
        "activation_authorized": False,
        "hardware_authorized": False,
    }
    router = _router_from_unsigned(unsigned, canonical_hash(unsigned))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(router.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return router


def load_g1_proprioceptive_expert_router(path: Path) -> G1ProprioceptiveExpertRouter:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    claimed = str(value.pop("router_hash", ""))
    if claimed != canonical_hash(value):
        raise ValueError("proprioceptive expert-router hash mismatch")
    router = _router_from_unsigned(value, claimed)
    if not router.accepted or router.failure_codes:
        raise ValueError("proprioceptive expert router was not development-accepted")
    return router


def _router_from_unsigned(value: dict[str, Any], router_hash: str) -> G1ProprioceptiveExpertRouter:
    if (
        value.get("schema_version") != "rosclaw.growth.g1_proprioceptive_expert_router.v1"
        or tuple(value.get("feature_names", ())) != G1_STRIKE_HANDOFF_FEATURE_NAMES
        or value.get("evidence_domain") != "SIM_ONLY_DEVELOPMENT"
        or value.get("sealed_generalization_evidence") is not False
        or value.get("promotion_truth_allowed") is not False
        or value.get("activation_authorized") is not False
        or value.get("hardware_authorized") is not False
        or value.get("miss_penalty_m") != _MISS_PENALTY_M
        or value.get("unsafe_penalty_m") != _UNSAFE_PENALTY_M
    ):
        raise ValueError("proprioceptive expert-router safety boundary is invalid")
    phases = tuple(int(item) for item in value["expert_phases"])
    fallback_phase = int(value["fallback_phase"])
    baseline_phase = int(value["baseline_phase"])
    location = tuple(float(item) for item in value["feature_location"])
    scale = tuple(float(item) for item in value["feature_scale"])
    centroids = tuple(tuple(float(item) for item in row) for row in value["centroids"])
    confidence_margin = float(value["confidence_margin"])
    maximum_centroid_distance = float(value["maximum_centroid_distance"])
    development_seeds = tuple(int(item) for item in value["development_seeds"])
    source_hashes = tuple(str(item) for item in value["source_evidence_hashes"])
    provenance_hashes = (
        str(value["body_hash"]),
        str(value["source_implementation_hash"]),
        str(value["experiment_context_hash"]),
        router_hash,
        *source_hashes,
    )
    failure_codes = tuple(str(item) for item in value["failure_codes"])
    accepted = bool(value["accepted"])
    metrics = (
        float(value["cross_validation_baseline_mean_error_m"]),
        float(value["cross_validation_selected_mean_error_m"]),
    )
    if (
        len(phases) != 3
        or len(set(phases)) != 3
        or any(not 100 <= item <= 240 for item in phases)
        or fallback_phase not in phases
        or baseline_phase not in phases
        or len(location) != len(G1_STRIKE_HANDOFF_FEATURE_NAMES)
        or len(scale) != len(location)
        or len(centroids) != len(phases)
        or any(len(row) != len(location) for row in centroids)
        or not all(math.isfinite(item) for item in (*location, *scale, *(x for row in centroids for x in row)))
        or any(item <= 0.0 for item in scale)
        or not 0.0 <= confidence_margin <= 1.0
        or not 0.5 <= maximum_centroid_distance <= 10.0
        or len(development_seeds) < 12
        or len(set(development_seeds)) != len(development_seeds)
        or any(item < 0 for item in development_seeds)
        or len(source_hashes) != len(phases) * len(development_seeds)
        or not all(_is_sha256(item) for item in provenance_hashes)
        or not all(math.isfinite(item) and item >= 0.0 for item in metrics)
        or accepted == bool(failure_codes)
    ):
        raise ValueError("proprioceptive expert-router geometry is invalid")
    return G1ProprioceptiveExpertRouter(
        expert_phases=phases,
        fallback_phase=fallback_phase,
        baseline_phase=baseline_phase,
        feature_location=location,
        feature_scale=scale,
        centroids=centroids,
        confidence_margin=confidence_margin,
        maximum_centroid_distance=maximum_centroid_distance,
        development_seeds=development_seeds,
        source_evidence_hashes=source_hashes,
        body_hash=str(value["body_hash"]),
        source_implementation_hash=str(value["source_implementation_hash"]),
        experiment_context_hash=str(value["experiment_context_hash"]),
        cross_validation_baseline_mean_error_m=float(value["cross_validation_baseline_mean_error_m"]),
        cross_validation_selected_mean_error_m=float(value["cross_validation_selected_mean_error_m"]),
        cross_validation_baseline_precision_hits=int(value["cross_validation_baseline_precision_hits"]),
        cross_validation_selected_precision_hits=int(value["cross_validation_selected_precision_hits"]),
        cross_validation_baseline_unsafe_episodes=int(value["cross_validation_baseline_unsafe_episodes"]),
        cross_validation_selected_unsafe_episodes=int(value["cross_validation_selected_unsafe_episodes"]),
        accepted=accepted,
        failure_codes=failure_codes,
        router_hash=router_hash,
    )


def _fit_centroids(
    features: np.ndarray, labels: np.ndarray, class_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    location = np.median(features, axis=0)
    scale = np.maximum(
        np.quantile(features, 0.75, axis=0) - np.quantile(features, 0.25, axis=0),
        1e-3,
    )
    normalized = (features - location) / scale
    centroids = np.asarray([normalized[labels == index].mean(axis=0) for index in range(class_count)])
    if not np.all(np.isfinite(centroids)):
        raise ValueError("expert-router centroid fit produced a non-finite value")
    return location, scale, centroids


def _select_index(
    features: np.ndarray,
    *,
    location: np.ndarray,
    scale: np.ndarray,
    centroids: np.ndarray,
    fallback_index: int,
    confidence_margin: float,
    maximum_centroid_distance: float,
) -> tuple[int, bool]:
    normalized = (features - location) / scale
    distances = np.linalg.norm(centroids - normalized, axis=1)
    order = np.argsort(distances)
    fallback = bool(
        distances[order[0]] > maximum_centroid_distance
        or distances[order[1]] - distances[order[0]] < confidence_margin
    )
    return (fallback_index if fallback else int(order[0])), fallback


def _trajectory_features(path: Path) -> G1StrikeHandoffFeatures:
    with np.load(path, allow_pickle=False) as archive:
        required = {"controller_mode", "pelvis_pose", "joint_velocity"}
        if not required.issubset(archive.files):
            raise ValueError("expert-router trajectory lacks handoff state arrays")
        mode = np.asarray(archive["controller_mode"])
        pose = np.asarray(archive["pelvis_pose"], dtype=np.float64)
        velocity = np.asarray(archive["joint_velocity"], dtype=np.float64)
    if mode.ndim != 1 or pose.shape != (len(mode), 7) or velocity.shape != (len(mode), 29):
        raise ValueError("expert-router trajectory handoff arrays have invalid shapes")
    approach = np.flatnonzero(mode == 5)
    if approach.size == 0:
        raise ValueError("expert-router trajectory has no SONIC approach state")
    index = int(approach[-1])
    return strike_handoff_features(pose[index], velocity[index])


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 71 and value.startswith("sha256:") and all(
        character in "0123456789abcdef" for character in value[7:]
    )


__all__ = [
    "G1ProprioceptiveExpertRouter",
    "G1StrikeHandoffFeatures",
    "derive_g1_proprioceptive_expert_router",
    "load_g1_proprioceptive_expert_router",
    "strike_handoff_features",
]
