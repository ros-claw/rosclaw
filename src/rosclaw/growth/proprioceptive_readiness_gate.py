"""Counterfactual safety support for G1 strike-expert selection.

The readiness gate is deliberately conservative.  A frozen strike router may
choose an expert only when nearby development states unanimously support that
expert as physically safe.  Otherwise the gate abstains.  It never emits
torque, turns an abstention into a successful shot, or authorizes hardware.
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
from rosclaw.growth.proprioceptive_expert_router import (
    G1ProprioceptiveExpertRouter,
    G1StrikeHandoffFeatures,
    load_g1_proprioceptive_expert_router,
)

_MAX_TRAJECTORY_BYTES = 2 * 1024 * 1024 * 1024
_MISS_PENALTY_M = 2.0


@dataclass(frozen=True)
class G1ReadinessDecision:
    """One runtime decision backed by nearby counterfactual outcomes."""

    abstained: bool
    selected_phase_start_frame: int | None
    router_phase_start_frame: int
    safe_supported_phases: tuple[int, ...]
    neighbor_seeds: tuple[int, ...]
    neighbor_distances: tuple[float, ...]
    used_router_phase: bool


@dataclass(frozen=True)
class G1ProprioceptiveReadinessGate:
    """Hash-bound k-nearest unanimity gate over paired expert outcomes."""

    router_hash: str
    expert_phases: tuple[int, ...]
    feature_location: tuple[float, ...]
    feature_scale: tuple[float, ...]
    expert_centroids: tuple[tuple[float, ...], ...]
    development_seeds: tuple[int, ...]
    development_features: tuple[tuple[float, ...], ...]
    development_safe_by_phase: tuple[tuple[bool, ...], ...]
    neighbor_count: int
    maximum_support_distance: float
    minimum_attempt_coverage: float
    cross_validation_attempted: int
    cross_validation_abstained: int
    cross_validation_unsafe_attempts: int
    cross_validation_precision_hits: int
    cross_validation_mean_penalized_error_m: float
    cross_validation_all_unsafe_states: int
    cross_validation_all_unsafe_abstained: int
    source_evidence_hashes: tuple[str, ...]
    source_implementation_hashes: tuple[str, ...]
    source_schema_versions: tuple[str, ...]
    body_hash: str
    experiment_context_hash: str
    accepted: bool
    failure_codes: tuple[str, ...]
    gate_hash: str
    schema_version: str = "rosclaw.growth.g1_proprioceptive_readiness_gate.v1"

    def decide(
        self,
        features: G1StrikeHandoffFeatures,
        router: G1ProprioceptiveExpertRouter,
    ) -> G1ReadinessDecision:
        if router.router_hash != self.router_hash:
            raise ValueError("readiness gate/router hash mismatch")
        if router.body_hash != self.body_hash:
            raise ValueError("readiness gate/router Body hash mismatch")
        if not self.accepted:
            raise ValueError("readiness gate was not accepted in development")
        return _decision(
            features=np.asarray(features.vector, dtype=np.float64),
            bank=np.asarray(self.development_features, dtype=np.float64),
            safe_by_phase=np.asarray(self.development_safe_by_phase, dtype=np.bool_),
            seeds=np.asarray(self.development_seeds, dtype=np.int64),
            phases=self.expert_phases,
            location=np.asarray(self.feature_location, dtype=np.float64),
            scale=np.asarray(self.feature_scale, dtype=np.float64),
            centroids=np.asarray(self.expert_centroids, dtype=np.float64),
            neighbor_count=self.neighbor_count,
            maximum_support_distance=self.maximum_support_distance,
            router=router,
        )

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            **asdict(self),
            "expert_phases": list(self.expert_phases),
            "feature_names": [
                "abs_pelvis_yaw_rad",
                "abs_pelvis_roll_rad",
                "abs_pelvis_pitch_rad",
                "pelvis_x_m",
                "pelvis_y_m",
                "joint_velocity_rms_rad_s",
            ],
            "feature_location": list(self.feature_location),
            "feature_scale": list(self.feature_scale),
            "expert_centroids": [list(row) for row in self.expert_centroids],
            "development_seeds": list(self.development_seeds),
            "development_features": [list(row) for row in self.development_features],
            "development_safe_by_phase": [
                list(row) for row in self.development_safe_by_phase
            ],
            "source_evidence_hashes": list(self.source_evidence_hashes),
            "source_implementation_hashes": list(self.source_implementation_hashes),
            "source_schema_versions": list(self.source_schema_versions),
            "failure_codes": list(self.failure_codes),
            "evidence_domain": "SIM_ONLY_DEVELOPMENT",
            "sealed_generalization_evidence": False,
            "promotion_truth_allowed": False,
            "activation_authorized": False,
            "hardware_authorized": False,
        }
        if not include_hash:
            value.pop("gate_hash")
        return value


@dataclass(frozen=True)
class _Outcome:
    features: G1StrikeHandoffFeatures
    safe: bool
    error_m: float
    precision_hit: bool
    evidence_hash: str


def derive_g1_proprioceptive_readiness_gate(
    *,
    evidence_paths: tuple[Path, ...],
    router_path: Path,
    output_path: Path,
    source_checkout: Path,
    neighbor_count: int = 2,
    maximum_support_distance: float = 2.0,
    minimum_attempt_coverage: float = 0.50,
) -> G1ProprioceptiveReadinessGate:
    """Fit and leave-one-seed-out gate counterfactual safety support."""

    router = load_g1_proprioceptive_expert_router(router_path)
    if not router.accepted:
        raise ValueError("readiness gate requires an accepted expert router")
    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("readiness-gate evidence must be outside the source checkout")
    if output.exists():
        raise FileExistsError("readiness-gate output already exists")
    if not 1 <= neighbor_count <= 8:
        raise ValueError("readiness neighbor count must be in [1, 8]")
    if not 0.5 <= maximum_support_distance <= 10.0:
        raise ValueError("maximum readiness support distance must be in [0.5, 10]")
    if not 0.25 <= minimum_attempt_coverage <= 1.0:
        raise ValueError("minimum readiness attempt coverage must be in [0.25, 1]")
    if len(evidence_paths) < 48 or len(evidence_paths) % len(router.expert_phases):
        raise ValueError("readiness gate requires at least 16 fully paired seeds")

    outcomes: dict[int, dict[int, _Outcome]] = {}
    body_hashes: set[str] = set()
    implementation_hashes: set[str] = set()
    schema_versions: set[str] = set()
    context_hashes: set[str] = set()
    for raw_path in evidence_paths:
        path = raw_path.expanduser().resolve()
        evidence = json.loads(path.read_text(encoding="utf-8"))
        if evidence.get("strict_replay") is not True:
            raise ValueError("readiness gate requires strict replay evidence")
        trajectory = Path(str(evidence.get("trajectory_path", ""))).resolve()
        if not trajectory.is_file() or evidence.get("trajectory_hash") != _file_hash(trajectory):
            raise ValueError("readiness-gate trajectory binding is invalid")
        if not 1 <= trajectory.stat().st_size <= _MAX_TRAJECTORY_BYTES:
            raise ValueError("readiness-gate trajectory is empty or too large")
        body_hashes.add(str(evidence.get("body_hash", "")))
        implementation_hashes.add(str(evidence.get("implementation_hash", "")))
        flow = dict(evidence.get("flow_config", {}))
        sonic = dict(evidence.get("sonic_runup_config", {}))
        runup_config = dict(evidence.get("runup_config", {}))
        goal_spec = dict(evidence.get("goal_spec", {}))
        for label, mapping in (
            ("flow", flow),
            ("sonic", sonic),
            ("runup", runup_config),
            ("goal", goal_spec),
        ):
            schema_versions.add(f"{label}:{mapping.pop('schema_version', '')}")
        result = dict(evidence.get("result", {}))
        phase = int(result.get("selected_kick_phase_start_frame", -1))
        if phase not in router.expert_phases:
            raise ValueError("readiness evidence executed an unknown expert")
        declared_phase = int(flow.get("kick_phase_start_frame", phase))
        routed = result.get("proprioceptive_router_executed") is True
        if not routed and declared_phase != phase:
            raise ValueError("readiness evidence declared and executed phases disagree")
        seed = int(sonic.pop("planner_seed", -1))
        if seed < 0 or phase in outcomes.setdefault(seed, {}):
            raise ValueError("readiness evidence has a duplicate seed/phase")
        for key in (
            "kick_phase_start_frame",
            "contextual_phase_yaw_threshold_rad",
            "contextual_high_yaw_kick_phase_start_frame",
            "contextual_phase_calibration_hash",
            "proprioceptive_router_hash",
        ):
            flow.pop(key, None)
        context_hashes.add(
            canonical_hash(
                {
                    "flow_config": flow,
                    "sonic_runup_config": sonic,
                    "runup_config": runup_config,
                    "goal_spec": goal_spec,
                }
            )
        )
        safe = _safe(result)
        crossed = result.get("goal_crossed") is True
        raw_error = result.get("goal_plane_target_error_m")
        error = (
            float(raw_error)
            if crossed
            and isinstance(raw_error, (int, float))
            and math.isfinite(float(raw_error))
            else _MISS_PENALTY_M
        )
        radius = float(result["precision_radius_m"])
        outcomes[seed][phase] = _Outcome(
            features=_trajectory_features(trajectory),
            safe=safe,
            error_m=error,
            precision_hit=bool(safe and crossed and error <= radius),
            evidence_hash=_file_hash(path),
        )

    if body_hashes != {router.body_hash}:
        raise ValueError("readiness-gate Body binding is invalid")
    if len(context_hashes) != 1:
        raise ValueError("readiness-gate experiment contexts disagree")
    if not implementation_hashes or not all(_is_sha256(item) for item in implementation_hashes):
        raise ValueError("readiness-gate implementation hashes are invalid")
    phases = router.expert_phases
    seeds = tuple(sorted(outcomes))
    if len(seeds) < 16 or any(set(outcomes[seed]) != set(phases) for seed in seeds):
        raise ValueError("readiness gate requires every expert for every seed")
    feature_rows: list[tuple[float, ...]] = []
    safe_rows: list[tuple[bool, ...]] = []
    for seed in seeds:
        vectors = [np.asarray(outcomes[seed][phase].features.vector) for phase in phases]
        if any(not np.allclose(vectors[0], item, atol=1e-9, rtol=0.0) for item in vectors[1:]):
            raise ValueError(f"readiness handoff features differ across phases for seed {seed}")
        feature_rows.append(tuple(float(item) for item in vectors[0]))
        safe_rows.append(tuple(outcomes[seed][phase].safe for phase in phases))

    features = np.asarray(feature_rows, dtype=np.float64)
    safe_by_phase = np.asarray(safe_rows, dtype=np.bool_)
    location = np.asarray(router.feature_location, dtype=np.float64)
    scale = np.asarray(router.feature_scale, dtype=np.float64)
    centroids = np.asarray(router.centroids, dtype=np.float64)
    attempted = 0
    abstained = 0
    unsafe_attempts = 0
    precision_hits = 0
    errors: list[float] = []
    all_unsafe_states = 0
    all_unsafe_abstained = 0
    for index, seed in enumerate(seeds):
        keep = np.asarray([item for item in range(len(seeds)) if item != index])
        decision = _decision(
            features=features[index],
            bank=features[keep],
            safe_by_phase=safe_by_phase[keep],
            seeds=np.asarray(seeds, dtype=np.int64)[keep],
            phases=phases,
            location=location,
            scale=scale,
            centroids=centroids,
            neighbor_count=neighbor_count,
            maximum_support_distance=maximum_support_distance,
            router=router,
        )
        all_unsafe = not bool(np.any(safe_by_phase[index]))
        all_unsafe_states += int(all_unsafe)
        if decision.abstained:
            abstained += 1
            all_unsafe_abstained += int(all_unsafe)
            continue
        attempted += 1
        assert decision.selected_phase_start_frame is not None
        outcome = outcomes[seed][decision.selected_phase_start_frame]
        unsafe_attempts += int(not outcome.safe)
        precision_hits += int(outcome.precision_hit)
        errors.append(outcome.error_m)

    failures: list[str] = []
    coverage = attempted / len(seeds)
    if unsafe_attempts:
        failures.append("CROSS_VALIDATION_UNSAFE_ATTEMPT")
    if coverage < minimum_attempt_coverage:
        failures.append("CROSS_VALIDATION_COVERAGE_TOO_LOW")
    if all_unsafe_states == 0:
        failures.append("NO_NEGATIVE_READINESS_EXAMPLES")
    elif all_unsafe_abstained != all_unsafe_states:
        failures.append("ALL_UNSAFE_STATE_NOT_ABSTAINED")
    accepted = not failures
    source_hashes = tuple(
        outcomes[seed][phase].evidence_hash for seed in seeds for phase in phases
    )
    unsigned: dict[str, Any] = {
        "schema_version": "rosclaw.growth.g1_proprioceptive_readiness_gate.v1",
        "router_hash": router.router_hash,
        "expert_phases": list(phases),
        "feature_names": [
            "abs_pelvis_yaw_rad",
            "abs_pelvis_roll_rad",
            "abs_pelvis_pitch_rad",
            "pelvis_x_m",
            "pelvis_y_m",
            "joint_velocity_rms_rad_s",
        ],
        "feature_location": list(router.feature_location),
        "feature_scale": list(router.feature_scale),
        "expert_centroids": [list(row) for row in router.centroids],
        "development_seeds": list(seeds),
        "development_features": [list(row) for row in feature_rows],
        "development_safe_by_phase": [list(row) for row in safe_rows],
        "neighbor_count": neighbor_count,
        "maximum_support_distance": maximum_support_distance,
        "minimum_attempt_coverage": minimum_attempt_coverage,
        "cross_validation_attempted": attempted,
        "cross_validation_abstained": abstained,
        "cross_validation_unsafe_attempts": unsafe_attempts,
        "cross_validation_precision_hits": precision_hits,
        "cross_validation_mean_penalized_error_m": (
            sum(errors) / len(errors) if errors else _MISS_PENALTY_M
        ),
        "cross_validation_all_unsafe_states": all_unsafe_states,
        "cross_validation_all_unsafe_abstained": all_unsafe_abstained,
        "source_evidence_hashes": list(source_hashes),
        "source_implementation_hashes": sorted(implementation_hashes),
        "source_schema_versions": sorted(schema_versions),
        "body_hash": router.body_hash,
        "experiment_context_hash": next(iter(context_hashes)),
        "accepted": accepted,
        "failure_codes": failures,
        "evidence_domain": "SIM_ONLY_DEVELOPMENT",
        "sealed_generalization_evidence": False,
        "promotion_truth_allowed": False,
        "activation_authorized": False,
        "hardware_authorized": False,
    }
    gate = _gate_from_unsigned(unsigned, canonical_hash(unsigned))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(gate.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return gate


def load_g1_proprioceptive_readiness_gate(path: Path) -> G1ProprioceptiveReadinessGate:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    gate_hash = str(value.get("gate_hash", ""))
    unsigned = dict(value)
    unsigned.pop("gate_hash", None)
    if gate_hash != canonical_hash(unsigned):
        raise ValueError("proprioceptive readiness-gate hash mismatch")
    return _gate_from_unsigned(unsigned, gate_hash)


def _gate_from_unsigned(value: dict[str, Any], gate_hash: str) -> G1ProprioceptiveReadinessGate:
    expected_names = (
        "abs_pelvis_yaw_rad",
        "abs_pelvis_roll_rad",
        "abs_pelvis_pitch_rad",
        "pelvis_x_m",
        "pelvis_y_m",
        "joint_velocity_rms_rad_s",
    )
    if (
        value.get("schema_version") != "rosclaw.growth.g1_proprioceptive_readiness_gate.v1"
        or tuple(value.get("feature_names", ())) != expected_names
        or value.get("evidence_domain") != "SIM_ONLY_DEVELOPMENT"
        or value.get("sealed_generalization_evidence") is not False
        or value.get("promotion_truth_allowed") is not False
        or value.get("activation_authorized") is not False
        or value.get("hardware_authorized") is not False
    ):
        raise ValueError("proprioceptive readiness-gate safety boundary is invalid")
    phases = tuple(int(item) for item in value["expert_phases"])
    location = tuple(float(item) for item in value["feature_location"])
    scale = tuple(float(item) for item in value["feature_scale"])
    centroids = tuple(tuple(float(item) for item in row) for row in value["expert_centroids"])
    seeds = tuple(int(item) for item in value["development_seeds"])
    features = tuple(tuple(float(item) for item in row) for row in value["development_features"])
    safe = tuple(tuple(bool(item) for item in row) for row in value["development_safe_by_phase"])
    source_hashes = tuple(str(item) for item in value["source_evidence_hashes"])
    implementation_hashes = tuple(str(item) for item in value["source_implementation_hashes"])
    schema_versions = tuple(str(item) for item in value["source_schema_versions"])
    failures = tuple(str(item) for item in value["failure_codes"])
    accepted = bool(value["accepted"])
    neighbor_count = int(value["neighbor_count"])
    maximum_distance = float(value["maximum_support_distance"])
    minimum_coverage = float(value["minimum_attempt_coverage"])
    if (
        len(phases) != 3
        or len(set(phases)) != 3
        or len(location) != len(expected_names)
        or len(scale) != len(expected_names)
        or len(centroids) != len(phases)
        or any(len(row) != len(expected_names) for row in centroids)
        or len(seeds) < 16
        or len(set(seeds)) != len(seeds)
        or len(features) != len(seeds)
        or any(len(row) != len(expected_names) for row in features)
        or len(safe) != len(seeds)
        or any(len(row) != len(phases) for row in safe)
        or len(source_hashes) != len(seeds) * len(phases)
        or not all(_is_sha256(item) for item in source_hashes)
        or not all(_is_sha256(item) for item in implementation_hashes)
        or not schema_versions
        or any(":" not in item for item in schema_versions)
        or not _is_sha256(str(value["router_hash"]))
        or not _is_sha256(str(value["body_hash"]))
        or not _is_sha256(str(value["experiment_context_hash"]))
        or not _is_sha256(gate_hash)
        or not all(math.isfinite(item) for item in (*location, *scale, *(x for row in centroids for x in row), *(x for row in features for x in row)))
        or any(item <= 0.0 for item in scale)
        or not 1 <= neighbor_count <= 8
        or not 0.5 <= maximum_distance <= 10.0
        or not 0.25 <= minimum_coverage <= 1.0
        or accepted == bool(failures)
    ):
        raise ValueError("proprioceptive readiness-gate geometry is invalid")
    return G1ProprioceptiveReadinessGate(
        router_hash=str(value["router_hash"]),
        expert_phases=phases,
        feature_location=location,
        feature_scale=scale,
        expert_centroids=centroids,
        development_seeds=seeds,
        development_features=features,
        development_safe_by_phase=safe,
        neighbor_count=neighbor_count,
        maximum_support_distance=maximum_distance,
        minimum_attempt_coverage=minimum_coverage,
        cross_validation_attempted=int(value["cross_validation_attempted"]),
        cross_validation_abstained=int(value["cross_validation_abstained"]),
        cross_validation_unsafe_attempts=int(value["cross_validation_unsafe_attempts"]),
        cross_validation_precision_hits=int(value["cross_validation_precision_hits"]),
        cross_validation_mean_penalized_error_m=float(value["cross_validation_mean_penalized_error_m"]),
        cross_validation_all_unsafe_states=int(value["cross_validation_all_unsafe_states"]),
        cross_validation_all_unsafe_abstained=int(value["cross_validation_all_unsafe_abstained"]),
        source_evidence_hashes=source_hashes,
        source_implementation_hashes=implementation_hashes,
        source_schema_versions=schema_versions,
        body_hash=str(value["body_hash"]),
        experiment_context_hash=str(value["experiment_context_hash"]),
        accepted=accepted,
        failure_codes=failures,
        gate_hash=gate_hash,
    )


def _decision(
    *,
    features: np.ndarray,
    bank: np.ndarray,
    safe_by_phase: np.ndarray,
    seeds: np.ndarray,
    phases: tuple[int, ...],
    location: np.ndarray,
    scale: np.ndarray,
    centroids: np.ndarray,
    neighbor_count: int,
    maximum_support_distance: float,
    router: G1ProprioceptiveExpertRouter,
) -> G1ReadinessDecision:
    if bank.shape[0] < neighbor_count:
        raise ValueError("readiness bank is smaller than the neighbor count")
    normalized = (features - location) / scale
    normalized_bank = (bank - location) / scale
    distances = np.linalg.norm(normalized_bank - normalized, axis=1)
    order = np.argsort(distances, kind="stable")[:neighbor_count]
    neighbors = tuple(int(seeds[item]) for item in order)
    neighbor_distances = tuple(float(distances[item]) for item in order)
    within_support = neighbor_distances[-1] <= maximum_support_distance
    supported = tuple(
        phase
        for phase_index, phase in enumerate(phases)
        if within_support and bool(np.all(safe_by_phase[order, phase_index]))
    )
    router_phase = router.select(
        G1StrikeHandoffFeatures(*tuple(float(item) for item in features))
    ).phase_start_frame
    if not supported:
        return G1ReadinessDecision(
            abstained=True,
            selected_phase_start_frame=None,
            router_phase_start_frame=router_phase,
            safe_supported_phases=(),
            neighbor_seeds=neighbors,
            neighbor_distances=neighbor_distances,
            used_router_phase=False,
        )
    if router_phase in supported:
        selected = router_phase
        used_router = True
    else:
        phase_distances = {
            phase: float(np.linalg.norm(normalized - centroids[phases.index(phase)]))
            for phase in supported
        }
        selected = min(supported, key=lambda phase: (phase_distances[phase], phase))
        used_router = False
    return G1ReadinessDecision(
        abstained=False,
        selected_phase_start_frame=selected,
        router_phase_start_frame=router_phase,
        safe_supported_phases=supported,
        neighbor_seeds=neighbors,
        neighbor_distances=neighbor_distances,
        used_router_phase=used_router,
    )


def _trajectory_features(path: Path) -> G1StrikeHandoffFeatures:
    from rosclaw.growth.proprioceptive_expert_router import _trajectory_features as load

    return load(path)


def _safe(result: dict[str, Any]) -> bool:
    return bool(
        result.get("finite_state") is True
        and result.get("post_kick_fall") is False
        and result.get("joint_limit_violation") is False
        and result.get("torque_limit_violation") is False
    )


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
    "G1ProprioceptiveReadinessGate",
    "G1ReadinessDecision",
    "derive_g1_proprioceptive_readiness_gate",
    "load_g1_proprioceptive_readiness_gate",
]
