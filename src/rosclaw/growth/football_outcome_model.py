"""Counterfactual football outcome memory for mandatory G1 shot selection."""

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
    G1StrikeHandoffFeatures,
    _trajectory_features,
)

_MAX_TRAJECTORY_BYTES = 2 * 1024 * 1024 * 1024
_MISS_PENALTY_M = 2.0


@dataclass(frozen=True)
class G1FootballOutcomeDecision:
    selected_phase_start_frame: int
    predicted_hard_safe_probability: float
    predicted_precision_probability: float
    predicted_stability_probability: float
    predicted_penalized_error_m: float
    predicted_saturation_score: float
    retry_recommended: bool
    neighbor_seeds: tuple[int, ...]
    neighbor_distances: tuple[float, ...]


@dataclass(frozen=True)
class G1FootballOutcomeModel:
    expert_phases: tuple[int, ...]
    feature_location: tuple[float, ...]
    feature_scale: tuple[float, ...]
    development_seeds: tuple[int, ...]
    development_features: tuple[tuple[float, ...], ...]
    hard_safe_by_phase: tuple[tuple[bool, ...], ...]
    precision_hit_by_phase: tuple[tuple[bool, ...], ...]
    stability_qualified_by_phase: tuple[tuple[bool, ...], ...]
    penalized_error_by_phase: tuple[tuple[float, ...], ...]
    saturation_score_by_phase: tuple[tuple[float, ...], ...]
    kick_tilt_score_by_phase: tuple[tuple[float, ...], ...]
    final_speed_score_by_phase: tuple[tuple[float, ...], ...]
    neighbor_count: int
    distance_power: float
    hard_failure_weight: float
    precision_weight: float
    saturation_weight: float
    maximum_retry_support_distance: float
    minimum_direct_attempt_safety_probability: float
    baseline_phase: int
    baseline_hard_safe_episodes: int
    baseline_precision_hits: int
    baseline_stability_qualified_episodes: int
    baseline_mean_penalized_error_m: float
    cross_validation_hard_safe_episodes: int
    cross_validation_precision_hits: int
    cross_validation_stability_qualified_episodes: int
    cross_validation_mean_penalized_error_m: float
    cross_validation_retry_recommendations: int
    all_experts_unsafe_states: int
    source_evidence_hashes: tuple[str, ...]
    source_implementation_hashes: tuple[str, ...]
    source_schema_versions: tuple[str, ...]
    body_hash: str
    experiment_context_hash: str
    accepted: bool
    failure_codes: tuple[str, ...]
    model_hash: str
    schema_version: str = "rosclaw.growth.g1_football_outcome_model.v1"
    baseline_mean_saturation_score: float | None = None
    cross_validation_mean_saturation_score: float | None = None
    stability_plasticity_guard_enforced: bool = False

    def decide(self, features: G1StrikeHandoffFeatures) -> G1FootballOutcomeDecision:
        if not self.accepted:
            raise ValueError("rejected football outcome model cannot select a shot")
        return _decision(
            features=np.asarray(features.vector, dtype=np.float64),
            bank=np.asarray(self.development_features, dtype=np.float64),
            seeds=np.asarray(self.development_seeds, dtype=np.int64),
            phases=self.expert_phases,
            outcomes=_outcome_tensor(self),
            location=np.asarray(self.feature_location, dtype=np.float64),
            scale=np.asarray(self.feature_scale, dtype=np.float64),
            neighbor_count=self.neighbor_count,
            distance_power=self.distance_power,
            hard_failure_weight=self.hard_failure_weight,
            precision_weight=self.precision_weight,
            saturation_weight=self.saturation_weight,
            maximum_retry_support_distance=self.maximum_retry_support_distance,
            minimum_direct_attempt_safety_probability=(
                self.minimum_direct_attempt_safety_probability
            ),
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
            "development_seeds": list(self.development_seeds),
            "development_features": [list(row) for row in self.development_features],
            "hard_safe_by_phase": [list(row) for row in self.hard_safe_by_phase],
            "precision_hit_by_phase": [list(row) for row in self.precision_hit_by_phase],
            "stability_qualified_by_phase": [
                list(row) for row in self.stability_qualified_by_phase
            ],
            "penalized_error_by_phase": [
                list(row) for row in self.penalized_error_by_phase
            ],
            "saturation_score_by_phase": [
                list(row) for row in self.saturation_score_by_phase
            ],
            "kick_tilt_score_by_phase": [
                list(row) for row in self.kick_tilt_score_by_phase
            ],
            "final_speed_score_by_phase": [
                list(row) for row in self.final_speed_score_by_phase
            ],
            "source_evidence_hashes": list(self.source_evidence_hashes),
            "source_implementation_hashes": list(self.source_implementation_hashes),
            "source_schema_versions": list(self.source_schema_versions),
            "failure_codes": list(self.failure_codes),
            "objective": {
                "football_success_requires_ball_contact": True,
                "recovery_only_is_task_success": False,
                "retry_recommendation_is_terminal_abstention": False,
                "selection_order": "hard_safety_then_precision_then_stability",
                "stability_plasticity_guard_enforced": (
                    self.stability_plasticity_guard_enforced
                ),
                "cross_validation_saturation_may_not_exceed_baseline": (
                    self.stability_plasticity_guard_enforced
                ),
            },
            "evidence_domain": "SIM_ONLY_DEVELOPMENT_COUNTERFACTUAL_MEMORY",
            "sealed_generalization_evidence": False,
            "promotion_truth_allowed": False,
            "activation_authorized": False,
            "hardware_authorized": False,
        }
        if not include_hash:
            value.pop("model_hash")
        return value


@dataclass(frozen=True)
class _EpisodeOutcome:
    features: G1StrikeHandoffFeatures
    hard_safe: bool
    precision_hit: bool
    stability_qualified: bool
    penalized_error_m: float
    saturation_score: float
    kick_tilt_score: float
    final_speed_score: float
    evidence_hash: str


def derive_g1_football_outcome_model(
    *,
    evidence_paths: tuple[Path, ...],
    output_path: Path,
    source_checkout: Path,
    expert_phases: tuple[int, ...] = (190, 205, 214),
    minimum_precision_improvement: int = 3,
) -> G1FootballOutcomeModel:
    """Fit a mandatory-attempt selector from paired success/failure outcomes."""

    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("football outcome model must be outside the checkout")
    if output.exists():
        raise FileExistsError("football outcome model output already exists")
    phases = tuple(sorted(set(expert_phases)))
    if len(phases) < 2 or any(phase < 150 or phase > 260 for phase in phases):
        raise ValueError("football outcome model requires distinct bounded experts")
    if len(evidence_paths) < 48 or len(evidence_paths) % len(phases):
        raise ValueError("football outcome model requires fully paired counterfactuals")
    if minimum_precision_improvement < 1:
        raise ValueError("minimum precision improvement must be positive")

    outcomes: dict[int, dict[int, _EpisodeOutcome]] = {}
    body_hashes: set[str] = set()
    implementation_hashes: set[str] = set()
    schema_versions: set[str] = set()
    context_hashes: set[str] = set()
    for raw_path in evidence_paths:
        path = raw_path.expanduser().resolve()
        evidence = json.loads(path.read_text(encoding="utf-8"))
        if evidence.get("strict_replay") is not True:
            raise ValueError("football outcome model requires strict replay evidence")
        trajectory = Path(str(evidence.get("trajectory_path", ""))).resolve()
        if (
            not trajectory.is_file()
            or evidence.get("trajectory_hash") != _file_hash(trajectory)
            or not 1 <= trajectory.stat().st_size <= _MAX_TRAJECTORY_BYTES
        ):
            raise ValueError("football outcome trajectory binding is invalid")
        body_hashes.add(str(evidence.get("body_hash", "")))
        implementation_hashes.add(str(evidence.get("implementation_hash", "")))
        flow = dict(evidence.get("flow_config", {}))
        sonic = dict(evidence.get("sonic_runup_config", {}))
        runup = dict(evidence.get("runup_config", {}))
        goal = dict(evidence.get("goal_spec", {}))
        for label, mapping in (
            ("flow", flow),
            ("sonic", sonic),
            ("runup", runup),
            ("goal", goal),
        ):
            schema_versions.add(f"{label}:{mapping.pop('schema_version', '')}")
        result = dict(evidence.get("result", {}))
        phase = int(result.get("selected_kick_phase_start_frame", -1))
        if phase not in phases:
            raise ValueError("football outcome evidence executed an unknown expert")
        seed = int(sonic.pop("planner_seed", -1))
        if seed < 0 or phase in outcomes.setdefault(seed, {}):
            raise ValueError("football outcome evidence has a duplicate seed/phase")
        for key in (
            "kick_phase_start_frame",
            "contextual_phase_yaw_threshold_rad",
            "contextual_high_yaw_kick_phase_start_frame",
            "contextual_phase_calibration_hash",
            "proprioceptive_router_hash",
            "football_outcome_model_hash",
            "football_retry_recovery_duration_sec",
            "football_retry_follow_through_gain_scale",
        ):
            flow.pop(key, None)
        if float(flow.get("aim_bias_z_m", 0.0)) == 0.0:
            flow.pop("aim_bias_z_m", None)
        context_hashes.add(
            canonical_hash(
                {
                    "flow_config": flow,
                    "sonic_runup_config": sonic,
                    "runup_config": runup,
                    "goal_spec": goal,
                }
            )
        )
        outcomes[seed][phase] = _episode_outcome(
            trajectory=trajectory,
            result=result,
            evidence_hash=_file_hash(path),
        )

    if len(body_hashes) != 1 or not next(iter(body_hashes)).startswith("sha256:"):
        raise ValueError("football outcome Body binding is invalid")
    if len(context_hashes) != 1:
        raise ValueError("football outcome experiment contexts disagree")
    if not implementation_hashes or not all(_is_sha256(item) for item in implementation_hashes):
        raise ValueError("football outcome implementation hashes are invalid")
    seeds = tuple(sorted(outcomes))
    if len(seeds) < 16 or any(set(outcomes[seed]) != set(phases) for seed in seeds):
        raise ValueError("football outcome model requires every expert for every seed")

    feature_rows: list[tuple[float, ...]] = []
    episode_rows: list[list[_EpisodeOutcome]] = []
    for seed in seeds:
        row = [outcomes[seed][phase] for phase in phases]
        vectors = [np.asarray(item.features.vector, dtype=np.float64) for item in row]
        if any(not np.allclose(vectors[0], item, atol=1e-9, rtol=0.0) for item in vectors[1:]):
            raise ValueError(f"football handoff features differ across phases for seed {seed}")
        feature_rows.append(tuple(float(item) for item in vectors[0]))
        episode_rows.append(row)

    features = np.asarray(feature_rows, dtype=np.float64)
    tensor = _episode_tensor(episode_rows)
    location, scale = _robust_scale(features)
    baseline_phase, baseline_metrics = _best_fixed_baseline(tensor, phases)
    baseline_saturation_sum = float(np.sum(tensor[:, phases.index(baseline_phase), 4]))
    (
        hyperparameters,
        cv_metrics,
        cv_saturation_sum,
        retry_count,
        retry_distances,
    ) = _select_hyperparameters(
        features=features,
        seeds=np.asarray(seeds, dtype=np.int64),
        phases=phases,
        outcomes=tensor,
        baseline_metrics=baseline_metrics,
        baseline_saturation_sum=baseline_saturation_sum,
        minimum_precision_improvement=minimum_precision_improvement,
    )
    maximum_retry_distance = float(max(0.5, np.quantile(retry_distances, 0.95)))
    minimum_safety_probability = 0.75
    failures: list[str] = []
    if cv_metrics[0] < baseline_metrics[0]:
        failures.append("CROSS_VALIDATION_HARD_SAFETY_REGRESSION")
    if cv_metrics[1] < baseline_metrics[1] + minimum_precision_improvement:
        failures.append("CROSS_VALIDATION_PRECISION_GAIN_TOO_SMALL")
    if cv_metrics[3] > baseline_metrics[3] - 0.02:
        failures.append("CROSS_VALIDATION_ERROR_GAIN_TOO_SMALL")
    if cv_saturation_sum > baseline_saturation_sum + 1e-12:
        failures.append("CROSS_VALIDATION_SATURATION_REGRESSION")
    accepted = not failures
    source_hashes = tuple(
        outcomes[seed][phase].evidence_hash for seed in seeds for phase in phases
    )
    hard_safe = tuple(tuple(item.hard_safe for item in row) for row in episode_rows)
    precision = tuple(tuple(item.precision_hit for item in row) for row in episode_rows)
    stability = tuple(
        tuple(item.stability_qualified for item in row) for row in episode_rows
    )
    errors = tuple(tuple(item.penalized_error_m for item in row) for row in episode_rows)
    saturation = tuple(tuple(item.saturation_score for item in row) for row in episode_rows)
    kick_tilt = tuple(tuple(item.kick_tilt_score for item in row) for row in episode_rows)
    final_speed = tuple(tuple(item.final_speed_score for item in row) for row in episode_rows)
    unsigned: dict[str, Any] = {
        "schema_version": "rosclaw.growth.g1_football_outcome_model.v2",
        "expert_phases": list(phases),
        "feature_names": [
            "abs_pelvis_yaw_rad",
            "abs_pelvis_roll_rad",
            "abs_pelvis_pitch_rad",
            "pelvis_x_m",
            "pelvis_y_m",
            "joint_velocity_rms_rad_s",
        ],
        "feature_location": list(location),
        "feature_scale": list(scale),
        "development_seeds": list(seeds),
        "development_features": [list(row) for row in feature_rows],
        "hard_safe_by_phase": [list(row) for row in hard_safe],
        "precision_hit_by_phase": [list(row) for row in precision],
        "stability_qualified_by_phase": [list(row) for row in stability],
        "penalized_error_by_phase": [list(row) for row in errors],
        "saturation_score_by_phase": [list(row) for row in saturation],
        "kick_tilt_score_by_phase": [list(row) for row in kick_tilt],
        "final_speed_score_by_phase": [list(row) for row in final_speed],
        "neighbor_count": hyperparameters[0],
        "distance_power": hyperparameters[1],
        "hard_failure_weight": hyperparameters[2],
        "precision_weight": hyperparameters[3],
        "saturation_weight": hyperparameters[4],
        "maximum_retry_support_distance": maximum_retry_distance,
        "minimum_direct_attempt_safety_probability": minimum_safety_probability,
        "baseline_phase": baseline_phase,
        "baseline_hard_safe_episodes": baseline_metrics[0],
        "baseline_precision_hits": baseline_metrics[1],
        "baseline_stability_qualified_episodes": baseline_metrics[2],
        "baseline_mean_penalized_error_m": baseline_metrics[3],
        "cross_validation_hard_safe_episodes": cv_metrics[0],
        "cross_validation_precision_hits": cv_metrics[1],
        "cross_validation_stability_qualified_episodes": cv_metrics[2],
        "cross_validation_mean_penalized_error_m": cv_metrics[3],
        "cross_validation_retry_recommendations": retry_count,
        "baseline_mean_saturation_score": baseline_saturation_sum / len(seeds),
        "cross_validation_mean_saturation_score": cv_saturation_sum / len(seeds),
        "stability_plasticity_guard_enforced": True,
        "all_experts_unsafe_states": int(np.sum(np.sum(tensor[:, :, 0], axis=1) == 0)),
        "source_evidence_hashes": list(source_hashes),
        "source_implementation_hashes": sorted(implementation_hashes),
        "source_schema_versions": sorted(schema_versions),
        "body_hash": next(iter(body_hashes)),
        "experiment_context_hash": next(iter(context_hashes)),
        "accepted": accepted,
        "failure_codes": failures,
        "objective": {
            "football_success_requires_ball_contact": True,
            "recovery_only_is_task_success": False,
            "retry_recommendation_is_terminal_abstention": False,
            "selection_order": "hard_safety_then_precision_then_stability",
            "stability_plasticity_guard_enforced": True,
            "cross_validation_saturation_may_not_exceed_baseline": True,
        },
        "evidence_domain": "SIM_ONLY_DEVELOPMENT_COUNTERFACTUAL_MEMORY",
        "sealed_generalization_evidence": False,
        "promotion_truth_allowed": False,
        "activation_authorized": False,
        "hardware_authorized": False,
    }
    model = _model_from_unsigned(unsigned, canonical_hash(unsigned))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return model


def load_g1_football_outcome_model(path: Path) -> G1FootballOutcomeModel:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    claimed = str(value.get("model_hash", ""))
    unsigned = dict(value)
    unsigned.pop("model_hash", None)
    if claimed != canonical_hash(unsigned):
        raise ValueError("football outcome model hash mismatch")
    return _model_from_unsigned(unsigned, claimed)


def _episode_outcome(
    *, trajectory: Path, result: dict[str, Any], evidence_hash: str
) -> _EpisodeOutcome:
    hard_safe = bool(
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
    precision = bool(hard_safe and error <= float(result["precision_radius_m"]))
    saturation_steps = int(result["actuator_saturation_steps"])
    kick_tilt = float(result["kick_peak_tilt_rad"])
    final_speed = float(result["final_speed_mps"])
    stability = bool(
        hard_safe
        and saturation_steps == 0
        and float(result["runup_peak_tilt_rad"]) <= 0.60
        and kick_tilt <= 0.40
        and float(result["final_pelvis_height_m"]) >= 0.70
        and final_speed <= 0.20
    )
    return _EpisodeOutcome(
        features=_trajectory_features(trajectory),
        hard_safe=hard_safe,
        precision_hit=precision,
        stability_qualified=stability,
        penalized_error_m=error,
        saturation_score=min(max(saturation_steps, 0), 200) / 200.0,
        kick_tilt_score=min(max(kick_tilt, 0.0), 1.20) / 1.20,
        final_speed_score=min(max(final_speed, 0.0), 1.0),
        evidence_hash=evidence_hash,
    )


def _episode_tensor(rows: list[list[_EpisodeOutcome]]) -> np.ndarray:
    return np.asarray(
        [
            [
                (
                    float(item.hard_safe),
                    float(item.precision_hit),
                    float(item.stability_qualified),
                    item.penalized_error_m,
                    item.saturation_score,
                    item.kick_tilt_score,
                    item.final_speed_score,
                )
                for item in row
            ]
            for row in rows
        ],
        dtype=np.float64,
    )


def _outcome_tensor(model: G1FootballOutcomeModel) -> np.ndarray:
    return np.stack(
        (
            np.asarray(model.hard_safe_by_phase, dtype=np.float64),
            np.asarray(model.precision_hit_by_phase, dtype=np.float64),
            np.asarray(model.stability_qualified_by_phase, dtype=np.float64),
            np.asarray(model.penalized_error_by_phase, dtype=np.float64),
            np.asarray(model.saturation_score_by_phase, dtype=np.float64),
            np.asarray(model.kick_tilt_score_by_phase, dtype=np.float64),
            np.asarray(model.final_speed_score_by_phase, dtype=np.float64),
        ),
        axis=2,
    )


def _select_hyperparameters(
    *,
    features: np.ndarray,
    seeds: np.ndarray,
    phases: tuple[int, ...],
    outcomes: np.ndarray,
    baseline_metrics: tuple[int, int, int, float],
    baseline_saturation_sum: float,
    minimum_precision_improvement: int,
) -> tuple[
    tuple[int, float, float, float, float],
    tuple[int, int, int, float],
    float,
    int,
    list[float],
]:
    candidates: list[
        tuple[
            tuple[int, int, int, int, float, float, int, float, float, float, float],
            tuple[int, float, float, float, float],
            tuple[int, int, int, float],
            float,
            int,
            list[float],
        ]
    ] = []
    for neighbor_count in (1, 3, 5, 7, 9, 11, 13, 15):
        if neighbor_count >= len(features):
            continue
        for distance_power in (1.0, 2.0, 3.0):
            for hard_failure_weight in (2.0, 4.0, 8.0, 16.0):
                for precision_weight in (0.5, 1.0, 2.0):
                    for saturation_weight in (0.05, 0.10, 0.5, 1.0, 2.0, 4.0, 8.0):
                        selected: list[np.ndarray] = []
                        retry_count = 0
                        kth_distances: list[float] = []
                        for index in range(len(features)):
                            keep = np.asarray(
                                [item for item in range(len(features)) if item != index]
                            )
                            location, scale = _robust_scale(features[keep])
                            decision = _decision(
                                features=features[index],
                                bank=features[keep],
                                seeds=seeds[keep],
                                phases=phases,
                                outcomes=outcomes[keep],
                                location=location,
                                scale=scale,
                                neighbor_count=neighbor_count,
                                distance_power=distance_power,
                                hard_failure_weight=hard_failure_weight,
                                precision_weight=precision_weight,
                                saturation_weight=saturation_weight,
                                maximum_retry_support_distance=math.inf,
                                minimum_direct_attempt_safety_probability=0.75,
                            )
                            phase_index = phases.index(decision.selected_phase_start_frame)
                            selected.append(outcomes[index, phase_index])
                            retry_count += int(decision.retry_recommended)
                            kth_distances.append(decision.neighbor_distances[-1])
                        metrics = _metrics(np.asarray(selected))
                        saturation_sum = float(sum(item[4] for item in selected))
                        stability_plasticity_eligible = bool(
                            metrics[0] >= baseline_metrics[0]
                            and metrics[1]
                            >= baseline_metrics[1] + minimum_precision_improvement
                            and metrics[3] <= baseline_metrics[3] - 0.02
                            and saturation_sum <= baseline_saturation_sum + 1e-12
                        )
                        objective = (
                            int(stability_plasticity_eligible),
                            metrics[0],
                            metrics[1],
                            metrics[2],
                            -metrics[3],
                            -saturation_sum,
                            -neighbor_count,
                            -distance_power,
                            -hard_failure_weight,
                            -precision_weight,
                            -saturation_weight,
                        )
                        hyperparameters = (
                            neighbor_count,
                            distance_power,
                            hard_failure_weight,
                            precision_weight,
                            saturation_weight,
                        )
                        candidates.append(
                            (
                                objective,
                                hyperparameters,
                                metrics,
                                saturation_sum,
                                retry_count,
                                kth_distances,
                            )
                        )
    _, hyperparameters, metrics, saturation_sum, retry_count, distances = max(
        candidates, key=lambda item: item[0]
    )
    return hyperparameters, metrics, saturation_sum, retry_count, distances


def _decision(
    *,
    features: np.ndarray,
    bank: np.ndarray,
    seeds: np.ndarray,
    phases: tuple[int, ...],
    outcomes: np.ndarray,
    location: np.ndarray,
    scale: np.ndarray,
    neighbor_count: int,
    distance_power: float,
    hard_failure_weight: float,
    precision_weight: float,
    saturation_weight: float,
    maximum_retry_support_distance: float,
    minimum_direct_attempt_safety_probability: float,
) -> G1FootballOutcomeDecision:
    if bank.shape[0] < neighbor_count or outcomes.shape[:2] != (
        len(bank),
        len(phases),
    ):
        raise ValueError("football outcome memory shape is invalid")
    normalized = (features - location) / scale
    distances = np.linalg.norm((bank - location) / scale - normalized, axis=1)
    order = np.argsort(distances, kind="stable")[:neighbor_count]
    selected_distances = distances[order]
    weights = 1.0 / np.maximum(selected_distances, 0.05) ** distance_power
    weights /= np.sum(weights)
    predicted = np.sum(outcomes[order] * weights[:, None, None], axis=0)
    scores = (
        hard_failure_weight * (1.0 - predicted[:, 0])
        + predicted[:, 3]
        - precision_weight * predicted[:, 1]
        + saturation_weight * predicted[:, 4]
        + 0.02 * predicted[:, 5]
        + 0.02 * predicted[:, 6]
    )
    phase_index = int(np.argmin(scores))
    retry = bool(
        predicted[phase_index, 0] < minimum_direct_attempt_safety_probability
        or selected_distances[-1] > maximum_retry_support_distance
    )
    return G1FootballOutcomeDecision(
        selected_phase_start_frame=phases[phase_index],
        predicted_hard_safe_probability=float(predicted[phase_index, 0]),
        predicted_precision_probability=float(predicted[phase_index, 1]),
        predicted_stability_probability=float(predicted[phase_index, 2]),
        predicted_penalized_error_m=float(predicted[phase_index, 3]),
        predicted_saturation_score=float(predicted[phase_index, 4]),
        retry_recommended=retry,
        neighbor_seeds=tuple(int(seeds[item]) for item in order),
        neighbor_distances=tuple(float(item) for item in selected_distances),
    )


def _best_fixed_baseline(
    outcomes: np.ndarray, phases: tuple[int, ...]
) -> tuple[int, tuple[int, int, int, float]]:
    candidates = []
    for index, phase in enumerate(phases):
        metrics = _metrics(outcomes[:, index])
        candidates.append(((metrics[0], metrics[1], metrics[2], -metrics[3], -phase), phase, metrics))
    _, phase, metrics = max(candidates, key=lambda item: item[0])
    return phase, metrics


def _metrics(selected: np.ndarray) -> tuple[int, int, int, float]:
    return (
        int(np.sum(selected[:, 0])),
        int(np.sum(selected[:, 1])),
        int(np.sum(selected[:, 2])),
        float(np.mean(selected[:, 3])),
    )


def _robust_scale(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    location = np.median(features, axis=0)
    scale = np.percentile(features, 75.0, axis=0) - np.percentile(features, 25.0, axis=0)
    fallback = np.std(features, axis=0)
    scale = np.where(scale >= 1e-6, scale, np.maximum(fallback, 1e-6))
    return location, scale


def _model_from_unsigned(value: dict[str, Any], model_hash: str) -> G1FootballOutcomeModel:
    schema_version = str(value.get("schema_version", ""))
    if (
        schema_version
        not in {
            "rosclaw.growth.g1_football_outcome_model.v1",
            "rosclaw.growth.g1_football_outcome_model.v2",
        }
        or value.get("evidence_domain")
        != "SIM_ONLY_DEVELOPMENT_COUNTERFACTUAL_MEMORY"
        or value.get("sealed_generalization_evidence") is not False
        or value.get("promotion_truth_allowed") is not False
        or value.get("activation_authorized") is not False
        or value.get("hardware_authorized") is not False
    ):
        raise ValueError("football outcome model safety boundary is invalid")
    objective = dict(value.get("objective", {}))
    if (
        objective.get("football_success_requires_ball_contact") is not True
        or objective.get("recovery_only_is_task_success") is not False
        or objective.get("retry_recommendation_is_terminal_abstention") is not False
        or (
            schema_version.endswith(".v2")
            and (
                objective.get("stability_plasticity_guard_enforced") is not True
                or objective.get(
                    "cross_validation_saturation_may_not_exceed_baseline"
                )
                is not True
            )
        )
    ):
        raise ValueError("football outcome model task objective is invalid")
    phases = tuple(int(item) for item in value["expert_phases"])
    seeds = tuple(int(item) for item in value["development_seeds"])
    features = tuple(tuple(float(item) for item in row) for row in value["development_features"])
    shape = (len(seeds), len(phases))

    def boolean_rows(key: str) -> tuple[tuple[bool, ...], ...]:
        rows = tuple(tuple(bool(item) for item in row) for row in value[key])
        if (len(rows), len(rows[0]) if rows else 0) != shape:
            raise ValueError(f"football outcome model {key} shape is invalid")
        return rows

    def float_rows(key: str) -> tuple[tuple[float, ...], ...]:
        rows = tuple(tuple(float(item) for item in row) for row in value[key])
        if (len(rows), len(rows[0]) if rows else 0) != shape or not all(
            math.isfinite(item) for row in rows for item in row
        ):
            raise ValueError(f"football outcome model {key} shape is invalid")
        return rows

    if len(features) != len(seeds) or any(len(row) != 6 for row in features):
        raise ValueError("football outcome development features are invalid")
    model = G1FootballOutcomeModel(
        expert_phases=phases,
        feature_location=tuple(float(item) for item in value["feature_location"]),
        feature_scale=tuple(float(item) for item in value["feature_scale"]),
        development_seeds=seeds,
        development_features=features,
        hard_safe_by_phase=boolean_rows("hard_safe_by_phase"),
        precision_hit_by_phase=boolean_rows("precision_hit_by_phase"),
        stability_qualified_by_phase=boolean_rows("stability_qualified_by_phase"),
        penalized_error_by_phase=float_rows("penalized_error_by_phase"),
        saturation_score_by_phase=float_rows("saturation_score_by_phase"),
        kick_tilt_score_by_phase=float_rows("kick_tilt_score_by_phase"),
        final_speed_score_by_phase=float_rows("final_speed_score_by_phase"),
        neighbor_count=int(value["neighbor_count"]),
        distance_power=float(value["distance_power"]),
        hard_failure_weight=float(value["hard_failure_weight"]),
        precision_weight=float(value["precision_weight"]),
        saturation_weight=float(value["saturation_weight"]),
        maximum_retry_support_distance=float(value["maximum_retry_support_distance"]),
        minimum_direct_attempt_safety_probability=float(
            value["minimum_direct_attempt_safety_probability"]
        ),
        baseline_phase=int(value["baseline_phase"]),
        baseline_hard_safe_episodes=int(value["baseline_hard_safe_episodes"]),
        baseline_precision_hits=int(value["baseline_precision_hits"]),
        baseline_stability_qualified_episodes=int(
            value["baseline_stability_qualified_episodes"]
        ),
        baseline_mean_penalized_error_m=float(value["baseline_mean_penalized_error_m"]),
        cross_validation_hard_safe_episodes=int(
            value["cross_validation_hard_safe_episodes"]
        ),
        cross_validation_precision_hits=int(value["cross_validation_precision_hits"]),
        cross_validation_stability_qualified_episodes=int(
            value["cross_validation_stability_qualified_episodes"]
        ),
        cross_validation_mean_penalized_error_m=float(
            value["cross_validation_mean_penalized_error_m"]
        ),
        cross_validation_retry_recommendations=int(
            value["cross_validation_retry_recommendations"]
        ),
        all_experts_unsafe_states=int(value["all_experts_unsafe_states"]),
        source_evidence_hashes=tuple(str(item) for item in value["source_evidence_hashes"]),
        source_implementation_hashes=tuple(
            str(item) for item in value["source_implementation_hashes"]
        ),
        source_schema_versions=tuple(str(item) for item in value["source_schema_versions"]),
        body_hash=str(value["body_hash"]),
        experiment_context_hash=str(value["experiment_context_hash"]),
        accepted=bool(value["accepted"]),
        failure_codes=tuple(str(item) for item in value["failure_codes"]),
        model_hash=model_hash,
        schema_version=schema_version,
        baseline_mean_saturation_score=(
            None
            if value.get("baseline_mean_saturation_score") is None
            else float(value["baseline_mean_saturation_score"])
        ),
        cross_validation_mean_saturation_score=(
            None
            if value.get("cross_validation_mean_saturation_score") is None
            else float(value["cross_validation_mean_saturation_score"])
        ),
        stability_plasticity_guard_enforced=bool(
            value.get("stability_plasticity_guard_enforced", False)
        ),
    )
    if (
        len(model.feature_location) != 6
        or len(model.feature_scale) != 6
        or not all(item > 0.0 and math.isfinite(item) for item in model.feature_scale)
        or model.neighbor_count < 1
        or model.neighbor_count > len(model.development_seeds)
        or model.accepted == bool(model.failure_codes)
        or (
            schema_version.endswith(".v2")
            and (
                model.baseline_mean_saturation_score is None
                or model.cross_validation_mean_saturation_score is None
                or not math.isfinite(model.baseline_mean_saturation_score)
                or not math.isfinite(model.cross_validation_mean_saturation_score)
                or model.cross_validation_mean_saturation_score
                > model.baseline_mean_saturation_score + 1e-12
                or not model.stability_plasticity_guard_enforced
            )
        )
    ):
        raise ValueError("football outcome model contract is invalid")
    return model


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
    "G1FootballOutcomeDecision",
    "G1FootballOutcomeModel",
    "derive_g1_football_outcome_model",
    "load_g1_football_outcome_model",
]
