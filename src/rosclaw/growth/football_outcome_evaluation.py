"""Sealed counterfactual evaluation for mandatory G1 football attempts."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.feedback.contracts import canonical_hash
from rosclaw.growth.football_outcome_model import load_g1_football_outcome_model
from rosclaw.growth.proprioceptive_expert_router import (
    G1StrikeHandoffFeatures,
    _trajectory_features,
)

_MISS_PENALTY_M = 2.0
_MAX_TRAJECTORY_BYTES = 2 * 1024 * 1024 * 1024


@dataclass(frozen=True)
class G1FootballOutcomeEvaluation:
    model_hash: str
    seeds: tuple[int, ...]
    baseline_phase: int
    mandatory_attempts: int
    terminal_abstentions: int
    retry_recommendations: int
    baseline_hard_safe_episodes: int
    selected_hard_safe_episodes: int
    oracle_hard_safe_episodes: int
    baseline_precision_hits: int
    selected_precision_hits: int
    oracle_precision_hits: int
    baseline_stability_qualified_episodes: int
    selected_stability_qualified_episodes: int
    oracle_stability_qualified_episodes: int
    baseline_mean_penalized_error_m: float
    selected_mean_penalized_error_m: float
    baseline_saturation_steps: int
    selected_saturation_steps: int
    selected_phase_counts: dict[int, int]
    strict_replay_all: bool
    saturation_guard_passed: bool
    measurable_improvement: bool
    accepted: bool
    failure_codes: tuple[str, ...]
    source_evidence_hashes: tuple[str, ...]
    source_implementation_hashes: tuple[str, ...]
    body_hash: str
    experiment_context_hash: str
    report_hash: str
    schema_version: str = "rosclaw.growth.g1_football_outcome_evaluation.v2"

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            **asdict(self),
            "seeds": list(self.seeds),
            "selected_phase_counts": {
                str(key): item for key, item in sorted(self.selected_phase_counts.items())
            },
            "failure_codes": list(self.failure_codes),
            "source_evidence_hashes": list(self.source_evidence_hashes),
            "source_implementation_hashes": list(self.source_implementation_hashes),
            "objective": {
                "football_success_requires_ball_contact": True,
                "recovery_only_is_task_success": False,
                "retry_recommendation_is_terminal_abstention": False,
                "every_seed_selects_and_scores_a_shot": True,
            },
            "evidence_domain": "SIM_ONLY_SEALED_COUNTERFACTUAL_HOLDOUT",
            "sealed_generalization_evidence": True,
            "continuous_retry_physically_validated": False,
            "promotion_truth_allowed": False,
            "activation_authorized": False,
            "hardware_authorized": False,
        }
        if not include_hash:
            value.pop("report_hash")
        return value


@dataclass(frozen=True)
class _ShotOutcome:
    hard_safe: bool
    precision_hit: bool
    stability_qualified: bool
    penalized_error_m: float
    saturation_steps: int


def evaluate_g1_football_outcome_model(
    *,
    evidence_paths: tuple[Path, ...],
    model_path: Path,
    output_path: Path,
    source_checkout: Path,
    minimum_precision_improvement: int = 1,
    minimum_mean_improvement_m: float = 0.02,
) -> G1FootballOutcomeEvaluation:
    """Score a frozen selector on unseen paired three-expert outcomes.

    A retry recommendation remains diagnostic: every sealed state still selects
    and scores one physical shot.  Therefore this evaluator cannot accidentally
    turn recovery or abstention into football success.
    """

    model = load_g1_football_outcome_model(model_path)
    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("football outcome evaluation must be outside the checkout")
    if output.exists():
        raise FileExistsError("football outcome evaluation output already exists")
    if minimum_precision_improvement < 1 or minimum_mean_improvement_m <= 0.0:
        raise ValueError("football outcome improvement thresholds must be positive")
    if len(evidence_paths) < 24 or len(evidence_paths) % len(model.expert_phases):
        raise ValueError("football outcome evaluation requires paired counterfactuals")

    evidence_by_seed: dict[
        int, dict[int, tuple[_ShotOutcome, G1StrikeHandoffFeatures, str]]
    ] = {}
    body_hashes: set[str] = set()
    implementation_hashes: set[str] = set()
    context_hashes: set[str] = set()
    strict_values: list[bool] = []
    for raw_path in evidence_paths:
        path = raw_path.expanduser().resolve()
        evidence = json.loads(path.read_text(encoding="utf-8"))
        trajectory = Path(str(evidence.get("trajectory_path", ""))).resolve()
        if (
            not trajectory.is_file()
            or not 1 <= trajectory.stat().st_size <= _MAX_TRAJECTORY_BYTES
            or evidence.get("trajectory_hash") != _file_hash(trajectory)
        ):
            raise ValueError("football evaluation trajectory binding is invalid")
        strict_values.append(evidence.get("strict_replay") is True)
        body_hashes.add(str(evidence.get("body_hash", "")))
        implementation_hashes.add(str(evidence.get("implementation_hash", "")))
        flow = dict(evidence.get("flow_config", {}))
        sonic = dict(evidence.get("sonic_runup_config", {}))
        runup = dict(evidence.get("runup_config", {}))
        goal = dict(evidence.get("goal_spec", {}))
        for mapping in (flow, sonic, runup, goal):
            mapping.pop("schema_version", None)
        result = dict(evidence.get("result", {}))
        phase = int(result.get("selected_kick_phase_start_frame", -1))
        seed = int(sonic.pop("planner_seed", -1))
        if (
            seed < 0
            or seed in model.development_seeds
            or phase not in model.expert_phases
            or phase in evidence_by_seed.setdefault(seed, {})
        ):
            raise ValueError("football evaluation seed/phase is invalid or leaked")
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
        evidence_by_seed[seed][phase] = (
            _shot_outcome(result),
            _trajectory_features(trajectory),
            _file_hash(path),
        )

    phases = set(model.expert_phases)
    if len(evidence_by_seed) < 8 or any(set(row) != phases for row in evidence_by_seed.values()):
        raise ValueError("football evaluation lacks a complete expert set")
    if len(body_hashes) != 1 or next(iter(body_hashes)) != model.body_hash:
        raise ValueError("football evaluation Body binding is invalid")
    if len(implementation_hashes) != 1 or not all(
        _is_sha256(item) for item in implementation_hashes
    ):
        raise ValueError("football evaluation implementation binding is invalid")
    if len(context_hashes) != 1 or next(iter(context_hashes)) != model.experiment_context_hash:
        raise ValueError("football evaluation experiment context differs")

    baseline: list[_ShotOutcome] = []
    selected: list[_ShotOutcome] = []
    oracle: list[_ShotOutcome] = []
    retry_count = 0
    phase_counts = dict.fromkeys(model.expert_phases, 0)
    source_hashes: list[str] = []
    for seed in sorted(evidence_by_seed):
        row = evidence_by_seed[seed]
        vectors = [
            np.asarray(row[phase][1].vector, dtype=np.float64)
            for phase in model.expert_phases
        ]
        if any(not np.allclose(vectors[0], item, atol=1e-9, rtol=0.0) for item in vectors[1:]):
            raise ValueError(f"football handoff differs across phases for seed {seed}")
        feature = row[model.expert_phases[0]][1]
        decision = model.decide(feature)
        chosen = row[decision.selected_phase_start_frame][0]
        selected.append(chosen)
        baseline.append(row[model.baseline_phase][0])
        oracle.append(max((item[0] for item in row.values()), key=_oracle_key))
        phase_counts[decision.selected_phase_start_frame] += 1
        retry_count += int(decision.retry_recommended)
        source_hashes.extend(row[phase][2] for phase in model.expert_phases)

    baseline_metrics = _metrics(baseline)
    selected_metrics = _metrics(selected)
    oracle_metrics = _metrics(oracle)
    measurable = bool(
        selected_metrics[0] >= baseline_metrics[0]
        and selected_metrics[1] >= baseline_metrics[1] + minimum_precision_improvement
        and selected_metrics[3] <= baseline_metrics[3] - minimum_mean_improvement_m
        and selected_metrics[4] <= baseline_metrics[4]
    )
    failures: list[str] = []
    if not all(strict_values):
        failures.append("STRICT_REPLAY_FAILED")
    if selected_metrics[0] < baseline_metrics[0]:
        failures.append("HARD_SAFETY_REGRESSION")
    if selected_metrics[1] < baseline_metrics[1] + minimum_precision_improvement:
        failures.append("SEALED_PRECISION_GAIN_TOO_SMALL")
    if selected_metrics[3] > baseline_metrics[3] - minimum_mean_improvement_m:
        failures.append("SEALED_ERROR_GAIN_TOO_SMALL")
    if selected_metrics[4] > baseline_metrics[4]:
        failures.append("SEALED_SATURATION_REGRESSION")
    if len(selected) != len(evidence_by_seed):
        failures.append("MANDATORY_ATTEMPT_COVERAGE_FAILED")

    unsigned: dict[str, Any] = {
        "schema_version": "rosclaw.growth.g1_football_outcome_evaluation.v2",
        "model_hash": model.model_hash,
        "seeds": sorted(evidence_by_seed),
        "baseline_phase": model.baseline_phase,
        "mandatory_attempts": len(selected),
        "terminal_abstentions": 0,
        "retry_recommendations": retry_count,
        "baseline_hard_safe_episodes": baseline_metrics[0],
        "selected_hard_safe_episodes": selected_metrics[0],
        "oracle_hard_safe_episodes": oracle_metrics[0],
        "baseline_precision_hits": baseline_metrics[1],
        "selected_precision_hits": selected_metrics[1],
        "oracle_precision_hits": oracle_metrics[1],
        "baseline_stability_qualified_episodes": baseline_metrics[2],
        "selected_stability_qualified_episodes": selected_metrics[2],
        "oracle_stability_qualified_episodes": oracle_metrics[2],
        "baseline_mean_penalized_error_m": baseline_metrics[3],
        "selected_mean_penalized_error_m": selected_metrics[3],
        "baseline_saturation_steps": baseline_metrics[4],
        "selected_saturation_steps": selected_metrics[4],
        "selected_phase_counts": {str(key): item for key, item in sorted(phase_counts.items())},
        "strict_replay_all": all(strict_values),
        "saturation_guard_passed": selected_metrics[4] <= baseline_metrics[4],
        "measurable_improvement": measurable,
        "accepted": not failures,
        "failure_codes": failures,
        "source_evidence_hashes": source_hashes,
        "source_implementation_hashes": sorted(implementation_hashes),
        "body_hash": next(iter(body_hashes)),
        "experiment_context_hash": next(iter(context_hashes)),
        "objective": {
            "football_success_requires_ball_contact": True,
            "recovery_only_is_task_success": False,
            "retry_recommendation_is_terminal_abstention": False,
            "every_seed_selects_and_scores_a_shot": True,
        },
        "evidence_domain": "SIM_ONLY_SEALED_COUNTERFACTUAL_HOLDOUT",
        "sealed_generalization_evidence": True,
        "continuous_retry_physically_validated": False,
        "promotion_truth_allowed": False,
        "activation_authorized": False,
        "hardware_authorized": False,
    }
    report = G1FootballOutcomeEvaluation(
        model_hash=model.model_hash,
        seeds=tuple(sorted(evidence_by_seed)),
        baseline_phase=model.baseline_phase,
        mandatory_attempts=len(selected),
        terminal_abstentions=0,
        retry_recommendations=retry_count,
        baseline_hard_safe_episodes=baseline_metrics[0],
        selected_hard_safe_episodes=selected_metrics[0],
        oracle_hard_safe_episodes=oracle_metrics[0],
        baseline_precision_hits=baseline_metrics[1],
        selected_precision_hits=selected_metrics[1],
        oracle_precision_hits=oracle_metrics[1],
        baseline_stability_qualified_episodes=baseline_metrics[2],
        selected_stability_qualified_episodes=selected_metrics[2],
        oracle_stability_qualified_episodes=oracle_metrics[2],
        baseline_mean_penalized_error_m=baseline_metrics[3],
        selected_mean_penalized_error_m=selected_metrics[3],
        baseline_saturation_steps=baseline_metrics[4],
        selected_saturation_steps=selected_metrics[4],
        selected_phase_counts=phase_counts,
        strict_replay_all=all(strict_values),
        saturation_guard_passed=selected_metrics[4] <= baseline_metrics[4],
        measurable_improvement=measurable,
        accepted=not failures,
        failure_codes=tuple(failures),
        source_evidence_hashes=tuple(source_hashes),
        source_implementation_hashes=tuple(sorted(implementation_hashes)),
        body_hash=next(iter(body_hashes)),
        experiment_context_hash=next(iter(context_hashes)),
        report_hash=canonical_hash(unsigned),
    )
    if report.report_hash != canonical_hash(report.to_dict(include_hash=False)):
        raise RuntimeError("football evaluation report hash construction diverged")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _shot_outcome(result: dict[str, Any]) -> _ShotOutcome:
    hard_safe = bool(
        result.get("finite_state") is True
        and result.get("post_kick_fall") is False
        and result.get("joint_limit_violation") is False
        and result.get("torque_limit_violation") is False
    )
    raw_error = result.get("goal_plane_target_error_m")
    crossed = result.get("goal_crossed") is True
    error = (
        float(raw_error)
        if crossed and isinstance(raw_error, (int, float)) and math.isfinite(float(raw_error))
        else _MISS_PENALTY_M
    )
    saturation_steps = int(result["actuator_saturation_steps"])
    stability = bool(
        hard_safe
        and saturation_steps == 0
        and float(result["runup_peak_tilt_rad"]) <= 0.60
        and float(result["kick_peak_tilt_rad"]) <= 0.40
        and float(result["final_pelvis_height_m"]) >= 0.70
        and float(result["final_speed_mps"]) <= 0.20
    )
    return _ShotOutcome(
        hard_safe=hard_safe,
        precision_hit=bool(hard_safe and error <= float(result["precision_radius_m"])),
        stability_qualified=stability,
        penalized_error_m=error,
        saturation_steps=saturation_steps,
    )


def _oracle_key(item: _ShotOutcome) -> tuple[int, int, int, float, int]:
    return (
        int(item.hard_safe),
        int(item.precision_hit),
        int(item.stability_qualified),
        -item.penalized_error_m,
        -item.saturation_steps,
    )


def _metrics(items: list[_ShotOutcome]) -> tuple[int, int, int, float, int]:
    return (
        sum(item.hard_safe for item in items),
        sum(item.precision_hit for item in items),
        sum(item.stability_qualified for item in items),
        sum(item.penalized_error_m for item in items) / len(items),
        sum(item.saturation_steps for item in items),
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


__all__ = ["G1FootballOutcomeEvaluation", "evaluate_g1_football_outcome_model"]
