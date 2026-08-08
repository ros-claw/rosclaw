"""Paired sealed evaluation for data-derived SONIC authority calibration."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rosclaw.feedback.contracts import canonical_hash
from rosclaw.growth.sonic_authority_calibration import (
    load_g1_sonic_authority_calibration,
)

_MISS_PENALTY_M = 2.0


@dataclass(frozen=True)
class G1SonicAuthorityEvaluation:
    baseline_calibration_hash: str
    candidate_calibration_hash: str
    seeds: tuple[int, ...]
    baseline_hard_safe_episodes: int
    candidate_hard_safe_episodes: int
    baseline_stability_qualified_episodes: int
    candidate_stability_qualified_episodes: int
    baseline_saturation_episodes: int
    candidate_saturation_episodes: int
    baseline_saturation_steps: int
    candidate_saturation_steps: int
    baseline_precision_hits: int
    candidate_precision_hits: int
    baseline_mean_penalized_error_m: float
    candidate_mean_penalized_error_m: float
    baseline_mean_runup_tilt_rad: float
    candidate_mean_runup_tilt_rad: float
    baseline_mean_kick_tilt_rad: float
    candidate_mean_kick_tilt_rad: float
    strict_replay_all: bool
    accepted: bool
    failure_codes: tuple[str, ...]
    report_hash: str
    schema_version: str = "rosclaw.growth.g1_sonic_authority_evaluation.v1"

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            **asdict(self),
            "seeds": list(self.seeds),
            "failure_codes": list(self.failure_codes),
            "stability_contract": {
                "actuator_saturation": False,
                "maximum_runup_tilt_rad": 0.60,
                "maximum_kick_tilt_rad": 0.40,
                "minimum_final_pelvis_height_m": 0.70,
                "maximum_final_speed_mps": 0.20,
            },
            "evidence_domain": "SIM_ONLY_SEALED_PAIRED_HOLDOUT",
            "promotion_truth_allowed": False,
            "activation_authorized": False,
            "hardware_authorized": False,
        }
        if not include_hash:
            value.pop("report_hash")
        return value


@dataclass(frozen=True)
class _Outcome:
    hard_safe: bool
    stability_qualified: bool
    saturation_steps: int
    precision_hit: bool
    error_m: float
    runup_tilt_rad: float
    kick_tilt_rad: float


def evaluate_g1_sonic_authority_holdout(
    *,
    baseline_paths: tuple[Path, ...],
    candidate_paths: tuple[Path, ...],
    baseline_calibration_path: Path,
    candidate_calibration_path: Path,
    output_path: Path,
    source_checkout: Path,
    maximum_mean_error_regression_m: float = 0.05,
    maximum_mean_tilt_regression_rad: float = 0.01,
) -> G1SonicAuthorityEvaluation:
    baseline_calibration = load_g1_sonic_authority_calibration(baseline_calibration_path)
    candidate_calibration = load_g1_sonic_authority_calibration(candidate_calibration_path)
    if candidate_calibration.base_calibration_hash != baseline_calibration.calibration_hash:
        raise ValueError("SONIC authority candidate is not derived from the baseline")
    if candidate_calibration.body_hash != baseline_calibration.body_hash:
        raise ValueError("SONIC authority calibration Body hashes disagree")
    if len(baseline_paths) != len(candidate_paths) or len(baseline_paths) < 8:
        raise ValueError("SONIC authority evaluation requires at least eight pairs")
    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("SONIC authority evaluation must be outside the checkout")
    if output.exists():
        raise FileExistsError("SONIC authority evaluation output already exists")
    if not 0.0 <= maximum_mean_error_regression_m <= 0.50:
        raise ValueError("maximum error regression must be in [0, 0.5]")
    if not 0.0 <= maximum_mean_tilt_regression_rad <= 0.10:
        raise ValueError("maximum tilt regression must be in [0, 0.1]")

    baseline: dict[int, _Outcome] = {}
    candidate: dict[int, _Outcome] = {}
    body_hashes: set[str] = set()
    implementation_hashes: set[str] = set()
    context_hashes: set[str] = set()
    replay_values: list[bool] = []
    for expected_hash, paths, target in (
        (baseline_calibration.calibration_hash, baseline_paths, baseline),
        (candidate_calibration.calibration_hash, candidate_paths, candidate),
    ):
        for raw_path in paths:
            path = raw_path.expanduser().resolve()
            evidence = json.loads(path.read_text(encoding="utf-8"))
            trajectory = Path(str(evidence.get("trajectory_path", ""))).resolve()
            if not trajectory.is_file() or evidence.get("trajectory_hash") != _file_hash(trajectory):
                raise ValueError("SONIC authority evaluation trajectory binding is invalid")
            replay_values.append(evidence.get("strict_replay") is True)
            body_hashes.add(str(evidence.get("body_hash", "")))
            implementation_hashes.add(str(evidence.get("implementation_hash", "")))
            flow = dict(evidence.get("flow_config", {}))
            sonic = dict(evidence.get("sonic_runup_config", {}))
            runup = dict(evidence.get("runup_config", {}))
            goal = dict(evidence.get("goal_spec", {}))
            for mapping in (flow, sonic, runup, goal):
                mapping.pop("schema_version", None)
            seed = int(sonic.pop("planner_seed", -1))
            if seed < 0 or seed in target:
                raise ValueError("SONIC authority evaluation seed is invalid or duplicated")
            if (
                flow.pop("authority_calibration_hash", None) != expected_hash
                or sonic.pop("authority_calibration_hash", None) != expected_hash
            ):
                raise ValueError("SONIC authority evidence calibration hash is invalid")
            flow.pop("strike_gain_scales", None)
            flow.pop("follow_through_gain_scales", None)
            sonic.pop("joint_gain_scales", None)
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
            target[seed] = _outcome(dict(evidence.get("result", {})))

    if set(baseline) != set(candidate):
        raise ValueError("SONIC authority paired seeds disagree")
    if body_hashes != {baseline_calibration.body_hash}:
        raise ValueError("SONIC authority evaluation Body binding is invalid")
    if len(implementation_hashes) != 1 or len(context_hashes) != 1:
        raise ValueError("SONIC authority implementation or experiment context differs")
    seeds = tuple(sorted(baseline))
    base = [baseline[seed] for seed in seeds]
    cand = [candidate[seed] for seed in seeds]
    baseline_hard = sum(item.hard_safe for item in base)
    candidate_hard = sum(item.hard_safe for item in cand)
    baseline_stable = sum(item.stability_qualified for item in base)
    candidate_stable = sum(item.stability_qualified for item in cand)
    baseline_saturation_episodes = sum(item.saturation_steps > 0 for item in base)
    candidate_saturation_episodes = sum(item.saturation_steps > 0 for item in cand)
    baseline_saturation_steps = sum(item.saturation_steps for item in base)
    candidate_saturation_steps = sum(item.saturation_steps for item in cand)
    baseline_hits = sum(item.precision_hit for item in base)
    candidate_hits = sum(item.precision_hit for item in cand)
    baseline_error = _mean(item.error_m for item in base)
    candidate_error = _mean(item.error_m for item in cand)
    baseline_runup_tilt = _mean(item.runup_tilt_rad for item in base)
    candidate_runup_tilt = _mean(item.runup_tilt_rad for item in cand)
    baseline_kick_tilt = _mean(item.kick_tilt_rad for item in base)
    candidate_kick_tilt = _mean(item.kick_tilt_rad for item in cand)
    strict = all(replay_values)
    failures: list[str] = []
    if not strict:
        failures.append("STRICT_REPLAY_FAILED")
    if candidate_hard < baseline_hard:
        failures.append("HARD_SAFETY_REGRESSION")
    if candidate_stable < baseline_stable:
        failures.append("STABILITY_QUALIFICATION_REGRESSION")
    if candidate_saturation_steps >= baseline_saturation_steps:
        failures.append("SATURATION_NOT_REDUCED")
    if candidate_saturation_episodes > baseline_saturation_episodes:
        failures.append("SATURATION_EPISODE_REGRESSION")
    if candidate_hits < baseline_hits:
        failures.append("PRECISION_REGRESSION")
    if candidate_error > baseline_error + maximum_mean_error_regression_m:
        failures.append("MEAN_ERROR_REGRESSION")
    if candidate_runup_tilt > baseline_runup_tilt + maximum_mean_tilt_regression_rad:
        failures.append("RUNUP_TILT_REGRESSION")
    if candidate_kick_tilt > baseline_kick_tilt + maximum_mean_tilt_regression_rad:
        failures.append("KICK_TILT_REGRESSION")
    accepted = not failures
    unsigned = {
        "schema_version": "rosclaw.growth.g1_sonic_authority_evaluation.v1",
        "baseline_calibration_hash": baseline_calibration.calibration_hash,
        "candidate_calibration_hash": candidate_calibration.calibration_hash,
        "seeds": list(seeds),
        "baseline_hard_safe_episodes": baseline_hard,
        "candidate_hard_safe_episodes": candidate_hard,
        "baseline_stability_qualified_episodes": baseline_stable,
        "candidate_stability_qualified_episodes": candidate_stable,
        "baseline_saturation_episodes": baseline_saturation_episodes,
        "candidate_saturation_episodes": candidate_saturation_episodes,
        "baseline_saturation_steps": baseline_saturation_steps,
        "candidate_saturation_steps": candidate_saturation_steps,
        "baseline_precision_hits": baseline_hits,
        "candidate_precision_hits": candidate_hits,
        "baseline_mean_penalized_error_m": baseline_error,
        "candidate_mean_penalized_error_m": candidate_error,
        "baseline_mean_runup_tilt_rad": baseline_runup_tilt,
        "candidate_mean_runup_tilt_rad": candidate_runup_tilt,
        "baseline_mean_kick_tilt_rad": baseline_kick_tilt,
        "candidate_mean_kick_tilt_rad": candidate_kick_tilt,
        "strict_replay_all": strict,
        "accepted": accepted,
        "failure_codes": failures,
        "stability_contract": {
            "actuator_saturation": False,
            "maximum_runup_tilt_rad": 0.60,
            "maximum_kick_tilt_rad": 0.40,
            "minimum_final_pelvis_height_m": 0.70,
            "maximum_final_speed_mps": 0.20,
        },
        "evidence_domain": "SIM_ONLY_SEALED_PAIRED_HOLDOUT",
        "promotion_truth_allowed": False,
        "activation_authorized": False,
        "hardware_authorized": False,
    }
    report = G1SonicAuthorityEvaluation(
        baseline_calibration_hash=baseline_calibration.calibration_hash,
        candidate_calibration_hash=candidate_calibration.calibration_hash,
        seeds=seeds,
        baseline_hard_safe_episodes=baseline_hard,
        candidate_hard_safe_episodes=candidate_hard,
        baseline_stability_qualified_episodes=baseline_stable,
        candidate_stability_qualified_episodes=candidate_stable,
        baseline_saturation_episodes=baseline_saturation_episodes,
        candidate_saturation_episodes=candidate_saturation_episodes,
        baseline_saturation_steps=baseline_saturation_steps,
        candidate_saturation_steps=candidate_saturation_steps,
        baseline_precision_hits=baseline_hits,
        candidate_precision_hits=candidate_hits,
        baseline_mean_penalized_error_m=baseline_error,
        candidate_mean_penalized_error_m=candidate_error,
        baseline_mean_runup_tilt_rad=baseline_runup_tilt,
        candidate_mean_runup_tilt_rad=candidate_runup_tilt,
        baseline_mean_kick_tilt_rad=baseline_kick_tilt,
        candidate_mean_kick_tilt_rad=candidate_kick_tilt,
        strict_replay_all=strict,
        accepted=accepted,
        failure_codes=tuple(failures),
        report_hash=canonical_hash(unsigned),
    )
    if report.report_hash != canonical_hash(report.to_dict(include_hash=False)):
        raise RuntimeError("SONIC authority evaluation report hash construction diverged")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _outcome(result: dict[str, Any]) -> _Outcome:
    hard_safe = bool(
        result.get("finite_state") is True
        and result.get("post_kick_fall") is False
        and result.get("joint_limit_violation") is False
        and result.get("torque_limit_violation") is False
    )
    saturation_steps = int(result.get("actuator_saturation_steps", 0))
    runup_tilt = float(result["runup_peak_tilt_rad"])
    kick_tilt = float(result["kick_peak_tilt_rad"])
    stability = bool(
        hard_safe
        and saturation_steps == 0
        and runup_tilt <= 0.60
        and kick_tilt <= 0.40
        and float(result["final_pelvis_height_m"]) >= 0.70
        and float(result["final_speed_mps"]) <= 0.20
    )
    crossed = result.get("goal_crossed") is True
    raw_error = result.get("goal_plane_target_error_m")
    error = (
        float(raw_error)
        if crossed and isinstance(raw_error, (int, float)) and math.isfinite(float(raw_error))
        else _MISS_PENALTY_M
    )
    return _Outcome(
        hard_safe=hard_safe,
        stability_qualified=stability,
        saturation_steps=saturation_steps,
        precision_hit=bool(
            hard_safe and crossed and error <= float(result["precision_radius_m"])
        ),
        error_m=error,
        runup_tilt_rad=runup_tilt,
        kick_tilt_rad=kick_tilt,
    )


def _mean(values: Any) -> float:
    items = tuple(float(item) for item in values)
    return sum(items) / len(items)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


__all__ = ["G1SonicAuthorityEvaluation", "evaluate_g1_sonic_authority_holdout"]
