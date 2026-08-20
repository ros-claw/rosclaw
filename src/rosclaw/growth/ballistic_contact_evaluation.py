"""Fail-closed multi-seed evaluation for a G1 ballistic contact action."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rosclaw.feedback.contracts import canonical_hash

_MAX_TRAJECTORY_BYTES = 2 * 1024 * 1024 * 1024


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


@dataclass(frozen=True)
class G1BallisticContactHoldoutEvaluation:
    policy_hash: str
    action_rad: tuple[float, ...]
    seeds: tuple[int, ...]
    episode_count: int
    strict_replay_episodes: int
    hard_safe_episodes: int
    continuous_episodes: int
    goal_crossed_episodes: int
    mean_goal_error_m: float
    worst_goal_error_m: float
    mean_goal_crossing_height_m: float
    minimum_goal_crossing_height_m: float
    maximum_saturation_steps: int
    maximum_actuator_demand_ratio: float
    accepted: bool
    failure_codes: tuple[str, ...]
    body_hash: str
    implementation_hash: str
    experiment_context_hash: str
    source_evidence_hashes: tuple[str, ...]
    report_hash: str
    schema_version: str = "rosclaw.growth.g1_ballistic_contact_holdout_evaluation.v1"

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            **asdict(self),
            "action_rad": list(self.action_rad),
            "seeds": list(self.seeds),
            "failure_codes": list(self.failure_codes),
            "source_evidence_hashes": list(self.source_evidence_hashes),
            "evidence_domain": "SIM_ONLY_SEALED_MULTI_SEED_HOLDOUT",
            "sealed_generalization_evidence": True,
            "promotion_authorized": False,
            "activation_authorized": False,
            "hardware_authorized": False,
        }
        if not include_hash:
            value.pop("report_hash")
        return value


def evaluate_g1_ballistic_contact_holdout(
    *,
    evidence_paths: tuple[Path, ...],
    output_path: Path,
    source_checkout: Path,
    maximum_worst_error_m: float = 0.75,
    minimum_crossing_height_m: float = 0.65,
    maximum_saturation_steps: int = 30,
) -> G1BallisticContactHoldoutEvaluation:
    """Evaluate one frozen action without allowing a seed-0 win to promote it."""

    if len(evidence_paths) < 4:
        raise ValueError("ballistic contact holdout requires at least four seeds")
    if not math.isfinite(maximum_worst_error_m) or maximum_worst_error_m <= 0.0:
        raise ValueError("ballistic holdout error ceiling must be positive")
    if not math.isfinite(minimum_crossing_height_m) or minimum_crossing_height_m <= 0.0:
        raise ValueError("ballistic holdout height floor must be positive")
    if maximum_saturation_steps < 0:
        raise ValueError("ballistic holdout saturation ceiling cannot be negative")
    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("ballistic holdout evidence must be outside the source checkout")
    if output.exists():
        raise FileExistsError("ballistic holdout evaluation output already exists")

    actions: set[tuple[float, ...]] = set()
    seeds: set[int] = set()
    body_hashes: set[str] = set()
    implementation_hashes: set[str] = set()
    context_hashes: set[str] = set()
    source_hashes: list[str] = []
    strict_count = 0
    hard_safe_count = 0
    continuous_count = 0
    goal_crossed_count = 0
    errors: list[float] = []
    heights: list[float] = []
    saturation_steps: list[int] = []
    demand_ratios: list[float] = []

    for raw_path in evidence_paths:
        path = raw_path.expanduser().resolve()
        evidence = json.loads(path.read_text(encoding="utf-8"))
        trajectory = Path(str(evidence.get("trajectory_path", ""))).resolve()
        if (
            not trajectory.is_file()
            or not 1 <= trajectory.stat().st_size <= _MAX_TRAJECTORY_BYTES
            or evidence.get("trajectory_hash") != _file_hash(trajectory)
        ):
            raise ValueError("ballistic holdout trajectory binding is invalid")
        source_hashes.append(_file_hash(path))
        strict_count += int(evidence.get("strict_replay") is True)
        body_hashes.add(str(evidence.get("body_hash", "")))
        implementation_hashes.add(str(evidence.get("implementation_hash", "")))

        flow = dict(evidence.get("flow_config", {}))
        raw_action = flow.get("ballistic_contact_residual_rad")
        if not isinstance(raw_action, list) or len(raw_action) != 6:
            raise ValueError("ballistic holdout evidence lacks a six-joint action")
        action = tuple(float(value) for value in raw_action)
        if not all(math.isfinite(value) for value in action):
            raise ValueError("ballistic holdout action must be finite")
        actions.add(action)

        sonic = dict(evidence.get("sonic_runup_config", {}))
        seed = int(sonic.pop("planner_seed", -1))
        if seed < 0 or seed in seeds:
            raise ValueError("ballistic holdout planner seeds must be unique and non-negative")
        seeds.add(seed)
        for mapping in (flow, sonic):
            mapping.pop("schema_version", None)
        context_hashes.add(
            canonical_hash(
                {
                    "flow_config": flow,
                    "sonic_runup_config_without_seed": sonic,
                    "runup_config": evidence.get("runup_config"),
                    "goal_spec": evidence.get("goal_spec"),
                    "approach_strike_candidate_hash": evidence.get(
                        "approach_strike_candidate_hash"
                    ),
                    "football_motion_prior_hash": evidence.get(
                        "football_motion_prior_hash"
                    ),
                }
            )
        )

        result = dict(evidence.get("result", {}))
        hard_safe = bool(
            result.get("finite_state") is True
            and result.get("post_kick_fall") is False
            and result.get("joint_limit_violation") is False
            and result.get("torque_limit_violation") is False
        )
        hard_safe_count += int(hard_safe)
        continuous_count += int(result.get("perceptual_continuity_passed") is True)
        crossed = result.get("goal_crossed") is True
        goal_crossed_count += int(crossed)
        raw_error = result.get("goal_plane_target_error_m")
        error = (
            float(raw_error)
            if isinstance(raw_error, (int, float)) and math.isfinite(float(raw_error))
            else maximum_worst_error_m + 1.0
        )
        errors.append(error)
        crossing = result.get("goal_crossing_xyz_m")
        height = (
            float(crossing[2])
            if isinstance(crossing, list)
            and len(crossing) == 3
            and isinstance(crossing[2], (int, float))
            and math.isfinite(float(crossing[2]))
            else 0.0
        )
        heights.append(height)
        saturation_steps.append(int(result.get("actuator_saturation_steps", 0)))
        demand = float(result.get("actuator_peak_demand_ratio", 10.0))
        demand_ratios.append(demand if math.isfinite(demand) else 10.0)

    if len(actions) != 1:
        raise ValueError("ballistic holdout evidence mixes candidate actions")
    if len(body_hashes) != 1 or not next(iter(body_hashes)).startswith("sha256:"):
        raise ValueError("ballistic holdout Body hashes disagree")
    if len(implementation_hashes) != 1 or not next(iter(implementation_hashes)).startswith(
        "sha256:"
    ):
        raise ValueError("ballistic holdout implementation hashes disagree")
    if len(context_hashes) != 1:
        raise ValueError("ballistic holdout experiment contexts disagree")

    count = len(evidence_paths)
    failures: list[str] = []
    if strict_count != count:
        failures.append("STRICT_REPLAY_FAILED")
    if hard_safe_count != count:
        failures.append("HARD_SAFETY_REGRESSION")
    if continuous_count != count:
        failures.append("CONTINUITY_GENERALIZATION_FAILED")
    if goal_crossed_count != count:
        failures.append("MANDATORY_SHOT_COVERAGE_FAILED")
    if max(errors) > maximum_worst_error_m:
        failures.append("WORST_CASE_TARGET_ERROR_FAILED")
    if min(heights) < minimum_crossing_height_m:
        failures.append("HIGH_TARGET_GENERALIZATION_FAILED")
    if max(saturation_steps) > maximum_saturation_steps:
        failures.append("SATURATION_GENERALIZATION_FAILED")

    action = next(iter(actions))
    policy_hash = canonical_hash(
        {
            "action_rad": action,
            "experiment_context_hash": next(iter(context_hashes)),
        }
    )
    unsigned: dict[str, Any] = {
        "schema_version": "rosclaw.growth.g1_ballistic_contact_holdout_evaluation.v1",
        "policy_hash": policy_hash,
        "action_rad": list(action),
        "seeds": sorted(seeds),
        "episode_count": count,
        "strict_replay_episodes": strict_count,
        "hard_safe_episodes": hard_safe_count,
        "continuous_episodes": continuous_count,
        "goal_crossed_episodes": goal_crossed_count,
        "mean_goal_error_m": sum(errors) / count,
        "worst_goal_error_m": max(errors),
        "mean_goal_crossing_height_m": sum(heights) / count,
        "minimum_goal_crossing_height_m": min(heights),
        "maximum_saturation_steps": max(saturation_steps),
        "maximum_actuator_demand_ratio": max(demand_ratios),
        "accepted": not failures,
        "failure_codes": failures,
        "body_hash": next(iter(body_hashes)),
        "implementation_hash": next(iter(implementation_hashes)),
        "experiment_context_hash": next(iter(context_hashes)),
        "source_evidence_hashes": source_hashes,
    }
    report_hash = canonical_hash(unsigned)
    report = G1BallisticContactHoldoutEvaluation(
        policy_hash=policy_hash,
        action_rad=action,
        seeds=tuple(sorted(seeds)),
        episode_count=count,
        strict_replay_episodes=strict_count,
        hard_safe_episodes=hard_safe_count,
        continuous_episodes=continuous_count,
        goal_crossed_episodes=goal_crossed_count,
        mean_goal_error_m=sum(errors) / count,
        worst_goal_error_m=max(errors),
        mean_goal_crossing_height_m=sum(heights) / count,
        minimum_goal_crossing_height_m=min(heights),
        maximum_saturation_steps=max(saturation_steps),
        maximum_actuator_demand_ratio=max(demand_ratios),
        accepted=not failures,
        failure_codes=tuple(failures),
        body_hash=next(iter(body_hashes)),
        implementation_hash=next(iter(implementation_hashes)),
        experiment_context_hash=next(iter(context_hashes)),
        source_evidence_hashes=tuple(source_hashes),
        report_hash=report_hash,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


__all__ = [
    "G1BallisticContactHoldoutEvaluation",
    "evaluate_g1_ballistic_contact_holdout",
]
