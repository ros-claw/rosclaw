"""Paired multi-context gate for a coupled G1 contact proposal.

One planner state can make a constant contact residual look excellent while a
different proprioceptive handoff makes the same residual harmful.  This gate
binds paired anchor/candidate strict replays, evaluates the worst context and
lower-tail CVaR, and refuses champion replacement on context regressions.
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
from rosclaw.growth.ballistic_contact_residual import G1BallisticContactResidualConfig

_ACTOR_SCHEMA = "rosclaw.growth.g1_ballistic_contact_coupled_actor_critic.v1"
_SCHEMA = "rosclaw.growth.g1_ballistic_contact_context_holdout_evaluation.v1"


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


@dataclass(frozen=True)
class G1BallisticContactContextCase:
    planner_seed_audit_label: int
    anchor_evidence_hash: str
    candidate_evidence_hash: str
    anchor_error_m: float
    candidate_error_m: float
    error_improvement_m: float
    candidate_hard_safe: bool
    stability_preserved: bool

    def __post_init__(self) -> None:
        if self.planner_seed_audit_label < 0:
            raise ValueError("contact context seed audit label is invalid")
        if not self.anchor_evidence_hash.startswith(
            "sha256:"
        ) or not self.candidate_evidence_hash.startswith("sha256:"):
            raise ValueError("contact context case requires evidence hashes")
        if (
            not all(
                math.isfinite(value)
                for value in (self.anchor_error_m, self.candidate_error_m, self.error_improvement_m)
            )
            or min(self.anchor_error_m, self.candidate_error_m) < 0.0
        ):
            raise ValueError("contact context case metrics are invalid")


@dataclass(frozen=True)
class G1BallisticContactContextHoldoutEvaluation:
    actor_critic_hash: str
    actor_critic_artifact_hash: str
    evaluation_implementation_hash: str
    body_hash: str
    implementation_hash: str
    context_family_hash: str
    anchor_action_rad: tuple[float, ...]
    candidate_action_rad: tuple[float, ...]
    cases: tuple[G1BallisticContactContextCase, ...]
    mean_error_improvement_m: float
    worst_context_improvement_m: float
    lower_tail_cvar_improvement_m: float
    minimum_mean_improvement_m: float
    maximum_context_regression_m: float
    improved_context_count: int
    required_improved_context_count: int
    all_contexts_hard_safe: bool
    all_contexts_stability_preserved: bool
    accepted: bool
    decision: str
    activation_ceiling: str = "SIM_ONLY"
    sealed_generalization_evidence: bool = False
    promotion_authorized: bool = False
    hardware_authorized: bool = False
    online_hot_swap_allowed: bool = False
    schema_version: str = _SCHEMA

    def __post_init__(self) -> None:
        hashes = (
            self.actor_critic_hash,
            self.actor_critic_artifact_hash,
            self.evaluation_implementation_hash,
            self.body_hash,
            self.implementation_hash,
            self.context_family_hash,
        )
        if any(not value.startswith("sha256:") for value in hashes):
            raise ValueError("contact context evaluation requires SHA-256 bindings")
        G1BallisticContactResidualConfig(right_leg_residual_rad=self.anchor_action_rad)
        G1BallisticContactResidualConfig(right_leg_residual_rad=self.candidate_action_rad)
        if len(self.cases) < 3 or len(
            {case.planner_seed_audit_label for case in self.cases}
        ) != len(self.cases):
            raise ValueError("contact context evaluation requires three distinct states")
        metrics = (
            self.mean_error_improvement_m,
            self.worst_context_improvement_m,
            self.lower_tail_cvar_improvement_m,
            self.minimum_mean_improvement_m,
            self.maximum_context_regression_m,
        )
        if not all(math.isfinite(value) for value in metrics):
            raise ValueError("contact context evaluation metrics must be finite")
        if not 1 <= self.required_improved_context_count <= len(self.cases):
            raise ValueError("contact context improvement quorum is invalid")
        if not 0 <= self.improved_context_count <= len(self.cases):
            raise ValueError("contact context improvement count is invalid")
        expected = bool(
            self.mean_error_improvement_m >= self.minimum_mean_improvement_m
            and self.worst_context_improvement_m >= -self.maximum_context_regression_m
            and self.lower_tail_cvar_improvement_m >= -self.maximum_context_regression_m
            and self.improved_context_count >= self.required_improved_context_count
            and self.all_contexts_hard_safe
            and self.all_contexts_stability_preserved
        )
        if self.accepted != expected or self.decision != ("ACCEPTED" if expected else "REJECTED"):
            raise ValueError("contact context evaluation decision is inconsistent")
        if (
            self.schema_version != _SCHEMA
            or self.activation_ceiling != "SIM_ONLY"
            or self.sealed_generalization_evidence
            or self.promotion_authorized
            or self.hardware_authorized
            or self.online_hot_swap_allowed
        ):
            raise ValueError("contact context evaluation cannot authorize promotion")

    @property
    def evaluation_hash(self) -> str:
        return canonical_hash(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            **asdict(self),
            "cases": [asdict(case) for case in self.cases],
            "risk_measure": "lower_tail_cvar_25_percent_plus_worst_context",
            "planner_seed_usage": "audit_label_only_not_policy_input",
        }
        if include_hash:
            value["evaluation_hash"] = self.evaluation_hash
        return value


def evaluate_g1_ballistic_contact_context_holdout(
    *,
    actor_critic_path: Path,
    anchor_evidence_paths: tuple[Path, ...],
    candidate_evidence_paths: tuple[Path, ...],
    output_path: Path,
    source_checkout: Path,
    minimum_mean_improvement_m: float = 0.005,
    maximum_context_regression_m: float = 0.005,
    maximum_peak_velocity_ratio: float = 1.05,
    maximum_settling_regression_sec: float = 0.10,
) -> G1BallisticContactContextHoldoutEvaluation:
    """Reject a candidate that fails safety, quorum, worst-case, or CVaR gates."""

    if (
        len(anchor_evidence_paths) != len(candidate_evidence_paths)
        or len(anchor_evidence_paths) < 3
    ):
        raise ValueError("contact context evaluation requires three paired replays")
    if not 0.0 < minimum_mean_improvement_m <= 0.10:
        raise ValueError("contact context mean improvement threshold is invalid")
    if not 0.0 <= maximum_context_regression_m <= 0.05:
        raise ValueError("contact context regression tolerance is invalid")
    if not 1.0 <= maximum_peak_velocity_ratio <= 1.25:
        raise ValueError("contact context peak velocity ratio is invalid")
    if not 0.0 <= maximum_settling_regression_sec <= 0.50:
        raise ValueError("contact context settling tolerance is invalid")
    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("contact context evaluation must remain outside the checkout")
    if output.exists():
        raise FileExistsError("contact context evaluation output already exists")

    actor_file = actor_critic_path.expanduser().resolve(strict=True)
    actor = json.loads(actor_file.read_text(encoding="utf-8"))
    if not isinstance(actor, dict):
        raise ValueError("contact context actor-critic must be an object")
    actor_hash = str(actor.get("candidate_hash", ""))
    actor_payload = dict(actor)
    actor_payload.pop("candidate_hash", None)
    if actor_hash != canonical_hash(actor_payload):
        raise ValueError("contact context actor-critic hash mismatch")
    if (
        actor.get("schema_version") != _ACTOR_SCHEMA
        or actor.get("sim_replay_recommended") is not True
        or actor.get("activation_ceiling") != "SIM_ONLY"
        or actor.get("promotion_authorized") is not False
        or actor.get("hardware_authorized") is not False
        or actor.get("online_hot_swap_allowed") is not False
    ):
        raise ValueError("contact context actor-critic is not replay-authorized SIM_ONLY evidence")
    anchor_action = _action(actor.get("best_observed_action_rad"))
    candidate_action = _action(actor.get("proposed_action_rad"))

    cases: list[G1BallisticContactContextCase] = []
    family_hashes: set[str] = set()
    seeds: set[int] = set()
    for anchor_path, candidate_path in zip(
        anchor_evidence_paths, candidate_evidence_paths, strict=True
    ):
        anchor = _strict_evidence(anchor_path)
        candidate = _strict_evidence(candidate_path)
        if anchor["seed"] != candidate["seed"]:
            raise ValueError("contact context paired planner seed labels disagree")
        seed = int(anchor["seed"])
        if seed in seeds:
            raise ValueError("contact context planner seed labels must be distinct")
        seeds.add(seed)
        if not _actions_equal(anchor["action"], anchor_action):
            raise ValueError("contact context anchor action does not replay")
        if not _actions_equal(candidate["action"], candidate_action):
            raise ValueError("contact context candidate action does not replay")
        for evidence in (anchor, candidate):
            if evidence["body_hash"] != actor.get("body_hash"):
                raise ValueError("contact context Body hash mismatch")
            if evidence["implementation_hash"] != actor.get("implementation_hash"):
                raise ValueError("contact context implementation hash mismatch")
            family_hashes.add(str(evidence["context_family_hash"]))
        anchor_result = anchor["result"]
        candidate_result = candidate["result"]
        anchor_error = _metric(anchor_result, "goal_plane_target_error_m")
        candidate_error = _metric(candidate_result, "goal_plane_target_error_m")
        candidate_safe = _hard_safe(candidate_result)
        stability = bool(
            _metric(candidate_result, "post_contact_peak_joint_velocity_rms_rad_s")
            <= _metric(anchor_result, "post_contact_peak_joint_velocity_rms_rad_s")
            * maximum_peak_velocity_ratio
            and _metric(candidate_result, "post_contact_settling_time_sec")
            <= _metric(anchor_result, "post_contact_settling_time_sec")
            + maximum_settling_regression_sec
            and _integer_metric(candidate_result, "actuator_saturation_steps")
            <= _integer_metric(anchor_result, "actuator_saturation_steps")
        )
        cases.append(
            G1BallisticContactContextCase(
                planner_seed_audit_label=seed,
                anchor_evidence_hash=str(anchor["evidence_hash"]),
                candidate_evidence_hash=str(candidate["evidence_hash"]),
                anchor_error_m=anchor_error,
                candidate_error_m=candidate_error,
                error_improvement_m=anchor_error - candidate_error,
                candidate_hard_safe=candidate_safe,
                stability_preserved=stability,
            )
        )
    if len(family_hashes) != 1:
        raise ValueError("contact context replay families disagree")
    improvements = np.asarray([case.error_improvement_m for case in cases], dtype=np.float64)
    tail_count = max(1, int(math.ceil(0.25 * len(cases))))
    lower_tail = np.sort(improvements)[:tail_count]
    required_improved = int(math.ceil((2.0 / 3.0) * len(cases)))
    mean_improvement = float(np.mean(improvements))
    worst_improvement = float(np.min(improvements))
    cvar_improvement = float(np.mean(lower_tail))
    improved_count = int(np.count_nonzero(improvements >= minimum_mean_improvement_m))
    all_safe = all(case.candidate_hard_safe for case in cases)
    all_stable = all(case.stability_preserved for case in cases)
    accepted = bool(
        mean_improvement >= minimum_mean_improvement_m
        and worst_improvement >= -maximum_context_regression_m
        and cvar_improvement >= -maximum_context_regression_m
        and improved_count >= required_improved
        and all_safe
        and all_stable
    )
    evaluation = G1BallisticContactContextHoldoutEvaluation(
        actor_critic_hash=actor_hash,
        actor_critic_artifact_hash=_file_hash(actor_file),
        evaluation_implementation_hash=_file_hash(Path(__file__).resolve()),
        body_hash=str(actor["body_hash"]),
        implementation_hash=str(actor["implementation_hash"]),
        context_family_hash=next(iter(family_hashes)),
        anchor_action_rad=anchor_action,
        candidate_action_rad=candidate_action,
        cases=tuple(cases),
        mean_error_improvement_m=mean_improvement,
        worst_context_improvement_m=worst_improvement,
        lower_tail_cvar_improvement_m=cvar_improvement,
        minimum_mean_improvement_m=minimum_mean_improvement_m,
        maximum_context_regression_m=maximum_context_regression_m,
        improved_context_count=improved_count,
        required_improved_context_count=required_improved,
        all_contexts_hard_safe=all_safe,
        all_contexts_stability_preserved=all_stable,
        accepted=accepted,
        decision="ACCEPTED" if accepted else "REJECTED",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(evaluation.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evaluation


def _strict_evidence(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    evidence = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(evidence, dict) or evidence.get("strict_replay") is not True:
        raise ValueError("contact context evaluation requires strict replay evidence")
    trajectory = Path(str(evidence.get("trajectory_path", ""))).resolve()
    if not trajectory.is_file() or evidence.get("trajectory_hash") != _file_hash(trajectory):
        raise ValueError("contact context trajectory binding is invalid")
    flow = dict(evidence.get("flow_config", {}))
    action = _action(flow.pop("ballistic_contact_residual_rad", None))
    sonic = dict(evidence.get("sonic_runup_config", {}))
    seed = sonic.pop("planner_seed", None)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("contact context evidence requires a planner seed audit label")
    context_family_hash = canonical_hash(
        {
            "flow_config_without_actor_action": flow,
            "goal_spec": evidence.get("goal_spec"),
            "runup_config": evidence.get("runup_config"),
            "sonic_runup_config_without_seed": sonic,
            "approach_strike_candidate_hash": evidence.get("approach_strike_candidate_hash"),
            "football_motion_prior_hash": flow.get("football_motion_prior_hash"),
        }
    )
    result = evidence.get("result")
    if not isinstance(result, dict):
        raise ValueError("contact context result is missing")
    return {
        "body_hash": evidence.get("body_hash"),
        "implementation_hash": evidence.get("implementation_hash"),
        "context_family_hash": context_family_hash,
        "seed": seed,
        "action": action,
        "result": result,
        "evidence_hash": _file_hash(resolved),
    }


def _action(value: Any) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != 6:
        raise ValueError("contact context action is invalid")
    action = tuple(float(item) for item in value)
    G1BallisticContactResidualConfig(right_leg_residual_rad=action)
    return action


def _actions_equal(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
    return all(
        math.isclose(left_value, right_value, rel_tol=0.0, abs_tol=1e-9)
        for left_value, right_value in zip(left, right, strict=True)
    )


def _metric(result: dict[str, Any], key: str) -> float:
    value = result.get(key)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"contact context metric {key} is invalid")
    return float(value)


def _integer_metric(result: dict[str, Any], key: str) -> int:
    value = result.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"contact context metric {key} is invalid")
    return value


def _hard_safe(result: dict[str, Any]) -> bool:
    return bool(
        result.get("finite_state") is True
        and result.get("post_kick_fall") is False
        and result.get("joint_limit_violation") is False
        and result.get("torque_limit_violation") is False
        and result.get("perceptual_continuity_passed") is True
        and result.get("actuator_saturation") is False
    )


__all__ = [
    "G1BallisticContactContextCase",
    "G1BallisticContactContextHoldoutEvaluation",
    "evaluate_g1_ballistic_contact_context_holdout",
]
