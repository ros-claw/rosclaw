"""Strict replay gate for a ballistic contact actor-critic proposal."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rosclaw.feedback.contracts import canonical_hash
from rosclaw.growth.ballistic_contact_residual import G1BallisticContactResidualConfig

_ACTOR_SCHEMAS = {
    "rosclaw.growth.g1_ballistic_contact_actor_critic.v4",
    "rosclaw.growth.g1_ballistic_contact_coupled_actor_critic.v1",
}
_SCHEMA = "rosclaw.growth.g1_ballistic_contact_candidate_evaluation.v1"


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


@dataclass(frozen=True)
class G1BallisticContactCandidateEvaluation:
    actor_critic_hash: str
    experiment_context_hash: str
    anchor_evidence_hash: str
    candidate_evidence_hash: str
    anchor_action_rad: tuple[float, ...]
    candidate_action_rad: tuple[float, ...]
    anchor_error_m: float
    candidate_error_m: float
    measured_error_improvement_m: float
    minimum_error_improvement_m: float
    anchor_peak_joint_velocity_rms_rad_s: float
    candidate_peak_joint_velocity_rms_rad_s: float
    anchor_settling_time_sec: float
    candidate_settling_time_sec: float
    precision_improved: bool
    stability_preserved: bool
    hard_safe: bool
    accepted: bool
    decision: str
    activation_ceiling: str = "SIM_ONLY"
    promotion_authorized: bool = False
    hardware_authorized: bool = False
    online_hot_swap_allowed: bool = False
    schema_version: str = _SCHEMA

    def __post_init__(self) -> None:
        for value in (
            self.actor_critic_hash,
            self.experiment_context_hash,
            self.anchor_evidence_hash,
            self.candidate_evidence_hash,
        ):
            if not value.startswith("sha256:"):
                raise ValueError("contact candidate evaluation requires SHA-256 bindings")
        G1BallisticContactResidualConfig(right_leg_residual_rad=self.anchor_action_rad)
        G1BallisticContactResidualConfig(right_leg_residual_rad=self.candidate_action_rad)
        metrics = (
            self.anchor_error_m,
            self.candidate_error_m,
            self.measured_error_improvement_m,
            self.minimum_error_improvement_m,
            self.anchor_peak_joint_velocity_rms_rad_s,
            self.candidate_peak_joint_velocity_rms_rad_s,
            self.anchor_settling_time_sec,
            self.candidate_settling_time_sec,
        )
        if not all(math.isfinite(value) for value in metrics):
            raise ValueError("contact candidate evaluation metrics must be finite")
        if min(metrics[0], metrics[1], metrics[3], *metrics[4:]) < 0.0:
            raise ValueError("contact candidate evaluation metrics must be non-negative")
        expected = self.precision_improved and self.stability_preserved and self.hard_safe
        if self.accepted != expected:
            raise ValueError("contact candidate evaluation decision is inconsistent")
        if self.decision != ("ACCEPTED" if self.accepted else "REJECTED"):
            raise ValueError("contact candidate evaluation label is inconsistent")
        if (
            self.schema_version != _SCHEMA
            or self.activation_ceiling != "SIM_ONLY"
            or self.promotion_authorized
            or self.hardware_authorized
            or self.online_hot_swap_allowed
        ):
            raise ValueError("contact candidate evaluation cannot authorize activation")

    @property
    def evaluation_hash(self) -> str:
        return canonical_hash(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "evaluation_hash": self.evaluation_hash}


def evaluate_g1_ballistic_contact_candidate(
    *,
    actor_critic_path: Path,
    anchor_evidence_path: Path,
    candidate_evidence_path: Path,
    output_path: Path,
    source_checkout: Path,
    minimum_error_improvement_m: float = 0.005,
    maximum_peak_velocity_ratio: float = 1.05,
    maximum_settling_regression_sec: float = 0.10,
) -> G1BallisticContactCandidateEvaluation:
    """Accept only a measured, hard-safe improvement over the bound anchor."""

    if not 0.0 < minimum_error_improvement_m <= 0.10:
        raise ValueError("contact candidate minimum improvement is invalid")
    if not 1.0 <= maximum_peak_velocity_ratio <= 1.25:
        raise ValueError("contact candidate peak velocity ratio is invalid")
    if not 0.0 <= maximum_settling_regression_sec <= 0.50:
        raise ValueError("contact candidate settling tolerance is invalid")
    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("contact candidate evaluation must be outside the checkout")
    if output.exists():
        raise FileExistsError("contact candidate evaluation output already exists")

    actor = _hashed_object(actor_critic_path, hash_field="candidate_hash")
    actor_hash = str(actor.pop("candidate_hash"))
    if (
        actor.get("schema_version") not in _ACTOR_SCHEMAS
        or actor.get("sim_replay_recommended") is not True
        or actor.get("activation_ceiling") != "SIM_ONLY"
        or actor.get("promotion_authorized") is not False
        or actor.get("hardware_authorized") is not False
        or actor.get("direct_torque_output") is not False
        or actor.get("online_hot_swap_allowed") is not False
    ):
        raise ValueError("contact candidate source is not replay-authorized SIM_ONLY evidence")

    anchor = _strict_evidence(anchor_evidence_path)
    candidate = _strict_evidence(candidate_evidence_path)
    for evidence in (anchor, candidate):
        if evidence["body_hash"] != actor.get("body_hash"):
            raise ValueError("contact candidate Body hash mismatch")
        if evidence["implementation_hash"] != actor.get("implementation_hash"):
            raise ValueError("contact candidate implementation hash mismatch")
        if evidence["context_hash"] != actor.get("experiment_context_hash"):
            raise ValueError("contact candidate experiment context mismatch")

    anchor_action = _action(actor.get("best_observed_action_rad"))
    proposed_action = _action(actor.get("proposed_action_rad"))
    if not _actions_equal(anchor["action"], anchor_action):
        raise ValueError("contact candidate anchor action does not replay")
    if not _actions_equal(candidate["action"], proposed_action):
        raise ValueError("contact candidate proposed action does not replay")
    probes = actor.get("probes")
    if not isinstance(probes, list) or not any(
        isinstance(probe, dict)
        and probe.get("evidence_hash") == anchor["evidence_hash"]
        and probe.get("hard_safe") is True
        for probe in probes
    ):
        raise ValueError("contact candidate anchor is not source-bound hard-safe evidence")

    anchor_result = anchor["result"]
    candidate_result = candidate["result"]
    anchor_error = _metric(anchor_result, "goal_plane_target_error_m")
    candidate_error = _metric(candidate_result, "goal_plane_target_error_m")
    anchor_peak = _metric(anchor_result, "post_contact_peak_joint_velocity_rms_rad_s")
    candidate_peak = _metric(candidate_result, "post_contact_peak_joint_velocity_rms_rad_s")
    anchor_settling = _metric(anchor_result, "post_contact_settling_time_sec")
    candidate_settling = _metric(candidate_result, "post_contact_settling_time_sec")
    improvement = anchor_error - candidate_error
    precision_improved = improvement >= minimum_error_improvement_m
    stability_preserved = bool(
        candidate_peak <= anchor_peak * maximum_peak_velocity_ratio
        and candidate_settling <= anchor_settling + maximum_settling_regression_sec
        and _integer_metric(candidate_result, "actuator_saturation_steps")
        <= _integer_metric(anchor_result, "actuator_saturation_steps")
    )
    hard_safe = _hard_safe(candidate_result)
    accepted = precision_improved and stability_preserved and hard_safe
    report = G1BallisticContactCandidateEvaluation(
        actor_critic_hash=actor_hash,
        experiment_context_hash=str(actor.get("experiment_context_hash")),
        anchor_evidence_hash=str(anchor["evidence_hash"]),
        candidate_evidence_hash=str(candidate["evidence_hash"]),
        anchor_action_rad=anchor_action,
        candidate_action_rad=proposed_action,
        anchor_error_m=anchor_error,
        candidate_error_m=candidate_error,
        measured_error_improvement_m=improvement,
        minimum_error_improvement_m=minimum_error_improvement_m,
        anchor_peak_joint_velocity_rms_rad_s=anchor_peak,
        candidate_peak_joint_velocity_rms_rad_s=candidate_peak,
        anchor_settling_time_sec=anchor_settling,
        candidate_settling_time_sec=candidate_settling,
        precision_improved=precision_improved,
        stability_preserved=stability_preserved,
        hard_safe=hard_safe,
        accepted=accepted,
        decision="ACCEPTED" if accepted else "REJECTED",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _hashed_object(path: Path, *, hash_field: str) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve(strict=True).read_text())
    if not isinstance(value, dict):
        raise ValueError("contact candidate artifact must be an object")
    declared = str(value.get(hash_field, ""))
    payload = dict(value)
    payload.pop(hash_field, None)
    if declared != canonical_hash(payload):
        raise ValueError("contact candidate artifact hash mismatch")
    return value


def _strict_evidence(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    evidence = json.loads(resolved.read_text())
    if not isinstance(evidence, dict) or evidence.get("strict_replay") is not True:
        raise ValueError("contact candidate requires strict replay evidence")
    trajectory = Path(str(evidence.get("trajectory_path", ""))).resolve()
    if not trajectory.is_file() or evidence.get("trajectory_hash") != _file_hash(trajectory):
        raise ValueError("contact candidate trajectory binding is invalid")
    flow = dict(evidence.get("flow_config", {}))
    action = _action(flow.pop("ballistic_contact_residual_rad", None))
    context_hash = canonical_hash(
        {
            "flow_config_without_actor_action": flow,
            "goal_spec": evidence.get("goal_spec"),
            "runup_config": evidence.get("runup_config"),
            "sonic_runup_config": evidence.get("sonic_runup_config"),
            "approach_strike_candidate_hash": evidence.get("approach_strike_candidate_hash"),
            "football_motion_prior_hash": flow.get("football_motion_prior_hash"),
        }
    )
    result = evidence.get("result")
    if not isinstance(result, dict):
        raise ValueError("contact candidate result is missing")
    return {
        "body_hash": evidence.get("body_hash"),
        "implementation_hash": evidence.get("implementation_hash"),
        "context_hash": context_hash,
        "action": action,
        "result": result,
        "evidence_hash": _file_hash(resolved),
    }


def _action(value: Any) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != 6:
        raise ValueError("contact candidate action is invalid")
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
        raise ValueError(f"contact candidate metric {key} is invalid")
    return float(value)


def _integer_metric(result: dict[str, Any], key: str) -> int:
    value = result.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"contact candidate metric {key} is invalid")
    return value


def _hard_safe(result: dict[str, Any]) -> bool:
    return bool(
        result.get("finite_state") is True
        and result.get("post_kick_fall") is False
        and result.get("joint_limit_violation") is False
        and result.get("torque_limit_violation") is False
        and result.get("perceptual_continuity_passed") is True
    )


__all__ = [
    "G1BallisticContactCandidateEvaluation",
    "evaluate_g1_ballistic_contact_candidate",
]
