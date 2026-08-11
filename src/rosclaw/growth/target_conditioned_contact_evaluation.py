"""Fail-closed evaluation for target-conditioned G1 contact actors."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rosclaw.feedback.contracts import canonical_hash
from rosclaw.growth.ballistic_contact_impulse_actor import (
    g1_ballistic_contact_impulse_context_hash,
    load_g1_ballistic_contact_impulse_actor,
)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _load_evidence(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    evidence = json.loads(resolved.read_text(encoding="utf-8"))
    if evidence.get("strict_replay") is not True:
        raise ValueError("target-conditioned evaluation requires strict replay evidence")
    trajectory = Path(str(evidence.get("trajectory_path", ""))).resolve()
    if not trajectory.is_file() or evidence.get("trajectory_hash") != _file_hash(trajectory):
        raise ValueError("target-conditioned evaluation trajectory binding is invalid")
    return evidence


def _context_hash(evidence: dict[str, Any]) -> str:
    return g1_ballistic_contact_impulse_context_hash(
        flow_config=dict(evidence.get("flow_config", {})),
        goal_spec=dict(evidence.get("goal_spec", {})),
        runup_config=dict(evidence.get("runup_config", {})),
        sonic_runup_config=(
            None
            if evidence.get("sonic_runup_config") is None
            else dict(evidence["sonic_runup_config"])
        ),
        approach_strike_candidate_hash=evidence.get("approach_strike_candidate_hash"),
        target_conditioned=True,
    )


def _hard_safe(result: dict[str, Any]) -> bool:
    return bool(
        result.get("kick_contact_observed") is True
        and result.get("goal_mouth_hit") is True
        and result.get("perceptual_continuity_passed") is True
        and result.get("post_kick_fall") is False
        and result.get("joint_limit_violation") is False
        and result.get("torque_limit_violation") is False
        and result.get("actuator_saturation") is False
        and result.get("torque_authority_projection_qualified") is True
        and float(result.get("contact_task_authority_scale_min", 0.0)) >= 0.95
        and float(result.get("post_contact_backward_displacement_m", math.inf)) <= 0.02
    )


@dataclass(frozen=True)
class G1TargetConditionedContactEvaluation:
    actor_hash: str
    actor_artifact_hash: str
    evaluation_implementation_hash: str
    body_hash: str
    implementation_hash: str
    baseline_evidence_hash: str
    candidate_evidence_hash: str
    stability_anchor_evidence_hash: str
    target_xyz_m: tuple[float, float, float]
    baseline_goal_plane_target_error_m: float
    candidate_goal_plane_target_error_m: float
    absolute_error_improvement_m: float
    relative_error_improvement: float
    precision_radius_m: float
    actor_active_frames: int
    actor_out_of_envelope_anchor_frames: int
    candidate_hard_safe: bool
    stability_anchor_preserved: bool
    precision_improved: bool
    development_breakthrough: bool
    verdict: str
    activation_ceiling: str = "SIM_ONLY"
    sealed_generalization_evidence: bool = False
    promotion_authorized: bool = False
    hardware_authorized: bool = False
    schema_version: str = "rosclaw.growth.g1_target_conditioned_contact_evaluation.v1"

    def __post_init__(self) -> None:
        hashes = (
            self.actor_hash,
            self.actor_artifact_hash,
            self.evaluation_implementation_hash,
            self.body_hash,
            self.implementation_hash,
            self.baseline_evidence_hash,
            self.candidate_evidence_hash,
            self.stability_anchor_evidence_hash,
        )
        if any(not value.startswith("sha256:") for value in hashes):
            raise ValueError("target-conditioned evaluation hashes must be SHA-256")
        values = (
            *self.target_xyz_m,
            self.baseline_goal_plane_target_error_m,
            self.candidate_goal_plane_target_error_m,
            self.absolute_error_improvement_m,
            self.relative_error_improvement,
            self.precision_radius_m,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("target-conditioned evaluation metrics must be finite")
        if self.verdict not in {"DEVELOPMENT", "REJECTED"}:
            raise ValueError("target-conditioned evaluation verdict is invalid")
        if self.development_breakthrough != (self.verdict == "DEVELOPMENT"):
            raise ValueError("target-conditioned evaluation verdict disagrees with gates")
        if (
            self.activation_ceiling != "SIM_ONLY"
            or self.sealed_generalization_evidence
            or self.promotion_authorized
            or self.hardware_authorized
        ):
            raise ValueError("target-conditioned evaluation must remain unpromoted SIM_ONLY")

    @property
    def evaluation_hash(self) -> str:
        return canonical_hash(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = {**asdict(self), "target_xyz_m": list(self.target_xyz_m)}
        if include_hash:
            value["evaluation_hash"] = self.evaluation_hash
        return value


def evaluate_g1_target_conditioned_contact_actor(
    *,
    actor_path: Path,
    baseline_evidence_path: Path,
    candidate_evidence_path: Path,
    stability_anchor_evidence_path: Path,
    output_path: Path,
    source_checkout: Path,
) -> G1TargetConditionedContactEvaluation:
    """Compare a candidate to its zero-force baseline and a fail-closed anchor."""

    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("target-conditioned evaluation must remain outside source checkout")
    if output.exists():
        raise FileExistsError("target-conditioned evaluation output already exists")
    actor_file = actor_path.expanduser().resolve()
    actor = load_g1_ballistic_contact_impulse_actor(actor_file)
    if actor.schema_version != "rosclaw.growth.g1_ballistic_contact_impulse_actor.v2":
        raise ValueError("target-conditioned evaluation requires a v2 actor")
    baseline_path = baseline_evidence_path.expanduser().resolve()
    candidate_path = candidate_evidence_path.expanduser().resolve()
    anchor_path = stability_anchor_evidence_path.expanduser().resolve()
    if len({baseline_path, candidate_path, anchor_path}) != 3:
        raise ValueError("target-conditioned evaluation evidence paths must be distinct")
    baseline = _load_evidence(baseline_path)
    candidate = _load_evidence(candidate_path)
    anchor = _load_evidence(anchor_path)
    for evidence in (baseline, candidate, anchor):
        if evidence.get("body_hash") != actor.body_hash:
            raise ValueError("target-conditioned evaluation Body hash mismatch")
        if evidence.get("implementation_hash") != actor.implementation_hash:
            raise ValueError("target-conditioned evaluation implementation hash mismatch")
        if _context_hash(evidence) != actor.experiment_context_hash:
            raise ValueError("target-conditioned evaluation context mismatch")
    if baseline.get("goal_spec") != candidate.get("goal_spec"):
        raise ValueError("target-conditioned baseline and candidate goals disagree")
    baseline_result = dict(baseline.get("result", {}))
    candidate_result = dict(candidate.get("result", {}))
    anchor_result = dict(anchor.get("result", {}))
    if baseline_result.get("ballistic_contact_impulse_actor_executed") is not False:
        raise ValueError("target-conditioned baseline is contaminated by an actor")
    if not _hard_safe(baseline_result):
        raise ValueError("target-conditioned baseline is not hard-safe")
    if (
        dict(candidate.get("flow_config", {})).get(
            "ballistic_contact_impulse_actor_hash"
        )
        != actor.actor_hash
        or candidate_result.get("ballistic_contact_impulse_actor_target_conditioned") is not True
        or int(candidate_result.get("ballistic_contact_impulse_actor_active_frames", 0)) < 1
    ):
        raise ValueError("target-conditioned candidate did not execute the bound actor")
    anchor_outside = int(
        anchor_result.get("ballistic_contact_impulse_actor_out_of_envelope_frames", 0)
    )
    anchor_preserved = bool(
        dict(anchor.get("flow_config", {})).get("ballistic_contact_impulse_actor_hash")
        == actor.actor_hash
        and anchor_result.get("ballistic_contact_impulse_actor_target_conditioned") is True
        and int(anchor_result.get("ballistic_contact_impulse_actor_active_frames", -1)) == 0
        and anchor_outside > 0
        and _hard_safe(anchor_result)
    )
    baseline_error = float(baseline_result.get("goal_plane_target_error_m", math.inf))
    candidate_error = float(candidate_result.get("goal_plane_target_error_m", math.inf))
    precision = float(candidate_result.get("precision_radius_m", 0.0))
    if not all(math.isfinite(value) for value in (baseline_error, candidate_error, precision)):
        raise ValueError("target-conditioned evaluation errors are invalid")
    absolute_improvement = baseline_error - candidate_error
    relative_improvement = absolute_improvement / max(baseline_error, 1e-12)
    candidate_safe = _hard_safe(candidate_result)
    precision_improved = bool(
        candidate_error <= precision and absolute_improvement >= 0.001
    )
    development = bool(candidate_safe and anchor_preserved and precision_improved)
    goal = dict(candidate["goal_spec"])
    evaluation = G1TargetConditionedContactEvaluation(
        actor_hash=actor.actor_hash,
        actor_artifact_hash=_file_hash(actor_file),
        evaluation_implementation_hash=_file_hash(Path(__file__).resolve()),
        body_hash=actor.body_hash,
        implementation_hash=actor.implementation_hash,
        baseline_evidence_hash=_file_hash(baseline_path),
        candidate_evidence_hash=_file_hash(candidate_path),
        stability_anchor_evidence_hash=_file_hash(anchor_path),
        target_xyz_m=(
            float(goal["plane_x_m"]),
            float(goal["target_y_m"]),
            float(goal["target_z_m"]),
        ),
        baseline_goal_plane_target_error_m=baseline_error,
        candidate_goal_plane_target_error_m=candidate_error,
        absolute_error_improvement_m=absolute_improvement,
        relative_error_improvement=relative_improvement,
        precision_radius_m=precision,
        actor_active_frames=int(
            candidate_result["ballistic_contact_impulse_actor_active_frames"]
        ),
        actor_out_of_envelope_anchor_frames=anchor_outside,
        candidate_hard_safe=candidate_safe,
        stability_anchor_preserved=anchor_preserved,
        precision_improved=precision_improved,
        development_breakthrough=development,
        verdict="DEVELOPMENT" if development else "REJECTED",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(evaluation.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evaluation


__all__ = [
    "G1TargetConditionedContactEvaluation",
    "evaluate_g1_target_conditioned_contact_actor",
]
