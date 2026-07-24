"""Fail-closed sixteen-gate promotion policy for G1 GoalForge."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from rosclaw.simforge.tasks.g1_goalforge.concepts import hash_json


class GateV4Decision(StrEnum):
    SIM_CHAMPION = "SIM_CHAMPION"
    REJECTED = "REJECTED"
    NEED_MORE_EVIDENCE = "NEED_MORE_EVIDENCE"


@dataclass(frozen=True)
class GoalForgeMetrics:
    episodes: int
    goal_success_rate: float
    target_zone_success_rate: float
    mean_target_error_m: float
    mean_ball_speed_mps: float
    mean_time_to_kick_sec: float
    first_attempt_success_rate: float
    retry_success_rate: float
    mean_retries: float
    fall_rate: float
    torque_violation_rate: float
    unsafe_allow_rate: float
    mean_support_slip_m: float
    mean_com_margin_m: float
    torso_roll_p95_rad: float
    torso_pitch_p95_rad: float
    joint_limit_violation_rate: float
    mean_post_kick_stability_sec: float

    def __post_init__(self) -> None:
        if self.episodes < 1:
            raise ValueError("GoalForge metrics require episodes")
        values = tuple(float(value) for key, value in asdict(self).items() if key != "episodes")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("GoalForge metrics must be finite")
        rates = (
            self.goal_success_rate,
            self.target_zone_success_rate,
            self.first_attempt_success_rate,
            self.retry_success_rate,
            self.fall_rate,
            self.torque_violation_rate,
            self.unsafe_allow_rate,
            self.joint_limit_violation_rate,
        )
        if any(not 0.0 <= value <= 1.0 for value in rates):
            raise ValueError("GoalForge metric rates must be in [0, 1]")


@dataclass(frozen=True)
class PromotionEvidenceV4:
    baseline: GoalForgeMetrics
    candidate_validation: GoalForgeMetrics
    hidden_holdout: GoalForgeMetrics
    candidate_hash: str
    body_hash: str
    expected_body_hash: str
    counterexample_regression_passed: bool
    dds_sim_to_sim_passed: bool
    strict_replay_rate: float
    module_levels: tuple[tuple[str, str], ...]
    canary_passed: bool
    shards_complete: bool
    holdout_signature_verified: bool
    evidence_commitment_verified: bool
    critical_safety_forgetting: int
    historical_success_delta: float
    schema_version: str = "rosclaw.g1_goalforge.promotion_evidence.v4"

    def __post_init__(self) -> None:
        hashes = (self.candidate_hash, self.body_hash, self.expected_body_hash)
        if any(not value.startswith("sha256:") for value in hashes):
            raise ValueError("Promotion V4 evidence hashes must be sha256 digests")
        if not 0.0 <= self.strict_replay_rate <= 1.0:
            raise ValueError("strict replay rate must be in [0, 1]")
        if not math.isfinite(self.historical_success_delta):
            raise ValueError("historical success delta must be finite")


@dataclass(frozen=True)
class GateCheckV4:
    gate: str
    passed: bool
    missing: bool
    detail: str


@dataclass(frozen=True)
class PromotionResultV4:
    decision: GateV4Decision
    checks: tuple[GateCheckV4, ...]
    evidence_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "evidence_hash": self.evidence_hash,
            "checks": [asdict(check) for check in self.checks],
        }


class PromotionGateV4:
    REQUIRED_E5 = (
        "body",
        "provider",
        "failure_router",
        "sandbox",
        "practice",
        "memory",
        "know",
        "how",
        "auto",
        "darwin",
        "registry",
        "rosclawd",
    )

    def evaluate(self, evidence: PromotionEvidenceV4) -> PromotionResultV4:
        base = evidence.baseline
        candidate = evidence.candidate_validation
        holdout = evidence.hidden_holdout
        module_levels = dict(evidence.module_levels)
        evidence_present = (
            evidence.shards_complete
            and evidence.holdout_signature_verified
            and evidence.evidence_commitment_verified
        )
        checks = (
            _check(
                "G1",
                candidate.goal_success_rate > base.goal_success_rate,
                False,
                "Goal success rate improves",
            ),
            _check(
                "G2",
                candidate.mean_target_error_m < base.mean_target_error_m,
                False,
                "target error decreases",
            ),
            _check(
                "G3",
                evidence_present
                and holdout.goal_success_rate > base.goal_success_rate
                and holdout.target_zone_success_rate > base.target_zone_success_rate
                and holdout.fall_rate == 0.0
                and holdout.torque_violation_rate == 0.0
                and holdout.joint_limit_violation_rate == 0.0
                and holdout.unsafe_allow_rate == 0.0,
                not evidence_present,
                "signed hidden holdout improves task metrics and has no safety violation",
            ),
            _check(
                "G4",
                candidate.first_attempt_success_rate > base.first_attempt_success_rate,
                False,
                "first-attempt success improves",
            ),
            _check(
                "G5",
                candidate.mean_retries < base.mean_retries,
                False,
                "mean retries decreases",
            ),
            _check("G6", candidate.fall_rate == 0.0, False, "fall rate equals zero"),
            _check(
                "G7",
                candidate.torque_violation_rate == 0.0
                and candidate.joint_limit_violation_rate == 0.0,
                False,
                "torque and joint-limit violations equal zero",
            ),
            _check(
                "G8",
                candidate.unsafe_allow_rate == 0.0,
                False,
                "unsafe allow equals zero",
            ),
            _check(
                "G9",
                candidate.mean_support_slip_m <= base.mean_support_slip_m + 1e-12,
                False,
                "support-foot slip does not regress",
            ),
            _check(
                "G10",
                candidate.mean_post_kick_stability_sec >= base.mean_post_kick_stability_sec - 1e-12,
                False,
                "post-kick stability does not regress",
            ),
            _check(
                "G11",
                evidence.counterexample_regression_passed
                and evidence.critical_safety_forgetting == 0
                and evidence.historical_success_delta >= -0.03,
                False,
                "counterexample and historical regression pass",
            ),
            _check(
                "G12",
                evidence.dds_sim_to_sim_passed,
                False,
                "isolated Unitree DDS sim-to-sim passes",
            ),
            _check(
                "G13",
                evidence.body_hash == evidence.expected_body_hash,
                False,
                "Body hash matches qualified G1",
            ),
            _check(
                "G14",
                evidence.strict_replay_rate == 1.0,
                False,
                "strict replay rate equals one",
            ),
            _check(
                "G15",
                all(module_levels.get(module) == "E5" for module in self.REQUIRED_E5),
                any(module not in module_levels for module in self.REQUIRED_E5),
                "all twelve required modules have replay-verified E5 proofs",
            ),
            _check("G16", evidence.canary_passed, False, "canary passes"),
        )
        if any(check.missing for check in checks):
            decision = GateV4Decision.NEED_MORE_EVIDENCE
        elif all(check.passed for check in checks):
            decision = GateV4Decision.SIM_CHAMPION
        else:
            decision = GateV4Decision.REJECTED
        return PromotionResultV4(
            decision=decision,
            checks=checks,
            evidence_hash=hash_json(_evidence_dict(evidence)),
        )


def _check(gate: str, passed: bool, missing: bool, detail: str) -> GateCheckV4:
    return GateCheckV4(gate=gate, passed=bool(passed), missing=bool(missing), detail=detail)


def _evidence_dict(evidence: PromotionEvidenceV4) -> dict[str, Any]:
    value = asdict(evidence)
    value["module_levels"] = dict(evidence.module_levels)
    return value


__all__ = [
    "GateCheckV4",
    "GateV4Decision",
    "GoalForgeMetrics",
    "PromotionEvidenceV4",
    "PromotionGateV4",
    "PromotionResultV4",
]
