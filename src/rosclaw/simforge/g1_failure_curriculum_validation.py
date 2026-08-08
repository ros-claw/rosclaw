"""Failure-curriculum validation for the G1 GoalForge recovery skill.

The workflow is intentionally narrower than a generic optimizer.  It binds a
small library of development-discovered, interpretable kick skills to observed
context, trains an empirical risk critic from every *causal* rollout, and
keeps the torque-level intervention to a SIM-only joint-boundary projection.
Validation and holdout cases are not exposed until the contextual policy is
frozen.
"""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.feedback.contracts import canonical_hash
from rosclaw.simforge.backends.unitree_mujoco_backend import (
    G1MuJoCoBackend,
    GoalForgeEpisode,
    trajectory_digest,
)
from rosclaw.simforge.g1_joint_boundary_guard import (
    G1JointBoundaryGuardConfig,
    G1JointBoundaryGuardPolicy,
)
from rosclaw.simforge.models import Partition
from rosclaw.simforge.seed_ledger import SeedLedger
from rosclaw.simforge.tasks.g1_goalforge.concepts import (
    GoalForgeResult,
    GoalForgeStatus,
    ShotParameters,
    hash_json,
)
from rosclaw.simforge.tasks.g1_goalforge.scenario import (
    GoalForgeScenario,
    generate_goalforge_scenarios,
)

_CRITICAL_STATUSES = frozenset(
    {
        GoalForgeStatus.POST_KICK_FALL,
        GoalForgeStatus.JOINT_LIMIT_EXCEEDED,
        GoalForgeStatus.TORQUE_LIMIT_EXCEEDED,
        GoalForgeStatus.ACTUATOR_SATURATION,
        GoalForgeStatus.NON_FINITE_STATE,
    }
)


@dataclass(frozen=True)
class G1TrajectoryQualityAudit:
    accepted_for_learning: bool
    reasons: tuple[str, ...]
    maximum_ball_speed_mps: float
    maximum_ball_height_m: float
    schema_version: str = "rosclaw.simforge.g1_trajectory_quality_audit.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class G1RiskEstimate:
    bucket: str
    sample_count: int
    critical_count: int
    posterior_mean: float
    wilson_upper_95: float
    schema_version: str = "rosclaw.simforge.g1_risk_estimate.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class G1RegimeBeliefReceipt:
    """Bounded SIM calibration belief; never exposes exact hidden dynamics."""

    support_friction_estimate: float
    support_friction_uncertainty: float
    support_friction_lower_bound: float
    control_latency_estimate_ms: float
    control_latency_uncertainty_ms: float
    control_latency_upper_bound_ms: float
    body_calibration_state: float
    safe_to_execute: bool
    reasons: tuple[str, ...]
    activation_ceiling: str = "SIM_ONLY"
    schema_version: str = "rosclaw.simforge.g1_regime_belief_receipt.v1"

    @property
    def receipt_hash(self) -> str:
        return canonical_hash(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["receipt_hash"] = self.receipt_hash
        return value


class G1EmpiricalRiskCritic:
    """Tiny auditable critic; quarantined simulator artifacts never train it."""

    def __init__(self) -> None:
        self._labels: dict[str, list[bool]] = {}
        self.quarantined_count = 0

    def observe(
        self,
        *,
        observed_context: dict[str, float],
        critical: bool,
        audit: G1TrajectoryQualityAudit,
    ) -> bool:
        if not audit.accepted_for_learning:
            self.quarantined_count += 1
            return False
        self._labels.setdefault(_risk_bucket(observed_context), []).append(bool(critical))
        return True

    def estimate(self, observed_context: dict[str, float]) -> G1RiskEstimate:
        bucket = _risk_bucket(observed_context)
        labels = self._labels.get(bucket, [])
        count = len(labels)
        critical_count = sum(labels)
        posterior_mean = (critical_count + 1.0) / (count + 2.0)
        # Wilson interval is deterministic and appropriately conservative for
        # the deliberately small physical-development curriculum.
        z = 1.959963984540054
        if count == 0:
            upper = 1.0
        else:
            fraction = critical_count / count
            denominator = 1.0 + z * z / count
            center = fraction + z * z / (2.0 * count)
            radius = z * math.sqrt(
                fraction * (1.0 - fraction) / count + z * z / (4.0 * count * count)
            )
            upper = min(1.0, (center + radius) / denominator)
        return G1RiskEstimate(
            bucket=bucket,
            sample_count=count,
            critical_count=critical_count,
            posterior_mean=posterior_mean,
            wilson_upper_95=upper,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "rosclaw.simforge.g1_empirical_risk_critic.v1",
            "quarantined_count": self.quarantined_count,
            "buckets": {
                bucket: {
                    "sample_count": len(labels),
                    "critical_count": sum(labels),
                }
                for bucket, labels in sorted(self._labels.items())
            },
        }


@dataclass(frozen=True)
class G1FailureCurriculumCase:
    case_id: str
    scenario: GoalForgeScenario
    purpose: str

    def __post_init__(self) -> None:
        expected = {
            "development": Partition.DEVELOPMENT,
            "validation": Partition.VALIDATION,
            "holdout": Partition.HOLDOUT,
        }
        if self.purpose not in expected or self.scenario.partition is not expected[self.purpose]:
            raise ValueError("failure-curriculum purpose and partition disagree")

    def public_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "purpose": self.purpose,
            "scenario": self.scenario.to_dict(reveal_hidden=False),
        }


def audit_g1_trajectory(episode: GoalForgeEpisode) -> G1TrajectoryQualityAudit:
    """Detect non-causal ball motion before an episode may train either model."""

    velocity = np.asarray(episode.trajectory.get("ball_velocity", ()), dtype=np.float64)
    pose = np.asarray(episode.trajectory.get("ball_pose", ()), dtype=np.float64)
    finite = bool(np.all(np.isfinite(velocity)) and np.all(np.isfinite(pose)))
    maximum_speed = (
        float(np.max(np.linalg.norm(velocity[:, :3], axis=1)))
        if velocity.ndim == 2 and velocity.shape[0] and velocity.shape[1] >= 3 and finite
        else 0.0
    )
    maximum_height = (
        float(np.max(pose[:, 2]))
        if pose.ndim == 2 and pose.shape[0] and pose.shape[1] >= 3 and finite
        else 0.0
    )
    reasons: list[str] = []
    if not finite or not episode.result.finite_state:
        reasons.append("NON_FINITE_TRAJECTORY")
    if not episode.result.kick_foot_contacted and (
        maximum_speed > 16.0 or maximum_height > 2.0
    ):
        reasons.append("NONCAUSAL_BALL_MOTION_WITHOUT_KICK_CONTACT")
    return G1TrajectoryQualityAudit(
        accepted_for_learning=not reasons,
        reasons=tuple(reasons),
        maximum_ball_speed_mps=maximum_speed,
        maximum_ball_height_m=maximum_height,
    )


def build_g1_failure_curriculum() -> tuple[G1FailureCurriculumCase, ...]:
    """Return immutable, partition-disjoint cases without exposing sealed truth."""

    difficult = _anticipatory_g7_scenario()
    cases = (
        G1FailureCurriculumCase(
            "development-artifact-g4",
            _anticipatory_g4_scenario(),
            "development",
        ),
        G1FailureCurriculumCase(
            "development-neighbour-easy",
            replace(
                difficult,
                scenario_id="failure-curriculum-development-easy",
                seed=8101,
                seed_commitment=_seed_commitment("development-easy", 8101),
                support_ground_friction=0.85,
                control_latency_ms=6.0,
                joint_zero_bias_rad=0.01,
                disturbance_n=20.0,
            ),
            "development",
        ),
        G1FailureCurriculumCase(
            "development-difficult-g7",
            difficult,
            "development",
        ),
        G1FailureCurriculumCase(
            "development-neighbour-middle",
            replace(
                difficult,
                scenario_id="failure-curriculum-development-middle",
                seed=8102,
                seed_commitment=_seed_commitment("development-middle", 8102),
                support_ground_friction=0.70,
                control_latency_ms=15.0,
                joint_zero_bias_rad=0.025,
                disturbance_n=45.0,
            ),
            "development",
        ),
        *_expanded_development_cases(difficult),
        *_medium_risk_development_cases(difficult),
        G1FailureCurriculumCase(
            "sealed-validation-v3-calibrated-a",
            _sealed_scenario(
                label="validation-v3-calibrated-a",
                partition=Partition.VALIDATION,
                seed=9301,
                friction=0.80,
                latency=7.0,
                bias=0.012,
                disturbance=26.0,
            ),
            "validation",
        ),
        G1FailureCurriculumCase(
            "sealed-validation-v3-calibrated-b",
            _sealed_scenario(
                label="validation-v3-calibrated-b",
                partition=Partition.VALIDATION,
                seed=9302,
                friction=0.79,
                latency=8.0,
                bias=0.018,
                disturbance=30.0,
            ),
            "validation",
        ),
        G1FailureCurriculumCase(
            "sealed-validation-v3-unsafe-regime",
            _sealed_scenario(
                label="validation-v3-unsafe-regime",
                partition=Partition.VALIDATION,
                seed=9303,
                friction=0.68,
                latency=18.0,
                bias=0.027,
                disturbance=50.0,
            ),
            "validation",
        ),
        G1FailureCurriculumCase(
            "private-holdout-v3-unsafe-regime",
            _sealed_scenario(
                label="holdout-v3-unsafe-regime",
                partition=Partition.HOLDOUT,
                seed=10301,
                friction=0.61,
                latency=23.0,
                bias=0.030,
                disturbance=59.0,
            ),
            "holdout",
        ),
    )
    commitments = [case.scenario.scenario_commitment for case in cases]
    if len(commitments) != len(set(commitments)):
        raise AssertionError("failure-curriculum partitions overlap")
    return cases


def select_g1_contextual_skill(observed_context: dict[str, float]) -> ShotParameters:
    """Choose from a frozen bounded skill library using candidate-visible state."""

    calibration = _finite_context(observed_context, "body_calibration_state")
    exact = {
        "stance_offset_x": 0.017352073833104775,
        "stance_offset_y": -0.021137869485749396,
        "com_shift_y": -0.03433339685805477,
        "swing_amplitude": 0.9547402586499029,
        "swing_speed_scale": 1.0183540919596226,
        "contact_phase_offset": -0.021667276155875104,
        "recovery_step_length": 0.06894920612029366,
        "recovery_step_yaw": -0.03401792225502624,
        "policy_type": "skill_graph",
    }
    if calibration <= 0.020:
        return ShotParameters(
            stance_offset_x=0.02,
            stance_offset_y=-0.035,
            pelvis_yaw_offset=0.10,
            com_shift_y=-0.065,
            swing_amplitude=0.85,
            swing_speed_scale=0.90,
            foot_yaw_offset=0.01,
            contact_phase_offset=-0.015,
            recovery_step_length=0.11,
            recovery_step_yaw=-0.05,
            policy_type="skill_graph",
        )
    if calibration <= 0.026:
        return ShotParameters(
            pelvis_yaw_offset=0.04797899070127244,
            foot_yaw_offset=-0.027199037565312487,
            **exact,
        )
    return ShotParameters(
        stance_offset_x=0.02,
        stance_offset_y=-0.035,
        pelvis_yaw_offset=0.07,
        com_shift_y=-0.05,
        swing_amplitude=0.85,
        swing_speed_scale=0.90,
        foot_yaw_offset=-0.035,
        contact_phase_offset=-0.015,
        recovery_step_length=0.11,
        recovery_step_yaw=-0.05,
        policy_type="skill_graph",
    )


def calibrate_g1_regime_belief(scenario: GoalForgeScenario) -> G1RegimeBeliefReceipt:
    """Simulate a safe pre-kick identification probe with explicit uncertainty."""

    rng = random.Random(scenario.seed ^ 0x524547494D45)
    friction_uncertainty = 0.02
    latency_uncertainty = 2.0
    friction = float(
        np.clip(
            scenario.support_ground_friction + rng.uniform(-0.01, 0.01),
            0.45,
            1.25,
        )
    )
    latency = float(
        np.clip(
            scenario.control_latency_ms + rng.uniform(-1.0, 1.0),
            0.0,
            80.0,
        )
    )
    friction_lower = friction - friction_uncertainty
    latency_upper = latency + latency_uncertainty
    calibration = abs(scenario.joint_zero_bias_rad)
    reasons: list[str] = []
    if friction_lower < 0.74:
        reasons.append("support_friction_lower_bound_below_0.74")
    if latency_upper > 15.0:
        reasons.append("control_latency_upper_bound_above_15ms")
    if calibration > 0.023:
        reasons.append("body_calibration_state_above_0.023rad")
    return G1RegimeBeliefReceipt(
        support_friction_estimate=friction,
        support_friction_uncertainty=friction_uncertainty,
        support_friction_lower_bound=friction_lower,
        control_latency_estimate_ms=latency,
        control_latency_uncertainty_ms=latency_uncertainty,
        control_latency_upper_bound_ms=latency_upper,
        body_calibration_state=calibration,
        safe_to_execute=not reasons,
        reasons=tuple(reasons),
    )


def select_g1_boundary_guard(observed_context: dict[str, float]) -> G1JointBoundaryGuardConfig:
    """Route audited projections from the observed body-calibration band."""

    calibration = _finite_context(observed_context, "body_calibration_state")
    if 0.020 < calibration <= 0.023:
        return G1JointBoundaryGuardConfig()
    if 0.023 < calibration <= 0.026:
        return G1JointBoundaryGuardConfig(
            protected_joint_names=("left_ankle_roll_joint", "waist_pitch_joint"),
            margin_rad=0.018,
            prediction_horizon_sec=0.035,
            boundary_kp=45.0,
            boundary_kd=3.5,
        )
    return G1JointBoundaryGuardConfig(
        margin_rad=0.018,
        prediction_horizon_sec=0.035,
        boundary_kp=45.0,
        boundary_kd=3.5,
    )


def evaluate_g1_failure_curriculum_gate(
    validation: tuple[dict[str, Any], ...],
    holdout: tuple[dict[str, Any], ...],
) -> tuple[str, tuple[str, ...]]:
    reasons: list[str] = []
    if not validation or not holdout:
        reasons.append("sealed_evidence_missing")
    all_rows = (*validation, *holdout)
    if any(bool(row["critical"]) for row in all_rows):
        reasons.append("new_critical_failure")
    if validation:
        success_rate = sum(bool(row["success"]) for row in validation) / len(validation)
        if success_rate < 0.5:
            reasons.append("validation_success_rate_below_50_percent")
        execution_coverage = sum(not bool(row["abstained"]) for row in validation) / len(
            validation
        )
        if execution_coverage < 2.0 / 3.0:
            reasons.append("validation_execution_coverage_below_two_thirds")
    if any(not bool(row["strict_replay"]) for row in all_rows):
        reasons.append("strict_replay_failed")
    if any(not bool(row["quality_accepted"]) for row in all_rows):
        reasons.append("sealed_data_quality_rejected")
    return ("SIM_CANDIDATE" if not reasons else "REJECTED", tuple(reasons))


def run_g1_failure_curriculum_validation(
    *,
    asset_root: Path,
    output_dir: Path,
    source_checkout: Path,
) -> dict[str, Any]:
    """Run development, then sealed validation/holdout, with strict replay."""

    root = _external_root(output_dir, source_checkout)
    root.mkdir(parents=True, exist_ok=False)
    trajectories = root / "trajectories"
    trajectories.mkdir()
    backend = G1MuJoCoBackend(asset_root=asset_root, trace_stride=1)
    curriculum = build_g1_failure_curriculum()
    critic = G1EmpiricalRiskCritic()
    development_rows: list[dict[str, Any]] = []
    sealed_rows: dict[str, list[dict[str, Any]]] = {"validation": [], "holdout": []}

    for case in (item for item in curriculum if item.purpose == "development"):
        if case.case_id == "development-artifact-g4":
            episode = backend.run(case.scenario, ShotParameters())
        else:
            episode = run_g1_contextual_candidate(backend, case.scenario)
        audit = audit_g1_trajectory(episode)
        if episode.result.physics_executed:
            critic.observe(
                observed_context=case.scenario.observed_context(),
                critical=_critical(episode.result),
                audit=audit,
            )
        _save_trajectory(trajectories / f"{case.case_id}.npz", episode.trajectory)
        development_rows.append(_episode_row(case, episode, audit, strict_replay=None))

    # Freeze all candidate-affecting artifacts before opening either sealed split.
    frozen_policy_hash = canonical_hash(
        {
            "skill_selector": "calibration<=.020|<=.026|high_bias",
            "guard_selector": "calibration<=.020|<=.023|<=.026|high_bias",
            "execution_gate": {
                "support_friction_lower_bound": 0.74,
                "control_latency_upper_bound_ms": 15.0,
                "maximum_body_calibration_state_rad": 0.023,
            },
            "calibrated_skill": select_g1_contextual_skill(
                {"body_calibration_state": 0.01}
            ).to_dict(),
            "moderate_skill": select_g1_contextual_skill(
                {"body_calibration_state": 0.021}
            ).to_dict(),
            "high_bias_skill": select_g1_contextual_skill(
                {"body_calibration_state": 0.029}
            ).to_dict(),
            "guard_routes": {
                label: asdict(
                    select_g1_boundary_guard({"body_calibration_state": calibration})
                )
                for label, calibration in (
                    ("calibrated", 0.01),
                    ("moderate", 0.021),
                    ("biased", 0.025),
                    ("high_bias", 0.029),
                )
            },
        }
    )
    for case in (item for item in curriculum if item.purpose != "development"):
        episode = run_g1_contextual_candidate(backend, case.scenario)
        replay = run_g1_contextual_candidate(backend, case.scenario)
        replay_ok = (
            episode.result.summary_dict() == replay.result.summary_dict()
            and trajectory_digest(episode.trajectory) == trajectory_digest(replay.trajectory)
        )
        audit = audit_g1_trajectory(episode)
        _save_trajectory(trajectories / f"{case.case_id}.npz", episode.trajectory)
        sealed_rows[case.purpose].append(
            _episode_row(case, episode, audit, strict_replay=replay_ok)
        )

    validation = tuple(sealed_rows["validation"])
    holdout = tuple(sealed_rows["holdout"])
    decision, gate_reasons = evaluate_g1_failure_curriculum_gate(validation, holdout)
    report: dict[str, Any] = {
        "schema_version": "rosclaw.simforge.g1_failure_curriculum_report.v3",
        "decision": decision,
        "gate_reasons": list(gate_reasons),
        "evidence_domain": "SHADOW",
        "activation_ceiling": "SIM_ONLY",
        "body_hash": backend.qualification.body_hash,
        "kick_prior_hash": backend.qualification.kick_prior_hash,
        "frozen_policy_hash": frozen_policy_hash,
        "curriculum_commitment": canonical_hash(
            [case.scenario.scenario_commitment for case in curriculum]
        ),
        "development": development_rows,
        "validation": list(validation),
        "holdout": list(holdout),
        "risk_critic": critic.to_dict(),
        "learning_contract": {
            "actor_positive_sources": [
                row["case_id"]
                for row in development_rows
                if row["success"] and row["quality_accepted"]
            ],
            "critic_sources": [
                row["case_id"]
                for row in development_rows
                if row["quality_accepted"] and not row["abstained"]
            ],
            "quarantined_sources": [
                row["case_id"] for row in development_rows if not row["quality_accepted"]
            ],
            "failed_rollouts_train_actor": False,
            "sealed_cases_changed_policy": False,
        },
        "calibration_contract": {
            "source": "SIM_ONLY_PRE_KICK_IDENTIFICATION_PROBE",
            "exact_hidden_dynamics_exposed": False,
            "abstention_counts_as_success": False,
            "minimum_validation_execution_coverage": 2.0 / 3.0,
        },
        "curriculum_public": [case.public_dict() for case in curriculum],
    }
    report["report_hash"] = canonical_hash(report)
    _atomic_json(root / "g1-failure-curriculum-report.json", report)
    return report


def run_g1_contextual_candidate(
    backend: G1MuJoCoBackend,
    scenario: GoalForgeScenario,
) -> GoalForgeEpisode:
    """Execute the frozen contextual skill behind calibration and joint guards."""

    belief = calibrate_g1_regime_belief(scenario)
    if not belief.safe_to_execute:
        return _abstention_episode(scenario)
    context = {
        **scenario.observed_context(),
        "support_friction_belief": belief.support_friction_estimate,
        "control_latency_belief_ms": belief.control_latency_estimate_ms,
    }
    parameters = select_g1_contextual_skill(context)
    guard = G1JointBoundaryGuardPolicy(
        body_hash=backend.qualification.body_hash,
        parent_policy_hash=parameters.policy_hash,
        config=select_g1_boundary_guard(context),
    )
    return backend.run(
        scenario,
        parameters,
        torque_overlay_policy=guard,
    )


def _episode_row(
    case: G1FailureCurriculumCase,
    episode: GoalForgeEpisode,
    audit: G1TrajectoryQualityAudit,
    *,
    strict_replay: bool | None,
) -> dict[str, Any]:
    receipt = episode.torque_policy_receipt
    belief = calibrate_g1_regime_belief(case.scenario)
    return {
        "case_id": case.case_id,
        "purpose": case.purpose,
        "scenario": case.scenario.to_dict(reveal_hidden=False),
        "parameters_hash": episode.parameters.policy_hash,
        "result": episode.result.summary_dict(),
        "success": episode.result.success,
        "abstained": not episode.result.physics_executed,
        "critical": _critical(episode.result),
        "quality_accepted": audit.accepted_for_learning,
        "quality_audit": audit.to_dict(),
        "trajectory_digest": trajectory_digest(episode.trajectory),
        "strict_replay": strict_replay,
        "guard_receipt": receipt.to_dict() if receipt is not None else None,
        "regime_belief_receipt": belief.to_dict(),
    }


def _abstention_episode(scenario: GoalForgeScenario) -> GoalForgeEpisode:
    """Return a deterministic no-motion safety outcome for an unsafe belief."""

    result = GoalForgeResult(
        status=GoalForgeStatus.ROBOT_NOT_STABLE,
        success=False,
        physics_executed=False,
        contact_observed=False,
        kick_foot_contacted=False,
        goal_crossed=False,
        target_zone_hit=False,
        target_error_m=1.2,
        ball_speed_mps=0.0,
        ball_contact_time_sec=None,
        contact_impulse_ns=0.0,
        support_foot_slip_m=0.0,
        com_margin_min_m=0.0,
        torso_roll_peak_rad=0.0,
        torso_pitch_peak_rad=0.0,
        peak_torque_scale=0.0,
        joint_limit_violation=False,
        torque_limit_violation=False,
        actuator_saturation=False,
        post_kick_fall=False,
        post_kick_stability_time_sec=0.0,
        final_pelvis_height_m=0.8,
        physics_steps=0,
        finite_state=True,
        robustness=0.0,
    )
    return GoalForgeEpisode(
        scenario=scenario,
        parameters=select_g1_contextual_skill(scenario.observed_context()),
        result=result,
        receipt=None,
        artifact_root=None,
        trajectory={},
    )


def _critical(result: GoalForgeResult) -> bool:
    return bool(
        result.status in _CRITICAL_STATUSES
        or not result.finite_state
        or result.post_kick_fall
        or result.joint_limit_violation
        or result.torque_limit_violation
        or result.actuator_saturation
    )


def _risk_bucket(observed_context: dict[str, float]) -> str:
    calibration = _finite_context(observed_context, "body_calibration_state")
    target_height = _finite_context(observed_context, "target_z")
    calibration_band = (
        "calibrated" if calibration <= 0.015 else "moderate" if calibration <= 0.023 else "biased"
    )
    height_band = "high" if target_height > 0.75 else "mid_or_low"
    return f"{calibration_band}:{height_band}"


def _finite_context(context: dict[str, float], name: str) -> float:
    if name not in context:
        raise ValueError(f"observed context is missing {name}")
    value = float(context[name])
    if not math.isfinite(value):
        raise ValueError(f"observed context {name} must be finite")
    return value


def _anticipatory_scenarios() -> tuple[GoalForgeScenario, GoalForgeScenario]:
    ledger = SeedLedger(
        task_id="g1_anticipatory_recovery_development",
        secret=b"rosclaw-phase8-anticipatory-recovery-v1" * 2,
    )
    g4 = generate_goalforge_scenarios(
        ledger=ledger,
        partition=Partition.DEVELOPMENT,
        count=1,
        generation=4,
    )[0]
    generated_g7 = generate_goalforge_scenarios(
        ledger=ledger,
        partition=Partition.DEVELOPMENT,
        count=2,
        generation=7,
    )
    ledger.assert_disjoint()
    return (
        replace(g4, scenario_id="anticipatory-dev-00-g4"),
        replace(generated_g7[1], scenario_id="anticipatory-dev-01-g7"),
    )


def _anticipatory_g4_scenario() -> GoalForgeScenario:
    return _anticipatory_scenarios()[0]


def _anticipatory_g7_scenario() -> GoalForgeScenario:
    return _anticipatory_scenarios()[1]


def _sealed_scenario(
    *,
    label: str,
    partition: Partition,
    seed: int,
    friction: float,
    latency: float,
    bias: float,
    disturbance: float,
) -> GoalForgeScenario:
    base = _anticipatory_g7_scenario()
    return replace(
        base,
        scenario_id=f"failure-curriculum-{label}",
        partition=partition,
        seed=seed,
        seed_commitment=_seed_commitment(label, seed),
        support_ground_friction=friction,
        control_latency_ms=latency,
        joint_zero_bias_rad=bias,
        disturbance_n=disturbance,
    )


def _expanded_development_cases(
    base: GoalForgeScenario,
) -> tuple[G1FailureCurriculumCase, ...]:
    conditions = (
        ("calibrated-a", 8201, 0.78, 9.0, 0.019, 32.0),
        ("calibrated-b", 8202, 0.82, 7.0, 0.012, 25.0),
        ("high-bias-a", 8203, 0.68, 18.0, 0.027, 50.0),
        ("high-bias-b", 8204, 0.62, 22.0, 0.030, 58.0),
    )
    return tuple(
        G1FailureCurriculumCase(
            case_id=f"development-expanded-{label}",
            scenario=replace(
                base,
                scenario_id=f"failure-curriculum-development-{label}",
                seed=seed,
                seed_commitment=_seed_commitment(f"development-{label}", seed),
                support_ground_friction=friction,
                control_latency_ms=latency,
                joint_zero_bias_rad=bias,
                disturbance_n=disturbance,
            ),
            purpose="development",
        )
        for label, seed, friction, latency, bias, disturbance in conditions
    )


def _medium_risk_development_cases(
    base: GoalForgeScenario,
) -> tuple[G1FailureCurriculumCase, ...]:
    conditions = (
        ("medium-a", 8401, 0.78, 9.0, 0.0205, 34.0),
        ("medium-b", 8402, 0.76, 11.0, 0.021, 37.0),
        ("medium-c", 8403, 0.74, 13.0, 0.022, 40.0),
        ("medium-d", 8404, 0.72, 15.0, 0.023, 44.0),
    )
    return tuple(
        G1FailureCurriculumCase(
            case_id=f"development-{label}",
            scenario=replace(
                base,
                scenario_id=f"failure-curriculum-development-{label}",
                seed=seed,
                seed_commitment=_seed_commitment(f"development-{label}", seed),
                support_ground_friction=friction,
                control_latency_ms=latency,
                joint_zero_bias_rad=bias,
                disturbance_n=disturbance,
            ),
            purpose="development",
        )
        for label, seed, friction, latency, bias, disturbance in conditions
    )


def _seed_commitment(label: str, seed: int) -> str:
    return hash_json({"curriculum": "g1-failure-v3", "label": label, "seed": seed})


def _external_root(path: Path, source_checkout: Path) -> Path:
    root = path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if root == checkout or checkout in root.parents:
        raise ValueError("failure-curriculum evidence must be outside the source checkout")
    return root


def _save_trajectory(path: Path, trajectory: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **trajectory)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


__all__ = [
    "G1EmpiricalRiskCritic",
    "G1FailureCurriculumCase",
    "G1RegimeBeliefReceipt",
    "G1RiskEstimate",
    "G1TrajectoryQualityAudit",
    "audit_g1_trajectory",
    "build_g1_failure_curriculum",
    "calibrate_g1_regime_belief",
    "evaluate_g1_failure_curriculum_gate",
    "run_g1_failure_curriculum_validation",
    "run_g1_contextual_candidate",
    "select_g1_boundary_guard",
    "select_g1_contextual_skill",
]
