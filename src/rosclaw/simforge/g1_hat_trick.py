"""GoalForge Hat Trick: target, moving-ball, and disturbance-rescue evidence."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.feedback.profiles.g1 import build_g1_balance_runtime
from rosclaw.feedback.replay import RecordedLatencyClock
from rosclaw.feedback.runtime import FeedbackRuntime
from rosclaw.simforge.backends.unitree_mujoco_backend import (
    G1MuJoCoBackend,
    GoalForgeEpisode,
    trajectory_digest,
)
from rosclaw.simforge.g1_cerebellar_recovery import G1CerebellarRecoveryConfig
from rosclaw.simforge.g1_moving_ball import MovingBallInterceptAdapter
from rosclaw.simforge.g1_recovery_evolution import G1MomentumUnloadingEvolution
from rosclaw.simforge.g1_recovery_quality import (
    G1MomentumUnloadingComparison,
    G1NaturalnessComparison,
    G1RecoveryComparison,
    G1RecoveryQuality,
    compare_g1_momentum_unloading,
    compare_g1_naturalness,
    compare_g1_recovery,
    evaluate_g1_absolute_recovery_gate,
    measure_g1_recovery_quality,
)
from rosclaw.simforge.models import Partition
from rosclaw.simforge.seed_ledger import SeedLedger
from rosclaw.simforge.tasks.g1_goalforge.concepts import ShotParameters
from rosclaw.simforge.tasks.g1_goalforge.scenario import (
    GoalForgeScenario,
    generate_goalforge_scenarios,
)

_SECRET = b"rosclaw-phase7.1-goalforge-hat-trick-v1"
_CLOCK_CAPACITY = 5000
_LATENCY_NS = 100_000


@dataclass(frozen=True)
class HatTrickShot:
    name: str
    title: str
    capability: str
    scenario: dict[str, Any]
    parameters: dict[str, Any]
    result: dict[str, Any]
    trajectory_path: str
    trajectory_hash: str
    strict_replay: bool
    feedback_receipt: dict[str, Any] | None = None
    comparison_result: dict[str, Any] | None = None
    comparison_trajectory_path: str | None = None
    comparison_trajectory_hash: str | None = None
    planner_receipt: dict[str, Any] | None = None
    recovery_receipt: dict[str, Any] | None = None
    recovery_metrics: dict[str, Any] | None = None
    recovery_baseline_result: dict[str, Any] | None = None
    recovery_baseline_metrics: dict[str, Any] | None = None
    recovery_baseline_trajectory_path: str | None = None
    recovery_baseline_trajectory_hash: str | None = None
    recovery_baseline_strict_replay: bool = False
    recovery_comparison: dict[str, Any] | None = None
    absolute_recovery_gate: dict[str, Any] | None = None
    momentum_parent_result: dict[str, Any] | None = None
    momentum_parent_metrics: dict[str, Any] | None = None
    momentum_parent_trajectory_path: str | None = None
    momentum_parent_trajectory_hash: str | None = None
    momentum_parent_strict_replay: bool = False
    momentum_comparison: dict[str, Any] | None = None
    momentum_evolution: dict[str, Any] | None = None
    momentum_route_receipt: dict[str, Any] | None = None
    naturalness_parent_result: dict[str, Any] | None = None
    naturalness_parent_metrics: dict[str, Any] | None = None
    naturalness_parent_trajectory_path: str | None = None
    naturalness_parent_trajectory_hash: str | None = None
    naturalness_parent_strict_replay: bool = False
    naturalness_comparison: dict[str, Any] | None = None


@dataclass(frozen=True)
class GoalForgeHatTrick:
    body_hash: str
    kick_prior_hash: str
    backend_commit: str
    shots: tuple[HatTrickShot, ...]
    schema_version: str = "rosclaw.g1_goalforge.hat_trick.v6"

    @property
    def passed(self) -> bool:
        if len(self.shots) != 3:
            return False
        target, moving, rescue = self.shots
        return bool(
            all(shot.result["success"] and shot.strict_replay for shot in self.shots)
            and target.result["target_zone_hit"]
            and target.scenario["target_z_m"] >= 0.55
            and target.result["target_error_m"] <= 0.30
            and target.result["ball_speed_mps"] >= 6.0
            and moving.scenario["ball_launch_delay_sec"] > 0.0
            and moving.result["kick_foot_contacted"]
            and moving.result["ball_contact_time_sec"] is not None
            and rescue.scenario["disturbance_n"] >= 80.0
            and rescue.comparison_result is not None
            and not rescue.comparison_result["success"]
            and rescue.momentum_parent_strict_replay
            and rescue.momentum_comparison is not None
            and rescue.momentum_comparison["passed"]
            and rescue.momentum_evolution is not None
            and rescue.momentum_evolution["decision"] == "SIM_CHAMPION"
            and rescue.momentum_route_receipt is not None
            and rescue.momentum_route_receipt["used_candidate"]
            and rescue.naturalness_parent_strict_replay
            and rescue.naturalness_comparison is not None
            and rescue.naturalness_comparison["passed"]
            and rescue.recovery_receipt is not None
            and rescue.recovery_receipt["peak_settling_fraction"] == 1.0
            and rescue.momentum_evolution.get("recovery_controller_hash")
            == rescue.recovery_receipt.get("controller_hash")
            and rescue.momentum_evolution.get("recovery_config_hash")
            == rescue.recovery_receipt.get("config_hash")
            and rescue.momentum_route_receipt.get("recovery_controller_hash")
            == rescue.recovery_receipt.get("controller_hash")
            and rescue.momentum_route_receipt.get("recovery_config_hash")
            == rescue.recovery_receipt.get("config_hash")
            and all(
                shot.recovery_receipt is not None
                and shot.recovery_receipt["strict_replay"]
                and shot.recovery_receipt["regime_eligible"]
                and shot.recovery_receipt["contact_latched"]
                and shot.recovery_receipt["kick_foot_landing_latched"]
                and shot.recovery_baseline_strict_replay
                and shot.recovery_comparison is not None
                and shot.recovery_comparison["passed"]
                for shot in self.shots
            )
            and not any(
                shot.result[name]
                for shot in self.shots
                for name in (
                    "post_kick_fall",
                    "joint_limit_violation",
                    "torque_limit_violation",
                )
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "body_hash": self.body_hash,
            "kick_prior_hash": self.kick_prior_hash,
            "backend_commit": self.backend_commit,
            "shots": [asdict(shot) for shot in self.shots],
            "passed": self.passed,
            "visualization_may_consume_evidence": self.passed,
            "claims": {
                "evidence_domain": "SIM",
                "static_nine_grid_shot": self.shots[0].result["success"],
                "moving_ball_first_time_shot": self.shots[1].result["success"],
                "disturbance_feedback_rescue": self.shots[2].result["success"],
                "post_kick_cerebellar_recovery": all(
                    bool(shot.recovery_comparison and shot.recovery_comparison["passed"])
                    for shot in self.shots
                ),
                "absolute_recovery_growth_ready": all(
                    bool(shot.absolute_recovery_gate and shot.absolute_recovery_gate["passed"])
                    for shot in self.shots
                ),
                "elevated_target_challenge": bool(
                    self.shots[0].scenario["target_z_m"] >= 0.55
                    and self.shots[0].result["target_error_m"] <= 0.30
                ),
                "momentum_unloading_sim_champion": bool(
                    self.shots[2].momentum_comparison
                    and self.shots[2].momentum_comparison["passed"]
                ),
                "stability_plasticity_exact_regime_routing": bool(
                    self.shots[2].momentum_route_receipt
                    and self.shots[2].momentum_route_receipt["used_candidate"]
                ),
                "natural_upper_body_follow_through": bool(
                    self.shots[2].naturalness_comparison
                    and self.shots[2].naturalness_comparison["passed"]
                ),
                "two_stage_upright_body_recovery": bool(
                    self.shots[2].recovery_receipt
                    and self.shots[2].recovery_receipt["peak_settling_fraction"] == 1.0
                ),
                "backward_and_lateral_reversal_reduced": bool(
                    self.shots[2].naturalness_comparison
                    and self.shots[2].naturalness_comparison["backward_reversal_reduction"] >= 0.10
                    and self.shots[2].naturalness_comparison["lateral_peak_return_reduction"]
                    >= 0.15
                ),
                "candidate_v3_promoted": False,
                "magnus_curve_claimed": False,
                "real_hardware": False,
            },
        }


@dataclass(frozen=True)
class _RecoveryPair:
    candidate: GoalForgeEpisode
    baseline: GoalForgeEpisode
    candidate_strict_replay: bool
    baseline_strict_replay: bool
    feedback_receipt: dict[str, Any] | None
    recovery_receipt: dict[str, Any]
    candidate_metrics: G1RecoveryQuality
    baseline_metrics: G1RecoveryQuality
    comparison: G1RecoveryComparison


@dataclass(frozen=True)
class _MomentumPair:
    parent: GoalForgeEpisode
    parent_strict_replay: bool
    parent_metrics: G1RecoveryQuality
    comparison: G1MomentumUnloadingComparison
    evolution: G1MomentumUnloadingEvolution
    route_receipt: dict[str, Any]


@dataclass(frozen=True)
class _NaturalnessPair:
    parent: GoalForgeEpisode
    parent_strict_replay: bool
    parent_metrics: G1RecoveryQuality
    comparison: G1NaturalnessComparison


def run_goalforge_hat_trick(
    *,
    asset_root: Path,
    output_dir: Path,
    source_checkout: Path,
) -> GoalForgeHatTrick:
    """Execute and strictly replay all three SIM-only flagship shots."""

    root = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if root == checkout or checkout in root.parents:
        raise ValueError("Hat Trick evidence must be outside the source checkout")
    root.mkdir(parents=True, exist_ok=False)
    backend = G1MuJoCoBackend(asset_root=asset_root, trace_stride=1)
    qualification = backend.qualification
    base = _base_scenario()
    retained_recovery = G1CerebellarRecoveryConfig()
    natural_recovery = _natural_recovery_config()
    selection_seed = 202607289
    target_y, target_z = 0.0, 0.55
    target_scenario = replace(
        base,
        scenario_id="goalforge-hat-trick-nine-grid",
        generation=5,
        ball_y_m=0.10,
        target_y_m=target_y,
        target_z_m=target_z,
    )
    target_parameters = ShotParameters(
        stance_offset_y=0.12,
        pelvis_yaw_offset=-0.20,
        foot_yaw_offset=-0.06,
        swing_amplitude=0.90,
        contact_phase_offset=0.08,
        policy_type="parameter",
    )
    target = _run_recovery_pair(
        backend=backend,
        scenario=target_scenario,
        parameters=target_parameters,
        feedback_enabled=False,
        recovery_config=retained_recovery,
    )
    target_path = _save_trajectory(root / "shot-1-nine-grid.npz", target.candidate)
    target_baseline_path = _save_trajectory(
        root / "shot-1-recovery-off.npz",
        target.baseline,
    )

    moving_scenario = replace(
        base,
        scenario_id="goalforge-hat-trick-moving-ball",
        ball_x_m=1.12,
        ball_y_m=0.0,
        ball_velocity_x_mps=-0.08,
        ball_velocity_y_mps=0.0,
        ball_launch_delay_sec=4.0,
        ball_ground_friction=0.03,
        target_y_m=0.0,
        target_z_m=0.20,
    )
    moving_plan = MovingBallInterceptAdapter().plan(moving_scenario)
    if not moving_plan.eligible:
        raise RuntimeError("Hat Trick moving-ball scenario is outside the adapter envelope")
    moving_parameters = moving_plan.parameters
    moving = _run_recovery_pair(
        backend=backend,
        scenario=moving_scenario,
        parameters=moving_parameters,
        feedback_enabled=False,
        recovery_config=retained_recovery,
    )
    moving_path = _save_trajectory(root / "shot-2-moving-ball.npz", moving.candidate)
    moving_baseline_path = _save_trajectory(
        root / "shot-2-recovery-off.npz",
        moving.baseline,
    )

    rescue_scenario = replace(
        base,
        scenario_id="goalforge-hat-trick-80n-rescue",
        disturbance_n=80.0,
    )
    rescue_parent_parameters = ShotParameters()
    rescue_parameters = ShotParameters(
        stance_offset_y=-0.08,
        swing_amplitude=1.125,
        policy_type="parameter",
    )
    rescue_baseline = backend.run(rescue_scenario, rescue_parent_parameters)
    rescue = _run_recovery_pair(
        backend=backend,
        scenario=rescue_scenario,
        parameters=rescue_parameters,
        feedback_enabled=True,
        recovery_config=natural_recovery,
    )
    rescue_path = _save_trajectory(root / "shot-3-feedback-on.npz", rescue.candidate)
    rescue_off_path = _save_trajectory(root / "shot-3-feedback-off.npz", rescue_baseline)
    rescue_recovery_baseline_path = _save_trajectory(
        root / "shot-3-feedback-on-recovery-off.npz",
        rescue.baseline,
    )
    naturalness = _run_naturalness_comparison(
        backend=backend,
        scenario=rescue_scenario,
        parameters=rescue_parameters,
        candidate_pair=rescue,
    )
    rescue_naturalness_parent_path = _save_trajectory(
        root / "shot-3-naturalness-parent.npz",
        naturalness.parent,
    )
    momentum = _run_momentum_evolution(
        backend=backend,
        scenario=rescue_scenario,
        parent_parameters=rescue_parent_parameters,
        candidate_parameters=rescue_parameters,
        candidate_pair=rescue,
        recovery_config=retained_recovery,
    )
    rescue_momentum_parent_path = _save_trajectory(
        root / "shot-3-momentum-parent.npz",
        momentum.parent,
    )

    shots = (
        _shot(
            name="nine_grid_power",
            title="SHOT 1 · 3×3 TARGET POWER",
            capability="precision_and_power",
            scenario=target_scenario,
            parameters=target_parameters,
            episode=target.candidate,
            path=target_path,
            strict=target.candidate_strict_replay,
            recovery_pair=target,
            recovery_baseline_path=target_baseline_path,
            planner_receipt={
                "schema_version": "rosclaw.g1_goalforge.nine_grid_selection.v1",
                "selection_seed": selection_seed,
                "selection_seed_commitment": "sha256:"
                + hashlib.sha256(str(selection_seed).encode()).hexdigest(),
                "selected_target_y_m": target_y,
                "selected_target_z_m": target_z,
                "candidate_zones": 3,
                "challenge_row": "elevated",
                "minimum_target_z_m": 0.55,
                "maximum_target_error_m": 0.30,
            },
        ),
        _shot(
            name="moving_ball_first_time",
            title="SHOT 2 · MOVING BALL FIRST-TIME",
            capability="moving_ball_intercept_adapter",
            scenario=moving_scenario,
            parameters=moving_parameters,
            episode=moving.candidate,
            path=moving_path,
            strict=moving.candidate_strict_replay,
            recovery_pair=moving,
            recovery_baseline_path=moving_baseline_path,
            planner_receipt=moving_plan.to_dict(),
        ),
        _shot(
            name="disturbance_feedback_rescue",
            title="SHOT 3 · 80 N TWO-STAGE BODY RECOVERY",
            capability="balance_unloading_upright_settling_and_disturbance_recovery",
            scenario=rescue_scenario,
            parameters=rescue_parameters,
            episode=rescue.candidate,
            path=rescue_path,
            strict=rescue.candidate_strict_replay,
            feedback_receipt=rescue.feedback_receipt,
            comparison_result=rescue_baseline.result.summary_dict(),
            comparison_path=rescue_off_path,
            recovery_pair=rescue,
            recovery_baseline_path=rescue_recovery_baseline_path,
            momentum_pair=momentum,
            momentum_parent_path=rescue_momentum_parent_path,
            naturalness_pair=naturalness,
            naturalness_parent_path=rescue_naturalness_parent_path,
        ),
    )
    result = GoalForgeHatTrick(
        body_hash=qualification.body_hash,
        kick_prior_hash=qualification.kick_prior_hash,
        backend_commit=qualification.backend_commit,
        shots=shots,
    )
    _atomic_json(root / "goalforge-hat-trick.json", result.to_dict())
    return result


def _base_scenario() -> GoalForgeScenario:
    return generate_goalforge_scenarios(
        ledger=SeedLedger(task_id="g1_penalty_kick", secret=_SECRET),
        partition=Partition.VALIDATION,
        count=1,
        generation=0,
    )[0]


def _natural_recovery_config() -> G1CerebellarRecoveryConfig:
    """Two-stage unloading and upright settling bound to matched SIM evidence."""

    return G1CerebellarRecoveryConfig(
        start_policy_frame=300,
        blend_frames=100,
        standing_pose_blend=0.30,
        roll_posture_bias_rad=-0.05,
        settling_start_policy_frame=400,
        settling_blend_frames=100,
        settling_standing_pose_blend=0.45,
        settling_roll_posture_bias_rad=-0.02,
        settling_waist_pitch_bias_rad=0.09,
        target_smoothing_alpha=0.60,
        target_smoothing_start_policy_frame=300,
        target_smoothing_joint_group="upper_body",
    )


def _retained_natural_recovery_config() -> G1CerebellarRecoveryConfig:
    """Immediate Phase 7.1 parent used by the body-effect promotion gate."""

    return G1CerebellarRecoveryConfig(
        target_smoothing_alpha=0.70,
        target_smoothing_start_policy_frame=400,
        target_smoothing_joint_group="upper_body",
    )


def _run_recovery_pair(
    *,
    backend: G1MuJoCoBackend,
    scenario: GoalForgeScenario,
    parameters: ShotParameters,
    feedback_enabled: bool,
    recovery_config: G1CerebellarRecoveryConfig,
) -> _RecoveryPair:
    baseline_runtime = _balance_runtime(backend) if feedback_enabled else None
    baseline = backend.run(
        scenario,
        parameters,
        feedback_runtime=baseline_runtime,
    )
    baseline_replay_runtime = _balance_runtime(backend) if feedback_enabled else None
    baseline_replay = backend.run(
        scenario,
        parameters,
        feedback_runtime=baseline_replay_runtime,
    )
    baseline_strict = _episodes_match(baseline, baseline_replay)

    runtime = _balance_runtime(backend) if feedback_enabled else None
    recovery = backend.build_cerebellar_recovery_controller(scenario, recovery_config)
    candidate = backend.run(
        scenario,
        parameters,
        feedback_runtime=runtime,
        recovery_controller=recovery,
    )
    replay_runtime = _balance_runtime(backend) if feedback_enabled else None
    replay_recovery = backend.build_cerebellar_recovery_controller(
        scenario,
        recovery_config,
    )
    replay = backend.run(
        scenario,
        parameters,
        feedback_runtime=replay_runtime,
        recovery_controller=replay_recovery,
    )
    candidate_strict = _episodes_match(candidate, replay)
    candidate_metrics = measure_g1_recovery_quality(candidate.trajectory)
    baseline_metrics = measure_g1_recovery_quality(baseline.trajectory)
    comparison = compare_g1_recovery(
        baseline=baseline_metrics,
        candidate=candidate_metrics,
        baseline_result=baseline.result.summary_dict(),
        candidate_result=candidate.result.summary_dict(),
    )
    feedback_receipt = (
        runtime.build_receipt(
            action_id=scenario.scenario_id + ":balance-reflex",
            strict_replay=candidate_strict,
            evidence_domain="SIM",
        ).to_dict()
        if runtime is not None
        else None
    )
    recovery_receipt = recovery.build_receipt(
        strict_replay=candidate_strict,
        evidence_domain="SIM",
    ).to_dict()
    return _RecoveryPair(
        candidate=candidate,
        baseline=baseline,
        candidate_strict_replay=candidate_strict,
        baseline_strict_replay=baseline_strict,
        feedback_receipt=feedback_receipt,
        recovery_receipt=recovery_receipt,
        candidate_metrics=candidate_metrics,
        baseline_metrics=baseline_metrics,
        comparison=comparison,
    )


def _run_momentum_evolution(
    *,
    backend: G1MuJoCoBackend,
    scenario: GoalForgeScenario,
    parent_parameters: ShotParameters,
    candidate_parameters: ShotParameters,
    candidate_pair: _RecoveryPair,
    recovery_config: G1CerebellarRecoveryConfig,
) -> _MomentumPair:
    """Gate a shorter recovery action against the retained Phase 7.1 parent."""

    parent_runtime = _balance_runtime(backend)
    parent_recovery = backend.build_cerebellar_recovery_controller(
        scenario,
        recovery_config,
    )
    parent = backend.run(
        scenario,
        parent_parameters,
        feedback_runtime=parent_runtime,
        recovery_controller=parent_recovery,
    )
    replay_runtime = _balance_runtime(backend)
    replay_recovery = backend.build_cerebellar_recovery_controller(
        scenario,
        recovery_config,
    )
    parent_replay = backend.run(
        scenario,
        parent_parameters,
        feedback_runtime=replay_runtime,
        recovery_controller=replay_recovery,
    )
    parent_strict = _episodes_match(parent, parent_replay)
    parent_metrics = measure_g1_recovery_quality(parent.trajectory)
    comparison = compare_g1_momentum_unloading(
        parent=parent_metrics,
        candidate=candidate_pair.candidate_metrics,
        parent_result=parent.result.summary_dict(),
        candidate_result=candidate_pair.candidate.result.summary_dict(),
        parent_strict_replay=parent_strict,
        candidate_strict_replay=candidate_pair.candidate_strict_replay,
    )
    candidate_controller_hash = str(candidate_pair.recovery_receipt["controller_hash"])
    candidate_config_hash = str(candidate_pair.recovery_receipt["config_hash"])
    if candidate_controller_hash != parent_recovery.controller_hash:
        raise RuntimeError("momentum evolution recovery controller hash mismatch")
    if candidate_config_hash != parent_recovery.config_hash:
        raise RuntimeError("momentum evolution recovery config hash mismatch")
    evolution = G1MomentumUnloadingEvolution.evaluate(
        body_hash=backend.qualification.body_hash,
        kick_prior_hash=backend.qualification.kick_prior_hash,
        recovery_controller_hash=candidate_controller_hash,
        recovery_config_hash=candidate_config_hash,
        regime_commitment=scenario.scenario_commitment,
        parent=parent_parameters,
        candidate=candidate_parameters,
        parent_metrics=parent_metrics,
        candidate_metrics=candidate_pair.candidate_metrics,
        comparison=comparison,
    )
    selected, route_receipt = evolution.route(regime_commitment=scenario.scenario_commitment)
    if selected.policy_hash != candidate_parameters.policy_hash:
        raise RuntimeError("momentum-unloading candidate did not pass its promotion route")
    return _MomentumPair(
        parent=parent,
        parent_strict_replay=parent_strict,
        parent_metrics=parent_metrics,
        comparison=comparison,
        evolution=evolution,
        route_receipt=route_receipt.to_dict(),
    )


def _run_naturalness_comparison(
    *,
    backend: G1MuJoCoBackend,
    scenario: GoalForgeScenario,
    parameters: ShotParameters,
    candidate_pair: _RecoveryPair,
) -> _NaturalnessPair:
    """Gate two-stage body recovery against the retained natural action."""

    parent_config = _retained_natural_recovery_config()
    parent = backend.run(
        scenario,
        parameters,
        feedback_runtime=_balance_runtime(backend),
        recovery_controller=backend.build_cerebellar_recovery_controller(
            scenario,
            parent_config,
        ),
    )
    parent_replay = backend.run(
        scenario,
        parameters,
        feedback_runtime=_balance_runtime(backend),
        recovery_controller=backend.build_cerebellar_recovery_controller(
            scenario,
            parent_config,
        ),
    )
    parent_strict = _episodes_match(parent, parent_replay)
    parent_metrics = measure_g1_recovery_quality(parent.trajectory)
    comparison = compare_g1_naturalness(
        parent=parent_metrics,
        candidate=candidate_pair.candidate_metrics,
        parent_result=parent.result.summary_dict(),
        candidate_result=candidate_pair.candidate.result.summary_dict(),
        parent_strict_replay=parent_strict,
        candidate_strict_replay=candidate_pair.candidate_strict_replay,
    )
    if not comparison.passed:
        raise RuntimeError(
            "natural upper-body recovery did not pass its promotion gate: "
            + ", ".join(comparison.reasons)
        )
    return _NaturalnessPair(
        parent=parent,
        parent_strict_replay=parent_strict,
        parent_metrics=parent_metrics,
        comparison=comparison,
    )


def _balance_runtime(backend: G1MuJoCoBackend) -> FeedbackRuntime:
    return build_g1_balance_runtime(
        body_hash=backend.qualification.body_hash,
        compute_clock_ns=RecordedLatencyClock((_LATENCY_NS,) * _CLOCK_CAPACITY),
    )


def _episodes_match(left: GoalForgeEpisode, right: GoalForgeEpisode) -> bool:
    return bool(
        right.result.summary_dict() == left.result.summary_dict()
        and trajectory_digest(right.trajectory) == trajectory_digest(left.trajectory)
    )


def _save_trajectory(path: Path, episode: GoalForgeEpisode) -> Path:
    np.savez_compressed(path, **episode.trajectory)  # type: ignore[arg-type]
    return path


def _shot(
    *,
    name: str,
    title: str,
    capability: str,
    scenario: GoalForgeScenario,
    parameters: ShotParameters,
    episode: GoalForgeEpisode,
    path: Path,
    strict: bool,
    recovery_pair: _RecoveryPair,
    recovery_baseline_path: Path,
    feedback_receipt: dict[str, Any] | None = None,
    comparison_result: dict[str, Any] | None = None,
    comparison_path: Path | None = None,
    planner_receipt: dict[str, Any] | None = None,
    momentum_pair: _MomentumPair | None = None,
    momentum_parent_path: Path | None = None,
    naturalness_pair: _NaturalnessPair | None = None,
    naturalness_parent_path: Path | None = None,
) -> HatTrickShot:
    return HatTrickShot(
        name=name,
        title=title,
        capability=capability,
        scenario=scenario.to_private_dict(),
        parameters=parameters.to_dict(),
        result=episode.result.summary_dict(),
        trajectory_path=str(path),
        trajectory_hash=_file_hash(path),
        strict_replay=strict,
        feedback_receipt=feedback_receipt,
        comparison_result=comparison_result,
        comparison_trajectory_path=str(comparison_path) if comparison_path else None,
        comparison_trajectory_hash=_file_hash(comparison_path) if comparison_path else None,
        planner_receipt=planner_receipt,
        recovery_receipt=recovery_pair.recovery_receipt,
        recovery_metrics=recovery_pair.candidate_metrics.to_dict(),
        recovery_baseline_result=recovery_pair.baseline.result.summary_dict(),
        recovery_baseline_metrics=recovery_pair.baseline_metrics.to_dict(),
        recovery_baseline_trajectory_path=str(recovery_baseline_path),
        recovery_baseline_trajectory_hash=_file_hash(recovery_baseline_path),
        recovery_baseline_strict_replay=recovery_pair.baseline_strict_replay,
        recovery_comparison=recovery_pair.comparison.to_dict(),
        absolute_recovery_gate=evaluate_g1_absolute_recovery_gate(
            quality=recovery_pair.candidate_metrics,
            result=episode.result.summary_dict(),
            strict_replay=strict,
        ).to_dict(),
        momentum_parent_result=(
            momentum_pair.parent.result.summary_dict() if momentum_pair else None
        ),
        momentum_parent_metrics=(momentum_pair.parent_metrics.to_dict() if momentum_pair else None),
        momentum_parent_trajectory_path=(
            str(momentum_parent_path) if momentum_parent_path else None
        ),
        momentum_parent_trajectory_hash=(
            _file_hash(momentum_parent_path) if momentum_parent_path else None
        ),
        momentum_parent_strict_replay=bool(momentum_pair and momentum_pair.parent_strict_replay),
        momentum_comparison=(momentum_pair.comparison.to_dict() if momentum_pair else None),
        momentum_evolution=(momentum_pair.evolution.to_dict() if momentum_pair else None),
        momentum_route_receipt=(momentum_pair.route_receipt if momentum_pair else None),
        naturalness_parent_result=(
            naturalness_pair.parent.result.summary_dict() if naturalness_pair else None
        ),
        naturalness_parent_metrics=(
            naturalness_pair.parent_metrics.to_dict() if naturalness_pair else None
        ),
        naturalness_parent_trajectory_path=(
            str(naturalness_parent_path) if naturalness_parent_path else None
        ),
        naturalness_parent_trajectory_hash=(
            _file_hash(naturalness_parent_path) if naturalness_parent_path else None
        ),
        naturalness_parent_strict_replay=bool(
            naturalness_pair and naturalness_pair.parent_strict_replay
        ),
        naturalness_comparison=(
            naturalness_pair.comparison.to_dict() if naturalness_pair else None
        ),
    )


def _file_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


__all__ = ["GoalForgeHatTrick", "HatTrickShot", "run_goalforge_hat_trick"]
