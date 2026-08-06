"""Rollout learning for contextual G1 post-kick recovery primitives.

CPU MuJoCo remains the sole physics authority.  The learner evaluates a
bounded library of recovery primitives on DEVELOPMENT regimes, then freezes a
nearest-prototype selector before opening VALIDATION and private HOLDOUT
suites.  The exported actor can only select a validated target-space primitive
or route to the retained parent; it cannot emit torque or authorize hardware.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.simforge.backends.unitree_mujoco_backend import (
    GoalForgeEpisode,
    trajectory_digest,
)
from rosclaw.simforge.g1_cerebellar_recovery import G1CerebellarRecoveryConfig
from rosclaw.simforge.g1_contextual_recovery import (
    G1_CONTEXTUAL_RECOVERY_FEATURES,
    G1ContextualRecoveryArtifact,
    G1ContextualRecoveryPrimitive,
)
from rosclaw.simforge.g1_moving_ball import MovingBallInterceptAdapter
from rosclaw.simforge.g1_muscle_memory import G1_MUSCLE_MEMORY_OBSERVATIONS
from rosclaw.simforge.g1_muscle_memory_training import (
    G1MuscleMemoryCase,
    G1MuscleMemoryCaseResult,
    G1MuscleMemoryTrainer,
    _case_score,
    _normalization,
    g1_muscle_memory_parent_config,
)
from rosclaw.simforge.g1_recovery_quality import (
    G1RecoveryQuality,
    measure_g1_recovery_quality,
)
from rosclaw.simforge.g1_temporal_muscle_memory_training import (
    _build_temporal_holdout_cases,
    _case_commitment_hash,
    _moving_reductions,
    build_g1_temporal_muscle_memory_cases,
)
from rosclaw.simforge.tasks.g1_goalforge.concepts import hash_json

_POINT_ESTIMATE_GATE = 0.05
_COMPONENT_REGRESSION_LIMIT = 0.02
_BOOTSTRAP_DRAWS = 10_000


@dataclass(frozen=True)
class G1ContextualRecoveryPrimitiveTrial:
    case_name: str
    primitive_hash: str
    primitive: dict[str, Any]
    score: float
    backward_reduction: float
    tail_wobble_reduction: float
    leg_jerk_reduction: float
    composite_reduction: float
    safe: bool
    goal_preserved: bool
    naturalness_preserved: bool
    selected: bool
    schema_version: str = "rosclaw.g1_goalforge.contextual_primitive_trial.v1"


@dataclass(frozen=True)
class G1ContextualRecoveryValidationSummary:
    suite_hash: str
    case_count: int
    parent_valid_count: int
    contextual_route_count: int
    strict_replay_count: int
    passed_count: int
    mean_composite_reduction: float
    minimum_composite_reduction: float
    bootstrap_lower_95: float
    mean_backward_reduction: float
    mean_tail_wobble_reduction: float
    mean_leg_jerk_reduction: float
    qualified: bool
    development_search_excluded: bool = True
    case_rows_disclosed: bool = False
    evidence_domain: str = "SIM"
    physics_authority: str = "CPU_MUJOCO"
    schema_version: str = "rosclaw.g1_goalforge.contextual_validation.v2"


@dataclass(frozen=True)
class G1ContextualRecoveryHoldoutSummary:
    suite_hash: str
    case_count: int
    parent_valid_count: int
    fallback_route_count: int
    strict_replay_count: int
    exact_parent_replay_count: int
    passed_count: int
    qualified: bool
    development_search_excluded: bool = True
    case_rows_disclosed: bool = False
    evidence_domain: str = "SIM"
    physics_authority: str = "CPU_MUJOCO"
    schema_version: str = "rosclaw.g1_goalforge.contextual_holdout.v1"


@dataclass(frozen=True)
class G1TerminalDampingDevelopmentSummary:
    case_count: int
    passed_count: int
    mean_backward_reduction: float
    mean_tail_wobble_reduction: float
    minimum_tail_wobble_reduction: float
    mean_tail_joint_jerk_reduction: float
    minimum_tail_joint_jerk_reduction: float
    qualified: bool
    development_only: bool = True
    evidence_domain: str = "SIM"
    physics_authority: str = "CPU_MUJOCO"
    schema_version: str = "rosclaw.g1_goalforge.terminal_damping_development.v1"


@dataclass(frozen=True)
class G1ContextualRecoveryTrainingReport:
    artifact: G1ContextualRecoveryArtifact
    development_cases: tuple[G1MuscleMemoryCaseResult, ...]
    primitive_trials: tuple[G1ContextualRecoveryPrimitiveTrial, ...]
    retained_recovery_config_hash: str
    retained_recovery_config: dict[str, Any]
    fixed_structured_config_hash: str
    fixed_structured_config: dict[str, Any]
    training_rollout_count: int
    validation_rollout_count: int
    holdout_rollout_count: int
    learned_component_backward_reduction: float
    learned_component_tail_wobble_reduction: float
    learned_component_leg_jerk_reduction: float
    learned_component_composite_reduction: float
    learned_component_worst_case_composite: float
    learned_component_bootstrap_lower_95: float
    terminal_damping: G1TerminalDampingDevelopmentSummary
    validation: G1ContextualRecoveryValidationSummary
    holdout: G1ContextualRecoveryHoldoutSummary
    qualified: bool
    rejection_reasons: tuple[str, ...]
    activation_ceiling: str = "SIM_ONLY"
    evidence_domain: str = "SIM"
    physics_authority: str = "CPU_MUJOCO"
    hardware_command_sent: bool = False
    schema_version: str = "rosclaw.g1_goalforge.contextual_recovery_training.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact": {
                **self.artifact.to_dict(),
                "artifact_hash": self.artifact.artifact_hash,
            },
            "development_cases": [asdict(item) for item in self.development_cases],
            "primitive_trials": [asdict(item) for item in self.primitive_trials],
            "retained_recovery_config_hash": self.retained_recovery_config_hash,
            "retained_recovery_config": self.retained_recovery_config,
            "fixed_structured_config_hash": self.fixed_structured_config_hash,
            "fixed_structured_config": self.fixed_structured_config,
            "training_rollout_count": self.training_rollout_count,
            "validation_rollout_count": self.validation_rollout_count,
            "holdout_rollout_count": self.holdout_rollout_count,
            "learned_component_backward_reduction": (self.learned_component_backward_reduction),
            "learned_component_tail_wobble_reduction": (
                self.learned_component_tail_wobble_reduction
            ),
            "learned_component_leg_jerk_reduction": (self.learned_component_leg_jerk_reduction),
            "learned_component_composite_reduction": (self.learned_component_composite_reduction),
            "learned_component_worst_case_composite": (self.learned_component_worst_case_composite),
            "learned_component_bootstrap_lower_95": (self.learned_component_bootstrap_lower_95),
            "terminal_damping": asdict(self.terminal_damping),
            "validation": asdict(self.validation),
            "holdout": asdict(self.holdout),
            "qualified": self.qualified,
            "rejection_reasons": list(self.rejection_reasons),
            "qualification_thresholds": {
                "learned_component_point_estimate": _POINT_ESTIMATE_GATE,
                "component_regression_limit": _COMPONENT_REGRESSION_LIMIT,
                "bootstrap_lower_95": 0.0,
            },
            "activation_ceiling": self.activation_ceiling,
            "evidence_domain": self.evidence_domain,
            "physics_authority": self.physics_authority,
            "hardware_command_sent": self.hardware_command_sent,
        }


def build_g1_contextual_recovery_cases() -> tuple[G1MuscleMemoryCase, ...]:
    """Expand DEVELOPMENT around three regimes without opening validation."""

    static, nominal, lateral, light, disturbed = build_g1_temporal_muscle_memory_cases()
    specifications = (
        ("moving_ball_nominal_velocity_070", nominal, {"ball_velocity_x_mps": -0.070}),
        ("moving_ball_light_402g", light, {"ball_mass_kg": 0.402}),
    )
    adapter = MovingBallInterceptAdapter()
    variants: list[G1MuscleMemoryCase] = []
    for name, source, changes in specifications:
        scenario = replace(source.scenario, scenario_id=name, **changes)
        plan = adapter.plan(scenario)
        if not plan.eligible:
            raise RuntimeError(f"contextual development case is ineligible: {name}")
        variants.append(
            G1MuscleMemoryCase(name=name, scenario=scenario, parameters=plan.parameters)
        )
    return (
        static,
        nominal,
        lateral,
        light,
        variants[0],
        variants[1],
        disturbed,
    )


def _build_contextual_validation_cases() -> tuple[G1MuscleMemoryCase, ...]:
    """Predeclare v2 VALIDATION interpolation cases before policy search."""

    _, nominal, _lateral, light, _ = build_g1_temporal_muscle_memory_cases()
    specifications = (
        (
            "contextual_validation_nominal_velocity_0725",
            nominal,
            {"ball_velocity_x_mps": -0.0725},
        ),
        ("contextual_validation_light_4015g", light, {"ball_mass_kg": 0.4015}),
        ("contextual_validation_light_4038g", light, {"ball_mass_kg": 0.4038}),
    )
    adapter = MovingBallInterceptAdapter()
    cases: list[G1MuscleMemoryCase] = []
    for name, source, changes in specifications:
        scenario = replace(source.scenario, scenario_id=name, **changes)
        plan = adapter.plan(scenario)
        if not plan.eligible:
            raise RuntimeError(f"contextual validation case is ineligible: {name}")
        cases.append(G1MuscleMemoryCase(name=name, scenario=scenario, parameters=plan.parameters))
    return tuple(cases)


class G1ContextualRecoveryTrainer:
    """Learn a proprioceptive router over bounded recovery primitives."""

    def __init__(self, *, asset_root: Path, seed: int = 20260802) -> None:
        if seed < 0:
            raise ValueError("contextual recovery seed must be non-negative")
        self.seed = seed
        self.cases = build_g1_contextual_recovery_cases()
        self.retained_config = g1_muscle_memory_parent_config()
        self.fixed_config = replace(
            self.retained_config,
            settling_standing_pose_blend=0.42,
            settling_waist_pitch_bias_rad=0.11,
            target_smoothing_alpha=0.54,
            terminal_damping_start_policy_frame=500,
            terminal_damping_blend_frames=80,
            terminal_kp_scale=0.93,
            terminal_kd_scale=1.50,
            terminal_damping_joint_group="legs",
        )
        self.base = G1MuscleMemoryTrainer(
            asset_root=asset_root,
            cases=self.cases,
            recovery_config=self.retained_config,
        )

    def train(self) -> G1ContextualRecoveryTrainingReport:
        reasons: list[str] = []
        moving_indices = tuple(
            index for index, case in enumerate(self.cases) if case.name.startswith("moving_ball")
        )
        if len(moving_indices) < 5:
            raise RuntimeError("contextual recovery requires a multi-regime development suite")

        parents = tuple(self._run_config(case, self.retained_config) for case in self.cases)
        parent_quality = tuple(
            measure_g1_recovery_quality(episode.trajectory) for episode in parents
        )
        fixed = tuple(
            self._run_config(self.cases[index], self.fixed_config) for index in moving_indices
        )
        fixed_quality = tuple(measure_g1_recovery_quality(episode.trajectory) for episode in fixed)
        mean, scale = _normalization(parents)
        retained_hash = self._config_hash(self.retained_config)
        fixed_hash = self._config_hash(self.fixed_config)
        validation_cases = _build_contextual_validation_cases()
        validation_suite_hash = _case_commitment_hash(
            "rosclaw.g1_goalforge.contextual_validation.v4",
            validation_cases,
        )

        selected_primitives: list[G1ContextualRecoveryPrimitive] = []
        trial_rows: list[G1ContextualRecoveryPrimitiveTrial] = []
        trial_commitments: list[dict[str, Any]] = []
        library = _primitive_library()
        for local_index, case_index in enumerate(moving_indices):
            case = self.cases[case_index]
            baseline_quality = fixed_quality[local_index]
            retained_parent = parents[case_index]
            retained_parent_quality = parent_quality[case_index]
            scored: list[
                tuple[
                    float,
                    str,
                    G1ContextualRecoveryPrimitive,
                    GoalForgeEpisode,
                    G1RecoveryQuality,
                    float,
                    bool,
                    bool,
                    bool,
                    tuple[float, float, float],
                ]
            ] = []
            for primitive in library:
                config = _config_from_primitive(self.fixed_config, primitive)
                episode = self._run_config(case, config)
                quality = measure_g1_recovery_quality(episode.trajectory)
                score, safe, goal, natural = _case_score(
                    parent=retained_parent,
                    parent_quality=retained_parent_quality,
                    candidate=episode,
                    candidate_quality=quality,
                )
                reductions = _moving_reductions(baseline_quality, quality)
                composite = _composite(reductions)
                eligible = bool(
                    safe
                    and goal
                    and natural
                    and min(reductions) >= -_COMPONENT_REGRESSION_LIMIT
                    and composite >= 0.0
                )
                selection_score = composite + 0.02 * score if eligible else -1_000_000.0
                scored.append(
                    (
                        selection_score,
                        primitive.primitive_hash,
                        primitive,
                        episode,
                        quality,
                        score,
                        safe,
                        goal,
                        natural,
                        reductions,
                    )
                )
                trial_commitments.append(
                    {
                        "case_name": case.name,
                        "primitive_hash": primitive.primitive_hash,
                        "trajectory_hash": trajectory_digest(episode.trajectory),
                        "result": episode.result.summary_dict(),
                        "score": score,
                        "reductions": list(reductions),
                        "composite": composite,
                        "eligible": eligible,
                    }
                )
            scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
            if scored[0][0] <= -1_000_000.0:
                raise RuntimeError(f"no contextual primitive passed for {case.name}")
            selected = scored[0]
            selected_primitives.append(selected[2])
            for row in scored:
                reductions = row[9]
                trial_rows.append(
                    G1ContextualRecoveryPrimitiveTrial(
                        case_name=case.name,
                        primitive_hash=row[2].primitive_hash,
                        primitive=row[2].to_dict(),
                        score=row[5],
                        backward_reduction=reductions[0],
                        tail_wobble_reduction=reductions[1],
                        leg_jerk_reduction=reductions[2],
                        composite_reduction=_composite(reductions),
                        safe=row[6],
                        goal_preserved=row[7],
                        naturalness_preserved=row[8],
                        selected=row[2].primitive_hash == selected[2].primitive_hash,
                    )
                )

        dataset_hash = hash_json(
            {
                "schema_version": "rosclaw.g1_goalforge.contextual_recovery_dataset.v1",
                "development_cases": [
                    {
                        "name": case.name,
                        "scenario_commitment": case.scenario.scenario_commitment,
                        "policy_hash": case.parameters.policy_hash,
                        "retained_trajectory_hash": trajectory_digest(parent.trajectory),
                    }
                    for case, parent in zip(self.cases, parents, strict=True)
                ],
                "fixed_structured_config_hash": fixed_hash,
                "retained_recovery_config_hash": retained_hash,
                "primitive_library": [primitive.to_dict() for primitive in library],
                "trial_commitments": trial_commitments,
                "observation_mean": list(mean),
                "observation_scale": list(scale),
                "validation_suite_hash": validation_suite_hash,
                "validation_excluded": True,
                "private_holdout_excluded": True,
            }
        )
        artifact = G1ContextualRecoveryArtifact(
            body_hash=self.base.backend.qualification.body_hash,
            motion_hash=self.base.backend.qualification.motion_hash,
            baseline_recovery_config_hash=fixed_hash,
            fallback_recovery_config_hash=retained_hash,
            training_dataset_hash=dataset_hash,
            observation_mean=mean,
            observation_scale=scale,
            regime_feature_names=(self._feature_names()),
            regime_prototypes=tuple(
                _contextual_route_prototype(
                    parents[index],
                    observation_mean=mean,
                    observation_scale=scale,
                )
                for index in moving_indices
            ),
            primitives=tuple(selected_primitives),
            maximum_regime_distance=0.50,
            maximum_feature_z=8.0,
            training_episode_count=(len(parents) + len(fixed) + len(library) * len(moving_indices)),
            training_seed=self.seed,
        )

        terminal_damping = self._evaluate_terminal_damping(
            artifact=artifact,
            moving_indices=moving_indices,
        )
        if not terminal_damping.qualified:
            reasons.append("terminal_damping_development_failed")

        development_rows: list[G1MuscleMemoryCaseResult] = []
        learned_rows: list[tuple[float, float, float]] = []
        learned_composites: list[float] = []
        for index, (case, parent, parent_metric) in enumerate(
            zip(self.cases, parents, parent_quality, strict=True)
        ):
            candidate = self._run_contextual(case, artifact)
            replay = self._run_contextual(case, artifact)
            strict = _strict_replay(candidate, replay)
            candidate_metric = measure_g1_recovery_quality(candidate.trajectory)
            expected_route: int | None
            if index in moving_indices:
                local_index = moving_indices.index(index)
                causal_baseline_metric = fixed_quality[local_index]
                expected_route = local_index
                reductions = _moving_reductions(causal_baseline_metric, candidate_metric)
                learned_rows.append(reductions)
                learned_composites.append(_composite(reductions))
            comparison = parent
            comparison_metric = parent_metric
            if index not in moving_indices:
                expected_route = None
            score, safe, goal, natural = _case_score(
                parent=comparison,
                parent_quality=comparison_metric,
                candidate=candidate,
                candidate_quality=candidate_metric,
            )
            route = _contextual_route(candidate)
            if route != expected_route:
                reasons.append(case.name + ":contextual_route_mismatch")
            if expected_route is None and (
                trajectory_digest(parent.trajectory) != trajectory_digest(candidate.trajectory)
                or parent.result.summary_dict() != candidate.result.summary_dict()
            ):
                reasons.append(case.name + ":retained_parent_replay_mismatch")
            if not safe:
                reasons.append(case.name + ":safety_regressed")
            if not goal:
                reasons.append(case.name + ":goal_regressed")
            if not natural:
                reasons.append(case.name + ":naturalness_regressed")
            if not strict:
                reasons.append(case.name + ":strict_replay_failed")
            development_rows.append(
                G1MuscleMemoryCaseResult(
                    name=case.name,
                    parent_result=comparison.result.summary_dict(),
                    candidate_result=candidate.result.summary_dict(),
                    parent_metrics=comparison_metric.to_dict(),
                    candidate_metrics=candidate_metric.to_dict(),
                    score=score,
                    safe=safe,
                    goal_preserved=goal,
                    naturalness_preserved=natural,
                    strict_replay=strict,
                )
            )

        learned = np.asarray(learned_rows, dtype=np.float64)
        means = np.mean(learned, axis=0)
        mean_composite = float(np.mean(learned_composites))
        worst_composite = float(np.min(learned_composites))
        bootstrap_lower = _bootstrap_lower_95(
            tuple(learned_composites),
            seed=self.seed,
        )
        if mean_composite < _POINT_ESTIMATE_GATE:
            reasons.append("learned_component_point_estimate_below_5pct")
        if float(np.min(learned)) < -_COMPONENT_REGRESSION_LIMIT:
            reasons.append("learned_component_regression_limit_exceeded")
        if np.any(means < 0.0):
            reasons.append("learned_component_mean_regressed")
        if worst_composite < 0.0:
            reasons.append("learned_component_case_regressed")
        if bootstrap_lower <= 0.0:
            reasons.append("learned_component_bootstrap_lower_not_positive")

        validation, validation_rollouts = self._evaluate_validation(
            artifact,
            validation_cases,
            expected_suite_hash=validation_suite_hash,
        )
        if not validation.qualified:
            reasons.append("sealed_validation_failed")
        holdout, holdout_rollouts = self._evaluate_holdout(artifact)
        if not holdout.qualified:
            reasons.append("private_holdout_failed")

        return G1ContextualRecoveryTrainingReport(
            artifact=artifact,
            development_cases=tuple(development_rows),
            primitive_trials=tuple(trial_rows),
            retained_recovery_config_hash=retained_hash,
            retained_recovery_config=asdict(self.retained_config),
            fixed_structured_config_hash=fixed_hash,
            fixed_structured_config=asdict(self.fixed_config),
            training_rollout_count=(
                len(parents)
                + len(fixed)
                + len(library) * len(moving_indices)
                + 2 * len(moving_indices)
                + 2 * len(self.cases)
            ),
            validation_rollout_count=validation_rollouts,
            holdout_rollout_count=holdout_rollouts,
            learned_component_backward_reduction=float(means[0]),
            learned_component_tail_wobble_reduction=float(means[1]),
            learned_component_leg_jerk_reduction=float(means[2]),
            learned_component_composite_reduction=mean_composite,
            learned_component_worst_case_composite=worst_composite,
            learned_component_bootstrap_lower_95=bootstrap_lower,
            terminal_damping=terminal_damping,
            validation=validation,
            holdout=holdout,
            qualified=not reasons,
            rejection_reasons=tuple(reasons),
        )

    def _evaluate_validation(
        self,
        artifact: G1ContextualRecoveryArtifact,
        cases: tuple[G1MuscleMemoryCase, ...],
        *,
        expected_suite_hash: str,
    ) -> tuple[G1ContextualRecoveryValidationSummary, int]:
        rows: list[tuple[float, float, float]] = []
        composites: list[float] = []
        parent_valid_count = 0
        route_count = 0
        strict_count = 0
        passed_count = 0
        for case in cases:
            parent = self._run_config(case, self.retained_config)
            fixed = self._run_config(case, self.fixed_config)
            candidate = self._run_contextual(case, artifact)
            replay = self._run_contextual(case, artifact)
            parent_metric = measure_g1_recovery_quality(parent.trajectory)
            fixed_metric = measure_g1_recovery_quality(fixed.trajectory)
            candidate_metric = measure_g1_recovery_quality(candidate.trajectory)
            parent_valid = _parent_valid(parent, parent_metric)
            parent_valid_count += int(parent_valid)
            route = _contextual_route(candidate)
            route_count += int(route is not None)
            strict = _strict_replay(candidate, replay)
            strict_count += int(strict)
            score, safe, goal, natural = _case_score(
                parent=parent,
                parent_quality=parent_metric,
                candidate=candidate,
                candidate_quality=candidate_metric,
            )
            reductions = _moving_reductions(fixed_metric, candidate_metric)
            composite = _composite(reductions)
            rows.append(reductions)
            composites.append(composite)
            passed_count += int(
                parent_valid
                and route is not None
                and strict
                and safe
                and goal
                and natural
                and score >= -0.03
                and min(reductions) >= -_COMPONENT_REGRESSION_LIMIT
                and composite > 0.0
            )
        values = np.asarray(rows, dtype=np.float64)
        means = np.mean(values, axis=0)
        bootstrap_lower = _bootstrap_lower_95(tuple(composites), seed=self.seed + 1)
        suite_hash = _case_commitment_hash(
            "rosclaw.g1_goalforge.contextual_validation.v4",
            cases,
        )

        qualified = bool(
            suite_hash == expected_suite_hash
            and passed_count == len(cases)
            and parent_valid_count == len(cases)
            and route_count == len(cases)
            and strict_count == len(cases)
            and float(np.mean(composites)) >= _POINT_ESTIMATE_GATE
            and bootstrap_lower > 0.0
        )
        return (
            G1ContextualRecoveryValidationSummary(
                suite_hash=suite_hash,
                case_count=len(cases),
                parent_valid_count=parent_valid_count,
                contextual_route_count=route_count,
                strict_replay_count=strict_count,
                passed_count=passed_count,
                mean_composite_reduction=float(np.mean(composites)),
                minimum_composite_reduction=float(np.min(composites)),
                bootstrap_lower_95=bootstrap_lower,
                mean_backward_reduction=float(means[0]),
                mean_tail_wobble_reduction=float(means[1]),
                mean_leg_jerk_reduction=float(means[2]),
                qualified=qualified,
            ),
            4 * len(cases),
        )

    def _evaluate_terminal_damping(
        self,
        *,
        artifact: G1ContextualRecoveryArtifact,
        moving_indices: tuple[int, ...],
    ) -> G1TerminalDampingDevelopmentSummary:
        """Measure the third-stage damping against an exact parameter ablation."""

        rows: list[tuple[float, float, float]] = []
        passed = 0
        for local_index, case_index in enumerate(moving_indices):
            case = self.cases[case_index]
            damped_config = _config_from_primitive(
                self.fixed_config,
                artifact.primitives[local_index],
            )
            ablated_config = replace(
                damped_config,
                terminal_damping_start_policy_frame=None,
                terminal_kp_scale=1.0,
                terminal_kd_scale=1.0,
            )
            ablated = self._run_config(case, ablated_config)
            damped = self._run_config(case, damped_config)
            ablated_quality = measure_g1_recovery_quality(ablated.trajectory)
            damped_quality = measure_g1_recovery_quality(damped.trajectory)
            _, safe, goal, natural = _case_score(
                parent=ablated,
                parent_quality=ablated_quality,
                candidate=damped,
                candidate_quality=damped_quality,
            )
            reductions = _terminal_damping_reductions(ablated_quality, damped_quality)
            rows.append(reductions)
            passed += int(
                safe
                and goal
                and natural
                and reductions[0] >= 0.0
                and reductions[1] >= -0.05
                and reductions[2] >= 0.12
            )
        values = np.asarray(rows, dtype=np.float64)
        means = np.mean(values, axis=0)
        qualified = bool(
            passed == len(moving_indices)
            and means[0] >= 0.0
            and means[1] >= -0.02
            and means[2] >= 0.20
        )
        return G1TerminalDampingDevelopmentSummary(
            case_count=len(moving_indices),
            passed_count=passed,
            mean_backward_reduction=float(means[0]),
            mean_tail_wobble_reduction=float(means[1]),
            minimum_tail_wobble_reduction=float(np.min(values[:, 1])),
            mean_tail_joint_jerk_reduction=float(means[2]),
            minimum_tail_joint_jerk_reduction=float(np.min(values[:, 2])),
            qualified=qualified,
        )

    def _evaluate_holdout(
        self,
        artifact: G1ContextualRecoveryArtifact,
    ) -> tuple[G1ContextualRecoveryHoldoutSummary, int]:
        cases = _build_temporal_holdout_cases()
        parent_valid_count = 0
        fallback_count = 0
        strict_count = 0
        exact_count = 0
        passed_count = 0
        for case in cases:
            parent = self._run_config(case, self.retained_config)
            candidate = self._run_contextual(case, artifact)
            replay = self._run_contextual(case, artifact)
            parent_metric = measure_g1_recovery_quality(parent.trajectory)
            candidate_metric = measure_g1_recovery_quality(candidate.trajectory)
            parent_valid = _parent_valid(parent, parent_metric)
            parent_valid_count += int(parent_valid)
            fallback = _contextual_route(candidate) is None
            fallback_count += int(fallback)
            strict = _strict_replay(candidate, replay)
            strict_count += int(strict)
            exact = bool(
                parent.result.summary_dict() == candidate.result.summary_dict()
                and trajectory_digest(parent.trajectory) == trajectory_digest(candidate.trajectory)
            )
            exact_count += int(exact)
            _, safe, goal, natural = _case_score(
                parent=parent,
                parent_quality=parent_metric,
                candidate=candidate,
                candidate_quality=candidate_metric,
            )
            passed_count += int(
                parent_valid and fallback and strict and exact and safe and goal and natural
            )
        suite_hash = _case_commitment_hash(
            "rosclaw.g1_goalforge.contextual_private_holdout.v1",
            cases,
        )
        qualified = bool(
            passed_count == len(cases)
            and parent_valid_count == len(cases)
            and fallback_count == len(cases)
            and strict_count == len(cases)
            and exact_count == len(cases)
        )
        return (
            G1ContextualRecoveryHoldoutSummary(
                suite_hash=suite_hash,
                case_count=len(cases),
                parent_valid_count=parent_valid_count,
                fallback_route_count=fallback_count,
                strict_replay_count=strict_count,
                exact_parent_replay_count=exact_count,
                passed_count=passed_count,
                qualified=qualified,
            ),
            3 * len(cases),
        )

    def _run_config(
        self,
        case: G1MuscleMemoryCase,
        config: G1CerebellarRecoveryConfig,
    ) -> GoalForgeEpisode:
        controller = self.base.backend.build_cerebellar_recovery_controller(
            case.scenario,
            config,
        )
        return self.base.backend.run(
            case.scenario,
            case.parameters,
            feedback_runtime=self.base._feedback_runtime(case),
            recovery_controller=controller,
        )

    def _run_contextual(
        self,
        case: G1MuscleMemoryCase,
        artifact: G1ContextualRecoveryArtifact,
    ) -> GoalForgeEpisode:
        controller = self.base.backend.build_cerebellar_recovery_controller(
            case.scenario,
            self.fixed_config,
            contextual_recovery_artifact=artifact,
            fallback_config=self.retained_config,
        )
        return self.base.backend.run(
            case.scenario,
            case.parameters,
            feedback_runtime=self.base._feedback_runtime(case),
            recovery_controller=controller,
        )

    def _config_hash(self, config: G1CerebellarRecoveryConfig) -> str:
        return self.base.backend.build_cerebellar_recovery_controller(
            self.cases[0].scenario,
            config,
        ).config_hash

    @staticmethod
    def _feature_names() -> tuple[str, ...]:
        return G1_CONTEXTUAL_RECOVERY_FEATURES


def _primitive_library() -> tuple[G1ContextualRecoveryPrimitive, ...]:
    """Predeclared bounded library; only DEVELOPMENT rollouts may select it."""

    raw = (
        (300, 100, 400, 100, 0.42, 0.11, 0.54),
        (300, 100, 400, 100, 0.38, 0.08, 0.56),
        (300, 80, 380, 60, 0.42182, 0.12, 0.54294),
        (300, 100, 400, 100, 0.42, 0.12, 0.52),
        (300, 100, 400, 100, 0.40, 0.10, 0.52),
        (300, 100, 400, 100, 0.44, 0.12, 0.56),
    )
    return tuple(
        G1ContextualRecoveryPrimitive(
            start_policy_frame=start,
            blend_frames=blend,
            settling_start_policy_frame=settle,
            settling_blend_frames=settle_blend,
            settling_standing_pose_blend=standing,
            settling_waist_pitch_bias_rad=waist,
            target_smoothing_alpha=smoothing,
        )
        for start, blend, settle, settle_blend, standing, waist, smoothing in raw
    )


def _config_from_primitive(
    baseline: G1CerebellarRecoveryConfig,
    primitive: G1ContextualRecoveryPrimitive,
) -> G1CerebellarRecoveryConfig:
    return replace(
        baseline,
        start_policy_frame=primitive.start_policy_frame,
        blend_frames=primitive.blend_frames,
        settling_start_policy_frame=primitive.settling_start_policy_frame,
        settling_blend_frames=primitive.settling_blend_frames,
        settling_standing_pose_blend=primitive.settling_standing_pose_blend,
        settling_waist_pitch_bias_rad=primitive.settling_waist_pitch_bias_rad,
        target_smoothing_alpha=primitive.target_smoothing_alpha,
        target_smoothing_start_policy_frame=primitive.start_policy_frame,
    )


def _contextual_route_prototype(
    episode: GoalForgeEpisode,
    *,
    observation_mean: tuple[float, ...],
    observation_scale: tuple[float, ...],
) -> tuple[float, ...]:
    """Capture the post-impact state at the first causal recovery frame."""

    rows = np.asarray(episode.trajectory["recovery_proprioception"], dtype=np.float64)
    active = np.asarray(episode.trajectory["recovery_active"], dtype=np.bool_)
    impulse_index = G1_MUSCLE_MEMORY_OBSERVATIONS.index("contact_impulse_ns")
    # ``recovery_active`` already proves that the controller's contact and
    # kick-foot-landing latches opened. Requiring the *current* right-support
    # bit again can skip the exact frame selected online after the foot lifts,
    # producing a training/deployment prototype mismatch.
    route_rows = np.flatnonzero(active & (rows[:, impulse_index] > 0.0))
    if not len(route_rows):
        raise ValueError("development episode has no causal recovery route state")
    normalized = (
        rows[int(route_rows[0])] - np.asarray(observation_mean, dtype=np.float64)
    ) / np.asarray(observation_scale, dtype=np.float64)
    indices = [
        G1_MUSCLE_MEMORY_OBSERVATIONS.index(name) for name in G1_CONTEXTUAL_RECOVERY_FEATURES
    ]
    return tuple(map(float, normalized[indices]))


def _contextual_route(episode: GoalForgeEpisode) -> int | None:
    receipt = episode.recovery_receipt
    if receipt is None or receipt.contextual_recovery_receipt is None:
        return None
    value = receipt.contextual_recovery_receipt["selected_primitive_index"]
    return int(value) if value is not None else None


def _strict_replay(first: GoalForgeEpisode, second: GoalForgeEpisode) -> bool:
    return bool(
        first.result.summary_dict() == second.result.summary_dict()
        and trajectory_digest(first.trajectory) == trajectory_digest(second.trajectory)
    )


def _parent_valid(episode: GoalForgeEpisode, quality: G1RecoveryQuality) -> bool:
    result = episode.result
    return bool(
        result.success
        and result.contact_observed
        and result.goal_crossed
        and result.target_zone_hit
        and not result.post_kick_fall
        and not result.joint_limit_violation
        and not result.torque_limit_violation
        and not result.actuator_saturation
        and result.support_foot_slip_m <= 0.04
        and quality.terminal_bilateral_support
    )


def _composite(reductions: tuple[float, float, float]) -> float:
    return float(0.40 * reductions[0] + 0.40 * reductions[1] + 0.20 * reductions[2])


def _terminal_damping_reductions(
    ablated: G1RecoveryQuality,
    damped: G1RecoveryQuality,
) -> tuple[float, float, float]:
    """Return physical gains for backstep, whole-body wobble, and tail twitch."""

    def reduction(parent: float, candidate: float, floor: float) -> float:
        return (parent - candidate) / max(abs(parent), floor)

    return (
        reduction(
            ablated.post_contact_backward_reversal_m,
            damped.post_contact_backward_reversal_m,
            0.05,
        ),
        reduction(ablated.tail_wobble_index, damped.tail_wobble_index, 0.05),
        reduction(
            ablated.tail_joint_jerk_rms_rad_s3,
            damped.tail_joint_jerk_rms_rad_s3,
            0.10,
        ),
    )


def _bootstrap_lower_95(values: tuple[float, ...], *, seed: int) -> float:
    sample = np.asarray(values, dtype=np.float64)
    if sample.ndim != 1 or len(sample) < 2 or not np.all(np.isfinite(sample)):
        raise ValueError("bootstrap requires at least two finite case values")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(sample), size=(_BOOTSTRAP_DRAWS, len(sample)))
    means = np.mean(sample[indices], axis=1)
    return float(np.quantile(means, 0.025))


def write_g1_contextual_recovery_report(
    report: G1ContextualRecoveryTrainingReport,
    *,
    output_dir: Path,
    source_checkout: Path,
) -> tuple[Path, Path]:
    root = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if root == checkout or checkout in root.parents:
        raise ValueError("contextual recovery evidence must remain outside the checkout")
    root.mkdir(parents=True, exist_ok=False)
    artifact_path = root / "g1-contextual-recovery.json"
    report_path = root / "g1-contextual-recovery-training.json"
    _atomic_json(artifact_path, report.artifact.to_dict())
    _atomic_json(report_path, report.to_dict())
    return artifact_path, report_path


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    temp_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


__all__ = [
    "build_g1_contextual_recovery_cases",
    "G1ContextualRecoveryHoldoutSummary",
    "G1ContextualRecoveryPrimitiveTrial",
    "G1ContextualRecoveryTrainer",
    "G1ContextualRecoveryTrainingReport",
    "G1ContextualRecoveryValidationSummary",
    "G1TerminalDampingDevelopmentSummary",
    "write_g1_contextual_recovery_report",
]
