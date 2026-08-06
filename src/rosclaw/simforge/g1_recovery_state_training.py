"""Evidence-gated training for short-horizon G1 recovery-state memory.

The trainer expands the DEVELOPMENT curriculum around ball velocity, lateral
offset, and mass.  It searches only a fixed library of bounded target-space
recovery primitives.  Both successful and failed searches become temporal
prototypes: failed examples cast explicit abstention votes.  CPU MuJoCo is the
only physics authority; exported artifacts remain SIM_ONLY.
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
from rosclaw.simforge.g1_contextual_recovery import G1ContextualRecoveryPrimitive
from rosclaw.simforge.g1_contextual_recovery_training import (
    _bootstrap_lower_95,
    _composite,
    _parent_valid,
    _primitive_library,
    _strict_replay,
)
from rosclaw.simforge.g1_moving_ball import MovingBallInterceptAdapter
from rosclaw.simforge.g1_muscle_memory_training import (
    G1MuscleMemoryCase,
    G1MuscleMemoryTrainer,
    _case_score,
    g1_muscle_memory_parent_config,
)
from rosclaw.simforge.g1_recovery_quality import measure_g1_recovery_quality
from rosclaw.simforge.g1_recovery_state_memory import (
    G1_RECOVERY_STATE_FEATURES,
    G1_RECOVERY_STATE_OBSERVATIONS,
    G1RecoveryStateArtifact,
)
from rosclaw.simforge.g1_temporal_muscle_memory_training import (
    _build_temporal_holdout_cases,
    _case_commitment_hash,
    _moving_reductions,
    build_g1_temporal_muscle_memory_cases,
)
from rosclaw.simforge.tasks.g1_goalforge.concepts import hash_json

_WINDOW_FRAMES = 5
_NEIGHBOR_COUNT = 2
_MAXIMUM_NEIGHBOR_DISTANCE = 0.25
# A single contradictory neighbor proved insufficient under contact-dynamics
# interpolation (v6).  Require both local exemplars to carry positive evidence;
# plasticity resumes only after the curriculum supplies a coherent region.
_MINIMUM_CONSENSUS = 1.0
_ADVANTAGE_GATE = 0.02
_COMPONENT_REGRESSION_LIMIT = 0.02
_RESCUE_BACKWARD_LIMIT_M = 0.35
_RESCUE_TAIL_WOBBLE_LIMIT = 0.50
_RESCUE_LEG_JERK_LIMIT_RAD_S3 = 1_000.0
_RESCUE_SETTLING_LIMIT_SEC = 6.0


@dataclass(frozen=True)
class G1RecoveryStatePrototypeResult:
    case_name: str
    route_primitive_index: int
    selected_primitive_hash: str | None
    composite_advantage: float
    component_minimum: float
    safe: bool
    goal_preserved: bool
    naturalness_preserved: bool
    abstention_reason: str | None
    schema_version: str = "rosclaw.g1_goalforge.recovery_state_prototype_result.v1"


@dataclass(frozen=True)
class G1RecoveryStateSuiteSummary:
    suite_hash: str
    case_count: int
    parent_valid_count: int
    learned_route_count: int
    fallback_route_count: int
    strict_replay_count: int
    exact_parent_fallback_count: int
    passed_count: int
    mean_composite_reduction: float
    minimum_composite_reduction: float
    bootstrap_lower_95: float
    mean_backward_reduction: float
    mean_tail_wobble_reduction: float
    mean_leg_jerk_reduction: float
    qualified: bool
    case_rows_disclosed: bool = False
    evidence_domain: str = "SIM"
    physics_authority: str = "CPU_MUJOCO"
    schema_version: str = "rosclaw.g1_goalforge.recovery_state_suite.v1"


@dataclass(frozen=True)
class G1RecoveryStateTrainingReport:
    artifact: G1RecoveryStateArtifact
    prototype_results: tuple[G1RecoveryStatePrototypeResult, ...]
    development: G1RecoveryStateSuiteSummary
    validation: G1RecoveryStateSuiteSummary
    holdout: G1RecoveryStateSuiteSummary
    retained_recovery_config_hash: str
    retained_recovery_config: dict[str, Any]
    baseline_recovery_config_hash: str
    baseline_recovery_config: dict[str, Any]
    primitive_trial_count: int
    training_rollout_count: int
    validation_rollout_count: int
    holdout_rollout_count: int
    qualified: bool
    rejection_reasons: tuple[str, ...]
    activation_ceiling: str = "SIM_ONLY"
    evidence_domain: str = "SIM"
    physics_authority: str = "CPU_MUJOCO"
    hardware_command_sent: bool = False
    schema_version: str = "rosclaw.g1_goalforge.recovery_state_training.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact": {**self.artifact.to_dict(), "artifact_hash": self.artifact.artifact_hash},
            "prototype_results": [asdict(item) for item in self.prototype_results],
            "development": asdict(self.development),
            "validation": asdict(self.validation),
            "holdout": asdict(self.holdout),
            "retained_recovery_config_hash": self.retained_recovery_config_hash,
            "retained_recovery_config": self.retained_recovery_config,
            "baseline_recovery_config_hash": self.baseline_recovery_config_hash,
            "baseline_recovery_config": self.baseline_recovery_config,
            "primitive_trial_count": self.primitive_trial_count,
            "training_rollout_count": self.training_rollout_count,
            "validation_rollout_count": self.validation_rollout_count,
            "holdout_rollout_count": self.holdout_rollout_count,
            "qualified": self.qualified,
            "rejection_reasons": list(self.rejection_reasons),
            "qualification_thresholds": {
                "maximum_neighbor_distance": _MAXIMUM_NEIGHBOR_DISTANCE,
                "minimum_primitive_consensus": _MINIMUM_CONSENSUS,
                "minimum_composite_advantage": _ADVANTAGE_GATE,
                "component_regression_limit": _COMPONENT_REGRESSION_LIMIT,
            },
            "activation_ceiling": self.activation_ceiling,
            "evidence_domain": self.evidence_domain,
            "physics_authority": self.physics_authority,
            "hardware_command_sent": self.hardware_command_sent,
        }


def build_g1_recovery_state_cases() -> tuple[G1MuscleMemoryCase, ...]:
    """Predeclare a dense DEVELOPMENT neighborhood before primitive search."""

    static, nominal, lateral, light, disturbed = build_g1_temporal_muscle_memory_cases()
    specifications = (
        ("moving_ball_velocity_068", nominal, {"ball_velocity_x_mps": -0.068}),
        ("moving_ball_velocity_070", nominal, {"ball_velocity_x_mps": -0.070}),
        ("moving_ball_velocity_072", nominal, {"ball_velocity_x_mps": -0.072}),
        ("moving_ball_velocity_075", nominal, {"ball_velocity_x_mps": -0.075}),
        ("moving_ball_velocity_085", nominal, {"ball_velocity_x_mps": -0.085}),
        ("moving_ball_lateral_0025", lateral, {"ball_y_m": 0.0025}),
        ("moving_ball_lateral_004", lateral, {"ball_y_m": 0.004}),
        ("moving_ball_lateral_006", lateral, {"ball_y_m": 0.006}),
        ("moving_ball_lateral_0075", lateral, {"ball_y_m": 0.0075}),
        ("moving_ball_mass_395g", light, {"ball_mass_kg": 0.395}),
        ("moving_ball_mass_398g", light, {"ball_mass_kg": 0.398}),
        ("moving_ball_mass_4015g", light, {"ball_mass_kg": 0.4015}),
        ("moving_ball_mass_4038g", light, {"ball_mass_kg": 0.4038}),
        ("moving_ball_mass_405g", light, {"ball_mass_kg": 0.405}),
        (
            "moving_ball_velocity_070_r545",
            nominal,
            {"ball_velocity_x_mps": -0.070, "restitution": 0.545},
        ),
        (
            "moving_ball_velocity_070_r547",
            nominal,
            {"ball_velocity_x_mps": -0.070, "restitution": 0.547},
        ),
        (
            "moving_ball_velocity_070_r549",
            nominal,
            {"ball_velocity_x_mps": -0.070, "restitution": 0.549},
        ),
        (
            "moving_ball_velocity_070_r551",
            nominal,
            {"ball_velocity_x_mps": -0.070, "restitution": 0.551},
        ),
        (
            "moving_ball_velocity_070_r553",
            nominal,
            {"ball_velocity_x_mps": -0.070, "restitution": 0.553},
        ),
        (
            "moving_ball_velocity_070_r555",
            nominal,
            {"ball_velocity_x_mps": -0.070, "restitution": 0.555},
        ),
        (
            "moving_ball_velocity_070_r557",
            nominal,
            {"ball_velocity_x_mps": -0.070, "restitution": 0.557},
        ),
        (
            "moving_ball_velocity_070_r559",
            nominal,
            {"ball_velocity_x_mps": -0.070, "restitution": 0.559},
        ),
    )
    return (static, nominal, lateral, light, *_variants(specifications), disturbed)


def build_g1_recovery_state_validation_cases() -> tuple[G1MuscleMemoryCase, ...]:
    """Precommit v6 restitution-interpolation cases not consumed by search."""

    development = {case.name: case for case in build_g1_recovery_state_cases()}
    velocity_070 = development["moving_ball_velocity_070"]
    specifications = (
        ("state_v6_validation_velocity_070_r5455", velocity_070, {"restitution": 0.5455}),
        ("state_v6_validation_velocity_070_r5465", velocity_070, {"restitution": 0.5465}),
        ("state_v6_validation_velocity_070_r5485", velocity_070, {"restitution": 0.5485}),
        ("state_v6_validation_velocity_070_r5505", velocity_070, {"restitution": 0.5505}),
        ("state_v6_validation_velocity_070_r5525", velocity_070, {"restitution": 0.5525}),
        ("state_v6_validation_velocity_070_r5545", velocity_070, {"restitution": 0.5545}),
        ("state_v6_validation_velocity_070_r5565", velocity_070, {"restitution": 0.5565}),
        ("state_v6_validation_velocity_070_r5585", velocity_070, {"restitution": 0.5585}),
    )
    return _variants(specifications)


def _variants(
    specifications: tuple[tuple[str, G1MuscleMemoryCase, dict[str, float]], ...],
) -> tuple[G1MuscleMemoryCase, ...]:
    adapter = MovingBallInterceptAdapter()
    cases: list[G1MuscleMemoryCase] = []
    for name, source, changes in specifications:
        scenario = replace(source.scenario, scenario_id=name, **changes)
        plan = adapter.plan(scenario)
        if not plan.eligible:
            raise RuntimeError(f"recovery-state curriculum case is ineligible: {name}")
        cases.append(G1MuscleMemoryCase(name=name, scenario=scenario, parameters=plan.parameters))
    return tuple(cases)


class G1RecoveryStateTrainer:
    """Learn a conservative temporal evidence router over recovery primitives."""

    def __init__(self, *, asset_root: Path, seed: int = 20260803) -> None:
        if seed < 0:
            raise ValueError("recovery-state seed must be non-negative")
        self.seed = seed
        self.cases = build_g1_recovery_state_cases()
        self.validation_cases = build_g1_recovery_state_validation_cases()
        self.retained_config = g1_muscle_memory_parent_config()
        self.baseline_config = replace(
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

    def train(self) -> G1RecoveryStateTrainingReport:
        reasons: list[str] = []
        moving_indices = tuple(
            index for index, case in enumerate(self.cases) if case.name.startswith("moving_ball")
        )
        parents = tuple(self._run_config(case, self.retained_config) for case in self.cases)
        parent_metrics = tuple(measure_g1_recovery_quality(row.trajectory) for row in parents)
        baselines = tuple(self._run_config(case, self.baseline_config) for case in self.cases)
        baseline_metrics = tuple(measure_g1_recovery_quality(row.trajectory) for row in baselines)
        mean, scale = _recovery_state_normalization(parents)
        retained_hash = self._config_hash(self.retained_config)
        baseline_hash = self._config_hash(self.baseline_config)
        validation_suite_hash = _case_commitment_hash(
            "rosclaw.g1_goalforge.recovery_state_validation.v6",
            self.validation_cases,
        )
        # Deployment runs the retained parent while the temporal window is
        # pending.  Build prototypes from that exact causal rollout.
        descriptors = tuple(
            _recovery_state_descriptor(
                episode,
                observation_mean=mean,
                observation_scale=scale,
            )
            for episode in parents
        )

        library = _primitive_library()
        labels: list[int] = []
        advantages: list[float] = []
        component_minimums: list[float] = []
        prototype_results: list[G1RecoveryStatePrototypeResult] = []
        trial_commitments: list[dict[str, Any]] = []
        primitive_trials = 0
        for index, case in enumerate(self.cases):
            if index not in moving_indices:
                labels.append(-1)
                advantages.append(0.0)
                component_minimums.append(0.0)
                prototype_results.append(
                    G1RecoveryStatePrototypeResult(
                        case_name=case.name,
                        route_primitive_index=-1,
                        selected_primitive_hash=None,
                        composite_advantage=0.0,
                        component_minimum=0.0,
                        safe=True,
                        goal_preserved=True,
                        naturalness_preserved=True,
                        abstention_reason="non_moving_retained_parent",
                    )
                )
                continue
            scored: list[
                tuple[
                    float,
                    int,
                    G1ContextualRecoveryPrimitive,
                    tuple[float, float, float],
                    bool,
                    bool,
                    bool,
                ]
            ] = []
            for primitive_index, primitive in enumerate(library):
                # Evaluate the primitive through the same five-frame pending
                # window and causal state router used after export.  A forced
                # local evidence artifact removes only the final neighborhood
                # decision; it does not alter controller timing or physics.
                trial_artifact = self._trial_artifact(
                    descriptor=descriptors[index],
                    primitive_index=primitive_index,
                    primitives=library,
                    observation_mean=mean,
                    observation_scale=scale,
                    baseline_hash=baseline_hash,
                    retained_hash=retained_hash,
                )
                candidate = self._run_state(
                    case,
                    trial_artifact,
                )
                quality = measure_g1_recovery_quality(candidate.trajectory)
                score, safe, goal, natural = _case_score(
                    parent=parents[index],
                    parent_quality=parent_metrics[index],
                    candidate=candidate,
                    candidate_quality=quality,
                )
                reductions = _moving_reductions(baseline_metrics[index], quality)
                safety_rescue = bool(
                    not _parent_valid(baselines[index], baseline_metrics[index])
                    and safe
                    and goal
                )
                if safety_rescue:
                    # Relative reductions become misleading once the baseline
                    # has fallen: a prone robot can appear to have zero
                    # backstep.  Compare successful rescues to bounded absolute
                    # recovery limits instead of rewarding the failed trace.
                    reductions, natural = _absolute_recovery_margins(quality)
                composite = _composite(reductions)
                eligible = bool(
                    safe
                    and goal
                    and natural
                    and min(reductions) >= -_COMPONENT_REGRESSION_LIMIT
                    and composite >= _ADVANTAGE_GATE
                )
                selection_score = composite + 0.02 * score if eligible else -1_000_000.0
                scored.append(
                    (
                        selection_score,
                        primitive_index,
                        primitive,
                        reductions,
                        safe,
                        goal,
                        natural,
                    )
                )
                primitive_trials += 1
                trial_commitments.append(
                    {
                        "case_name": case.name,
                        "primitive_hash": primitive.primitive_hash,
                        "trajectory_hash": trajectory_digest(candidate.trajectory),
                        "result": candidate.result.summary_dict(),
                        "score": score,
                        "reductions": list(reductions),
                        "safety_rescue": safety_rescue,
                        "eligible": eligible,
                    }
                )
            scored.sort(key=lambda item: (item[0], item[2].primitive_hash), reverse=True)
            selected = scored[0]
            if selected[0] <= -1_000_000.0:
                best_observed = max(
                    scored, key=lambda item: (_composite(item[3]), item[2].primitive_hash)
                )
                labels.append(-1)
                advantages.append(_composite(best_observed[3]))
                component_minimums.append(min(best_observed[3]))
                prototype_results.append(
                    G1RecoveryStatePrototypeResult(
                        case_name=case.name,
                        route_primitive_index=-1,
                        selected_primitive_hash=None,
                        composite_advantage=_composite(best_observed[3]),
                        component_minimum=min(best_observed[3]),
                        safe=best_observed[4],
                        goal_preserved=best_observed[5],
                        naturalness_preserved=best_observed[6],
                        abstention_reason="no_primitive_passed_all_gates",
                    )
                )
            else:
                labels.append(selected[1])
                advantages.append(_composite(selected[3]))
                component_minimums.append(min(selected[3]))
                prototype_results.append(
                    G1RecoveryStatePrototypeResult(
                        case_name=case.name,
                        route_primitive_index=selected[1],
                        selected_primitive_hash=selected[2].primitive_hash,
                        composite_advantage=_composite(selected[3]),
                        component_minimum=min(selected[3]),
                        safe=selected[4],
                        goal_preserved=selected[5],
                        naturalness_preserved=selected[6],
                        abstention_reason=None,
                    )
                )

        dataset_hash = hash_json(
            {
                "schema_version": "rosclaw.g1_goalforge.recovery_state_dataset.v1",
                "development_cases": [
                    {
                        "name": case.name,
                        "scenario_commitment": case.scenario.scenario_commitment,
                        "policy_hash": case.parameters.policy_hash,
                        "retained_trajectory_hash": trajectory_digest(parent.trajectory),
                        "baseline_trajectory_hash": trajectory_digest(baseline.trajectory),
                    }
                    for case, parent, baseline in zip(
                        self.cases,
                        parents,
                        baselines,
                        strict=True,
                    )
                ],
                "primitive_library": [primitive.to_dict() for primitive in library],
                "trial_commitments": trial_commitments,
                "observation_mean": list(mean),
                "observation_scale": list(scale),
                "validation_suite_hash": validation_suite_hash,
                "validation_excluded": True,
                "private_holdout_excluded": True,
            }
        )
        artifact = G1RecoveryStateArtifact(
            body_hash=self.base.backend.qualification.body_hash,
            motion_hash=self.base.backend.qualification.motion_hash,
            baseline_recovery_config_hash=baseline_hash,
            fallback_recovery_config_hash=retained_hash,
            training_dataset_hash=dataset_hash,
            observation_mean=mean,
            observation_scale=scale,
            descriptor_feature_names=G1_RECOVERY_STATE_FEATURES,
            descriptor_prototypes=descriptors,
            prototype_primitive_indices=tuple(labels),
            prototype_composite_advantages=tuple(advantages),
            prototype_component_minimums=tuple(component_minimums),
            primitives=library,
            selection_window_frames=_WINDOW_FRAMES,
            neighbor_count=_NEIGHBOR_COUNT,
            maximum_neighbor_distance=_MAXIMUM_NEIGHBOR_DISTANCE,
            minimum_primitive_consensus=_MINIMUM_CONSENSUS,
            minimum_advantage_lower_bound=_ADVANTAGE_GATE,
            minimum_component_lower_bound=-_COMPONENT_REGRESSION_LIMIT,
            maximum_feature_z=8.0,
            training_episode_count=len(parents) + len(baselines) + primitive_trials,
            training_seed=self.seed,
        )

        development, development_rollouts = self._evaluate_suite(
            artifact=artifact,
            cases=self.cases,
            suite_hash=_case_commitment_hash(
                "rosclaw.g1_goalforge.recovery_state_development.v1",
                self.cases,
            ),
            require_all_learned=False,
            require_all_fallback=False,
        )
        if not development.qualified:
            reasons.append("development_replay_failed")
        validation, validation_rollouts = self._evaluate_suite(
            artifact=artifact,
            cases=self.validation_cases,
            suite_hash=validation_suite_hash,
            require_all_learned=True,
            require_all_fallback=False,
        )
        if not validation.qualified:
            reasons.append("sealed_local_physics_validation_failed")
        holdout_cases = _build_temporal_holdout_cases()
        holdout, holdout_rollouts = self._evaluate_suite(
            artifact=artifact,
            cases=holdout_cases,
            suite_hash=_case_commitment_hash(
                "rosclaw.g1_goalforge.recovery_state_private_holdout.v1",
                holdout_cases,
            ),
            require_all_learned=False,
            require_all_fallback=True,
        )
        if not holdout.qualified:
            reasons.append("private_holdout_failed")
        return G1RecoveryStateTrainingReport(
            artifact=artifact,
            prototype_results=tuple(prototype_results),
            development=development,
            validation=validation,
            holdout=holdout,
            retained_recovery_config_hash=retained_hash,
            retained_recovery_config=asdict(self.retained_config),
            baseline_recovery_config_hash=baseline_hash,
            baseline_recovery_config=asdict(self.baseline_config),
            primitive_trial_count=primitive_trials,
            training_rollout_count=(
                len(parents) + len(baselines) + primitive_trials + development_rollouts
            ),
            validation_rollout_count=validation_rollouts,
            holdout_rollout_count=holdout_rollouts,
            qualified=not reasons,
            rejection_reasons=tuple(reasons),
        )

    def _evaluate_suite(
        self,
        *,
        artifact: G1RecoveryStateArtifact,
        cases: tuple[G1MuscleMemoryCase, ...],
        suite_hash: str,
        require_all_learned: bool,
        require_all_fallback: bool,
    ) -> tuple[G1RecoveryStateSuiteSummary, int]:
        parent_valid_count = 0
        route_count = 0
        fallback_count = 0
        strict_count = 0
        exact_fallback_count = 0
        passed_count = 0
        routed_composites: list[float] = []
        routed_reductions: list[tuple[float, float, float]] = []
        for case in cases:
            parent = self._run_config(case, self.retained_config)
            baseline = self._run_config(case, self.baseline_config)
            candidate = self._run_state(case, artifact)
            replay = self._run_state(case, artifact)
            parent_metric = measure_g1_recovery_quality(parent.trajectory)
            baseline_metric = measure_g1_recovery_quality(baseline.trajectory)
            candidate_metric = measure_g1_recovery_quality(candidate.trajectory)
            parent_valid = _parent_valid(parent, parent_metric)
            parent_valid_count += int(parent_valid)
            route = _recovery_state_route(candidate)
            routed = route is not None
            route_count += int(routed)
            fallback_count += int(not routed)
            strict = _strict_replay(candidate, replay)
            strict_count += int(strict)
            exact_fallback = bool(
                not routed
                and parent.result.summary_dict() == candidate.result.summary_dict()
                and trajectory_digest(parent.trajectory) == trajectory_digest(candidate.trajectory)
            )
            exact_fallback_count += int(exact_fallback)
            _, safe, goal, natural = _case_score(
                parent=parent,
                parent_quality=parent_metric,
                candidate=candidate,
                candidate_quality=candidate_metric,
            )
            if routed:
                reductions = _moving_reductions(baseline_metric, candidate_metric)
                composite = _composite(reductions)
                routed_reductions.append(reductions)
                routed_composites.append(composite)
                passed = bool(
                    parent_valid
                    and strict
                    and safe
                    and goal
                    and natural
                    and min(reductions) >= -_COMPONENT_REGRESSION_LIMIT
                    and composite >= _ADVANTAGE_GATE
                )
            else:
                # Boundary negatives can have an invalid retained parent.  The
                # router passes only by reproducing it exactly; it cannot claim
                # to repair a regime for which no bounded primitive qualified.
                passed = bool(strict and exact_fallback)
            passed_count += int(passed)
        mean_composite = float(np.mean(routed_composites)) if routed_composites else 0.0
        minimum_composite = float(np.min(routed_composites)) if routed_composites else 0.0
        bootstrap = (
            _bootstrap_lower_95(tuple(routed_composites), seed=self.seed + 1)
            if len(routed_composites) >= 2
            else 0.0
        )
        mean_components = (
            np.mean(np.asarray(routed_reductions, dtype=np.float64), axis=0)
            if routed_reductions
            else np.zeros(3, dtype=np.float64)
        )
        route_requirement = not require_all_learned or route_count == len(cases)
        fallback_requirement = not require_all_fallback or fallback_count == len(cases)
        require_valid_parents = require_all_learned or require_all_fallback
        qualified = bool(
            passed_count == len(cases)
            and strict_count == len(cases)
            and route_requirement
            and fallback_requirement
            and (not require_valid_parents or parent_valid_count == len(cases))
            and (not require_all_learned or bootstrap > 0.0)
        )
        return (
            G1RecoveryStateSuiteSummary(
                suite_hash=suite_hash,
                case_count=len(cases),
                parent_valid_count=parent_valid_count,
                learned_route_count=route_count,
                fallback_route_count=fallback_count,
                strict_replay_count=strict_count,
                exact_parent_fallback_count=exact_fallback_count,
                passed_count=passed_count,
                mean_composite_reduction=mean_composite,
                minimum_composite_reduction=minimum_composite,
                bootstrap_lower_95=bootstrap,
                mean_backward_reduction=float(mean_components[0]),
                mean_tail_wobble_reduction=float(mean_components[1]),
                mean_leg_jerk_reduction=float(mean_components[2]),
                qualified=qualified,
            ),
            4 * len(cases),
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

    def _trial_artifact(
        self,
        *,
        descriptor: tuple[float, ...],
        primitive_index: int,
        primitives: tuple[G1ContextualRecoveryPrimitive, ...],
        observation_mean: tuple[float, ...],
        observation_scale: tuple[float, ...],
        baseline_hash: str,
        retained_hash: str,
    ) -> G1RecoveryStateArtifact:
        return G1RecoveryStateArtifact(
            body_hash=self.base.backend.qualification.body_hash,
            motion_hash=self.base.backend.qualification.motion_hash,
            baseline_recovery_config_hash=baseline_hash,
            fallback_recovery_config_hash=retained_hash,
            training_dataset_hash="sha256:" + "0" * 64,
            observation_mean=observation_mean,
            observation_scale=observation_scale,
            descriptor_feature_names=G1_RECOVERY_STATE_FEATURES,
            descriptor_prototypes=(descriptor, descriptor, descriptor),
            prototype_primitive_indices=(primitive_index, primitive_index, -1),
            prototype_composite_advantages=(1.0, 1.0, -1.0),
            prototype_component_minimums=(0.0, 0.0, -1.0),
            primitives=primitives,
            selection_window_frames=_WINDOW_FRAMES,
            neighbor_count=_NEIGHBOR_COUNT,
            maximum_neighbor_distance=_MAXIMUM_NEIGHBOR_DISTANCE,
            minimum_primitive_consensus=_MINIMUM_CONSENSUS,
            minimum_advantage_lower_bound=_ADVANTAGE_GATE,
            minimum_component_lower_bound=-_COMPONENT_REGRESSION_LIMIT,
            maximum_feature_z=8.0,
            training_episode_count=1,
            training_seed=self.seed,
        )

    def _run_state(
        self,
        case: G1MuscleMemoryCase,
        artifact: G1RecoveryStateArtifact,
    ) -> GoalForgeEpisode:
        controller = self.base.backend.build_cerebellar_recovery_controller(
            case.scenario,
            self.baseline_config,
            recovery_state_artifact=artifact,
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


def _recovery_state_descriptor(
    episode: GoalForgeEpisode,
    *,
    observation_mean: tuple[float, ...],
    observation_scale: tuple[float, ...],
) -> tuple[float, ...]:
    """Reconstruct the first five distinct causal policy observations."""

    rows = np.asarray(episode.trajectory["recovery_state_observation"], dtype=np.float64)
    updated = np.asarray(
        episode.trajectory["recovery_observation_updated"],
        dtype=np.bool_,
    )
    impulse_index = G1_RECOVERY_STATE_OBSERVATIONS.index("contact_impulse_ns")
    support_index = G1_RECOVERY_STATE_OBSERVATIONS.index("right_support")
    indices = np.asarray(
        [G1_RECOVERY_STATE_OBSERVATIONS.index(name) for name in G1_RECOVERY_STATE_FEATURES],
        dtype=np.int64,
    )
    landing_rows = np.flatnonzero(
        updated & (rows[:, impulse_index] > 0.0) & (rows[:, support_index] > 0.5)
    )
    if not len(landing_rows):
        raise ValueError("development episode has no causal recovery-state window")
    # The runtime latches landing.  Support may lift again during the next
    # observations, so only the first row requires right support; subsequent
    # rows must mirror the latched online policy exactly.
    eligible = np.flatnonzero(
        updated & (np.arange(len(rows), dtype=np.int64) >= int(landing_rows[0]))
    )
    unique = [rows[int(row_index)].copy() for row_index in eligible[:_WINDOW_FRAMES]]
    if len(unique) != _WINDOW_FRAMES:
        raise ValueError("development episode has an incomplete recovery-state window")
    normalized = (np.asarray(unique) - np.asarray(observation_mean)) / np.asarray(observation_scale)
    window = normalized[:, indices]
    descriptor = np.concatenate((np.mean(window, axis=0), window[-1] - window[0]))
    return tuple(map(float, descriptor))


def _recovery_state_normalization(
    episodes: tuple[GoalForgeEpisode, ...],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    rows = np.concatenate(
        [
            np.asarray(item.trajectory["recovery_state_observation"], dtype=np.float64)
            for item in episodes
        ],
        axis=0,
    )
    if rows.ndim != 2 or rows.shape[1] != len(G1_RECOVERY_STATE_OBSERVATIONS):
        raise ValueError("recorded recovery-state observations have an invalid shape")
    if not np.all(np.isfinite(rows)):
        raise ValueError("recorded recovery-state observations must be finite")
    mean = np.mean(rows, axis=0)
    scale = np.maximum(
        np.std(rows, axis=0),
        np.asarray(
            (
                0.50,
                0.05,
                0.10,
                0.10,
                0.10,
                0.05,
                0.05,
                0.10,
                0.10,
                0.10,
                0.03,
                0.25,
                0.25,
                0.10,
                0.10,
                0.25,
                0.10,
                0.10,
                0.10,
                0.25,
                0.25,
                0.25,
            ),
            dtype=np.float64,
        ),
    )
    return tuple(map(float, mean)), tuple(map(float, scale))


def _recovery_state_route(episode: GoalForgeEpisode) -> int | None:
    receipt = episode.recovery_receipt
    if receipt is None or receipt.recovery_state_receipt is None:
        return None
    value = receipt.recovery_state_receipt["selected_primitive_index"]
    return int(value) if value is not None else None


def _absolute_recovery_margins(
    quality: Any,
) -> tuple[tuple[float, float, float], bool]:
    """Score a standing rescue against physical limits, not a fallen parent."""

    reductions = (
        1.0 - quality.post_contact_backward_reversal_m / _RESCUE_BACKWARD_LIMIT_M,
        1.0 - quality.tail_wobble_index / _RESCUE_TAIL_WOBBLE_LIMIT,
        1.0
        - quality.post_contact_leg_joint_jerk_rms_rad_s3
        / _RESCUE_LEG_JERK_LIMIT_RAD_S3,
    )
    settling = quality.settling_time_sec
    natural = bool(
        quality.terminal_bilateral_support
        and settling is not None
        and settling <= _RESCUE_SETTLING_LIMIT_SEC
        and min(reductions) >= 0.0
    )
    return reductions, natural


def write_g1_recovery_state_report(
    report: G1RecoveryStateTrainingReport,
    *,
    output_dir: Path,
    source_checkout: Path,
) -> tuple[Path, Path]:
    root = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if root == checkout or checkout in root.parents:
        raise ValueError("recovery-state evidence must remain outside the checkout")
    root.mkdir(parents=True, exist_ok=False)
    artifact_path = root / "g1-recovery-state.json"
    report_path = root / "g1-recovery-state-training.json"
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
    "build_g1_recovery_state_cases",
    "build_g1_recovery_state_validation_cases",
    "G1RecoveryStatePrototypeResult",
    "G1RecoveryStateSuiteSummary",
    "G1RecoveryStateTrainer",
    "G1RecoveryStateTrainingReport",
    "write_g1_recovery_state_report",
]
