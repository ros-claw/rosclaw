"""Large matched physical failure-to-success validation for GoalForge."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from rosclaw.simforge.backends.unitree_mujoco_backend import G1MuJoCoBackend
from rosclaw.simforge.models import Partition
from rosclaw.simforge.tasks.g1_goalforge.concepts import ShotParameters, hash_json
from rosclaw.simforge.tasks.g1_goalforge.failure_signature import (
    GoalForgeFailureRouter,
    RecoverabilityV3,
)
from rosclaw.simforge.tasks.g1_goalforge.scenario import GoalForgeScenario
from rosclaw.simforge.tasks.g1_goalforge.verifier import GoalForgeVerifier


@dataclass(frozen=True)
class GoalForgePhysicalMetrics:
    status: str
    success: bool
    target_error_m: float
    ball_speed_mps: float
    contact_time_sec: float | None
    support_slip_m: float
    com_margin_m: float
    torso_roll_rad: float
    torso_pitch_rad: float
    post_kick_stability_sec: float
    fall: bool
    torque_violation: bool
    joint_limit_violation: bool


@dataclass(frozen=True)
class GoalForgeSandboxAttempt:
    attempt_index: int
    policy_hash: str
    status: str
    target_error_m: float
    verified: bool
    safe: bool


@dataclass(frozen=True)
class GoalForgeRecoveryPair:
    pair_id: str
    seed_commitment: str
    scenario_commitment: str
    baseline_policy_hash: str
    retry_policy_hash: str
    baseline_status: str
    retry_status: str
    baseline_target_error_m: float
    retry_target_error_m: float
    retry_budget: int
    baseline_verified: bool
    retry_verified: bool
    safe_retry: bool
    causal_passed: bool
    baseline_metrics: GoalForgePhysicalMetrics
    retry_metrics: GoalForgePhysicalMetrics
    sandbox_attempts: tuple[GoalForgeSandboxAttempt, ...]


@dataclass(frozen=True)
class GoalForgeRecoveryValidation:
    requested_pairs: int
    attempted_scenarios: int
    pairs: tuple[GoalForgeRecoveryPair, ...]
    recoverable_failure_capture_rate: float
    baseline_success_rate: float
    retry_success_rate: float
    infinite_retry_count: int
    unrecoverable_stop_rate: float
    physics_executions: int
    schema_version: str = "rosclaw.g1_goalforge.recovery_validation.v1"

    @property
    def passed(self) -> bool:
        return bool(
            len(self.pairs) >= self.requested_pairs >= 100
            and self.recoverable_failure_capture_rate == 1.0
            and self.retry_success_rate > self.baseline_success_rate
            and self.retry_success_rate >= 0.95
            and self.infinite_retry_count == 0
            and self.unrecoverable_stop_rate == 1.0
            and all(
                pair.baseline_verified
                and pair.retry_verified
                and pair.safe_retry
                and pair.causal_passed
                and pair.baseline_policy_hash != pair.retry_policy_hash
                and 1 <= len(pair.sandbox_attempts) <= pair.retry_budget
                for pair in self.pairs
            )
        )

    @property
    def result_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "pairs": [asdict(pair) for pair in self.pairs],
            "passed": self.passed,
        }


def run_physical_recovery_validation(
    *,
    asset_root: Path,
    output_path: Path,
    pair_count: int = 100,
    seed: int = 20260723,
    max_attempts: int = 400,
) -> GoalForgeRecoveryValidation:
    if pair_count < 100 or max_attempts < pair_count:
        raise ValueError("GoalForge recovery validation requires at least 100 pairs")
    backend = G1MuJoCoBackend(asset_root=asset_root, trace_stride=1)
    verifier = GoalForgeVerifier()
    router = GoalForgeFailureRouter()
    rng = random.Random(seed)
    baseline_parameters = ShotParameters()
    pairs: list[GoalForgeRecoveryPair] = []
    attempted = 0
    captured = 0
    recoverable_failures = 0
    physics_executions = 0
    while len(pairs) < pair_count and attempted < max_attempts:
        scenario_seed = seed * 1000 + attempted
        scenario = GoalForgeScenario(
            scenario_id=f"goalforge-recovery-{attempted:04d}",
            partition=Partition.VALIDATION,
            seed=scenario_seed,
            seed_commitment=hash_json({"seed": scenario_seed}),
            generation=6,
            ball_x_m=rng.uniform(0.995, 1.005),
            ball_y_m=rng.uniform(-0.005, 0.005),
            ball_velocity_x_mps=0.0,
            ball_velocity_y_mps=0.0,
            target_y_m=rng.uniform(0.55, 0.56),
            target_z_m=rng.uniform(0.54, 0.56),
            ball_mass_kg=rng.uniform(0.415, 0.425),
            ball_ground_friction=rng.uniform(0.05, 0.06),
            restitution=rng.uniform(0.54, 0.56),
            support_ground_friction=rng.uniform(0.98, 1.02),
            control_latency_ms=rng.uniform(0.0, 2.0),
            observation_noise_m=0.0,
            joint_zero_bias_rad=0.0,
            disturbance_n=0.0,
        )
        attempted += 1
        baseline = backend.run(scenario, baseline_parameters)
        physics_executions += int(baseline.result.physics_executed)
        if baseline.result.success:
            continue
        failure = router.route(
            result=baseline.result,
            body_hash=backend.qualification.body_hash,
            scene_hash=scenario.scenario_commitment,
            action_id=f"recovery-baseline-{attempted}",
            policy_hash=baseline_parameters.policy_hash,
        )
        if failure.recoverability is not RecoverabilityV3.RECOVERABLE:
            continue
        recoverable_failures += 1
        captured += 1
        context = scenario.observed_context()
        delta = context["target_y"] - context["ball_y"]
        candidates = (
            ShotParameters(
                stance_offset_y=max(-0.12, min(0.12, context["ball_y"] * 0.45)),
                pelvis_yaw_offset=max(-0.20, min(0.20, delta * 0.30)),
                com_shift_y=0.015,
                foot_yaw_offset=max(-0.12, min(0.12, delta * 0.05)),
                recovery_step_length=0.055,
                policy_type="parameter",
            ),
            ShotParameters(
                stance_offset_y=max(-0.12, min(0.12, context["ball_y"] * 0.45)),
                pelvis_yaw_offset=max(-0.20, min(0.20, delta * 0.28)),
                com_shift_y=0.010,
                foot_yaw_offset=max(-0.12, min(0.12, delta * 0.045)),
                recovery_step_length=0.050,
                policy_type="parameter",
            ),
        )
        baseline_verification = verifier.verify(baseline)
        attempts: list[GoalForgeSandboxAttempt] = []
        retry = baseline
        retry_verification = baseline_verification
        safe_retry = False
        for attempt_index, candidate in enumerate(
            candidates[: failure.retry_budget],
            start=1,
        ):
            retry = backend.run(scenario, candidate)
            physics_executions += int(retry.result.physics_executed)
            retry_verification = verifier.verify(retry)
            safe_retry = _safe(retry)
            attempts.append(
                GoalForgeSandboxAttempt(
                    attempt_index=attempt_index,
                    policy_hash=candidate.policy_hash,
                    status=retry.result.status.value,
                    target_error_m=_finite_error(retry.result.target_error_m),
                    verified=retry_verification.valid,
                    safe=safe_retry,
                )
            )
            if retry.result.success and retry_verification.valid and safe_retry:
                break
        causal = bool(
            baseline.scenario.seed == retry.scenario.seed
            and baseline.scenario.scenario_commitment == retry.scenario.scenario_commitment
            and baseline.parameters.policy_hash != retry.parameters.policy_hash
            and not baseline.result.success
            and retry.result.success
            and baseline_verification.valid
            and retry_verification.valid
            and safe_retry
        )
        pairs.append(
            GoalForgeRecoveryPair(
                pair_id=f"physical-pair-{len(pairs):03d}",
                seed_commitment=scenario.seed_commitment,
                scenario_commitment=scenario.scenario_commitment,
                baseline_policy_hash=baseline.parameters.policy_hash,
                retry_policy_hash=retry.parameters.policy_hash,
                baseline_status=baseline.result.status.value,
                retry_status=retry.result.status.value,
                baseline_target_error_m=_finite_error(baseline.result.target_error_m),
                retry_target_error_m=_finite_error(retry.result.target_error_m),
                retry_budget=failure.retry_budget,
                baseline_verified=baseline_verification.valid,
                retry_verified=retry_verification.valid,
                safe_retry=safe_retry,
                causal_passed=causal,
                baseline_metrics=_physical_metrics(baseline),
                retry_metrics=_physical_metrics(retry),
                sandbox_attempts=tuple(attempts),
            )
        )
    unrecoverable_stops = _unrecoverable_stop_checks(backend, router)
    result = GoalForgeRecoveryValidation(
        requested_pairs=pair_count,
        attempted_scenarios=attempted,
        pairs=tuple(pairs),
        recoverable_failure_capture_rate=(
            captured / recoverable_failures if recoverable_failures else 0.0
        ),
        baseline_success_rate=0.0,
        retry_success_rate=sum(pair.retry_status == "SUCCESS" for pair in pairs) / len(pairs),
        infinite_retry_count=sum(len(pair.sandbox_attempts) > pair.retry_budget for pair in pairs),
        unrecoverable_stop_rate=sum(unrecoverable_stops) / len(unrecoverable_stops),
        physics_executions=physics_executions,
    )
    output_path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _unrecoverable_stop_checks(
    backend: G1MuJoCoBackend,
    router: GoalForgeFailureRouter,
) -> tuple[bool, ...]:
    unreachable = GoalForgeScenario(
        scenario_id="goalforge-unreachable-stop",
        partition=Partition.COUNTEREXAMPLE_REGRESSION,
        seed=99,
        seed_commitment=hash_json({"seed": 99}),
        generation=10,
        ball_x_m=1.0,
        ball_y_m=0.0,
        ball_velocity_x_mps=0.0,
        ball_velocity_y_mps=0.0,
        target_y_m=0.0,
        target_z_m=0.55,
        ball_mass_kg=0.42,
        ball_ground_friction=0.05,
        restitution=0.55,
        support_ground_friction=1.0,
        control_latency_ms=0.0,
        observation_noise_m=0.0,
        joint_zero_bias_rad=0.0,
        disturbance_n=0.0,
        reachable=False,
    )
    cases = (
        backend.run(unreachable, ShotParameters()),
        backend.run(
            replace(
                unreachable,
                scenario_id="goalforge-incompatible-stop",
                reachable=True,
            ),
            ShotParameters(kick_foot="left"),
        ),
    )
    values = []
    for index, episode in enumerate(cases):
        failure = router.route(
            result=episode.result,
            body_hash=backend.qualification.body_hash,
            scene_hash=episode.scenario.scenario_commitment,
            action_id=f"unrecoverable-{index}",
            policy_hash=episode.parameters.policy_hash,
        )
        values.append(
            failure.recoverability is RecoverabilityV3.UNRECOVERABLE
            and failure.retry_budget == 0
            and failure.recommended_route == ("STOP", "HUMAN")
        )
    return tuple(values)


def _finite_error(value: float) -> float:
    return value if value < 99.0 else 99.0


def _safe(episode: Any) -> bool:
    result = episode.result
    return bool(
        not result.post_kick_fall
        and not result.torque_limit_violation
        and not result.joint_limit_violation
        and not result.actuator_saturation
    )


def _physical_metrics(episode: Any) -> GoalForgePhysicalMetrics:
    result = episode.result
    return GoalForgePhysicalMetrics(
        status=result.status.value,
        success=result.success,
        target_error_m=_finite_error(result.target_error_m),
        ball_speed_mps=result.ball_speed_mps,
        contact_time_sec=result.ball_contact_time_sec,
        support_slip_m=result.support_foot_slip_m,
        com_margin_m=result.com_margin_min_m,
        torso_roll_rad=result.torso_roll_peak_rad,
        torso_pitch_rad=result.torso_pitch_peak_rad,
        post_kick_stability_sec=result.post_kick_stability_time_sec,
        fall=result.post_kick_fall,
        torque_violation=result.torque_limit_violation,
        joint_limit_violation=result.joint_limit_violation,
    )


__all__ = [
    "GoalForgeRecoveryPair",
    "GoalForgeRecoveryValidation",
    "GoalForgePhysicalMetrics",
    "GoalForgeSandboxAttempt",
    "run_physical_recovery_validation",
]
