"""Balanced CPU-MuJoCo validation for GoalForge nominal penalty kicks."""

from __future__ import annotations

import json
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rosclaw.auto.g1_kick.parameter_search import build_parameter_candidates
from rosclaw.simforge.backends.unitree_mujoco_backend import (
    G1MuJoCoBackend,
    qualify_g1_assets,
    trajectory_digest,
)
from rosclaw.simforge.models import Partition
from rosclaw.simforge.tasks.g1_goalforge.concepts import ShotParameters, hash_json
from rosclaw.simforge.tasks.g1_goalforge.robustness import score_goalforge_result
from rosclaw.simforge.tasks.g1_goalforge.scenario import GoalForgeScenario
from rosclaw.simforge.tasks.g1_goalforge.verifier import GoalForgeVerifier
from rosclaw.twin.g1_kick.belief import KickTwinBelief


@dataclass(frozen=True)
class GoalForgeSuccessCase:
    scenario_id: str
    scenario_commitment: str
    ball_y_m: float
    target_y_m: float
    target_z_m: float
    fixed_status: str
    fixed_success: bool
    fixed_target_error_m: float
    contextual_status: str
    contextual_success: bool
    contextual_target_error_m: float
    optimized_status: str
    optimized_success: bool
    optimized_safe: bool
    optimized_target_error_m: float
    optimized_source: str
    optimized_policy_hash: str
    optimized_parameters: dict[str, Any]
    candidate_count: int
    safe_candidate_count: int
    successful_candidate_count: int
    independently_verified: bool
    strict_replay: bool


@dataclass(frozen=True)
class GoalForgeSuccessValidation:
    body_hash: str
    kick_prior_hash: str
    cases: tuple[GoalForgeSuccessCase, ...]
    fixed_success_rate: float
    contextual_success_rate: float
    optimized_success_rate: float
    safe_selection_rate: float
    independently_verified_rate: float
    strict_replay_rate: float
    fixed_mean_target_error_m: float
    optimized_mean_target_error_m: float
    evidence_domain: str = "SHADOW"
    schema_version: str = "rosclaw.g1_goalforge.nominal_success_validation.v1"

    @property
    def passed(self) -> bool:
        return bool(
            len(self.cases) == 30
            and self.optimized_success_rate >= 0.95
            and self.optimized_success_rate >= self.fixed_success_rate + 0.30
            and self.safe_selection_rate == 1.0
            and self.independently_verified_rate == 1.0
            and self.strict_replay_rate == 1.0
            and self.optimized_mean_target_error_m < self.fixed_mean_target_error_m
        )

    @property
    def result_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "cases": [asdict(case) for case in self.cases],
            "passed": self.passed,
        }


@dataclass(frozen=True)
class _EvaluatedCandidate:
    parameters: ShotParameters
    source: str
    status: str
    success: bool
    safe: bool
    score: float
    target_error_m: float
    episode: Any


def run_nominal_success_validation(
    *,
    asset_root: Path,
    output_path: Path,
    source_checkout: Path,
    workers: int = 4,
) -> GoalForgeSuccessValidation:
    """Compare fixed, contextual, and optimized candidates on a balanced grid."""

    if not 1 <= workers <= 4:
        raise ValueError("GoalForge nominal validation workers must be in [1, 4]")
    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if checkout == output or checkout in output.parents:
        raise ValueError("GoalForge success evidence must stay outside the source checkout")
    if output.exists():
        raise FileExistsError(f"GoalForge success evidence already exists: {output}")
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    scenarios = _nominal_scenarios()
    chunks = tuple(tuple(scenarios[index::workers]) for index in range(workers))
    nested: tuple[tuple[GoalForgeSuccessCase, ...], ...]
    if workers == 1:
        nested = (_evaluate_chunk(str(asset_root), chunks[0]),)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            nested = tuple(
                executor.map(_evaluate_chunk_for_pool, ((str(asset_root), c) for c in chunks))
            )
    cases = tuple(
        sorted((case for chunk in nested for case in chunk), key=lambda case: case.scenario_id)
    )
    result = GoalForgeSuccessValidation(
        body_hash=qualification.body_hash,
        kick_prior_hash=qualification.kick_prior_hash,
        cases=cases,
        fixed_success_rate=_rate(case.fixed_success for case in cases),
        contextual_success_rate=_rate(case.contextual_success for case in cases),
        optimized_success_rate=_rate(case.optimized_success for case in cases),
        safe_selection_rate=_rate(case.optimized_safe for case in cases),
        independently_verified_rate=_rate(case.independently_verified for case in cases),
        strict_replay_rate=_rate(case.strict_replay for case in cases),
        fixed_mean_target_error_m=_mean(case.fixed_target_error_m for case in cases),
        optimized_mean_target_error_m=_mean(case.optimized_target_error_m for case in cases),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _evaluate_chunk_for_pool(
    arguments: tuple[str, tuple[GoalForgeScenario, ...]],
) -> tuple[GoalForgeSuccessCase, ...]:
    return _evaluate_chunk(*arguments)


def _evaluate_chunk(
    asset_root: str,
    scenarios: tuple[GoalForgeScenario, ...],
) -> tuple[GoalForgeSuccessCase, ...]:
    backend = G1MuJoCoBackend(asset_root=Path(asset_root), trace_stride=10)
    verifier = GoalForgeVerifier()
    twin = KickTwinBelief.initial()
    base = ShotParameters()
    cases = []
    for scenario in scenarios:
        evaluated = []
        for parameters, source in build_parameter_candidates(
            base=base,
            public_context=scenario.observed_context(),
            twin=twin,
            memory=None,
            intervention=None,
        ):
            episode = backend.run(scenario, parameters)
            scored = score_goalforge_result(episode.result)
            evaluated.append(
                _EvaluatedCandidate(
                    parameters=parameters,
                    source=source,
                    status=episode.result.status.value,
                    success=episode.result.success,
                    safe=scored.safe,
                    score=scored.score,
                    target_error_m=_finite_error(episode.result.target_error_m),
                    episode=episode,
                )
            )
        fixed = next(item for item in evaluated if item.source == "fixed_prior")
        contextual = next(item for item in evaluated if item.source == "context_alignment")
        selected = max(
            evaluated,
            key=lambda item: (
                item.safe,
                item.success,
                item.score,
                item.parameters.policy_hash,
            ),
        )
        replay = backend.run(scenario, selected.parameters)
        selected_verification = verifier.verify(selected.episode)
        replay_verification = verifier.verify(replay)
        cases.append(
            GoalForgeSuccessCase(
                scenario_id=scenario.scenario_id,
                scenario_commitment=scenario.scenario_commitment,
                ball_y_m=scenario.ball_y_m,
                target_y_m=scenario.target_y_m,
                target_z_m=scenario.target_z_m,
                fixed_status=fixed.status,
                fixed_success=fixed.success,
                fixed_target_error_m=fixed.target_error_m,
                contextual_status=contextual.status,
                contextual_success=contextual.success,
                contextual_target_error_m=contextual.target_error_m,
                optimized_status=selected.status,
                optimized_success=selected.success,
                optimized_safe=selected.safe,
                optimized_target_error_m=selected.target_error_m,
                optimized_source=selected.source,
                optimized_policy_hash=selected.parameters.policy_hash,
                optimized_parameters=selected.parameters.to_dict(),
                candidate_count=len(evaluated),
                safe_candidate_count=sum(item.safe for item in evaluated),
                successful_candidate_count=sum(item.success for item in evaluated),
                independently_verified=(selected_verification.valid and replay_verification.valid),
                strict_replay=(
                    selected.episode.result.summary_dict() == replay.result.summary_dict()
                    and trajectory_digest(selected.episode.trajectory)
                    == trajectory_digest(replay.trajectory)
                ),
            )
        )
    return tuple(cases)


def _nominal_scenarios() -> tuple[GoalForgeScenario, ...]:
    settings = (
        (ball_y, target_y, target_z)
        for ball_y in (-0.10, 0.0, 0.10)
        for target_y in (-0.75, -0.40, 0.0, 0.40, 0.75)
        for target_z in (0.20, 0.55)
    )
    return tuple(
        GoalForgeScenario(
            scenario_id=f"goalforge-nominal-{index:03d}",
            partition=Partition.VALIDATION,
            seed=820000 + index,
            seed_commitment=hash_json({"goalforge_nominal_seed": 820000 + index}),
            generation=4,
            ball_x_m=1.0,
            ball_y_m=ball_y,
            ball_velocity_x_mps=0.0,
            ball_velocity_y_mps=0.0,
            target_y_m=target_y,
            target_z_m=target_z,
            ball_mass_kg=0.42,
            ball_ground_friction=0.05,
            restitution=0.55,
            support_ground_friction=1.0,
            control_latency_ms=0.0,
            observation_noise_m=0.0,
            joint_zero_bias_rad=0.0,
            disturbance_n=0.0,
        )
        for index, (ball_y, target_y, target_z) in enumerate(settings)
    )


def _finite_error(value: float) -> float:
    return value if math.isfinite(value) else 99.0


def _rate(values: Any) -> float:
    normalized = list(values)
    return sum(map(bool, normalized)) / len(normalized) if normalized else 0.0


def _mean(values: Any) -> float:
    normalized = list(values)
    return sum(normalized) / len(normalized) if normalized else 0.0


__all__ = [
    "GoalForgeSuccessCase",
    "GoalForgeSuccessValidation",
    "run_nominal_success_validation",
]
