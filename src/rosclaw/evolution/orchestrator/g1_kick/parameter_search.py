"""Bounded, safety-first parameter search around the fixed G1 kick prior."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, replace
from typing import Protocol

from rosclaw.how.g1_goalforge import GoalForgeHowIntervention
from rosclaw.know.g1_goalforge import GoalForgeKnowledge
from rosclaw.memory.g1_goalforge import GoalForgeMemoryRecall
from rosclaw.simforge.backends.unitree_mujoco_backend import GoalForgeEpisode
from rosclaw.simforge.tasks.g1_goalforge.concepts import (
    GoalForgeResult,
    ShotParameters,
    hash_json,
)
from rosclaw.simforge.tasks.g1_goalforge.robustness import score_goalforge_result
from rosclaw.simforge.tasks.g1_goalforge.scenario import GoalForgeScenario
from rosclaw.twin.g1_kick.belief import KickTwinBelief


class GoalForgeRunner(Protocol):
    def run(
        self,
        scenario: GoalForgeScenario,
        parameters: ShotParameters,
    ) -> GoalForgeEpisode: ...


@dataclass(frozen=True)
class ParameterCandidateEvaluation:
    candidate: ShotParameters
    result: GoalForgeResult
    safe: bool
    score: float
    source: str
    elapsed_ms: float

    @property
    def evaluation_hash(self) -> str:
        return hash_json(
            {
                "candidate": self.candidate.to_dict(),
                "result": self.result.summary_dict(),
                "safe": self.safe,
                "score": self.score,
                "source": self.source,
            }
        )


@dataclass(frozen=True)
class ParameterSearchOutcome:
    context_hash: str
    twin_belief_hash: str
    knowledge_hash: str
    evaluations: tuple[ParameterCandidateEvaluation, ...]
    winner: ShotParameters | None
    winner_evaluation_hash: str | None
    attempts: int
    hidden_truth_accessed_by_generator: bool = False
    schema_version: str = "rosclaw.g1_goalforge.parameter_search.v1"

    def __post_init__(self) -> None:
        if self.hidden_truth_accessed_by_generator:
            raise ValueError("parameter search generator accessed hidden truth")
        if self.attempts != len(self.evaluations):
            raise ValueError("parameter search attempt count is inconsistent")

    @property
    def search_hash(self) -> str:
        value = asdict(self)
        value["evaluations"] = [
            {
                "evaluation_hash": item.evaluation_hash,
                "candidate_hash": item.candidate.policy_hash,
            }
            for item in self.evaluations
        ]
        return hash_json(value)


class GoalForgeParameterSearch:
    """Generate from public context; execute against private physics boundary."""

    def __init__(self, *, max_candidates: int = 32) -> None:
        if not 1 <= max_candidates <= 32:
            raise ValueError("GoalForge search budget must be in [1, 32]")
        self.max_candidates = max_candidates

    def run(
        self,
        *,
        runner: GoalForgeRunner,
        scenario: GoalForgeScenario,
        base: ShotParameters,
        twin: KickTwinBelief,
        knowledge: GoalForgeKnowledge,
        memory: GoalForgeMemoryRecall | None = None,
        intervention: GoalForgeHowIntervention | None = None,
    ) -> ParameterSearchOutcome:
        public_context = {
            **scenario.observed_context(),
            **twin.public_context(),
            "memory_query_hash": memory.query_hash if memory is not None else "none",
        }
        candidates = build_parameter_candidates(
            base=base,
            public_context=scenario.observed_context(),
            twin=twin,
            memory=memory,
            intervention=intervention,
        )[: self.max_candidates]
        evaluations: list[ParameterCandidateEvaluation] = []
        for candidate, source in candidates:
            eligible, _errors = knowledge.validate_candidate(candidate=candidate)
            if not eligible:
                continue
            started = time.perf_counter()
            episode = runner.run(scenario, candidate)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            safe = _safe(episode.result)
            evaluations.append(
                ParameterCandidateEvaluation(
                    candidate=candidate,
                    result=episode.result,
                    safe=safe,
                    score=(score_goalforge_result(episode.result).score if safe else -1_000_000.0),
                    source=source,
                    elapsed_ms=elapsed_ms,
                )
            )
        eligible_evaluations = [item for item in evaluations if item.safe]
        winner_eval = (
            max(
                eligible_evaluations,
                key=lambda item: (
                    item.result.success,
                    item.score,
                    -item.elapsed_ms,
                    item.candidate.policy_hash,
                ),
            )
            if eligible_evaluations
            else None
        )
        return ParameterSearchOutcome(
            context_hash=hash_json(public_context),
            twin_belief_hash=twin.belief_hash,
            knowledge_hash=knowledge.knowledge_hash,
            evaluations=tuple(evaluations),
            winner=(winner_eval.candidate if winner_eval is not None else None),
            winner_evaluation_hash=(
                winner_eval.evaluation_hash if winner_eval is not None else None
            ),
            attempts=len(evaluations),
        )


def build_parameter_candidates(
    *,
    base: ShotParameters,
    public_context: dict[str, float],
    twin: KickTwinBelief,
    memory: GoalForgeMemoryRecall | None,
    intervention: GoalForgeHowIntervention | None,
) -> list[tuple[ShotParameters, str]]:
    values: list[tuple[ShotParameters, str]] = [(base, "fixed_prior")]
    if intervention is not None:
        values.append((intervention.patch, "how_intervention"))
    if memory is not None:
        values.extend((entry.safe_patch, "memory") for entry in memory.entries)
    ball_y = public_context["ball_y"]
    target_y = public_context["target_y"]
    lateral = max(-0.12, min(0.12, 0.055 * (target_y - ball_y)))
    heading = max(-0.20, min(0.20, 0.35 * (target_y - ball_y)))
    stance = max(-0.12, min(0.12, ball_y * 0.45))
    low_friction = twin.support_ground_friction.mean < 0.70
    values.extend(
        (
            (
                replace(
                    base,
                    stance_offset_y=stance,
                    pelvis_yaw_offset=heading,
                    foot_yaw_offset=lateral,
                    policy_type="parameter",
                ),
                "context_alignment",
            ),
            (
                replace(
                    base,
                    stance_offset_y=stance,
                    pelvis_yaw_offset=heading,
                    foot_yaw_offset=lateral,
                    com_shift_y=(0.03 if low_friction else 0.015),
                    swing_speed_scale=(0.88 if low_friction else 1.0),
                    recovery_step_length=(0.09 if low_friction else 0.055),
                    policy_type="parameter",
                ),
                "twin_balanced",
            ),
        )
    )
    for yaw_delta in (-0.08, -0.04, 0.04, 0.08):
        values.append(
            (
                replace(
                    base,
                    stance_offset_y=stance,
                    foot_yaw_offset=max(
                        -0.12,
                        min(0.12, lateral + yaw_delta),
                    ),
                    policy_type="parameter",
                ),
                "bounded_yaw_sweep",
            )
        )
    direction = -1.0 if target_y - ball_y < 0.0 else 1.0
    for pelvis_yaw in (0.0, 0.06, 0.12, 0.18, 0.20):
        for foot_yaw in (0.04, 0.08, 0.12):
            values.append(
                (
                    replace(
                        base,
                        stance_offset_y=stance,
                        pelvis_yaw_offset=direction * pelvis_yaw,
                        foot_yaw_offset=direction * foot_yaw,
                        policy_type="parameter",
                    ),
                    "coupled_lateral_sweep",
                )
            )
    for stance_delta in (-0.06, -0.03, 0.03, 0.06):
        values.append(
            (
                replace(
                    base,
                    stance_offset_y=max(-0.12, min(0.12, stance + stance_delta)),
                    pelvis_yaw_offset=max(-0.20, min(0.20, 0.28 * (target_y - ball_y))),
                    foot_yaw_offset=max(-0.12, min(0.12, 0.08 * (target_y - ball_y))),
                    policy_type="parameter",
                ),
                "stance_lateral_sweep",
            )
        )
    if abs(ball_y) >= 0.08:
        values.append(
            (
                replace(
                    base,
                    stance_offset_y=math.copysign(0.12, ball_y),
                    pelvis_yaw_offset=max(-0.20, min(0.20, 0.28 * (target_y - ball_y))),
                    foot_yaw_offset=max(-0.12, min(0.12, 0.07 * (target_y - ball_y))),
                    policy_type="parameter",
                ),
                "edge_stance_lateral_sweep",
            )
        )
    for phase in (-0.06, -0.03, 0.03):
        values.append(
            (
                replace(
                    base,
                    contact_phase_offset=phase,
                    policy_type="parameter",
                ),
                "bounded_contact_sweep",
            )
        )
    deduplicated: dict[str, tuple[ShotParameters, str]] = {}
    for candidate, source in values:
        deduplicated.setdefault(candidate.policy_hash, (candidate, source))
    return list(deduplicated.values())


def _safe(result: GoalForgeResult) -> bool:
    return bool(
        result.physics_executed
        and result.finite_state
        and not result.post_kick_fall
        and not result.joint_limit_violation
        and not result.torque_limit_violation
        and not result.actuator_saturation
        and result.support_foot_slip_m <= 0.08
        and result.final_pelvis_height_m >= 0.55
    )


__all__ = [
    "build_parameter_candidates",
    "GoalForgeParameterSearch",
    "GoalForgeRunner",
    "ParameterCandidateEvaluation",
    "ParameterSearchOutcome",
]
