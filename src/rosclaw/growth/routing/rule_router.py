"""Evidence-conditioned routing across complementary learning mechanisms."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from rosclaw.feedback.contracts import canonical_hash


def _score(label: str, value: float) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} must be finite and in [0, 1]")


@dataclass(frozen=True)
class GrowthProblemSignals:
    repeated_error: float = 0.0
    regime_shift: float = 0.0
    local_physics_residual: float = 0.0
    out_of_distribution: float = 0.0
    gradient_conflict: float = 0.0
    safety_model_complete: bool = True
    schema_version: str = "rosclaw.growth.problem_signals.v1"

    def __post_init__(self) -> None:
        for label in (
            "repeated_error",
            "regime_shift",
            "local_physics_residual",
            "out_of_distribution",
            "gradient_conflict",
        ):
            _score(label, getattr(self, label))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DataProfile:
    has_state: bool = False
    has_executed_action: bool = False
    has_next_state: bool = False
    has_reward_vector: bool = False
    has_cost_vector: bool = False
    has_kinematic_reference: bool = False
    has_chunk_feedback: bool = False
    fixed_dataset: bool = False
    online_rollout_allowed: bool = False
    schema_version: str = "rosclaw.growth.data_profile.v1"

    @property
    def transition_ready(self) -> bool:
        return all(
            (
                self.has_state,
                self.has_executed_action,
                self.has_next_state,
                self.has_reward_vector,
                self.has_cost_vector,
            )
        )

    @property
    def offline_rl_ready(self) -> bool:
        return self.fixed_dataset and self.transition_ready

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["transition_ready"] = self.transition_ready
        value["offline_rl_ready"] = self.offline_rl_ready
        return value


class RouteDisposition(StrEnum):
    SELECTED = "selected"
    NEED_MORE_EVIDENCE = "need_more_evidence"
    BLOCKED_SAFETY_MODEL = "blocked_safety_model"


@dataclass(frozen=True)
class LearnerRoute:
    disposition: RouteDisposition
    learner_ids: tuple[str, ...]
    reasons: tuple[str, ...]
    missing_requirements: tuple[str, ...]
    schema_version: str = "rosclaw.growth.learner_route.v1"

    @property
    def route_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "disposition": self.disposition.value,
            "learner_ids": list(self.learner_ids),
            "reasons": list(self.reasons),
            "missing_requirements": list(self.missing_requirements),
            "evidence_domain": "LEARNING_CONTROL_PLANE",
            "activation_authorized": False,
            "hardware_authorized": False,
        }


def route_learners(signals: GrowthProblemSignals, data: DataProfile) -> LearnerRoute:
    """Choose mechanisms from evidence shape instead of task or robot name."""

    if not isinstance(signals, GrowthProblemSignals) or not isinstance(data, DataProfile):
        raise ValueError("route_learners requires GrowthProblemSignals and DataProfile")
    if not signals.safety_model_complete:
        return LearnerRoute(
            disposition=RouteDisposition.BLOCKED_SAFETY_MODEL,
            learner_ids=(),
            reasons=("safety_model_incomplete",),
            missing_requirements=("complete_cost_and_safety_projection_contract",),
        )

    learners: list[str] = []
    reasons: list[str] = []
    missing: list[str] = []
    if signals.regime_shift >= 0.60:
        learners.append("system_identification")
        reasons.append("body_or_environment_regime_shift")
    if signals.repeated_error >= 0.70 and signals.regime_shift < 0.60:
        learners.append("ilc")
        reasons.append("stable_repeatable_error")
    if data.has_kinematic_reference and not data.transition_ready:
        learners.append("motion_tracking")
        reasons.append("kinematic_reference_without_physics_transition")
    if data.offline_rl_ready:
        learners.append("iql")
        reasons.append("complete_fixed_transition_dataset")
    elif data.fixed_dataset and any(
        (data.has_state, data.has_executed_action, data.has_next_state, data.has_reward_vector)
    ):
        missing.extend(_missing_transition_fields(data))
    if signals.local_physics_residual >= 0.60:
        if data.transition_ready and data.online_rollout_allowed:
            learners.append("residual_sac")
            reasons.append("bounded_local_physics_residual")
        else:
            missing.extend(_missing_residual_requirements(data))
    if data.has_chunk_feedback:
        learners.append("advantage_sft")
        reasons.append("chunk_level_feedback_available")
    if signals.out_of_distribution >= 0.70:
        learners.append("world_model_observer")
        reasons.append("world_prediction_or_state_ood")
    if signals.gradient_conflict >= 0.70:
        learners.append("new_expert_adapter")
        reasons.append("shared_parameter_gradient_conflict")

    learner_ids = tuple(dict.fromkeys(learners))
    missing_requirements = tuple(dict.fromkeys(missing))
    if learner_ids:
        disposition = RouteDisposition.SELECTED
    else:
        disposition = RouteDisposition.NEED_MORE_EVIDENCE
        if not missing_requirements:
            missing_requirements = ("diagnostic_signal_or_qualified_learning_data",)
    return LearnerRoute(
        disposition=disposition,
        learner_ids=learner_ids,
        reasons=tuple(reasons),
        missing_requirements=missing_requirements,
    )


def _missing_transition_fields(data: DataProfile) -> list[str]:
    fields = (
        ("state", data.has_state),
        ("executed_action", data.has_executed_action),
        ("next_state", data.has_next_state),
        ("reward_vector", data.has_reward_vector),
        ("cost_vector", data.has_cost_vector),
    )
    return [f"transition.{name}" for name, present in fields if not present]


def _missing_residual_requirements(data: DataProfile) -> list[str]:
    missing = _missing_transition_fields(data)
    if not data.online_rollout_allowed:
        missing.append("simulation_online_rollout")
    return missing


__all__ = [
    "DataProfile",
    "GrowthProblemSignals",
    "LearnerRoute",
    "RouteDisposition",
    "route_learners",
]
