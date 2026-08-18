"""Continual adaptation components for the G1 GoalForge task."""

from rosclaw.evolution.orchestrator.g1_kick.curriculum import GoalForgeCurriculum
from rosclaw.evolution.orchestrator.g1_kick.parameter_search import GoalForgeParameterSearch
from rosclaw.evolution.orchestrator.g1_kick.shot_adapter_train import (
    G1ShotAdapter,
    ShotAdapterTeacherSample,
)
from rosclaw.evolution.orchestrator.g1_kick.trajectory_search import TrajectoryCandidate

__all__ = [
    "G1ShotAdapter",
    "GoalForgeCurriculum",
    "GoalForgeParameterSearch",
    "ShotAdapterTeacherSample",
    "TrajectoryCandidate",
]
