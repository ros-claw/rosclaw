"""Continual adaptation components for the G1 GoalForge task."""

from rosclaw.auto.g1_kick.curriculum import GoalForgeCurriculum
from rosclaw.auto.g1_kick.parameter_search import GoalForgeParameterSearch
from rosclaw.auto.g1_kick.shot_adapter_train import (
    G1ShotAdapter,
    ShotAdapterTeacherSample,
)
from rosclaw.auto.g1_kick.trajectory_search import TrajectoryCandidate

__all__ = [
    "G1ShotAdapter",
    "GoalForgeCurriculum",
    "GoalForgeParameterSearch",
    "ShotAdapterTeacherSample",
    "TrajectoryCandidate",
]
