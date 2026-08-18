"""Ten-generation GoalForge continual-evolution curriculum."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CurriculumStage:
    generation: int
    difficulty: str
    focus: str


class GoalForgeCurriculum:
    stages = (
        CurriculumStage(0, "fixed_ball_center_goal", "stable_kick"),
        CurriculumStage(1, "lateral_ball_randomization", "stance_and_foot"),
        CurriculumStage(2, "three_target_zones", "shot_direction"),
        CurriculumStage(3, "ball_mass_and_friction", "twin_belief"),
        CurriculumStage(4, "support_friction", "balance_and_slip"),
        CurriculumStage(5, "body_calibration", "body_aware_adaptation"),
        CurriculumStage(6, "latency_and_noise", "contact_timing"),
        CurriculumStage(7, "external_disturbance", "recovery_step"),
        CurriculumStage(8, "ball_observation_noise", "sense_provider"),
        CurriculumStage(9, "slow_rolling_ball", "trigger_prediction"),
        CurriculumStage(10, "mixed_hidden_tournament", "continual_learning"),
    )

    def stage(self, generation: int) -> CurriculumStage:
        if not 0 <= generation <= 10:
            raise ValueError("GoalForge curriculum generation must be in [0, 10]")
        return self.stages[generation]

    def required_loop(self) -> tuple[str, ...]:
        return (
            "Practice",
            "Dataset Snapshot",
            "Memory / Know update",
            "Candidate Generation",
            "Development",
            "Validation",
            "Hidden Holdout",
            "Falsification",
            "Historical Regression",
            "Canary",
            "Activate / Reject",
        )


__all__ = ["CurriculumStage", "GoalForgeCurriculum"]
