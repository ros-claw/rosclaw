"""Causal same-seed GoalForge task loop."""

from __future__ import annotations

from dataclasses import dataclass

from rosclaw.simforge.backends.unitree_mujoco_backend import (
    G1MuJoCoBackend,
    GoalForgeEpisode,
)
from rosclaw.simforge.tasks.g1_goalforge.concepts import ShotParameters
from rosclaw.simforge.tasks.g1_goalforge.scenario import GoalForgeScenario


@dataclass(frozen=True)
class SameSeedKickPair:
    baseline: GoalForgeEpisode
    retry: GoalForgeEpisode
    same_seed: bool
    same_scenario: bool
    only_candidate_changed: bool
    outcome_improved: bool

    @property
    def causal_passed(self) -> bool:
        return (
            self.same_seed
            and self.same_scenario
            and self.only_candidate_changed
            and self.outcome_improved
        )


def run_same_seed_pair(
    *,
    backend: G1MuJoCoBackend,
    scenario: GoalForgeScenario,
    baseline: ShotParameters,
    candidate: ShotParameters,
) -> SameSeedKickPair:
    control = backend.run(scenario, baseline)
    treatment = backend.run(scenario, candidate)
    return SameSeedKickPair(
        baseline=control,
        retry=treatment,
        same_seed=control.scenario.seed == treatment.scenario.seed,
        same_scenario=(
            control.scenario.scenario_commitment == treatment.scenario.scenario_commitment
        ),
        only_candidate_changed=(control.parameters.policy_hash != treatment.parameters.policy_hash),
        outcome_improved=(
            treatment.result.success
            and (
                not control.result.success
                or treatment.result.target_error_m < control.result.target_error_m
            )
        ),
    )


__all__ = ["SameSeedKickPair", "run_same_seed_pair"]
