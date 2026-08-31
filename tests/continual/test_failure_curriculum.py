from __future__ import annotations

import pytest

from rosclaw.continual.failure_curriculum import (
    CapabilityBin,
    CapabilityFrontierScheduler,
    CurriculumMixture,
    CurriculumSource,
    DreamPerturbation,
    FailureConditionedDream,
    PerturbationDistribution,
)
from tests.continual.helpers import digest


def test_curriculum_mix_allocates_exact_batch_without_rounding_loss() -> None:
    allocation = CurriculumMixture().allocate(17)

    assert sum(allocation.values()) == 17
    assert allocation[CurriculumSource.CAPABILITY_FRONTIER] >= 6
    assert allocation[CurriculumSource.RECENT_FAILURE] >= 4


def test_failure_conditioned_dream_is_deterministic_and_sim_only() -> None:
    contract = FailureConditionedDream(
        failure_code="goalkeeper.getup_failure",
        source_snapshot_hash=digest("snapshot"),
        body_hash=digest("body"),
        scenario_hash=digest("scene"),
        perturbations=(
            DreamPerturbation("root_momentum", -0.2, 0.2),
            DreamPerturbation(
                "friction",
                -0.1,
                0.1,
                PerturbationDistribution.NORMAL_CLIPPED,
            ),
        ),
        maximum_variants=32,
    )

    assert contract.sample(count=4, seed=7) == contract.sample(count=4, seed=7)
    assert contract.training_use_only
    assert contract.activation_ceiling == "SIM_ONLY"
    with pytest.raises(ValueError, match="sampling"):
        contract.sample(count=33, seed=7)


def test_capability_frontier_prioritizes_the_learning_edge() -> None:
    probabilities = CapabilityFrontierScheduler().probabilities(
        (
            CapabilityBin("mastered", difficulty=0.2, successes=99, attempts=100),
            CapabilityBin("frontier", difficulty=0.6, successes=50, attempts=100),
            CapabilityBin("impossible", difficulty=1.0, successes=0, attempts=100),
        )
    )

    assert probabilities["frontier"] > probabilities["mastered"]
    assert probabilities["frontier"] > probabilities["impossible"]
    assert sum(probabilities.values()) == pytest.approx(1.0)
