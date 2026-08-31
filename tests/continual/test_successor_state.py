from __future__ import annotations

import pytest

from rosclaw.continual.successor_state import (
    SkillSuccessorState,
    SuccessorMetricSpec,
    SuccessorStateGrowthObjective,
    SuccessorStateSample,
    SuccessorStateTracker,
)
from tests.continual.helpers import digest


def _contract(*, hold_steps: int = 3, maximum_steps: int = 6) -> SkillSuccessorState:
    return SkillSuccessorState(
        contract_id="recovery.to.locomotion-ready",
        source_skill_id="recovery",
        successor_skill_id="locomotion",
        source_policy_hash=digest("recovery"),
        successor_policy_hash=digest("locomotion"),
        body_hash=digest("body"),
        metrics=(
            SuccessorMetricSpec("pelvis_height_m", minimum=0.70),
            SuccessorMetricSpec("root_angular_speed_rad_s", maximum=0.50),
        ),
        hold_steps=hold_steps,
        maximum_transition_steps=maximum_steps,
        control_period_s=0.02,
    )


def _sample(step: int, *, height: float, angular: float) -> SuccessorStateSample:
    return SuccessorStateSample(
        step=step,
        values={"pelvis_height_m": height, "root_angular_speed_rad_s": angular},
    )


def test_successor_requires_a_contiguous_hold_window() -> None:
    tracker = SuccessorStateTracker(_contract())

    assert not tracker.update(_sample(10, height=0.72, angular=0.2)).achieved
    reset = tracker.update(_sample(11, height=0.60, angular=0.2))
    assert reset.consecutive_hold_steps == 0
    tracker.update(_sample(12, height=0.72, angular=0.2))
    tracker.update(_sample(13, height=0.73, angular=0.3))
    passed = tracker.update(_sample(14, height=0.74, angular=0.4))

    assert passed.achieved
    assert passed.entry_step == 12
    assert passed.achieved_step == 14
    assert passed.transition_time_s == pytest.approx(0.10)


def test_successor_times_out_without_fabricating_success() -> None:
    tracker = SuccessorStateTracker(_contract(hold_steps=2, maximum_steps=3))

    tracker.update(_sample(0, height=0.5, angular=1.0))
    tracker.update(_sample(1, height=0.5, angular=1.0))
    report = tracker.update(_sample(2, height=0.5, angular=1.0))

    assert report.timed_out
    assert not report.achieved
    assert set(report.failed_metrics) == {"pelvis_height_m", "root_angular_speed_rad_s"}


def test_successor_rejects_non_finite_or_wrong_order_samples() -> None:
    with pytest.raises(ValueError, match="finite"):
        SuccessorStateSample(step=0, values={"pelvis_height_m": float("nan")})

    tracker = SuccessorStateTracker(_contract())
    with pytest.raises(ValueError, match="order"):
        tracker.update(
            SuccessorStateSample(
                step=0,
                values={"root_angular_speed_rad_s": 0.1, "pelvis_height_m": 0.8},
            )
        )


def test_successor_growth_objective_values_the_next_skill() -> None:
    objective = SuccessorStateGrowthObjective(successor_value_weight=0.4)

    poor_handoff = objective.score(task_return=1.0, successor_value=0.0, transition_cost=0.2)
    good_handoff = objective.score(task_return=1.0, successor_value=0.9, transition_cost=0.1)

    assert good_handoff > poor_handoff
