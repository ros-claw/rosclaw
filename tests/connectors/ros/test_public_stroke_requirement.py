"""A point repair cannot satisfy an immutable continuous-stroke requirement."""

from dataclasses import replace

import numpy as np
import pytest

from rosclaw.connectors.ros.public_coverage.coverage_ledger import CoverageLedger
from rosclaw.connectors.ros.public_coverage.coverage_work import select
from rosclaw.connectors.ros.public_coverage.online import AnchoredCoverage, Transform2D
from rosclaw.kernel import ActionState
from tests.connectors.ros.test_ported_coverage_work import WorkGrid
from tests.connectors.ros.test_public_coverage_execution import setup_executor
from tests.connectors.ros.test_public_online_coverage import grid


def test_high_scoring_point_repair_does_not_starve_required_continuous_work():
    ledger = CoverageLedger()
    robot = np.array([0.4, 0.4])
    point = robot.reshape(1, 2)
    stroke = np.array([robot, [1.0, 0.4]])
    ledger.opportunities = ledger._sweep(*stroke)
    debt = set(ledger.opportunities)
    ordinary = select([point, stroke], WorkGrid(), ledger, robot, {})
    required = select([point, stroke], WorkGrid(), ledger, robot, {}, continuous_only=True)
    assert len(ordinary[0]) == 1  # Original yield/cost ranking genuinely prefers this.
    np.testing.assert_array_equal(required[0], stroke)
    assert ledger.opportunities == debt and not ledger.cleaned


def test_requirement_applies_after_controller_clipping_and_keeps_debt():
    ledger = CoverageLedger()
    stroke = np.array([[0.4, 0.4], [1.0, 0.4]])
    ledger.opportunities = ledger._sweep(*stroke)
    debt = set(ledger.opportunities)
    assert (
        select(
            [stroke],
            WorkGrid(),
            ledger,
            stroke[0],
            {},
            prepare_path=lambda path: path[:1],
            continuous_only=True,
        )
        is None
    )
    assert select([stroke[:1]], WorkGrid(), ledger, stroke[0], {}, continuous_only=True) is None
    assert ledger.opportunities == debt and not ledger.cleaned


@pytest.mark.parametrize("invalid", [1, None, "true"])
def test_stroke_requirement_is_an_explicit_boolean(invalid):
    state = AnchoredCoverage(odom_epoch="fixture", brush_radius=0.15)
    with pytest.raises(ValueError, match="Explicit continuous"):
        state.next_work(
            grid(), [0, 0], Transform2D(0, 0, 0), now=0, require_continuous_stroke=invalid
        )


def test_executor_passes_remaining_original_stroke_goal_without_changing_counts(monkeypatch):
    _, executor, port, action = setup_executor()
    requirements = []

    def propose(*args, **kwargs):
        requirements.append(kwargs["require_continuous_stroke"])
        x, y = port.xy
        return {"kind": "WORK", "path_xy": [[x, y], [x, y + 0.5]]}

    monkeypatch.setattr(executor.ledger, "next_work", propose)
    action = replace(
        action,
        arguments={"maximum_work": 3, "minimum_completed_work": 2, "minimum_completed_strokes": 1},
    )
    deadline = action.deadline_at
    result = executor(action)
    assert result.final_state is ActionState.COMPLETED, result.errors
    assert requirements == [True, False]
    assert result.dispatch_result["completed_work"] == 2
    assert result.dispatch_result["completed_continuous_strokes"] == 2
    assert action.deadline_at == deadline and action.arguments["minimum_completed_strokes"] == 1
    assert not port.brush
