"""A clipped point cannot borrow gain or retry identity from an unexecuted lane."""

import numpy as np
import pytest

from rosclaw.connectors.ros.public_coverage.coverage_ledger import CoverageLedger
from rosclaw.connectors.ros.public_coverage.coverage_work import (
    FailedWork,
    select,
    select_phase,
    work_key,
)
from tests.connectors.ros.test_ported_coverage_work import WorkGrid

BAD = np.array([[0.4, 0.4], [0.4, 0.49], [1.5, 0.49]])
GOOD = np.array([[1.0, 1.2], [1.5, 1.2]])


def prepared(path):
    """Fixture body stops before its first corner; short motion becomes a point."""
    if len(path) == 1:
        return path
    return path[1:2] if np.linalg.norm(path[1] - path[0]) < 0.10 else path[:2]


def ledger(*, point_already_cleaned=True):
    result = CoverageLedger(resolution=0.025, brush_radius=0.15)
    result.opportunities = (
        result._sweep(BAD[0], BAD[1]) | result._sweep(BAD[1], BAD[2]) | result._sweep(*GOOD)
    )
    if point_already_cleaned:
        result.cleaned = result._sweep(BAD[1], BAD[1])
    return result


def test_positive_whole_lane_gain_cannot_select_its_already_cleaned_prepared_point():
    state = ledger()
    assert state.path_gain(BAD) and not state.path_gain(prepared(BAD))
    assert select([BAD], WorkGrid(), state, BAD[1], {}) is not None  # legacy selection
    before = (state.opportunities.copy(), state.cleaned.copy())
    assert select([BAD], WorkGrid(), state, BAD[1], {}, prepare_path=prepared) is None
    assert before == (state.opportunities, state.cleaned)  # no debt erased or fake execution


def test_other_phase_with_executable_gain_survives_a_false_full_lane_opportunity():
    state = ledger()
    phase, choice = select_phase(
        {"interior": [BAD], "boundary": [GOOD]},
        WorkGrid(),
        state,
        BAD[1],
        {},
        prepare_path=prepared,
    )
    assert phase == "boundary"
    assert np.array_equal(choice[0], GOOD)
    assert choice[1] == work_key(GOOD)
    assert choice[-1] == len(state.path_gain(GOOD))


def test_opposite_orientation_is_prepared_and_scored_independently():
    state = ledger()
    choice = select([BAD], WorkGrid(), state, BAD[1], {}, bidirectional=True, prepare_path=prepared)
    assert choice is not None
    expected = BAD[::-1][:2]
    assert np.array_equal(choice[0], expected)
    assert choice[-1] == len(state.path_gain(expected))
    assert choice[1] == work_key(expected)


def test_retry_uses_actual_prepared_target_despite_shift_beyond_old_match_radius():
    state = ledger(point_already_cleaned=False)
    retries = FailedWork()
    actual = np.array([0.4, 0.49])
    retries.record(prepared(BAD), actual, now=0)
    assert np.linalg.norm(BAD[0] - prepared(BAD)[0]) > 0.075
    assert not retries.blocked(BAD, actual, now=1)  # legacy original intent misses record
    assert select([BAD], WorkGrid(), state, actual, {}, retries=retries, now=1) is not None
    assert (
        select([BAD], WorkGrid(), state, actual, {}, retries=retries, now=1, prepare_path=prepared)
        is None
    )


@pytest.mark.parametrize(
    "adapted", [[], [[float("nan"), 0]], [[0, 0], [4, 0]], np.zeros((1025, 2)), [[1, 2, 3]]]
)
def test_invalid_prepared_work_cannot_expand_original_bounds(adapted):
    with pytest.raises(ValueError, match="Prepared work"):
        select([GOOD], WorkGrid(), ledger(), BAD[1], {}, prepare_path=lambda _: adapted)


def test_untrackable_candidate_can_be_declined_without_deleting_obligations():
    state = ledger()
    count = len(state.opportunities)
    assert select([GOOD], WorkGrid(), state, BAD[1], {}, prepare_path=lambda _: None) is None
    assert len(state.opportunities) == count


def test_preparation_does_not_mutate_shared_raw_candidates():
    path = GOOD.copy()
    original = path.copy()

    def mutate(candidate):
        candidate[0] += 0.025
        return candidate

    choice = select([path], WorkGrid(), ledger(), BAD[1], {}, prepare_path=mutate)
    assert choice is not None
    assert np.array_equal(path, original)
    assert not np.array_equal(choice[0], original)


def test_adaptation_still_requires_complete_grid_path_safety():
    class BlockedGrid(WorkGrid):
        def safe_path(self, path):
            return False

    assert select([GOOD], BlockedGrid(), ledger(), BAD[1], {}, prepare_path=prepared) is None


def test_online_entry_applies_body_preparation_and_retains_unserved_debt():
    from rosclaw.connectors.ros.public_coverage.online import AnchoredCoverage, Transform2D
    from tests.connectors.ros.test_public_online_coverage import grid

    state = AnchoredCoverage(odom_epoch="fixture", brush_radius=0.15)
    seen = []

    def decline(path):
        seen.append(path.copy())
        return None

    work = state.next_work(grid(), [0, 0], Transform2D(0, 0, 0), now=0, prepare_path=decline)
    assert seen
    assert work["kind"] == "NO_CURRENT_OBSERVED_WORK"
    assert state.anchor.opportunities and not state.anchor.cleaned
    assert work["whole_house_success"] is None
