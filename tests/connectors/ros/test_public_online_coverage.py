"""Public-only geometry, actual brush credit and persistent map registration."""

from types import SimpleNamespace as Namespace

import numpy as np
import pytest

from rosclaw.connectors.ros.public_coverage.grid import Grid
from rosclaw.connectors.ros.public_coverage.online import AnchoredCoverage, Transform2D


def grid(origin=(-1.0, -1.0), size=20, robot=(0.0, 0.0), occupied=None):
    data = np.zeros((size, size), dtype=int)
    if occupied is not None:
        data[occupied] = 100
    message = Namespace(
        info=Namespace(
            resolution=0.1,
            width=size,
            height=size,
            origin=Namespace(
                position=Namespace(x=origin[0], y=origin[1]),
                orientation=Namespace(x=0.0, y=0.0, z=0.0),
            ),
        ),
        data=data.ravel(),
    )
    return Grid(message, robot, clearance=0.2)


def sample(ledger, stamp, point, enabled=True, fresh=True):
    ledger.sample(
        stamp=stamp,
        odom_xy=point,
        cleaner=enabled,
        odom_epoch="continuous",
        odom_fresh=fresh,
        cleaner_fresh=fresh,
    )


def ledger():
    return AnchoredCoverage(odom_epoch="continuous", brush_radius=0.15)


def test_extra_planning_margin_does_not_change_the_physical_footprint():
    def build(margin):
        message = Namespace(
            info=Namespace(
                resolution=0.1,
                width=20,
                height=20,
                origin=Namespace(
                    position=Namespace(x=-1.0, y=-1.0), orientation=Namespace(x=0.0, y=0.0, z=0.0)
                ),
            ),
            data=np.zeros(400, dtype=int),
        )
        return Grid(message, [0, 0], clearance=0.20, tracking_margin=margin)

    ordinary = build(0.01)
    buffered = build(0.03)
    assert ordinary.physical_pose_safe([0.78, 0]) and buffered.physical_pose_safe([0.78, 0])
    assert ordinary.safe_pose([0.78, 0]) and not buffered.safe_pose([0.78, 0])
    with pytest.raises(ValueError, match="tracking margin"):
        build(-0.01)


def test_planning_has_no_execution_credit_and_checks_whole_footprint():
    state = ledger()
    geometry = grid()
    work = state.next_work(geometry, [0.0, 0.0], Transform2D(0, 0, 0), now=0, maximum_length=0.5)
    assert work["kind"] in ["interior", "boundary", "repair"]
    assert work["execution_credit"] is False
    assert geometry.safe_path(work["path_xy"])
    assert np.linalg.norm(np.diff(work["path_xy"], axis=0), axis=1).sum() <= 0.5 + 1e-10
    assert not state.anchor.cleaned
    assert state.status(geometry, Transform2D(0, 0, 0))["whole_house_success"] is None


def test_silent_cleaner_and_teleport_never_create_swept_bridge():
    state = ledger()
    sample(state, 1, [0.0, 0.0])
    sample(state, 2, [0.1, 0.0], fresh=False)
    sample(state, 3, [0.5, 0.0])
    sample(state, 4, [0.6, 0.0], enabled=False)
    assert (0, 10) not in state.anchor.cleaned  # x=.2625, between actual disks
    assert (
        not state.anchor._sweep([0.6, 0.0], [0.6, 0.0])
        - state.anchor._sweep([0.5, 0.0], [0.5, 0.0])
        <= state.anchor.cleaned
    )


@pytest.mark.parametrize("transform", [Transform2D(2.0, -3.0, 0), Transform2D(0.4, 0.6, 1.1)])
def test_map_corrections_reproject_same_odom_credit_and_debt(transform):
    state = ledger()
    geometry = grid()
    identity = Transform2D(0, 0, 0)
    state.observe(geometry, identity)
    sample(state, 1, [0.0, 0.0])
    sample(state, 1.5, [0.1, 0.0])
    debts = set(state.anchor.opportunities)
    cleaned = set(state.anchor.cleaned)
    view = state.view(transform)
    expected = state._keys(transform.apply(state._points(cleaned)))
    assert view.cleaned == expected
    assert state.anchor.opportunities == debts and state.anchor.cleaned == cleaned
    points = np.array([[0.1, 0.3], [-0.7, 0.9]])
    assert np.allclose(transform.inverse(transform.apply(points)), points)


def test_old_obligations_survive_unknown_map_and_restart(tmp_path):
    state = ledger()
    identity = Transform2D(0, 0, 0)
    geometry = grid()
    state.observe(geometry, identity)
    sample(state, 1, [0, 0])
    count = len(state.anchor.opportunities)
    state.failed([[0.3, 0], [0.4, 0]], [0, 0], identity, now=5)
    geometry.data[:] = -1
    status = state.status(geometry, identity)
    assert status["states"]["unobserved_again"] > 0
    assert status["discovered_obligations"] == count
    assert status["unobserved_space_extent"] == "UNKNOWN"
    path = tmp_path / "ledger.json"
    state.save(path)
    reopened = AnchoredCoverage.load(path, odom_epoch="continuous")
    assert reopened.anchor.opportunities == state.anchor.opportunities
    assert reopened.anchor.cleaned == state.anchor.cleaned
    assert reopened.retry.records[0]["attempts"] == 1
    assert reopened.last_sample is None
    with pytest.raises(ValueError, match="epoch mismatch"):
        AnchoredCoverage.load(path, odom_epoch="new-process-is-not-new-odom")


def test_failed_attempts_survive_rigid_map_change_and_keep_debt():
    state = ledger()
    identity = Transform2D(0, 0, 0)
    geometry = grid()
    state.observe(geometry, identity)
    count = len(state.anchor.opportunities)
    state.failed([[0.3, 0], [0.4, 0]], [0, 0], identity, now=5)
    assert state.status(geometry, identity)["states"]["pending_retry"] > 0
    moved = Transform2D(2, 3, 0.5)
    assert state.retry.blocked(moved.inverse(moved.apply([[0.3, 0], [0.4, 0]])), [0, 0], now=6)
    assert state.retry.blocked([[0.3, 0], [0.4, 0]], [0, 0], now=36) is False
    for t in [40, 80]:
        state.failed([[0.3, 0], [0.4, 0]], [0, 0], identity, now=t)
    assert state.retry.blocked([[0.3, 0], [0.4, 0]], [0, 0], now=200)
    assert len(state.anchor.opportunities) == count


def test_epoch_reset_and_out_of_order_data_are_explicit_failure():
    state = ledger()
    sample(state, 1, [0, 0])
    with pytest.raises(ValueError, match="Out-of-order"):
        sample(state, 0.5, [0, 0])
    with pytest.raises(ValueError, match="epoch changed"):
        state.sample(
            stamp=2,
            odom_xy=[0, 0],
            cleaner=True,
            odom_epoch="reset",
            odom_fresh=True,
            cleaner_fresh=True,
        )


def test_minimum_useful_work_defers_residuals_without_erasing_debt_or_claiming_completion():
    state = ledger()
    geometry = grid()
    identity = Transform2D(0, 0, 0)
    work = state.next_work(geometry, [0.0, 0.0], identity, now=0, minimum_new_area_m2=100.0)
    assert work["kind"] == "NO_CURRENT_OBSERVED_WORK"
    assert work["whole_house_success"] is None
    assert state.anchor.opportunities and not state.anchor.cleaned
    original_debt = set(state.anchor.opportunities)
    resumed = state.next_work(geometry, [0.0, 0.0], identity, now=1)
    assert resumed["kind"] in ["interior", "boundary", "repair"]
    assert original_debt <= state.anchor.opportunities and not state.anchor.cleaned


@pytest.mark.parametrize("area", [-1.0, float("nan"), float("inf")])
def test_invalid_useful_area_threshold_is_rejected(area):
    with pytest.raises(ValueError, match="new brush area"):
        ledger().next_work(
            grid(), [0.0, 0.0], Transform2D(0, 0, 0), now=0, minimum_new_area_m2=area
        )
