"""Exact-edge graph regressions, not a ROS scene or physical geometry claim."""

import numpy as np

from rosclaw.connectors.ros.public_coverage.route_costs import RouteCosts


class EdgeGrid:
    resolution = 0.05
    origin = np.zeros(2)

    def __init__(self, isolate_preferred=True, isolate_all=False):
        self.reachable = np.ones((2, 3), dtype=bool)
        self.goal = np.array([0.12, 0.075])
        self.isolate_preferred = isolate_preferred
        self.isolate_all = isolate_all

    def xy(self, cell):
        return (np.asarray(cell[::-1]) + 0.5) * self.resolution

    def cell(self, point):
        return tuple(np.floor(np.asarray(point)[::-1] / self.resolution).astype(int))

    def inside(self, cell):
        return 0 <= cell[0] < 2 and 0 <= cell[1] < 3

    def safe_pose(self, point):
        return True

    def safe_segment(self, a, b):
        a, b = np.asarray(a), np.asarray(b)
        if np.array_equal(a, b):
            return True
        if np.array_equal(a, self.goal) or np.array_equal(b, self.goal):
            p = b if np.array_equal(a, self.goal) else a
            return any(np.allclose(p, self.xy(c), atol=1e-15, rtol=0) for c in [(1, 1), (1, 2)])
        ca, cb = self.cell(a), self.cell(b)
        if self.isolate_all or (self.isolate_preferred and (ca == (1, 1) or cb == (1, 1))):
            return False
        return max(abs(ca[0] - cb[0]), abs(ca[1] - cb[1])) <= 1

    def motion_path_safe(self, path):
        return all(self.safe_segment(a, b) for a, b in zip(path, path[1:], strict=False))


def test_unreachable_preferred_attachment_does_not_hide_reachable_same_target():
    grid = EdgeGrid()
    start = grid.xy((1, 0))
    path = RouteCosts(grid, start).path(grid.goal)
    assert path is not None
    assert np.array_equal(path[0], start)
    assert np.array_equal(path[-1], grid.goal)
    assert grid.motion_path_safe(path)
    assert not any(np.allclose(p, grid.xy((1, 1)), rtol=0, atol=1e-15) for p in path)


def test_complete_exact_disconnection_is_still_rejected():
    grid = EdgeGrid(isolate_all=True)
    assert RouteCosts(grid, grid.xy((1, 0))).path(grid.goal) is None


def test_reachable_original_preference_is_retained():
    grid = EdgeGrid(isolate_preferred=False)
    path = RouteCosts(grid, grid.xy((1, 0))).path(grid.goal)
    assert path is not None and np.array_equal(path[-1], grid.goal)
    assert any(np.allclose(p, grid.xy((1, 1)), rtol=0, atol=1e-15) for p in path)


def test_actual_sdk_grid_raster_finds_same_target_after_unreachable_preference():
    import math
    from types import SimpleNamespace

    from rosclaw.connectors.ros.public_coverage.grid import Grid

    def pose():
        return SimpleNamespace(
            position=SimpleNamespace(x=0.0, y=0.0),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        )

    res = 0.001
    n = 1850
    extra = 0.0008
    a = math.radians(94.0)
    start = np.array([0.475, 0.825])
    goal = np.array([0.925, 0.925]) + 0.03 * np.array(
        [math.cos(math.radians(5)), math.sin(math.radians(5))]
    )
    msg = SimpleNamespace(
        info=SimpleNamespace(resolution=0.05, width=37, height=37, origin=pose()), data=[0] * 1369
    )
    yy, xx = np.mgrid[:n, :n]
    px = (xx + 0.5) * res
    py = (yy + 0.5) * res
    costs = np.zeros((n, n), np.uint8)
    for s in [-1, 1]:
        x = 0.925 + s * (0.25 + extra) * math.cos(a)
        y = 0.925 + s * (0.25 + extra) * math.sin(a)
        dx = np.maximum(np.abs(px - x) - res / 2, 0)
        dy = np.maximum(np.abs(py - y) - res / 2, 0)
        costs[dx * dx + dy * dy <= 0.02**2] = 254
    nav = SimpleNamespace(
        metadata=SimpleNamespace(resolution=res, size_y=n, size_x=n, origin=pose()),
        data=costs.ravel().tolist(),
    )
    g = Grid(msg, start, clearance=0.2, tracking_margin=0.03, navigation=nav)
    rc = RouteCosts(g, start)
    path = rc.path(goal)
    assert path is not None
    assert np.array_equal(path[0], start) and np.array_equal(path[-1], goal)
    assert g.motion_path_safe(path)
    assert rc.last_path_diagnostics["fallback_attachment_used"] is True
    assert rc.last_path_diagnostics["candidate_count"] == 3


def test_missing_start_attachment_is_distinguished_from_target_disconnection():
    grid = EdgeGrid()
    grid.reachable[:] = False
    costs = RouteCosts(grid, [0.025, 0.025])
    assert costs.path(grid.goal) is None
    assert costs.last_path_diagnostics["reason"] == "start_cannot_attach_safe_grid"
