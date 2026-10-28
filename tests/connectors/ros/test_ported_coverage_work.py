"""Preserve unexecuted work across map revisions without counting planned area."""

import unittest
from types import SimpleNamespace

import numpy as np

from rosclaw.connectors.ros.public_coverage.coverage_ledger import CoverageLedger
from rosclaw.connectors.ros.public_coverage.coverage_work import (
    FailedWork,
    WorkProgress,
    chunks,
    corners,
    entry_connector,
    geometry_key,
    margin_reentry,
    repair_options,
    select,
    select_phase,
    visited_corners,
    work_key,
)
from rosclaw.connectors.ros.public_coverage.geometry import CircleSpace


def message(data, resolution=0.05):
    height, width = data.shape
    return SimpleNamespace(
        data=data.ravel().tolist(),
        info=SimpleNamespace(
            width=width,
            height=height,
            resolution=resolution,
            origin=SimpleNamespace(
                position=SimpleNamespace(x=0.0, y=0.0),
                orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
        ),
    )


class WorkGrid:
    resolution = 0.05
    origin = np.array([0.0, 0.0])
    data = np.zeros((40, 40), np.int16)
    reachable = np.ones((40, 40), bool)

    def safe_path(self, path):
        return True


class CoverageWork(unittest.TestCase):
    def test_chunks_keep_all_bends_and_do_not_duplicate_or_drop_distance(self):
        path = np.array([[0, 0], [0.6, 0], [0.6, 0.25], [0, 0.25]])
        result = list(chunks(path, 0.7))
        self.assertAlmostEqual(
            sum(np.linalg.norm(np.diff(p, axis=0), axis=1).sum() for p in result), 1.45
        )
        self.assertEqual(sum(len(corners(p)) for p in result), 2)
        for a, b in zip(result, result[1:], strict=False):
            np.testing.assert_array_equal(a[-1], b[0])

    def test_action_endpoint_success_does_not_prove_turns_were_executed(self):
        path = [[0, 0], [1, 0], [1, 1], [0, 1]]
        self.assertFalse(visited_corners(path, [[0, 0], [0, 1]]))
        self.assertFalse(visited_corners(path, [[1, 1], [1, 0]]))
        self.assertTrue(visited_corners(path, [[0, 0], [1, 0], [1, 1], [0, 1]]))

    def test_canceled_plan_retains_work_and_completed_brush_removes_it(self):
        ledger = CoverageLedger()
        path = np.array([[0.4, 0.4], [1.0, 0.4]])
        ledger.opportunities = ledger._sweep(*path)
        count = len(ledger.opportunities)
        choice = select([path], WorkGrid(), ledger, [0.4, 0.4], {})
        self.assertEqual(choice[-1], count)
        self.assertFalse(ledger.cleaned)
        ledger._integrate([(0.4, 0.4, True), (0.5, 0.4, True)])
        choice = select([path], WorkGrid(), ledger, [0.5, 0.4], {})
        self.assertGreater(choice[-1], 0)
        self.assertLess(choice[-1], count)
        ledger.cleaned.update(ledger.opportunities)
        self.assertIsNone(select([path], WorkGrid(), ledger, [0.5, 0.4], {}))

    def test_failed_geometry_can_be_retried_after_public_map_revision(self):
        grid = WorkGrid()
        ledger = CoverageLedger()
        path = np.array([[0.4, 0.4], [1.0, 0.4]])
        ledger.opportunities = ledger._sweep(*path)
        failed = {work_key(path): geometry_key(grid)}
        self.assertIsNone(select([path], grid, ledger, [0.4, 0.4], failed))
        grid.data = grid.data.copy()
        grid.data[0, 0] = 100
        self.assertIsNotNone(select([path], grid, ledger, [0.4, 0.4], failed))

    def test_moving_without_new_brush_area_does_not_reset_work_watchdog(self):
        progress = WorkProgress(0, 10)
        self.assertFalse(progress.stalled(110, 10))
        self.assertTrue(progress.stalled(120, 10))
        self.assertFalse(progress.stalled(125, 11))
        self.assertFalse(progress.stalled(240, 11))
        self.assertTrue(progress.stalled(245, 11))

    def test_unexecuted_boundary_is_not_starved_by_almost_clean_interior(self):
        ledger = CoverageLedger()
        interior = np.array([[0.4, 0.4], [1.0, 0.4]])
        boundary = interior + [0, 0.6]
        inside = ledger._sweep(*interior)
        edge = ledger._sweep(*boundary)
        ledger.opportunities = inside | edge
        ledger.cleaned = set(list(inside)[2:])
        phase, candidate = select_phase(
            {"interior": [interior], "boundary": [boundary]}, WorkGrid(), ledger, [0.4, 0.4], {}
        )
        self.assertEqual(phase, "boundary")
        self.assertGreater(candidate[-1], 2)

    def test_small_chunk_handoff_preserves_remaining_opportunities(self):
        ledger = CoverageLedger()
        path = np.array([[0.4, 0.4], [1.0, 0.4]])
        ledger.opportunities = ledger._sweep(*path)
        ledger.cleaned = set(list(ledger.opportunities)[2:])
        before = ledger.summary().copy()
        self.assertIsNone(
            select_phase({"interior": [path]}, WorkGrid(), ledger, [0.4, 0.4], {}, minimum_cells=40)
        )
        self.assertEqual(ledger.summary(), before)

    def test_entry_connector_checks_whole_sweep_not_only_clear_endpoints(self):
        free = np.ones((80, 80), bool)
        free[40, 40] = False
        space = CircleSpace(free, 0.025, [0, 0], 0.21)
        grid = SimpleNamespace(safe_path=lambda path: space.safe_segment(*path))
        a, b = [0.6, 0.8], [1.4, 0.8]
        self.assertTrue(space.safe_point(a))
        self.assertTrue(space.safe_point(b))
        self.assertIsNone(entry_connector(grid, a, b))
        self.assertIsNotNone(entry_connector(grid, [0.6, 0.75], [1.4, 0.75]))

    def test_spatial_cooldown_survives_hash_and_endpoint_jitter(self):
        retries = FailedWork()
        path = np.array([[0.4, 0.4], [1.0, 0.4]])
        retries.record(path, [0.3, 0.3], 10)
        self.assertTrue(retries.blocked(path + 0.01, [0.31, 0.3], 20))
        self.assertFalse(retries.blocked(path, [0.3, 0.3], 40))
        retries.record(path, [0.3, 0.3], 40)
        retries.record(path, [0.3, 0.3], 70)
        self.assertTrue(retries.blocked(path, [0.3, 0.3], 1000))
        self.assertFalse(retries.blocked(path, [0.6, 0.3], 1000))

    def test_retry_cooldown_selects_other_work_without_erasing_ledger(self):
        ledger = CoverageLedger()
        path = np.array([[0.4, 0.4], [1.0, 0.4]])
        other = path + [0, 0.6]
        ledger.opportunities = ledger._sweep(*path) | ledger._sweep(*other)
        before = ledger.summary()
        retries = FailedWork()
        retries.record(path, [0.4, 0.4], 10)
        choice = select([path, other], WorkGrid(), ledger, [0.4, 0.4], {}, retries=retries, now=11)
        np.testing.assert_array_equal(choice[0], other)
        self.assertEqual(before, ledger.summary())

    def test_tiny_trickle_does_not_extend_no_effective_area_watchdog(self):
        progress = WorkProgress(0, 100, minimum_cells=40)
        self.assertFalse(progress.stalled(110, 139))
        self.assertTrue(progress.stalled(120, 139))
        self.assertFalse(progress.stalled(125, 140))

    def test_margin_reentry_obeys_existing_prefix_and_rejects_distant_escape(self):
        from rosclaw.connectors.ros.public_coverage.grid import Grid

        grid = Grid(message(np.zeros((80, 80), dtype=int)), [0.205, 1.0])
        path = margin_reentry(grid, [0.205, 1.0])
        self.assertIsNotNone(path)
        self.assertTrue(grid.motion_path_safe(path))
        self.assertTrue(grid.safe_pose(path[-1]))
        self.assertLessEqual(np.linalg.norm(path[-1] - path[0]), 0.10)
        grid.centers = np.array([[30, 30]])
        self.assertIsNone(margin_reentry(grid, [0.205, 1.0]))

    def test_point_repair_competes_without_granting_unexecuted_credit(self):
        ledger = CoverageLedger()
        stroke = np.array([[0.4, 0.4], [1.0, 0.4]])
        point = np.array([[0.4, 1.0]])
        ledger.opportunities = ledger._sweep(*stroke) | ledger._sweep(point[0], point[0])
        ledger.cleaned = ledger._sweep(*stroke)
        before = ledger.summary()
        phase, choice = select_phase(
            {"interior": [stroke], "repair": [point]}, WorkGrid(), ledger, [0.4, 0.9], {}
        )
        self.assertEqual(phase, "repair")
        self.assertEqual(len(choice[0]), 1)
        self.assertEqual(before, ledger.summary())
        ledger._integrate([(0.4, 1.0, True)])
        self.assertIsNone(select_phase({"repair": [point]}, WorkGrid(), ledger, [0.4, 1.0], {}))

    def test_repair_options_are_spaced_safe_and_disappear_after_actual_cleaning(self):
        from rosclaw.connectors.ros.public_coverage.grid import Grid

        grid = Grid(message(np.zeros((40, 40), dtype=int)), [1.0, 1.0])
        ledger = CoverageLedger()
        ledger.observe(grid, [])
        before = ledger.summary()
        options = repair_options(grid, ledger, [1.0, 1.0], maximum=6)
        self.assertEqual(len(options), 6)
        self.assertEqual(before, ledger.summary())
        for i, path in enumerate(options):
            self.assertTrue(grid.safe_path(path))
            for other in options[:i]:
                self.assertGreaterEqual(np.linalg.norm(path[0] - other[0]), 0.15 - 1e-12)
        ledger.cleaned.update(ledger.opportunities)
        self.assertEqual(repair_options(grid, ledger, [1.0, 1.0]), [])
