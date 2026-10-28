"""A changed current component cannot erase observed unfinished cleaning."""

import unittest
from types import SimpleNamespace

import numpy as np

from rosclaw.connectors.ros.public_coverage.coverage_ledger import CoverageLedger
from rosclaw.connectors.ros.public_coverage.grid import Grid


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


class PersistentCoverage(unittest.TestCase):
    def test_disconnection_and_unknown_do_not_erase_discovered_work(self):
        data = np.full((70, 100), 100, dtype=int)
        data[5:65, 5:95] = 0
        ledger = CoverageLedger()
        ledger.observe(Grid(message(data), (1.0, 1.0)), [])
        original = len(ledger.opportunities)
        data[5:65, 50:95] = -1
        ledger.observe(Grid(message(data), (1.0, 1.0)), [])
        self.assertEqual(len(ledger.opportunities), original)
        self.assertEqual(ledger.summary()["estimated_coverage"], 0.0)

    def test_only_explicit_occupied_cells_revise_floor_opportunities(self):
        data = np.zeros((60, 60), dtype=int)
        ledger = CoverageLedger()
        ledger.observe(Grid(message(data), (1.0, 1.0)), [])
        before = len(ledger.opportunities)
        data[30:35, 30:35] = 100
        ledger.observe(Grid(message(data), (1.0, 1.0)), [])
        self.assertLess(len(ledger.opportunities), before)
        keys, values = ledger._classification(Grid(message(data), (1.0, 1.0)), ledger.opportunities)
        self.assertFalse((values >= 65).any())

    def test_stationary_cleaner_and_continuous_sweep_are_incremental(self):
        grid = Grid(message(np.zeros((60, 60), dtype=int)), (1.0, 1.0))
        ledger = CoverageLedger()
        ledger.observe(grid, [(1.0, 1.0, True)])
        initial = ledger.summary()["estimated_cleaned_cells"]
        self.assertGreater(initial, 0)
        samples = [(1.0, 1.0, True), (1.15, 1.0, True)]
        ledger.observe(grid, samples)
        after = ledger.summary()["estimated_cleaned_cells"]
        self.assertGreater(after, initial)
        ledger.observe(grid, samples)
        self.assertEqual(ledger.summary()["estimated_cleaned_cells"], after)
        ledger.observe(grid, samples + [(2.0, 1.0, False)])
        self.assertEqual(ledger.summary()["estimated_cleaned_cells"], after)

    def test_cropped_shifted_map_retains_old_world_coordinates(self):
        ledger = CoverageLedger()
        ledger.observe(Grid(message(np.zeros((80, 100), dtype=int)), (1.0, 1.0)), [])
        original = set(ledger.opportunities)
        cropped = message(np.zeros((40, 40), dtype=int))
        cropped.info.origin.position.x = 0.5
        cropped.info.origin.position.y = 0.5
        ledger.observe(Grid(cropped, (1.0, 1.0)), [])
        self.assertEqual(ledger.opportunities, original)

    def test_teleport_does_not_paint_between_actual_brush_disks(self):
        ledger = CoverageLedger()
        grid = Grid(message(np.zeros((80, 100), dtype=int)), (1.0, 1.0))
        ledger.observe(grid, [(1.0, 1.0, True), (3.0, 1.0, True)])
        midpoint = (int(1.0 / ledger.resolution), int(2.0 / ledger.resolution))
        self.assertNotIn(midpoint, ledger.cleaned)

    def test_stale_boundary_cannot_reseed_explicitly_reclassified_floor(self):
        data = np.zeros((80, 80), dtype=int)
        ledger = CoverageLedger()
        witness = np.array([[0.6125, 0.6125]])
        ledger.observe(Grid(message(data, resolution=0.025), (1.3, 1.3)), [], extra_paths=[witness])
        victim = (25, 24)
        self.assertIn(victim, ledger.opportunities)
        data[25, 24] = 100
        data[24, 24] = 100
        ledger.observe(Grid(message(data, resolution=0.025), (1.3, 1.3)), [])
        self.assertFalse(
            victim in ledger.opportunities,
            "Reclassified floor was seeded from an obsolete boundary center",
        )
        retained = set(ledger.opportunities)
        data[25, 24] = 0
        grid = Grid(message(data, resolution=0.025), (1.3, 1.3))
        self.assertFalse(grid.safe_pose(witness[0]))
        ledger.observe(grid, [])
        self.assertFalse(
            victim in ledger.opportunities,
            "Reclassified floor was seeded from an obsolete boundary center",
        )
        self.assertTrue(retained <= ledger.opportunities)
        self.assertIn(tuple(witness[0]), ledger.extra_centers)
        self.assertFalse(ledger.cleaned)


if __name__ == "__main__":
    unittest.main()
