"""Physical-size regression cases for public-map continuous footprint safety."""

import unittest
from types import SimpleNamespace

import numpy as np

from rosclaw.connectors.ros.public_coverage.geometry import CircleSpace
from rosclaw.connectors.ros.public_coverage.grid import Grid, UnsafeRobotPose


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


class ContinuousGeometry(unittest.TestCase):
    def test_obstacle_square_intersects_circle_when_its_center_does_not(self):
        free = np.ones((100, 100), dtype=bool)
        free[50, 50] = False
        space = CircleSpace(free, 0.025, [0.0, 0.0], 0.20)
        pose = [1.05, 1.2625]
        self.assertGreater(np.linalg.norm(np.array([1.2625, 1.2625]) - pose), 0.20)
        self.assertFalse(space.safe_point(pose))
        self.assertTrue(space.safe_point([1.049, 1.2625]))

    def test_swept_disk_cannot_jump_over_obstacle_between_valid_endpoints(self):
        free = np.ones((80, 80), dtype=bool)
        free[40, 40] = False
        space = CircleSpace(free, 0.025, [0.0, 0.0], 0.20)
        a, b = [0.6, 1.0125], [1.4, 1.0125]
        self.assertTrue(space.safe_point(a))
        self.assertTrue(space.safe_point(b))
        self.assertFalse(space.safe_segment(a, b))
        self.assertTrue(space.safe_segment([0.6, 0.799], [1.4, 0.799]))
        self.assertFalse(space.safe_segment([0.6, 0.8], [1.4, 0.8]))

    def test_diagonal_sweep_checks_rectangle_corners(self):
        free = np.ones((80, 80), dtype=bool)
        free[40, 40] = False
        space = CircleSpace(free, 0.025, [0.0, 0.0], 0.20)
        self.assertFalse(space.safe_segment([0.6, 0.6], [1.4, 1.4]))
        self.assertTrue(space.safe_segment([0.6, 0.25], [1.4, 1.05]))

    def test_unknown_and_map_boundary_constrain_entire_body(self):
        free = np.ones((60, 60), dtype=bool)
        free[:, 40:] = False
        space = CircleSpace(free, 0.05, [0.0, 0.0], 0.21)
        self.assertFalse(space.safe_point([1.8, 1.0]))
        self.assertTrue(space.safe_point([1.789, 1.0]))
        self.assertFalse(space.safe_segment([0.2, 1.0], [1.0, 1.0]))

    def test_off_center_positions_are_checked_in_continuous_coordinates(self):
        free = np.ones((60, 60), dtype=bool)
        free[:, 40:] = False
        space = CircleSpace(free, 0.05, [0.0, 0.0], 0.21)
        # Both points lie in the same grid cell; only one has body clearance.
        self.assertEqual(int(1.78 / 0.05), int(1.795 / 0.05))
        self.assertTrue(space.safe_point([1.78, 1.0]))
        self.assertFalse(space.safe_point([1.795, 1.0]))

    def test_centers_batch_and_scalar_agree_on_wall_and_corner(self):
        free = np.ones((80, 80), dtype=bool)
        free[30:60, 40:50] = False
        space = CircleSpace(free, 0.025, [-0.2, 0.1], 0.21)
        yy, xx = np.mgrid[:80, :80]
        points = space.origin + (np.column_stack([xx.ravel(), yy.ravel()]) + 0.5) * 0.025
        np.testing.assert_array_equal(
            space.safe_points(points).reshape(free.shape), space.safe_centers()
        )
        for point in points[::43]:
            self.assertEqual(space.safe_point(point), bool(space.safe_points([point])[0]))

    def test_actual_blocked_start_cannot_be_replaced_by_nearby_free_cell(self):
        data = np.zeros((60, 60), dtype=int)
        data[:, 40:] = 100
        with self.assertRaises(UnsafeRobotPose):
            Grid(message(data), (1.805, 1.0))
        self.assertTrue(Grid(message(data), (1.78, 1.0)).safe_pose((1.78, 1.0)))

    def test_consumed_tracking_margin_allows_only_short_full_footprint_connector(self):
        data = np.zeros((60, 60), dtype=int)
        data[:, 40:] = 100
        grid = Grid(message(data), (1.795, 1.0))
        self.assertTrue(grid.physical_pose_safe((1.795, 1.0)))
        self.assertFalse(grid.safe_pose((1.795, 1.0)))
        self.assertFalse(grid.safe_path([(1.795, 1.0), (1.775, 1.0)]))
        self.assertTrue(grid.motion_path_safe([(1.795, 1.0), (1.775, 1.0), (1.5, 1.0)]))
        self.assertFalse(grid.motion_path_safe([(1.795, 1.0), (1.805, 1.0), (1.775, 1.0)]))
        self.assertFalse(grid.motion_path_safe([(1.795, 1.0), (1.795, 1.15), (1.775, 1.15)]))
        self.assertFalse(grid.motion_path_safe([(1.795, 1.0), (1.795, 1.01)]))
        self.assertFalse(grid.motion_path_safe([(1.775, 1.0), (1.795, 1.0)]))

    def test_virtual_native_center_cost_does_not_replace_actual_footprint_check(self):
        data = np.zeros((60, 60), dtype=int)
        data[:, 40:] = 100
        costs = np.zeros_like(data, dtype=np.uint8)
        costs[:, 35] = 253
        navigation = SimpleNamespace(
            data=costs.ravel().tolist(),
            metadata=SimpleNamespace(
                size_x=60, size_y=60, resolution=0.05, origin=message(data).info.origin
            ),
        )
        grid = Grid(message(data), (1.795, 1.0), navigation=navigation)
        self.assertTrue(grid.motion_path_safe([(1.795, 1.0), (1.725, 1.0)]))
        self.assertFalse(grid.safe_path([(1.725, 1.0), (1.795, 1.0)]))
        costs[:, 35] = 254
        navigation.data = costs.ravel().tolist()
        with self.assertRaises(UnsafeRobotPose):
            Grid(message(data), (1.795, 1.0), navigation=navigation)


if __name__ == "__main__":
    unittest.main()
