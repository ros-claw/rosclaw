"""Continuous circular footprint checks on public occupancy/costmap cells.

Cells represent closed squares, rather than point obstacles at their centers.
The segment test checks the full swept disk, without gaps between samples.
"""

import math

import numpy as np
from scipy import ndimage

PLANNING_RADIUS = 0.20
TRACKING_MARGIN = 0.01


class CircleSpace:
    def __init__(self, free, resolution, origin, radius):
        self.free = np.asarray(free, dtype=bool)
        self.resolution = float(resolution)
        self.origin = np.asarray(origin, dtype=float)
        self.radius = float(radius)
        if (
            self.free.ndim != 2
            or min(self.free.shape) == 0
            or not math.isfinite(self.resolution)
            or self.resolution <= 0
            or not math.isfinite(self.radius)
            or self.radius < 0
            or self.origin.shape != (2,)
            or not np.isfinite(self.origin).all()
        ):
            raise ValueError("Invalid public footprint geometry")

    def safe_centers(self):
        n = math.ceil(self.radius / self.resolution + 0.5)
        yy, xx = np.mgrid[-n : n + 1, -n : n + 1]
        dx = np.maximum(np.abs(xx) * self.resolution - self.resolution / 2, 0)
        dy = np.maximum(np.abs(yy) * self.resolution - self.resolution / 2, 0)
        kernel = dx * dx + dy * dy <= self.radius**2 + 1e-14
        return ~ndimage.binary_dilation(~self.free, structure=kernel, border_value=1)

    def safe_points(self, points):
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        height, width = self.free.shape
        relative = points - self.origin
        result = (
            np.isfinite(points).all(axis=1)
            & (relative > self.radius).all(axis=1)
            & (
                relative
                < [width * self.resolution - self.radius, height * self.resolution - self.radius]
            ).all(axis=1)
        )
        # Only finite rows reach integer conversion.
        finite = np.isfinite(points).all(axis=1)
        cells = np.zeros(points.shape, dtype=int)
        cells[finite] = np.floor(relative[finite] / self.resolution).astype(int)
        n = math.ceil(self.radius / self.resolution) + 1
        for dy in range(-n, n + 1):
            for dx in range(-n, n + 1):
                indices = cells + [dx, dy]
                inside = (
                    (indices[:, 0] >= 0)
                    & (indices[:, 0] < width)
                    & (indices[:, 1] >= 0)
                    & (indices[:, 1] < height)
                )
                blocked = np.ones(len(points), dtype=bool)
                blocked[inside] = ~self.free[indices[inside, 1], indices[inside, 0]]
                centers = self.origin + (indices + 0.5) * self.resolution
                delta = np.maximum(np.abs(centers - points) - self.resolution / 2, 0)
                result &= ~(blocked & (np.sum(delta * delta, axis=1) <= self.radius**2 + 1e-14))
        return result

    def safe_point(self, point):
        point = np.asarray(point, dtype=float)
        relative = point - self.origin
        height, width = self.free.shape
        if (
            point.shape != (2,)
            or not np.isfinite(point).all()
            or (relative <= self.radius).any()
            or (
                relative
                >= [width * self.resolution - self.radius, height * self.resolution - self.radius]
            ).any()
        ):
            return False
        lower = np.floor((relative - self.radius) / self.resolution).astype(int)
        upper = np.floor((relative + self.radius) / self.resolution).astype(int) + 1
        cells = np.argwhere(~self.free[lower[1] : upper[1], lower[0] : upper[0]])
        if not len(cells):
            return True
        centers = self.origin + (cells[:, ::-1] + lower + 0.5) * self.resolution
        delta = np.maximum(np.abs(centers - point) - self.resolution / 2, 0)
        return bool((np.sum(delta * delta, axis=1) > self.radius**2 + 1e-14).all())

    def safe_segment(self, start, finish):
        start, finish = np.asarray(start, dtype=float), np.asarray(finish, dtype=float)
        if not self.safe_point(start) or not self.safe_point(finish):
            return False
        lower = np.floor(
            (np.minimum(start, finish) - self.radius - self.origin) / self.resolution
        ).astype(int)
        upper = (
            np.floor(
                (np.maximum(start, finish) + self.radius - self.origin) / self.resolution
            ).astype(int)
            + 1
        )
        cells = np.argwhere(~self.free[lower[1] : upper[1], lower[0] : upper[0]])
        if not len(cells):
            return True
        low = self.origin + (cells[:, ::-1] + lower) * self.resolution
        high = low + self.resolution
        direction = finish - start
        length2 = float(direction @ direction)
        if length2 == 0:
            return True  # endpoint checks already passed
        # A segment intersects a rectangle when its two slab intervals overlap.
        enter = np.zeros(len(cells))
        leave = np.ones(len(cells))
        possible = np.ones(len(cells), dtype=bool)
        for axis in range(2):
            if direction[axis] == 0:
                possible &= (start[axis] >= low[:, axis]) & (start[axis] <= high[:, axis])
            else:
                a = (low[:, axis] - start[axis]) / direction[axis]
                b = (high[:, axis] - start[axis]) / direction[axis]
                enter = np.maximum(enter, np.minimum(a, b))
                leave = np.minimum(leave, np.maximum(a, b))
        if (possible & (enter <= leave)).any():
            return False
        # For disjoint convex segment/rectangle, the minimum is achieved by
        # a segment endpoint against a rectangle or a corner against the segment.
        distances = []
        for point in [start, finish]:
            delta = np.maximum(np.maximum(low - point, point - high), 0)
            distances.append(np.sum(delta * delta, axis=1))
        for xhigh, yhigh in [(False, False), (False, True), (True, False), (True, True)]:
            corner = np.column_stack(
                [high[:, 0] if xhigh else low[:, 0], high[:, 1] if yhigh else low[:, 1]]
            )
            fraction = np.clip(((corner - start) @ direction) / length2, 0, 1)
            delta = corner - (start + fraction[:, None] * direction)
            distances.append(np.sum(delta * delta, axis=1))
        return bool((np.minimum.reduce(distances) > self.radius**2 + 1e-14).all())


class RegionalCircleConstraint:
    """Additional observed-window constraint, always paired with a global space.

    Unobserved cells *inside* this local message remain blocked. Outside its
    finite extent there is no additional constraint: the global occupancy and
    global costmap still authorize every part of the path. Checks use exact
    closed cell squares and preserve rigid frame registration.
    """

    def __init__(self, free, resolution, origin, radius, map_from_local):
        registration = np.asarray(map_from_local, dtype=float)
        if registration.shape != (3,) or not np.isfinite(registration).all():
            raise ValueError("Explicit finite local-to-map registration required")
        self.translation = registration[:2]
        yaw = registration[2]
        self.rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        original = CircleSpace(free, resolution, origin, radius)
        self.low = original.origin - radius
        self.high = original.origin + np.array(original.free.shape[::-1]) * resolution + radius
        padding = math.ceil(2 * radius / resolution) + 3
        self.space = CircleSpace(
            np.pad(original.free, padding, constant_values=True),
            resolution,
            original.origin - padding * resolution,
            radius,
        )

    def local(self, points):
        return (np.asarray(points, dtype=float) - self.translation) @ self.rotation

    def safe_points(self, points):
        local = self.local(np.asarray(points, dtype=float).reshape(-1, 2))
        finite = np.isfinite(local).all(axis=1)
        inside = finite & (local >= self.low).all(axis=1) & (local <= self.high).all(axis=1)
        result = finite.copy()
        result[inside] = self.space.safe_points(local[inside])
        return result

    def safe_point(self, point):
        point = np.asarray(point, dtype=float)
        return bool(point.shape == (2,) and self.safe_points([point])[0])

    def safe_segment(self, start, finish):
        points = np.asarray([start, finish], dtype=float)
        if points.shape != (2, 2) or not np.isfinite(points).all():
            return False
        start, finish = self.local(points)
        direction = finish - start
        enter, leave = 0.0, 1.0
        for axis in range(2):
            if abs(direction[axis]) < 1e-15:
                if start[axis] < self.low[axis] or start[axis] > self.high[axis]:
                    return True  # No intersection with this observation window.
            else:
                bounds = sorted(
                    [
                        (self.low[axis] - start[axis]) / direction[axis],
                        (self.high[axis] - start[axis]) / direction[axis],
                    ]
                )
                enter, leave = max(enter, bounds[0]), min(leave, bounds[1])
                if enter > leave:
                    return True
        return self.space.safe_segment(start + enter * direction, start + leave * direction)


class FootprintPathSpace:
    """Continuous footprint checks without constructing room/frontier rasters.

    The planning and physical spaces are explicit immutable input bindings.
    Time, source identity and actuation authority remain with the caller.
    Grid uses these identical predicates; live path monitors need no room graph.
    """

    def __init__(self, *, planning_spaces, physical_spaces):
        self.planning_spaces = tuple(space for space in planning_spaces if space is not None)
        self.physical_spaces = tuple(space for space in physical_spaces if space is not None)
        if not self.planning_spaces or not self.physical_spaces:
            raise ValueError("Both planning and physical footprint spaces required")

    def safe_segment(self, start, finish):
        """Check the exact swept disk and native blocked center-cell squares."""
        return all(space.safe_segment(start, finish) for space in self.planning_spaces)

    def safe_pose(self, xy):
        return all(space.safe_point(xy) for space in self.planning_spaces)

    def physical_pose_safe(self, xy):
        return all(space.safe_point(xy) for space in self.physical_spaces)

    def motion_path_safe(self, points, max_connector=0.10):
        """Admit a short physical-safe prefix back into the planning margin.

        Actual tracking may consume the 1 cm planning margin. The prefix must
        still contain the entire native 20 cm footprint, reach a margin-safe
        waypoint within 10 cm, and leave the remainder fully margin-safe.
        Virtual center costs cannot replace this physical footprint check.
        """
        if not len(points) or not self.physical_pose_safe(points[0]):
            return False
        if self.safe_pose(points[0]):
            return self.safe_path(points)
        length = 0.0
        for index, (start, finish) in enumerate(zip(points, points[1:], strict=False), 1):
            length += float(np.linalg.norm(np.asarray(finish) - start))
            if length > max_connector + 1e-12:
                return False
            if not all(space.safe_segment(start, finish) for space in self.physical_spaces):
                return False
            if self.safe_pose(finish):
                return self.safe_path(points[index:])
        return False

    def safe_path(self, points):
        return (
            bool(len(points))
            and self.safe_pose(points[0])
            and all(self.safe_segment(a, b) for a, b in zip(points, points[1:], strict=False))
        )
