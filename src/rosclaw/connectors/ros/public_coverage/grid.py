"""Online occupancy geometry. All inputs are received SLAM grids and SDK poses."""

import math

import numpy as np
from scipy import ndimage

from rosclaw.connectors.ros.public_coverage.geometry import (
    PLANNING_RADIUS,
    TRACKING_MARGIN,
    CircleSpace,
)


class UnsafeRobotPose(ValueError):  # noqa: N818 - preserve the ported geometry API
    """The actual public pose does not have a valid full circular footprint."""


class Grid:
    def __init__(self, message, robot_xy, clearance=PLANNING_RADIUS, navigation=None):
        info = message.info
        q = info.origin.orientation
        if abs(q.x) + abs(q.y) + abs(q.z) > 1e-6:
            raise ValueError("Rotated occupancy origin requires an explicit transform")
        self.resolution = info.resolution
        self.origin = np.array([info.origin.position.x, info.origin.position.y])
        self.data = np.asarray(message.data, dtype=np.int16).reshape(info.height, info.width)
        self.free = (self.data >= 0) & (self.data <= 25)
        self.occupied = self.data >= 65
        # Unknown cells constrain navigation, while observed free margins define frontiers.
        distance = ndimage.distance_transform_edt(np.pad(self.free, 1))[1:-1, 1:-1]
        radius = clearance + TRACKING_MARGIN
        self.occupancy_space = CircleSpace(self.free, self.resolution, self.origin, radius)
        self.physical_occupancy_space = CircleSpace(
            self.free, self.resolution, self.origin, clearance
        )
        self.physical_navigation_space = None
        self.navigation_space = None
        self.navigation_centers = None
        self.safe = self.occupancy_space.safe_centers()
        if navigation is not None:
            # Also require the native planner's center-cell traversability.
            # This alone does not establish full-footprint collision freedom.
            # Online SLAM/depth occupancy alone can admit an inflated/unknown
            # endpoint, causing the complete lane to be skipped by the planner.
            meta = navigation.metadata
            if (
                abs(meta.origin.orientation.x)
                + abs(meta.origin.orientation.y)
                + abs(meta.origin.orientation.z)
                > 1e-6
            ):
                raise ValueError("Rotated native costmap requires an explicit transform")
            costs = np.asarray(navigation.data, dtype=np.uint8).reshape(meta.size_y, meta.size_x)
            native_origin = [meta.origin.position.x, meta.origin.position.y]
            self.navigation_space = CircleSpace(costs < 254, meta.resolution, native_origin, radius)
            self.physical_navigation_space = CircleSpace(
                costs < 254, meta.resolution, native_origin, clearance
            )
            self.navigation_centers = CircleSpace(costs < 253, meta.resolution, native_origin, 0.0)
            yy, xx = np.mgrid[: info.height, : info.width]
            nx = np.floor(
                (self.origin[0] + (xx + 0.5) * self.resolution - meta.origin.position.x)
                / meta.resolution
            ).astype(int)
            ny = np.floor(
                (self.origin[1] + (yy + 0.5) * self.resolution - meta.origin.position.y)
                / meta.resolution
            ).astype(int)
            inside = (nx >= 0) & (nx < meta.size_x) & (ny >= 0) & (ny < meta.size_y)
            navigable = np.zeros_like(self.safe)
            navigable[inside] = costs[ny[inside], nx[inside]] < 253
            if (
                costs.shape == self.safe.shape
                and abs(meta.resolution - self.resolution) < 1e-10
                and np.linalg.norm(np.array(native_origin) - self.origin) < 1e-10
            ):
                navigable &= self.navigation_space.safe_centers()
            else:
                points = (
                    self.origin
                    + (np.column_stack([xx.ravel(), yy.ravel()]) + 0.5) * self.resolution
                )
                navigable &= self.navigation_space.safe_points(points).reshape(self.safe.shape)
            self.safe &= navigable
        if not self.physical_pose_safe(robot_xy):
            raise UnsafeRobotPose("Actual robot footprint is blocked in public observed geometry")
        labels, _ = ndimage.label(self.safe)
        sy, sx = self.cell(robot_xy)
        if not self.inside((sy, sx)) or not self.safe[sy, sx]:
            cells = np.argwhere(self.safe)
            if not len(cells):
                raise ValueError("No observed safe robot centers")
            ordered = sorted(cells, key=lambda cell: np.linalg.norm(self.xy(cell) - robot_xy))
            connection = next(
                (
                    cell
                    for cell in ordered
                    if np.linalg.norm(self.xy(cell) - robot_xy) <= 0.10
                    and self.motion_path_safe([robot_xy, self.xy(cell)])
                ),
                None,
            )
            if connection is None:
                raise UnsafeRobotPose("Actual robot cannot connect to an observed safe center")
            sy, sx = connection
        self.reachable = labels == labels[sy, sx]
        self.centers = np.argwhere(self.reachable)
        unknown = self.data < 0
        frontier = self.free & ndimage.binary_dilation(unknown)
        nearby = ndimage.distance_transform_edt(~frontier) * self.resolution
        self.frontiers = (
            self.reachable & (nearby <= 0.40) if frontier.any() else np.zeros_like(self.safe)
        )
        # Door-separated regions from erosion, with nearest core labeling for narrow links.
        cores, count = ndimage.label(self.reachable & (distance * self.resolution >= 0.45))
        self.rooms = np.zeros_like(cores)
        if count:
            nearest = ndimage.distance_transform_edt(
                cores == 0, return_distances=False, return_indices=True
            )
            self.rooms[self.reachable] = cores[tuple(nearest)][self.reachable]

    def inside(self, point):
        y, x = point
        return 0 <= y < self.data.shape[0] and 0 <= x < self.data.shape[1]

    def cell(self, xy):
        x, y = np.floor((np.asarray(xy) - self.origin) / self.resolution).astype(int)
        return int(y), int(x)

    def xy(self, cell):
        y, x = cell
        return self.origin + (np.array([x, y]) + 0.5) * self.resolution

    def refresh_target(self, xy, max_shift=0.30):
        """Revalidate a stored lane endpoint against current observed geometry."""
        cell = self.cell(xy)
        if self.inside(cell) and self.reachable[cell] and self.safe_pose(xy):
            return np.asarray(xy, dtype=float)
        points = self.origin + (self.centers[:, ::-1] + 0.5) * self.resolution
        distances = np.linalg.norm(points - xy, axis=1)
        index = int(np.argmin(distances))
        return points[index] if distances[index] <= max_shift else None

    def safe_segment(self, start, finish):
        """Check the exact swept disk and native blocked center-cell squares."""
        return all(
            space.safe_segment(start, finish)
            for space in [self.occupancy_space, self.navigation_space, self.navigation_centers]
            if space is not None
        )

    def safe_pose(self, xy):
        return all(
            space.safe_point(xy)
            for space in [self.occupancy_space, self.navigation_space, self.navigation_centers]
            if space is not None
        )

    def physical_pose_safe(self, xy):
        return all(
            space.safe_point(xy)
            for space in [self.physical_occupancy_space, self.physical_navigation_space]
            if space is not None
        )

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
            if not all(
                space.safe_segment(start, finish)
                for space in [self.physical_occupancy_space, self.physical_navigation_space]
                if space is not None
            ):
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

    def brush_footprint(self, radius=0.15):
        """Include tangent cells despite ROS float32 resolution roundoff."""
        n = math.ceil(radius / self.resolution)
        yy, xx = np.mgrid[-n : n + 1, -n : n + 1]
        tolerance = np.finfo(np.float32).eps * radius
        return (xx * self.resolution) ** 2 + (yy * self.resolution) ** 2 <= (
            radius + tolerance
        ) ** 2

    def estimated_cleaned(self, samples, radius=0.15):
        cleaned = np.zeros_like(self.free)
        for start, finish in zip(samples, samples[1:], strict=False):
            if not (start[2] and finish[2]):
                continue
            length = float(np.linalg.norm(np.asarray(finish[:2]) - start[:2]))
            if length > 0.20:
                continue
            for t in np.linspace(0, 1, max(2, math.ceil(length / 0.025) + 1)):
                xy = np.asarray(start[:2]) * (1 - t) + np.asarray(finish[:2]) * t
                cy, cx = self.cell(xy)
                n = math.ceil(radius / self.resolution) + 1
                y0, y1 = max(0, cy - n), min(self.data.shape[0], cy + n + 1)
                x0, x1 = max(0, cx - n), min(self.data.shape[1], cx + n + 1)
                yy, xx = np.mgrid[y0:y1, x0:x1]
                cleaned[y0:y1, x0:x1] |= (
                    self.origin[0] + (xx + 0.5) * self.resolution - xy[0]
                ) ** 2 + (self.origin[1] + (yy + 0.5) * self.resolution - xy[1]) ** 2 <= radius**2
        return cleaned & self.free

    def frontier_targets(self, robot_xy, excluded=()):
        """Rank spaced safe viewpoints by nearby unknown area and travel distance."""
        cells = np.argwhere(self.frontiers)
        if not len(cells):
            return []
        xy = self.origin + (cells[:, ::-1] + 0.5) * self.resolution
        distances = np.linalg.norm(xy - robot_xy, axis=1)
        keep = distances >= 0.50
        for target in excluded:
            keep &= np.linalg.norm(xy - target, axis=1) > 0.45
        cells, xy, distances = cells[keep], xy[keep], distances[keep]
        if not len(cells):
            return []
        radius = math.ceil(1.0 / self.resolution)
        yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
        disk = (xx * self.resolution) ** 2 + (yy * self.resolution) ** 2 <= 1.0
        gain = ndimage.convolve(
            (self.data < 0).astype(np.int32), disk.astype(np.int32), mode="constant", cval=0
        )
        scores = gain[tuple(cells.T)] * self.resolution**2 / (1.0 + distances)
        order = np.argsort(-scores, kind="stable")
        chosen = []
        for index in order:
            if gain[tuple(cells[index])] * self.resolution**2 < 0.10:
                continue
            target = xy[index]
            if all(np.linalg.norm(target - other) >= 0.65 for other in chosen):
                chosen.append(target)
        return chosen

    def repair_target(self, missed, robot_xy, attempted=()):
        """Preserve nearby cleaning opportunities after an attempted viewpoint.

        Exploration spacing does not imply that a surrounding brush area was
        cleaned. Exclude only each attempted center, and let the measured
        cleaned mask determine which neighboring centers still offer gain.
        """
        scores = ndimage.convolve(missed.astype(np.int16), self.brush_footprint().astype(np.int16))
        scores[~self.reachable] = 0
        for target in attempted:
            cell = self.cell(target)
            if self.inside(cell):
                scores[cell] = 0
        if int(scores.max()) < 1:
            return None
        cells = np.argwhere(scores >= 1)
        xy = self.origin + (cells[:, ::-1] + 0.5) * self.resolution
        # One metre amortizes turning/settling between goals. Nearby useful
        # brush work can be cheaper than a distant maximum-gain viewpoint.
        utility = scores[tuple(cells.T)] / (1.0 + np.linalg.norm(xy - robot_xy, axis=1))
        return xy[np.argmax(utility)]

    def lanes(self, spacing=0.20, robot_xy=None):
        """Visit observed room regions consecutively, with serpentine rows within each."""
        lanes = []
        step = max(1, round(spacing / self.resolution))
        regions = list(np.unique(self.rooms[self.reachable]))
        current = np.asarray(robot_xy) if robot_xy is not None else self.xy(self.centers[0])
        while regions:
            region = min(
                regions,
                key=lambda label: np.linalg.norm(
                    self.xy(np.argwhere(self.reachable & (self.rooms == label)).mean(axis=0))
                    - current
                ),
            )
            regions.remove(region)
            mask = self.reachable & (self.rooms == region)
            room_lanes = []
            for index, y in enumerate(range(0, self.data.shape[0], step)):
                xs = np.flatnonzero(mask[y])
                if not len(xs):
                    continue
                splits = np.split(xs, np.flatnonzero(np.diff(xs) > 1) + 1)
                pairs = [((y, int(s[0])), (y, int(s[-1]))) for s in splits]
                if index % 2:
                    pairs = [(b, a) for a, b in pairs[::-1]]
                room_lanes.extend(pairs)
            if not room_lanes:
                continue
            if np.linalg.norm(self.xy(room_lanes[-1][1]) - current) < np.linalg.norm(
                self.xy(room_lanes[0][0]) - current
            ):
                room_lanes = [(b, a) for a, b in room_lanes[::-1]]
            lanes.extend(room_lanes)
            current = self.xy(room_lanes[-1][1])
        return lanes
