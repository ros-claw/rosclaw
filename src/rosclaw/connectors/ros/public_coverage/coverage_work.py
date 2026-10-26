"""Replan unfinished brush work from public maps and executed coverage only."""

import hashlib
import math

import numpy as np

from rosclaw.connectors.ros.public_coverage.coverage_paths import densify
from rosclaw.connectors.ros.public_coverage.coverage_variants import prefix


def chunks(path, maximum=3.0):
    """Partition a polyline by arc length, keeping every turn in its own chunk."""
    rest = np.asarray(path, float)
    if not math.isfinite(maximum) or maximum <= 0:
        raise ValueError("Invalid work chunk length")
    while len(rest) > 1:
        part = prefix(rest, maximum)
        if len(part) < 2:
            return
        yield part
        lengths = np.linalg.norm(np.diff(rest, axis=0), axis=1)
        total = float(lengths.sum())
        if total <= maximum + 1e-10:
            return
        end = int(np.searchsorted(np.cumsum(lengths), maximum, side="right")) + 1
        rest = np.vstack([part[-1], rest[end:]])


def corners(path):
    path = np.asarray(path, float)
    delta = np.diff(path, axis=0)
    length = np.linalg.norm(delta, axis=1)
    indices = []
    for i in range(1, len(path) - 1):
        if length[i - 1] > 1e-9 and length[i] > 1e-9:
            cosine = np.clip(delta[i - 1] @ delta[i] / length[i - 1] / length[i], -1, 1)
            if math.acos(cosine) > 0.3:
                indices.append(i)
    return path[indices]


def visited_corners(path, samples, tolerance=0.10):
    """Ordered public pose evidence; action success alone does not prove turns."""
    targets = corners(path)
    samples = np.asarray(samples, float).reshape(-1, 2)
    cursor = 0
    for target in targets:
        indices = np.flatnonzero(np.linalg.norm(samples[cursor:] - target, axis=1) <= tolerance)
        if not len(indices):
            return False
        cursor += int(indices[0]) + 1
    return True


def geometry_key(grid):
    h = hashlib.sha256(grid.data.tobytes())
    h.update(grid.reachable.tobytes())
    h.update(np.asarray([*grid.origin, grid.resolution], float).tobytes())
    for name in ["navigation_space", "navigation_centers"]:
        space = getattr(grid, name, None)
        h.update(name.encode())
        if space is not None:
            h.update(np.asarray(space.free.shape, np.int64).tobytes())
            h.update(space.free.tobytes())
            h.update(np.asarray([*space.origin, space.resolution], float).tobytes())
    return h.hexdigest()


def work_key(path):
    return tuple(np.rint(np.asarray(path).ravel() / 0.025).astype(int))


def entry_connector(grid, robot_xy, target):
    """Use a direct native-controller connector only when its whole body is safe."""
    path = np.asarray([robot_xy, target], float)
    return path if grid.safe_path(path) else None


def margin_reentry(grid, robot_xy):
    """Return to a legal center using the existing <=10cm physical-safe prefix.

    No occupancy or cost is cleared. The shared motion policy verifies the
    20cm disk throughout the prefix and the full planning margin thereafter.
    """
    robot_xy = np.asarray(robot_xy, float)
    if grid.safe_pose(robot_xy):
        return None
    centers = grid.origin + (grid.centers[:, ::-1] + 0.5) * grid.resolution
    distances = np.linalg.norm(centers - robot_xy, axis=1)
    for index in np.argsort(distances):
        if distances[index] > 0.10:
            break
        path = densify([robot_xy, centers[index]], step=0.0125)
        if grid.motion_path_safe(path):
            return path
    return None


class FailedWork:
    """Spatial retry cooldown survives map hash and endpoint jitter.

    Work remains in the coverage ledger. Moving to another entrance changes
    the context; a small SLAM revision alone does not erase failed attempts.
    """

    def __init__(self, cooldown=30.0, maximum_attempts=3):
        self.cooldown = cooldown
        self.maximum_attempts = maximum_attempts
        self.records = []

    def record(self, path, robot_xy, now):
        target = np.asarray(path[0], float)
        robot = np.asarray(robot_xy, float)
        record = self._find(target, robot)
        if record is None:
            record = {"target": target.copy(), "robot": robot.copy(), "attempts": 0, "last": now}
            self.records.append(record)
        record.update(attempts=record["attempts"] + 1, last=now)

    def _find(self, target, robot):
        return next(
            (
                r
                for r in self.records
                if np.linalg.norm(r["target"] - target) <= 0.075
                and np.linalg.norm(r["robot"] - robot) <= 0.20
            ),
            None,
        )

    def blocked(self, path, robot_xy, now):
        record = self._find(np.asarray(path[0]), np.asarray(robot_xy))
        return record is not None and (
            record["attempts"] >= self.maximum_attempts or now - record["last"] < self.cooldown
        )


def work_score(path, gain, robot_xy, transit_cost=None):
    """Relative executed brush yield per route cost, not a calibrated SOC rate."""
    path = np.asarray(path, float)
    length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    transit = (
        float(np.linalg.norm(path[0] - robot_xy))
        if transit_cost is None
        else float(transit_cost(path[0]))
    )
    if not math.isfinite(transit):
        return -math.inf
    if transit < 0:
        raise ValueError("Negative transit estimate")
    return gain / (length + transit + 0.20 * len(corners(path)) + 0.10)


def repair_options(
    grid,
    ledger,
    robot_xy,
    maximum=12,
    transit_distances=None,
    extra_paths=(),
    attempted=(),
    transit_cost=None,
):
    """Offer spaced public brush viewpoints alongside continuous work.

    These are proposed service locations, never executed cleaning credit.
    The ledger and full footprint admission still determine actual progress.
    """
    cells = grid.centers
    xy = grid.origin + (cells[:, ::-1] + 0.5) * grid.resolution
    distances = (
        np.linalg.norm(xy - robot_xy, axis=1)
        if transit_distances is None
        else np.asarray(transit_distances, float)
    )
    if distances.shape != (len(cells),) or np.isnan(distances).any() or (distances < 0).any():
        raise ValueError("Invalid center transit estimates")
    if extra_paths:
        extra = np.unique(np.vstack(extra_paths), axis=0)
        extra_distances = (
            np.linalg.norm(extra - robot_xy, axis=1)
            if transit_cost is None
            else np.asarray([transit_cost(point) for point in extra])
        )
        if np.isnan(extra_distances).any() or (extra_distances < 0).any():
            raise ValueError("Invalid boundary transit estimates")
        xy = np.vstack([xy, extra])
        distances = np.concatenate([distances, extra_distances])
    gains = ledger.point_gains(xy)
    for target in attempted:
        gains[np.linalg.norm(xy - target, axis=1) <= 0.005] = 0
    utility = gains / (0.10 + distances)
    options = []
    for _ in range(maximum):
        if not len(utility) or float(utility.max()) <= 0:
            break
        index = int(np.argmax(utility))
        target = xy[index]
        if grid.safe_pose(target):
            options.append(target.reshape(1, 2))
        utility[np.linalg.norm(xy - target, axis=1) < 0.15] = 0
    return options


def select(
    paths,
    grid,
    ledger,
    robot_xy,
    failed,
    maximum=3.0,
    service=None,
    minimum_cells=1,
    retries=None,
    now=0.0,
    transit_cost=None,
    bidirectional=False,
):
    """Rank actual remaining brush opportunity, never commanded endpoint count."""
    version = geometry_key(grid)
    best = None
    best_score = -1.0
    for path in paths:
        # A repair location has no sweep arc. Native navigation and actual
        # brush observations must still service it before it can be completed.
        for part in [np.asarray(path, float)] if len(path) == 1 else chunks(path, maximum):
            if service:
                service()
            eligible = []
            for oriented in [part, part[::-1]] if bidirectional and len(part) > 1 else [part]:
                key = work_key(oriented)
                if retries is not None and retries.blocked(oriented, robot_xy, now):
                    continue
                if failed.get(key) != version:
                    eligible.append((oriented, key))
            if not eligible or not grid.safe_path(part):
                continue
            gain = ledger.path_gain(part)
            if len(gain) < minimum_cells:
                continue
            for oriented, key in eligible:
                score = work_score(oriented, len(gain), robot_xy, transit_cost)
                if score > best_score:
                    best = (oriented, key, version, len(gain))
                    best_score = score
    return best


def select_phase(
    phases,
    grid,
    ledger,
    robot_xy,
    failed,
    service=None,
    minimum_cells=1,
    retries=None,
    now=0.0,
    transit_cost=None,
    bidirectional=False,
    maximum=3.0,
):
    """Boundary work must not starve behind diminishing interior opportunities."""
    candidates = []
    for phase, paths in phases.items():
        choice = select(
            paths,
            grid,
            ledger,
            robot_xy,
            failed,
            maximum=maximum,
            service=service,
            minimum_cells=minimum_cells,
            retries=retries,
            now=now,
            transit_cost=transit_cost,
            bidirectional=bidirectional,
        )
        if choice is not None:
            candidates.append((phase, choice))
    return (
        max(
            candidates, key=lambda item: work_score(item[1][0], item[1][-1], robot_xy, transit_cost)
        )
        if candidates
        else None
    )


class WorkProgress:
    """Stop zero-new-area cycling even when the robot is physically moving."""

    def __init__(self, active_seconds, cleaned_cells, timeout=120.0, minimum_cells=1):
        self.since = active_seconds
        self.cells = cleaned_cells
        self.timeout = timeout
        self.minimum_cells = minimum_cells

    def stalled(self, active_seconds, cleaned_cells):
        if cleaned_cells - self.cells >= self.minimum_cells:
            self.cells = cleaned_cells
            self.since = active_seconds
        return active_seconds - self.since >= self.timeout
