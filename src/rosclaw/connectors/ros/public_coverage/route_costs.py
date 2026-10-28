"""Public-grid transit estimates for work ordering; never authorize motion.

Graph costs use only the current public geometry snapshot.
Graph distances estimate transit around observed walls. Actual motion must still
use the native planner and exact swept-footprint verification.
"""

import heapq
import math

import numpy as np


class RouteCosts:
    def __init__(self, grid, robot_xy, service=None):
        self.grid = grid
        self.robot_xy = np.asarray(robot_xy, float)
        self.distance = np.full(grid.reachable.shape, np.inf)
        self.seed = None
        cells = np.argwhere(grid.reachable)
        if not len(cells):
            return
        points = grid.origin + (cells[:, ::-1] + 0.5) * grid.resolution
        offsets = np.linalg.norm(points - self.robot_xy, axis=1)
        for index in np.argsort(offsets):
            if offsets[index] > 0.10 + 1e-12:
                break
            cell = tuple(cells[index])
            if grid.motion_path_safe([self.robot_xy, points[index]]):
                self.seed = cell
                self.distance[cell] = offsets[index]
                break
        if self.seed is None:
            return
        queue = [(self.distance[self.seed], *self.seed)]
        count = 0
        height, width = grid.reachable.shape
        while queue:
            distance, y, x = heapq.heappop(queue)
            if distance > self.distance[y, x]:
                continue
            count += 1
            if service and count % 2048 == 0:
                service()
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                ny, nx = y + dy, x + dx
                if not (0 <= ny < height and 0 <= nx < width):
                    continue
                if not grid.reachable[ny, nx]:
                    continue
                if dy and dx and not (grid.reachable[y, nx] and grid.reachable[ny, x]):
                    continue
                candidate = distance + grid.resolution * math.hypot(dy, dx)
                if candidate < self.distance[ny, nx]:
                    self.distance[ny, nx] = candidate
                    heapq.heappush(queue, (candidate, ny, nx))

    def __call__(self, target):
        target = np.asarray(target, float)
        if self.grid.safe_segment(self.robot_xy, target):
            return float(np.linalg.norm(target - self.robot_xy))
        cy, cx = self.grid.cell(target)
        radius = math.ceil(0.10 / self.grid.resolution)
        height, width = self.distance.shape
        candidates = []
        for y in range(max(0, cy - radius), min(height, cy + radius + 1)):
            for x in range(max(0, cx - radius), min(width, cx + radius + 1)):
                if not math.isfinite(self.distance[y, x]):
                    continue
                center = self.grid.xy((y, x))
                offset = float(np.linalg.norm(center - target))
                if offset <= 0.10 + 1e-12:
                    candidates.append((self.distance[y, x] + offset, center))
        for distance, center in sorted(candidates, key=lambda item: item[0]):
            if self.grid.safe_segment(center, target):
                return float(distance)
        return math.inf

    def path(self, target, service=None):
        """A* on observed centers, checking the exact swept footprint per edge."""
        target = np.asarray(target, float)
        grid = self.grid
        if self.seed is None or target.shape != (2,) or not np.isfinite(target).all():
            return None
        if not grid.safe_pose(target):
            return None
        if grid.motion_path_safe([self.robot_xy, target]):
            return np.asarray([self.robot_xy, target])
        candidates = []
        cy, cx = grid.cell(target)
        radius = math.ceil(0.10 / grid.resolution)
        for y in range(max(0, cy - radius), min(grid.reachable.shape[0], cy + radius + 1)):
            for x in range(max(0, cx - radius), min(grid.reachable.shape[1], cx + radius + 1)):
                cell = (y, x)
                if not grid.reachable[cell]:
                    continue
                offset = float(np.linalg.norm(grid.xy(cell) - target))
                if offset <= 0.10 and grid.safe_segment(grid.xy(cell), target):
                    candidates.append((self.distance[cell] + offset, cell))
        if not candidates:
            return None
        goal = min(candidates)[1]
        costs = {self.seed: 0.0}
        parent = {}
        queue = [(0.0, 0.0, self.seed)]
        expanded = 0
        while queue:
            _, cost, cell = heapq.heappop(queue)
            if cost != costs.get(cell):
                continue
            if cell == goal:
                break
            expanded += 1
            if service and expanded % 128 == 0:
                service()
            y, x = cell
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                neighbor = (y + dy, x + dx)
                if not grid.inside(neighbor) or not grid.reachable[neighbor]:
                    continue
                candidate = cost + grid.resolution * math.hypot(dy, dx)
                if candidate >= costs.get(neighbor, math.inf):
                    continue
                if not grid.safe_segment(grid.xy(cell), grid.xy(neighbor)):
                    continue
                costs[neighbor] = candidate
                parent[neighbor] = cell
                heuristic = float(np.linalg.norm(grid.xy(neighbor) - grid.xy(goal)))
                heapq.heappush(queue, (candidate + heuristic, candidate, neighbor))
        else:
            return None
        cells = [goal]
        while cells[-1] != self.seed:
            cells.append(parent[cells[-1]])
        raw = np.vstack([self.robot_xy, [grid.xy(c) for c in cells[::-1]], target])
        # Preserve the existing <=10 cm physical-safe prefix when tracking
        # consumed the planning margin; shorten only safe later spans.
        short = [raw[0]]
        index = 0
        if not grid.safe_pose(raw[0]):
            short.append(raw[1])
            index = 1
        while index < len(raw) - 1:
            finish = next(
                (
                    j
                    for j in range(len(raw) - 1, index, -1)
                    if grid.safe_segment(raw[index], raw[j])
                ),
                None,
            )
            if finish is None:
                return None
            short.append(raw[finish])
            index = finish
        result = np.asarray(short)
        return result if grid.motion_path_safe(result) else None

    def attachment_paths(self, global_route, *, maximum_candidates=48, service=None):
        """Yield bounded live-safe routes joining the same global intent.

        A global route can have a locally blocked first tangent even when a
        current local detour exists. Its sampled attachment points are temporary
        routing proposals, not replacement mission goals or motion admission.
        Every yielded path passes this snapshot's original swept-footprint
        predicates; callers must still validate each committed action afresh.
        The existing optional planning service callback can observe cancellation
        and maintain its already-authorized lifecycle; exceptions propagate.
        This pure planner does not read ROS, execute actions, or award coverage.
        """
        if type(maximum_candidates) is not int or not 1 <= maximum_candidates <= 48:
            raise ValueError("Route attachment search requires 1..48 candidates")
        route = np.asarray(global_route, float)
        if (
            route.ndim != 2
            or route.shape[1] != 2
            or not 2 <= len(route) <= 1024
            or not np.isfinite(route).all()
            or np.linalg.norm(route[0] - self.robot_xy) > 0.10
        ):
            raise ValueError("Finite current global route required for local attachment")
        lengths = np.linalg.norm(np.diff(route, axis=0), axis=1)
        total = float(lengths.sum())
        if not math.isfinite(total) or total <= 0.005:
            raise ValueError("Global route attachment requires finite translation")
        cumulative = np.r_[0.0, np.cumsum(lengths)]
        count = (
            maximum_candidates
            if total >= maximum_candidates * 0.05
            else max(1, math.ceil(total / 0.05))
        )
        samples = np.linspace(0.0, total, count + 1)[1:]
        points = np.column_stack(
            [np.interp(samples, cumulative, route[:, axis]) for axis in range(2)]
        )
        for index, point in enumerate(points):
            if service:
                service()
            if np.linalg.norm(point - self.robot_xy) <= 0.005 or not self.grid.safe_pose(point):
                continue
            candidate = self.path(point, service=service)
            if candidate is None:
                continue
            yield (
                candidate,
                {
                    "kind": "live-safe attachment to held global route",
                    "attachment_xy": point.tolist(),
                    "attachment_arclength_m": float(samples[index]),
                    "sampled_attachments": index + 1,
                    "maximum_attachments": maximum_candidates,
                    "mission_target_changed": False,
                },
            )
