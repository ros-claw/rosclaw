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
