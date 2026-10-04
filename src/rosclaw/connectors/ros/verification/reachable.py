"""Derive a fixed cleanable denominator from a measured occupancy grid."""

import math
from collections import deque


def cleanable_cells(
    *, width, height, resolution, occupancy, start_cell, robot_radius, cleaning_radius
):
    if (
        not isinstance(width, int)
        or not isinstance(height, int)
        or width <= 0
        or height <= 0
        or len(occupancy) != width * height
        or not all(math.isfinite(v) and v > 0 for v in (resolution, robot_radius, cleaning_radius))
        or not isinstance(start_cell, int)
        or not 0 <= start_cell < width * height
    ):
        raise ValueError("valid map and physical robot radius required")
    free = {i for i, v in enumerate(occupancy) if 0 <= v < 50}
    radius = math.ceil(robot_radius / resolution)
    offsets = [
        (x, y)
        for y in range(-radius, radius + 1)
        for x in range(-radius, radius + 1)
        if math.hypot(x, y) * resolution <= robot_radius + resolution / math.sqrt(2)
    ]
    centers = set()
    for cell in free:
        x, y = cell % width, cell // width
        if all(
            0 <= x + dx < width and 0 <= y + dy < height and (y + dy) * width + x + dx in free
            for dx, dy in offsets
        ):
            centers.add(cell)
    if start_cell not in centers:
        raise ValueError("observed robot pose is not in a reachable map component")
    reachable, queue = {start_cell}, deque([start_cell])
    while queue:
        cell = queue.popleft()
        x, y = cell % width, cell // width
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            n = (y + dy) * width + x + dx
            if 0 <= x + dx < width and 0 <= y + dy < height and n in centers and n not in reachable:
                reachable.add(n)
                queue.append(n)
    # Inscribed cleaning radius is conservative for arbitrary robot headings.
    span = math.ceil(cleaning_radius / resolution)
    brush = [
        (x, y)
        for y in range(-span, span + 1)
        for x in range(-span, span + 1)
        if math.hypot(x, y) * resolution <= cleaning_radius
    ]
    cleanable = set()
    for cell in reachable:
        x, y = cell % width, cell // width
        cleanable.update(
            (y + dy) * width + x + dx
            for dx, dy in brush
            if 0 <= x + dx < width and 0 <= y + dy < height and (y + dy) * width + x + dx in free
        )
    return sorted(cleanable)
