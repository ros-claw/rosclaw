"""Coverage polylines generated only from the solver's observed free space."""

import math

import numpy as np


def densify(points, step=0.025):
    points = np.asarray(points, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or not len(points)
        or not np.isfinite(points).all()
        or step <= 0
    ):
        raise ValueError("Invalid observed coverage polyline")
    result = [points[0]]
    for start, finish in zip(points, points[1:], strict=False):
        length = float(np.linalg.norm(finish - start))
        if length > 1e-10:
            result.extend(np.linspace(start, finish, max(1, math.ceil(length / step)) + 1)[1:])
    return np.asarray(result)


def remaining(points, xy):
    points = np.asarray(points, dtype=float)
    index = int(np.argmin(np.linalg.norm(points - np.asarray(xy), axis=1)))
    suffix = points[index + 1 :]
    return np.vstack([xy, suffix if len(suffix) else points[-1:]])


def boundary_strokes(grid, service=None, simplify_tolerance=0.005, turn_limit=0.35):
    """Follow observed safe-space boundaries, projecting onto legal positions.

    Each edge is directed with its free cell on the left. Splits preserve
    safety at raster corners; no unknown or private floor geometry is used.
    """
    edges = {}
    outgoing = {}
    for index, (y, x) in enumerate(grid.centers):
        if service is not None and index % 256 == 0:
            service()
        choices = [
            ((x, y), (x + 1, y), (y - 1, x)),
            ((x + 1, y), (x + 1, y + 1), (y, x + 1)),
            ((x + 1, y + 1), (x, y + 1), (y + 1, x)),
            ((x, y + 1), (x, y), (y, x - 1)),
        ]
        for a, b, neighbor in choices:
            if grid.inside(neighbor) and grid.reachable[neighbor]:
                continue
            edges[(a, b)] = (y, x)
            outgoing.setdefault(a, set()).add(b)
    strokes = []
    while edges:
        if service is not None:
            service()
        edge = min(edges)
        loop = []
        while edge in edges:
            a, b = edge
            cell = edges.pop(edge)
            outgoing[a].remove(b)
            center = grid.xy(cell)
            edgepoint = grid.origin + (np.asarray(a) + b) * grid.resolution / 2
            low, high = 0.0, 1.0
            for _ in range(9):
                fraction = (low + high) / 2
                candidate = center + fraction * (edgepoint - center)
                if grid.safe_segment(center, candidate):
                    low = fraction
                else:
                    high = fraction
            loop.append((center + low * (edgepoint - center), center))
            candidates = outgoing.get(b, set())
            if not candidates:
                break
            direction = np.asarray(b) - a

            def turn(c, b=b, direction=direction):
                d = np.asarray(c) - b
                return math.atan2(direction[0] * d[1] - direction[1] * d[0], direction @ d)

            edge = (b, max(candidates, key=turn))
        path = []
        previous_center = None
        for point, center in loop:
            if service is not None and len(path) % 64 == 0:
                service()
            if path and not grid.safe_segment(path[-1], point):
                bridge = [previous_center, center, point]
                if grid.safe_path([path[-1]] + bridge):
                    path.extend(bridge)
                else:
                    if len(path) > 1:
                        strokes.extend(
                            _simplify_and_split(path, grid, simplify_tolerance, turn_limit)
                        )
                    path = []
            path.append(point)
            previous_center = center
        if len(path) > 1:
            if grid.safe_segment(path[-1], path[0]):
                path.append(path[0])
            strokes.extend(_simplify_and_split(path, grid, simplify_tolerance, turn_limit))
    return strokes


def _simplify_and_split(points, grid, tolerance=0.005, turn_limit=0.35):
    points = np.asarray(points)
    simplified = [points[0]]
    i = 0
    while i < len(points) - 1:
        last = i + 1
        for j in range(i + 2, len(points)):
            delta = points[j] - points[i]
            length2 = delta @ delta
            if length2 < 1e-12:
                break
            values = points[i : j + 1]
            fraction = np.clip((values - points[i]) @ delta / length2, 0, 1)
            deviation = np.linalg.norm(values - (points[i] + fraction[:, None] * delta), axis=1)
            if deviation.max() > tolerance or not grid.safe_segment(points[i], points[j]):
                break
            last = j
        simplified.append(points[last])
        i = last
    points = np.asarray(simplified)
    strokes = []
    start = 0
    for i in range(1, len(points) - 1):
        a, b = points[i] - points[i - 1], points[i + 1] - points[i]
        if np.linalg.norm(a) * np.linalg.norm(b) < 1e-10:
            continue
        turn = math.acos(float(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1)))
        if turn > turn_limit:
            strokes.append(points[start : i + 1])
            start = i
    if len(points[start:]) > 1:
        strokes.append(points[start:])
    opened = []
    for stroke in strokes:
        if len(stroke) > 2 and np.linalg.norm(stroke[0] - stroke[-1]) <= 0.005:
            middle = len(stroke) // 2
            opened.extend([stroke[: middle + 1], stroke[middle:]])
        else:
            opened.append(stroke)
    return [path for path in map(densify, opened) if len(path) > 1]
