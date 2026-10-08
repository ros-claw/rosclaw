"""Approved boundary targets for Nav2; this module computes no motion path."""

import math


def rectangular_boundary_targets(legal_centers, current_pose, *, edge_midpoints=False):
    """Choose a closed ring of four existing legal corners and tangent hints.

    Applicability is deliberately limited to a rectangular center envelope.
    Nav2 must calculate the actual route; corners or straight reference lines
    alone prove neither live reachability nor coverage. No Body inset is reduced.
    """
    if type(edge_midpoints) is not bool:
        raise ValueError("boundary midpoint mode must be boolean")
    if not legal_centers:
        return ()
    if not all(math.isfinite(current_pose[k]) for k in ["x", "y"]):
        raise ValueError("boundary entry pose must be finite")
    centers = frozenset(tuple(p) for p in legal_centers)
    if not all(len(p) == 2 and all(math.isfinite(x) for x in p) for p in centers):
        raise ValueError("boundary targets require finite legal center pairs")
    xmin, xmax = min(p[0] for p in centers), max(p[0] for p in centers)
    ymin, ymax = min(p[1] for p in centers), max(p[1] for p in centers)
    if xmin == xmax or ymin == ymax:
        return ()
    corners = [(xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)]
    if not all(p in centers for p in corners):
        return ()
    start = min(
        range(4),
        key=lambda i: (
            (corners[i][0] - current_pose["x"]) ** 2 + (corners[i][1] - current_pose["y"]) ** 2
        ),
    )
    corners = corners[start:] + corners[:start]
    if edge_midpoints:
        ring = []
        for index, corner in enumerate(corners):
            following = corners[(index + 1) % 4]
            edge = [
                p
                for p in centers
                if p[0] == corner[0] == following[0] or p[1] == corner[1] == following[1]
            ]
            midpoint = ((corner[0] + following[0]) / 2, (corner[1] + following[1]) / 2)
            interior = [p for p in edge if p not in (corner, following)]
            if not interior:
                return ()
            target = min(
                interior, key=lambda p: ((p[0] - midpoint[0]) ** 2 + (p[1] - midpoint[1]) ** 2, p)
            )
            ring.extend([corner, target])
        corners = ring
    targets = []
    for index, (x, y) in enumerate(corners):
        nx, ny = corners[(index + 1) % len(corners)]
        targets.append({"x": x, "y": y, "yaw": math.atan2(ny - y, nx - x)})
    return tuple(targets + [dict(targets[0])])
