"""Simple room sweeps on public observed geometry, without task ground truth."""

import numpy as np


def prefix(path, maximum):
    """Clip by traveled arc length, preserving corners rather than their chord."""
    path = np.asarray(path, float)
    if (
        path.ndim != 2
        or path.shape[1] != 2
        or not len(path)
        or not np.isfinite(path).all()
        or not np.isfinite(maximum)
        or maximum <= 0
    ):
        raise ValueError("Invalid smoke path or length")
    result = [path[0]]
    left = maximum
    for a, b in zip(path, path[1:], strict=False):
        length = float(np.linalg.norm(b - a))
        if length < 1e-12:
            continue
        if length >= left:
            result.append(a + (b - a) * (left / length))
            break
        result.append(b)
        left -= length
    return np.asarray(result)


def room_sweeps(grid, robot_xy, spacing=0.25):
    """Sweep along each room's long axis and join only footprint-safe turns.

    Rooms are Grid's online doorway-separated components. No world labels enter.
    Geometry changes are revalidated by the shared native execution layer.
    """
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("Positive sweep spacing required")
    current = np.asarray(robot_xy, float)
    paths = []
    regions = list(np.unique(grid.rooms[grid.reachable]))
    while regions:
        region = min(
            regions,
            key=lambda label: np.linalg.norm(
                grid.xy(np.argwhere(grid.reachable & (grid.rooms == label)).mean(axis=0)) - current
            ),
        )
        regions.remove(region)
        mask = grid.reachable & (grid.rooms == region)
        cells = np.argwhere(mask)
        if not len(cells):
            continue
        # Near-square online extents jitter by cells as SLAM grows. Keep their
        # orientation deterministic instead of alternating perpendicular sweeps.
        vertical = np.ptp(cells[:, 0]) > 1.15 * np.ptp(cells[:, 1])
        work = mask.T if vertical else mask
        strokes = []
        reverse = False
        occupied_rows = np.flatnonzero(work.any(axis=1))
        first, last = int(occupied_rows[0]), int(occupied_rows[-1])
        axis = 0 if vertical else 1
        low = grid.xy((0, first) if vertical else (first, 0))[axis]
        high = grid.xy((0, last) if vertical else (last, 0))[axis]
        anchors = np.arange(np.ceil(low / spacing), np.floor(high / spacing) + 1) * spacing
        # Map-coordinate anchors survive newly observed rows and origin shifts.
        # Avoid floor(spacing/resolution): native .02500000037 made .25 become
        # nine cells, increasing overlap and changing line placement each pass.
        indices = first + np.floor((anchors - low) / grid.resolution + 0.5 + 1e-7).astype(int)
        rows = sorted({first, last, *[int(row) for row in indices if first <= row <= last]})
        for row in rows:
            xs = np.flatnonzero(work[row])
            if not len(xs):
                continue
            groups = np.split(xs, np.flatnonzero(np.diff(xs) > 1) + 1)
            pairs = [
                [(int(s[0]), row), (int(s[-1]), row)]
                if vertical
                else [(row, int(s[0])), (row, int(s[-1]))]
                for s in groups
                if len(s) > 1
            ]
            if reverse:
                pairs = [pair[::-1] for pair in pairs[::-1]]
            strokes.extend(np.array([grid.xy(c) for c in pair]) for pair in pairs)
            reverse = not reverse
        if not strokes:
            continue
        if np.linalg.norm(strokes[-1][-1] - current) < np.linalg.norm(strokes[0][0] - current):
            strokes = [p[::-1] for p in strokes[::-1]]
        chain = []
        for stroke in strokes:
            if not grid.safe_path(stroke):
                continue
            if chain and not grid.safe_segment(chain[-1], stroke[0]):
                paths.append(np.asarray(chain))
                chain = []
            chain.extend(stroke)
        if chain:
            paths.append(np.asarray(chain))
            current = chain[-1]
    return paths
