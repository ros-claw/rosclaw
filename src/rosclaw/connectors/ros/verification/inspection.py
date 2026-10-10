"""Conservative, read-only sensor-grid passage evidence. No motion authority.

The caller supplies measured ground-free and obstacle cells in a declared region,
with y as the passage direction. This does not infer visibility from a map, nor
classify an unmeasured volume as free. A result is separate from task success.
"""

import math
from collections import deque
from collections.abc import Iterable

Cell = tuple[int, int]


def _connects(cells: set[Cell], starts: set[Cell], ends: set[Cell]) -> bool:
    queue = deque(cells & starts)
    seen = set(queue)
    while queue:
        x, y = queue.popleft()
        if (x, y) in ends:
            return True
        for point in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if point in cells and point not in seen:
                seen.add(point)
                queue.append(point)
    return False


def classify_passage(
    *,
    width: int,
    height: int,
    observed_free: Iterable[Cell],
    occupied: Iterable[Cell],
    clearance_cells: int,
    minimum_coverage: float,
    evidence_valid: bool = True,
) -> dict:
    """Return CLEAR / OBSTRUCTED / UNKNOWN with measured coverage and reasons.

    CLEAR requires measured-free clearance along a south-to-north path and
    sufficient whole-region coverage. OBSTRUCTED requires a measured obstacle
    barrier across the passage after clearance inflation. Unobserved cells can
    prevent CLEAR, but cannot themselves prove OBSTRUCTED. Clearance uses a
    conservative square stencil; the caller must bind its size to the Body.
    """
    if (
        type(width) is not int
        or type(height) is not int
        or width < 1
        or height < 1
        or width * height > 262144
        or type(clearance_cells) is not int
        or not 0 <= clearance_cells <= max(width, height)
        or not isinstance(minimum_coverage, (int, float))
        or isinstance(minimum_coverage, bool)
        or not math.isfinite(minimum_coverage)
        or not 0 < minimum_coverage <= 1
        or type(evidence_valid) is not bool
    ):
        raise ValueError("Bounded grid, finite coverage and integer clearance required")

    def checked(values: Iterable[Cell]) -> set[Cell]:
        result = set()
        for value in values:
            if len(value) != 2 or any(type(v) is not int for v in value):
                raise ValueError("Evidence cells require integer x/y")
            x, y = value
            if not 0 <= x < width or not 0 <= y < height:
                raise ValueError("Evidence cell outside inspection region")
            result.add((x, y))
        return result

    obstacles = checked(occupied)
    free = checked(observed_free) - obstacles
    observed = free | obstacles
    all_cells = {(x, y) for x in range(width) for y in range(height)}
    radius = clearance_cells
    blocked = {
        (x + dx, y + dy)
        for x, y in obstacles
        for dx in range(-radius, radius + 1)
        for dy in range(-radius, radius + 1)
        if 0 <= x + dx < width and 0 <= y + dy < height
    }
    safe = {
        (x, y)
        for x, y in free
        if radius <= x < width - radius
        and all(
            (x + dx, y + dy) in free
            for dx in range(-radius, radius + 1)
            for dy in range(-radius, radius + 1)
            if 0 <= y + dy < height
        )
    }
    clear_path = _connects(
        safe, {(x, 0) for x in range(width)}, {(x, height - 1) for x in range(width)}
    )
    barrier = _connects(
        blocked, {(0, y) for y in range(height)}, {(width - 1, y) for y in range(height)}
    )
    ratio = len(observed) / len(all_cells)
    result, reason = "UNKNOWN", "Insufficient measured coverage or no verified free passage"
    if not evidence_valid:
        reason = "Sensor/TF/completeness gate failed"
    elif barrier:
        result, reason = "OBSTRUCTED", "Measured obstacle clearance barrier spans passage"
    elif ratio >= minimum_coverage and clear_path:
        result, reason = "CLEAR", "Measured free path satisfies declared clearance and coverage"
    return {
        "schema_version": "rosclaw.passage_inspection.v1",
        "result": result,
        "reason": reason,
        "scope": "Declared 2D sensor grid and caller-bound height band; not full-volume or industrial safety certification",
        "evidence_valid": evidence_valid,
        "observed_cells": len(observed),
        "unknown_cells": len(all_cells - observed),
        "obstacle_cells": len(obstacles),
        "coverage_ratio": ratio,
        "minimum_coverage": minimum_coverage,
        "clearance_cells": radius,
        "verified_free_path": clear_path if evidence_valid else False,
        "measured_obstacle_barrier": barrier if evidence_valid else False,
    }
