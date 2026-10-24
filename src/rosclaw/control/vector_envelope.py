"""Pure numeric intersections of simultaneous per-axis control bounds.

No robot, actuator, runtime, transport or authorization is provided. An empty
intersection is explicit; callers must use their declared fail-closed policy,
not sequential clipping that can invalidate an earlier bound.
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def intersect_vector_bounds(
    bounds: Sequence[tuple[NDArray[np.float64], NDArray[np.float64]]],
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """Return owned, read-only intersection bounds, or None if infeasible."""
    if not bounds:
        raise ValueError("at least one finite vector envelope required")
    lower_values, upper_values = [], []
    shape = np.asarray(bounds[0][0]).shape
    if len(shape) != 1 or not 1 <= shape[0] <= 4096:
        raise ValueError("bounded nonempty flat vector envelopes required")
    for lower, upper in bounds:
        lo, hi = np.asarray(lower), np.asarray(upper)
        if (
            lo.shape != shape
            or hi.shape != shape
            or lo.dtype.kind not in "fi"
            or hi.dtype.kind not in "fi"
            or not np.isfinite(lo).all()
            or not np.isfinite(hi).all()
            or np.any(lo > hi)
        ):
            raise ValueError("aligned finite ordered vector bounds required")
        lower_values.append(lo.astype(np.float64, copy=True))
        upper_values.append(hi.astype(np.float64, copy=True))
    low: NDArray[np.float64] = np.maximum.reduce(lower_values)
    high: NDArray[np.float64] = np.minimum.reduce(upper_values)
    if np.any(low > high):
        return None
    low.flags.writeable = high.flags.writeable = False
    return low, high
