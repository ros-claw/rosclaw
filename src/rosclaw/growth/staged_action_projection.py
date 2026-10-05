"""Pure offline staged action math, not a motor command or safety validator."""

from typing import Any

import numpy as np


def staged_action_projection(
    raw: Any, previous: Any, lower: Any, upper: Any, *, cap: float, slew: float
) -> np.ndarray[Any, Any]:
    """Squash, limit change, then apply the final box, in that exact order.

    Final bounds may move between observations and override the earlier slew
    limit. The caller derives and authenticates its bounds; this function cannot
    prove physical feasibility, training validity, or authorization.
    """
    arrays = [np.asarray(v) for v in (raw, previous, lower, upper)]
    if (
        any(a.dtype.kind not in "fiu" for a in arrays)
        or not 1 <= arrays[0].ndim <= 3
        or not 1 <= arrays[0].shape[-1] <= 128
        or not 1 <= arrays[0].size <= 25000000
        or any(a.shape != arrays[0].shape or not np.isfinite(a).all() for a in arrays)
        or any(type(v) is not float or not np.isfinite(v) for v in (cap, slew))
        or not 0 < slew <= cap <= 1
    ):
        raise ValueError("complete finite aligned staged projection operands required")
    value, prior, low, high = [np.array(v, dtype=np.float64, copy=True) for v in arrays]
    if (
        np.any(np.abs(value) > 1e6)
        or np.any(np.abs(prior) > cap + 1e-5)
        or np.any(low > 0)
        or np.any(high < 0)
        or np.any(low < -cap)
        or np.any(high > cap)
    ):
        raise ValueError("finite bounded prior and final box containing zero required")
    desired = cap * np.tanh(value)
    proposed = prior + np.clip(desired - prior, -slew, slew)
    return np.asarray(np.clip(proposed, low, high), dtype=np.float64)
