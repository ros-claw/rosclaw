"""Task-neutral stable statewise advantage weights for training replay.

No policy activation or evidence qualification. Avoiding an absolute positive
clip preserves return ranking when a stale value baseline underestimates an
entire state. Equal states are observation vectors, never task/episode IDs.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def statewise_advantage_weights(
    observations: Any, returns: Any, values: Any, *, temperature: float = 0.5
) -> list[float]:
    states = np.asarray(observations, dtype=np.float64)
    target = np.asarray(returns, dtype=np.float64)
    baseline = np.asarray(values, dtype=np.float64)
    if (
        states.ndim != 2
        or not 1 <= states.shape[0] <= 4096
        or not 1 <= states.shape[1] <= 512
        or target.shape != (states.shape[0],)
        or baseline.shape != target.shape
        or not all(np.isfinite(v).all() for v in (states, target, baseline))
        or type(temperature) not in (float, int)
        or not np.isfinite(temperature)
        or not 0 < temperature <= 100
    ):
        raise ValueError("finite bounded aligned statewise advantage problem required")
    groups: dict[tuple[float, ...], list[int]] = {}
    for index, row in enumerate(states):
        groups.setdefault(tuple(float(v) for v in row), []).append(index)
    weights = np.empty_like(target)
    for ids in groups.values():
        # A state-value baseline must be identical for identical observations.
        if not np.array_equal(baseline[ids], np.full(len(ids), baseline[ids[0]])):
            raise ValueError("state value differs for identical observations")
        # exp((R-V)/T) normalized by its state's maximum equals exp((R-maxR)/T).
        # Subtracting R first also avoids huge common baseline cancellation.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            relative = (target[ids] - np.max(target[ids])) / float(temperature)
        if not np.isfinite(relative).all():
            raise ValueError("nonfinite relative advantages")
        weights[ids] = np.exp(np.maximum(relative, -64.0))
    return [float(value) for value in weights]
