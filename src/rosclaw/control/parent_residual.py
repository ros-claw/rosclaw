"""Pure vector composition, not a stability certificate or hardware executor."""

import math

import numpy as np


def parent_context(observation: np.ndarray, parent: np.ndarray, limits: np.ndarray) -> np.ndarray:
    state, command, hard = (np.asarray(x) for x in (observation, parent, limits))
    if (
        state.ndim != 1
        or not 1 <= len(state) <= 10000
        or command.ndim != 1
        or not 1 <= len(command) <= 4096
        or hard.shape != command.shape
        or not all(np.isfinite(x).all() for x in (state, command, hard))
        or np.any(hard <= 0)
        or np.any(hard > 1e6)
        or np.any(np.abs(command) > hard)
    ):
        raise ValueError("finite explicit state, parent and positive vector limits required")
    result = np.concatenate((state, command / hard)).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("parent-conditioned observation overflow")
    return result


def compose_parent_residual(
    parent: np.ndarray, residual: np.ndarray, limits: np.ndarray, *, maximum_ratio: float
) -> np.ndarray:
    command, delta, hard = (np.asarray(x, dtype=np.float64) for x in (parent, residual, limits))
    if (
        command.ndim != 1
        or not 1 <= len(command) <= 4096
        or delta.shape != command.shape
        or hard.shape != command.shape
        or not math.isfinite(maximum_ratio)
        or not 0 < maximum_ratio <= 0.1
        or not all(np.isfinite(x).all() for x in (command, delta, hard))
        or np.any(hard <= 0)
        or np.any(hard > 1e6)
        or np.any(np.abs(command) > hard)
        or np.any(np.abs(delta) > hard * maximum_ratio)
    ):
        raise ValueError("finite bounded parent residual required; no clipping or repair")
    # Preserve even signed zeros in the exact baseline, not just approximate equality.
    result = command.copy() if not np.any(delta) else command + delta
    if not np.isfinite(result).all():
        raise ValueError("composed residual overflow")
    return result
