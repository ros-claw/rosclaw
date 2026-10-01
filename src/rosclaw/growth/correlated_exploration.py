"""Pure numeric stationary AR(1) exploration; grants no execution authority.

For learning, conditional means must include the candidate's previous mean:
mu_t + rho * (actual_previous_action - candidate_mu_previous).
Using a fixed old noise offset for a new candidate gives incorrect likelihoods.
This module has no robot, actuator, environment or policy backend dependency.
"""

from typing import Any

import numpy as np


def _rho(value: float) -> float:
    if type(value) not in (float, int) or not np.isfinite(value) or not 0 <= value < 1:
        raise ValueError("finite correlation in [0, 1) required")
    return float(value)


def conditional_mean(current: Any, previous_mean: Any, previous_action: Any, rho: float) -> Any:
    """Gaussian mean conditioned on the actual past, not on future outcomes."""
    correlation = _rho(rho)
    arrays = [np.asarray(v, dtype=np.float64) for v in (current, previous_mean, previous_action)]
    if (
        not arrays[0].size
        or any(a.shape != arrays[0].shape for a in arrays)
        or not all(np.isfinite(a).all() for a in arrays)
    ):
        raise ValueError("finite aligned current mean, previous mean and action required")
    with np.errstate(over="ignore", invalid="ignore"):
        result = arrays[0] + correlation * (arrays[2] - arrays[1])
    if not np.isfinite(result).all():
        raise ValueError("nonfinite conditional mean arithmetic")
    return result


def stationary_noise(
    *, seed: int, rho: float, count: int, dimension: int, first_frame: int = 0
) -> Any:
    """Reproducible unit-variance process, initialized in its stationary law.

    At rho=0, each frame equals its independent SeedSequence draw exactly.
    Precomputed random draws carry no physical observation or outcome.
    """
    correlation = _rho(rho)
    if (
        type(seed) is not int
        or not 0 <= seed < 2**32
        or type(count) is not int
        or not 1 <= count <= 3000
        or type(dimension) is not int
        or not 1 <= dimension <= 1024
        or type(first_frame) is not int
        or not 0 <= first_frame < 2**32 - count
    ):
        raise ValueError("bounded explicit seed, frames and dimension required")
    result = np.empty((count, dimension), dtype=np.float64)
    scale = np.sqrt(1 - correlation**2)
    for index in range(count):
        rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(first_frame + index,)))
        innovation = rng.normal(size=dimension)
        result[index] = (
            innovation if index == 0 else correlation * result[index - 1] + scale * innovation
        )
    result.flags.writeable = False
    return result


def conditional_scale(std: float, rho: float, *, first: bool) -> float:
    """Initial marginal scale differs from subsequent innovation scale."""
    correlation = _rho(rho)
    if type(std) not in (float, int) or not np.isfinite(std) or std <= 0 or type(first) is not bool:
        raise ValueError("finite positive scale and explicit first-step flag required")
    return float(std if first else std * np.sqrt(1 - correlation**2))
