"""Seeded, dimension-independent SIM-only latent exploration.

This is a numeric random stream, not an executor. Consumers must retain their
own action envelope and keep the complete configuration in behavior identity.
"""

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CorrelatedExplorationConfig:
    seed: int
    scale: float = 0.25
    correlation: float = 0.95
    activation_ceiling: str = "SIM_ONLY"

    def __post_init__(self) -> None:
        if (
            type(self.seed) is not int
            or not 0 <= self.seed < 2**32
            or type(self.scale) not in (int, float)
            or type(self.correlation) not in (int, float)
            or not math.isfinite(self.scale)
            or not 0 <= self.scale <= 0.25
            or not math.isfinite(self.correlation)
            or not 0 <= self.correlation <= 0.999
            or self.activation_ceiling != "SIM_ONLY"
        ):
            raise ValueError("bounded seeded SIM_ONLY correlated exploration required")


class CorrelatedGaussianExploration:
    def __init__(self, dimension: int, config: CorrelatedExplorationConfig) -> None:
        if (
            type(dimension) is not int
            or not 1 <= dimension <= 4096
            or type(config) is not CorrelatedExplorationConfig
        ):
            raise ValueError("bounded numeric dimension and immutable exploration config required")
        self.dimension, self.config = dimension, config
        self.reset()

    def reset(self) -> None:
        self._rng = np.random.Generator(np.random.PCG64(self.config.seed))
        self._state = np.zeros(self.dimension, dtype=np.float64)
        self.draw_steps = 0

    def perturb(self, mean: np.ndarray, log_std: np.ndarray) -> np.ndarray:
        mean, log_std = np.asarray(mean), np.asarray(log_std)
        if (
            mean.shape != (self.dimension,)
            or log_std.shape != mean.shape
            or mean.dtype.kind not in "fi"
            or log_std.dtype.kind not in "fi"
            or not np.isfinite(mean).all()
            or not np.isfinite(log_std).all()
            or np.any(log_std < -5)
            or np.any(log_std > 0.5)
        ):
            raise ValueError(
                "finite aligned latent mean and bounded log standard deviation required"
            )
        if self.config.scale == 0:
            return mean.copy()
        rho = self.config.correlation
        self._state = rho * self._state + math.sqrt(1 - rho * rho) * self._rng.standard_normal(
            self.dimension
        )
        self.draw_steps += 1
        return (
            mean.astype(np.float64)
            + self.config.scale * np.exp(log_std.astype(np.float64)) * self._state
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema": "rosclaw.control.correlated_exploration_state.v1",
            "dimension": self.dimension,
            "config": asdict(self.config),
            "state": self._state.tolist(),
            "draw_steps": self.draw_steps,
            "rng": copy.deepcopy(self._rng.bit_generator.state),
        }

    def restore(self, value: dict[str, Any]) -> None:
        if (
            not isinstance(value, dict)
            or set(value) != {"schema", "dimension", "config", "state", "draw_steps", "rng"}
            or value["schema"] != "rosclaw.control.correlated_exploration_state.v1"
            or value["dimension"] != self.dimension
            or value["config"] != asdict(self.config)
            or type(value["draw_steps"]) is not int
            or not 0 <= value["draw_steps"] <= 1_000_000_000
        ):
            raise ValueError("exact bounded exploration-state commitment required")
        state = np.asarray(value["state"], dtype=np.float64)
        if (
            state.shape != (self.dimension,)
            or not np.isfinite(state).all()
            or not isinstance(value["rng"], dict)
            or value["rng"].get("bit_generator") != "PCG64"
        ):
            raise ValueError("finite aligned exploration state and PCG64 RNG required")
        rng = np.random.Generator(np.random.PCG64())
        rng.bit_generator.state = copy.deepcopy(value["rng"])
        self._state, self._rng, self.draw_steps = state.copy(), rng, value["draw_steps"]
