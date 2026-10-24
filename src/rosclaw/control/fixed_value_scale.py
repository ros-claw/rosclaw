"""Frozen scalar target scaling, with output-preserving affine conversion.

Same fixed-normalization principle as growth.normalized_value_regression;
the helper is model/robot independent and does not import an optimizer.
Inspired by https://arxiv.org/abs/1602.07714, not adaptive online PopArt.
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FixedValueScale:
    center: float
    scale: float

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.center)
            or abs(self.center) > 1e6
            or not math.isfinite(self.scale)
            or not 1e-6 <= self.scale <= 1e6
        ):
            raise ValueError("finite bounded value center and positive scale required")

    @classmethod
    def from_targets(cls, targets: np.ndarray, *, minimum_scale: float = 1.0) -> "FixedValueScale":
        value = np.asarray(targets)
        if (
            value.ndim != 1
            or not 1 <= len(value) <= 2_000_000
            or value.dtype.kind not in "fi"
            or not np.isfinite(value).all()
            or np.any(np.abs(value) > 1e6)
            or not math.isfinite(minimum_scale)
            or not 1e-6 <= minimum_scale <= 1e6
        ):
            raise ValueError(
                "complete finite scalar training targets and bounded scale floor required"
            )
        return cls(
            float(value.mean(dtype=np.float64)),
            max(float(value.std(dtype=np.float64)), minimum_scale),
        )

    def normalize_affine(
        self, weight: np.ndarray, bias: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        w, b = np.asarray(weight), np.asarray(bias)
        if (
            w.ndim != 2
            or w.shape[0] != 1
            or not 1 <= w.shape[1] <= 4096
            or b.shape != (1,)
            or w.dtype.kind not in "fi"
            or b.dtype.kind not in "fi"
            or not np.isfinite(w).all()
            or not np.isfinite(b).all()
        ):
            raise ValueError("finite scalar affine output layer required")
        return w.astype(np.float64) / self.scale, (b.astype(np.float64) - self.center) / self.scale
