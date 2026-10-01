"""Task-neutral frozen-state gate for a new plastic policy residual.

Unlike null-space projection this does not remove a whole linear span. It
zeros a new residual exactly at protected observations and smoothly attenuates
it nearby. No claim is made about unseen trajectories or safety. The encoder,
base policy, feature metric and anchor bank must remain frozen.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def _hash(value: dict[str, Any]) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    )


class AnchorKernelGuard:
    """Multiplier in [0,1] for a learned finite residual; never an approval."""

    def __init__(self, anchors: Any, *, bandwidth: float, accelerated: bool = True) -> None:
        values = np.asarray(anchors, dtype=np.float64)
        if (
            values.ndim != 2
            or not 1 <= values.shape[0] <= 32768
            or not 1 <= values.shape[1] <= 512
            or not np.isfinite(values).all()
            or np.max(np.abs(values)) > 1e6
            or type(bandwidth) not in (float, int)
            or not np.isfinite(bandwidth)
            or not 1e-4 <= bandwidth <= 1000
            or type(accelerated) is not bool
        ):
            raise ValueError("bounded finite frozen anchor metric required")
        self._anchors = values.copy()
        self._anchors.flags.writeable = False
        self.dimension = values.shape[1]
        self.bandwidth = float(bandwidth)
        self._tree: Any = None
        if accelerated:
            try:
                from scipy.spatial import cKDTree
            except ImportError:
                pass  # Optional acceleration; NumPy is the full reference path.
            else:
                self._tree = cKDTree(self._anchors, copy_data=True)

    def gate(self, latent: Any) -> float:
        values = np.asarray(latent, dtype=np.float64)
        if (
            values.shape != (self.dimension,)
            or not np.isfinite(values).all()
            or np.max(np.abs(values)) > 1e6
        ):
            raise ValueError("finite aligned frozen observation required")
        if self._tree is None:
            delta = self._anchors - values
            distance_squared = float(np.min(np.einsum("ni,ni->n", delta, delta)))
        else:
            # Recompute from differences: squared-norm dot-product subtraction
            # can lose precision and misidentify a known anchor as novel.
            _, index = self._tree.query(values, k=1)
            delta = self._anchors[int(index)] - values
            distance_squared = float(delta @ delta)
        if distance_squared <= 1e-20:
            return 0.0
        return float(-np.expm1(-distance_squared / (2 * self.bandwidth**2)))

    def gates(self, latents: Any) -> np.ndarray[Any, Any]:
        values = np.asarray(latents, dtype=np.float64)
        if values.ndim != 2 or not 1 <= len(values) <= 200000:
            raise ValueError("bounded nonempty frozen observation batch required")
        return np.asarray([self.gate(row) for row in values], dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": "rosclaw.growth.anchor_kernel_guard.v1",
            "anchors": self._anchors.tolist(),
            "dimension": self.dimension,
            "bandwidth": self.bandwidth,
            "zero_distance_tolerance": 1e-10,
            "requires_frozen_encoder": True,
            "requires_frozen_parent": True,
            "metric": "euclidean_frozen_latent",
            "local_guarantee_only": True,
            "distributional_retention_guaranteed": False,
            "training_only": True,
            "promotion_authorized": False,
            "hardware_authorized": False,
        }
        value["guard_hash"] = _hash(value)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any], *, accelerated: bool = True) -> AnchorKernelGuard:
        if value.get("guard_hash") != _hash({k: v for k, v in value.items() if k != "guard_hash"}):
            raise ValueError("unsealed anchor kernel guard")
        guard = cls(value.get("anchors"), bandwidth=value.get("bandwidth"), accelerated=accelerated)
        if value != guard.to_dict():
            raise ValueError("anchor metric or authority contract drift")
        return guard
