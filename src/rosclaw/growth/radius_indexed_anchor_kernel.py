"""Opt-in exact saturated-gate lookup with a bounded private search radius.

The guard's existing bandwidth, zero tolerance and logical bank are unchanged.
At distances beyond 16 bandwidths the reference float64 expm1 gate is already
exactly one. We need not find which distant anchor attained that saturated gate.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rosclaw.growth.anchor_kernel import AnchorKernelGuard
from rosclaw.growth.indexed_anchor_kernel import IndexedAnchorKernelGuard


class RadiusIndexedAnchorKernelGuard(IndexedAnchorKernelGuard):
    """No approximate nearest-neighbour query or approximate gate output."""

    def gate(self, latent: Any) -> float:
        if self._coordinate_tree is None:
            return AnchorKernelGuard.gate(self, latent)
        values = np.asarray(latent, dtype=np.float64)
        if (
            values.shape != (self.dimension,)
            or not np.isfinite(values).all()
            or np.max(np.abs(values)) > 1e6
        ):
            raise ValueError("finite aligned frozen observation required")
        distances, indices = self._coordinate_tree.query(
            values, k=2, distance_upper_bound=16 * self.bandwidth
        )
        if not np.isfinite(distances[0]):
            # d^2 / (2 * bandwidth^2) >= 128. expm1(-128) rounds to
            # exactly -1 in the reference float64 implementation. The wide
            # margin also avoids a gate-rounding boundary near the radius.
            return 1.0
        if np.isfinite(distances[1]) and abs(distances[0] - distances[1]) <= 1e-12 * max(
            distances[0], distances[1]
        ):
            return AnchorKernelGuard.gate(self, values)
        index = int(self._first_logical_indices[int(indices[0])])
        delta = self._anchors[index] - values
        distance_squared = float(delta @ delta)
        if distance_squared <= 1e-20:
            return 0.0
        return float(-np.expm1(-distance_squared / (2 * self.bandwidth**2)))
