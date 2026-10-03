"""Opt-in duplicate-coordinate search index over an intact frozen guard.

Logical anchors, their order, their multiplicity and the serialized contract
are unchanged. Ambiguous nearest-coordinate queries retain the original tree
and arithmetic. This is a lookup optimization, not a retention certificate.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rosclaw.growth.anchor_kernel import AnchorKernelGuard


class IndexedAnchorKernelGuard(AnchorKernelGuard):
    """Explicitly compiled index; optional SciPy keeps its reference fallback."""

    def __init__(self, anchors: Any, *, bandwidth: float, accelerated: bool = True) -> None:
        super().__init__(anchors, bandwidth=bandwidth, accelerated=accelerated)
        first: dict[bytes, int] = {}
        for index, row in enumerate(self._anchors):
            # Normalize only the private key, never the logical anchor storage.
            key = row.copy()
            key[key == 0] = 0.0
            first.setdefault(key.tobytes(), index)
        self._first_logical_indices = np.asarray(list(first.values()), dtype=np.int64)
        self._first_logical_indices.flags.writeable = False
        self._coordinate_tree: Any = None
        if self._tree is not None:
            from scipy.spatial import cKDTree

            self._coordinate_tree = cKDTree(
                self._anchors[self._first_logical_indices], copy_data=True
            )

    def gate(self, latent: Any) -> float:
        if self._coordinate_tree is None:
            return super().gate(latent)
        values = np.asarray(latent, dtype=np.float64)
        if (
            values.shape != (self.dimension,)
            or not np.isfinite(values).all()
            or np.max(np.abs(values)) > 1e6
        ):
            raise ValueError("finite aligned frozen observation required")
        if len(self._first_logical_indices) == 1:
            logical_index = int(self._first_logical_indices[0])
        else:
            distances, indices = self._coordinate_tree.query(values, k=2)
            if abs(distances[0] - distances[1]) <= 1e-12 * max(distances[0], distances[1]):
                # Original tree, rather than a different tie-breaking rule.
                return super().gate(values)
            logical_index = int(self._first_logical_indices[int(indices[0])])
        delta = self._anchors[logical_index] - values
        distance_squared = float(delta @ delta)
        if distance_squared <= 1e-20:
            return 0.0
        return float(-np.expm1(-distance_squared / (2 * self.bandwidth**2)))
