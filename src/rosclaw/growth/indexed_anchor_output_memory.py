"""Opt-in exact-coordinate query index; logical prediction memory is unchanged.

Duplicate coordinates remain in the serialized bank with their original order
and provenance. Only the private search index stores each coordinate once.
Distinct-coordinate ties retain the original NumPy lowest-index reference path.
This is an inference optimization, not consolidation or runtime authority.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rosclaw.growth.anchor_output_memory import AnchorOutputMemory


class IndexedAnchorOutputMemory(AnchorOutputMemory):
    """An explicitly selected compiled view of an intact logical memory.

    Inherited serialization and validation retain the original schema/hash.
    Inherited ``extend`` returns the ordinary reference implementation: callers
    must explicitly compile a new index after any logical-bank extension.
    SciPy is optional; without it, the full NumPy reference path is used.
    """

    def __init__(
        self,
        observations: Any,
        predictions: Any,
        *,
        bandwidth: float,
        encoder_hash: str,
        parent_policy_hash: str,
        evidence_hash: str,
        predecessor_memory_hash: str | None = None,
        accelerated: bool = True,
    ) -> None:
        super().__init__(
            observations,
            predictions,
            bandwidth=bandwidth,
            encoder_hash=encoder_hash,
            parent_policy_hash=parent_policy_hash,
            evidence_hash=evidence_hash,
            predecessor_memory_hash=predecessor_memory_hash,
            accelerated=accelerated,
        )
        # Base validation has already normalized signed zero and rejected
        # conflicting predictions at identical coordinates. No rounding here.
        first: dict[bytes, int] = {}
        for index, row in enumerate(self._observations):
            first.setdefault(row.tobytes(), index)
        self._first_logical_indices = np.asarray(list(first.values()), dtype=np.int64)
        self._first_logical_indices.flags.writeable = False
        if self._tree is not None:
            from scipy.spatial import cKDTree

            self._tree = cKDTree(self._observations[self._first_logical_indices], copy_data=True)

    def _nearest(self, observation: Any) -> int:
        if self._tree is not None:
            if len(self._first_logical_indices) == 1:
                return int(self._first_logical_indices[0])
            distances, indices = self._tree.query(observation, k=2)
            if abs(distances[0] - distances[1]) > 1e-12 * max(distances[0], distances[1]):
                return int(self._first_logical_indices[int(indices[0])])
        # Keep the original arithmetic and original bank order for true and
        # numerically ambiguous ties between distinct coordinates.
        delta = self._observations - observation
        return int(np.argmin(np.einsum("ni,ni->n", delta, delta)))
