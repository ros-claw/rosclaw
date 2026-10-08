"""Opt-in exact saturated blend; retain the entire logical prediction memory.

When the original float64 gate is exactly one and every proposal component is
nonzero, the reference coefficient is zero and the nearest prediction cannot
change any output bit. Zero components still take the original path to retain
signed-zero arithmetic. No bank consolidation, policy change or authority.
"""

from typing import Any

import numpy as np

from rosclaw.growth.bounded_query_anchor_guard import BoundedQueryAnchorGuard
from rosclaw.growth.indexed_anchor_output_memory import IndexedAnchorOutputMemory


class BoundedBlendOutputMemory(IndexedAnchorOutputMemory):
    """Explicit private numeric view; old constructors/defaults are unchanged."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._guard = BoundedQueryAnchorGuard.from_dict(
            self._guard.to_dict(), accelerated=kwargs.get("accelerated", True)
        )

    def blend(self, observation: Any, proposal: Any, *, encoder_hash: str) -> np.ndarray[Any, Any]:
        if encoder_hash != self.encoder_hash:
            raise ValueError("frozen observation encoder identity changed")
        gate = self._guard.gate(observation)
        predicted = np.asarray(proposal, dtype=np.float64)
        if (
            predicted.shape != (self.output_dimension,)
            or not np.isfinite(predicted).all()
            or np.max(np.abs(predicted)) > 1e6
        ):
            raise ValueError("finite aligned bounded numeric proposal required")
        if gate == 1.0 and np.all(predicted != 0):
            return predicted.copy()
        reference = self._predictions[self._nearest(observation)]
        if gate == 0:
            return reference.copy()
        return np.asarray(gate * predicted + (1 - gate) * reference, dtype=np.float64)
