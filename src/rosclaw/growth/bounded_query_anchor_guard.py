"""Opt-in numeric query acceleration, retaining the complete logical bank.

The reference gate saturates at exactly one in float64 well before sixteen
bandwidths. A bounded exact spatial query avoids expensive distant searches;
nearby queries and the no-SciPy path retain the original scalar arithmetic.
This does not change the metric, remove anchors, approve a policy or guarantee
trajectory retention. Adoption requires separate source-bound parity evidence.
"""

from typing import Any

import numpy as np

from rosclaw.growth.anchor_kernel import AnchorKernelGuard


class BoundedQueryAnchorGuard(AnchorKernelGuard):
    """Explicit alternate compiled query; never installed into a live guard."""

    def gate(self, latent: Any) -> float:
        values = np.asarray(latent, dtype=np.float64)
        if (
            values.shape != (self.dimension,)
            or not np.isfinite(values).all()
            or np.max(np.abs(values)) > 1e6
        ):
            raise ValueError("finite aligned frozen observation required")
        if self._tree is None:
            return super().gate(values)
        distance, index = self._tree.query(
            values, k=1, eps=0.0, distance_upper_bound=16 * self.bandwidth
        )
        if np.isinf(distance):
            # q >= 128, conservatively beyond float64 expm1 saturation.
            return 1.0
        delta = self._anchors[int(index)] - values
        distance_squared = float(delta @ delta)
        if distance_squared <= 1e-20:
            return 0.0
        return float(-np.expm1(-distance_squared / (2 * self.bandwidth**2)))
