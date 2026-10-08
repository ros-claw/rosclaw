"""Task-neutral offline rehearsal loss mass, not policy safety or RL."""

from typing import Any

import numpy as np


def context_balanced_rehearsal_weights(
    primary_weights: Any, rehearsal_episodes: Any, context_ids: Any, *, rehearsal_mass_ratio: float
) -> np.ndarray[Any, Any]:
    """Add disjoint rehearsal sequences without altering primary loss mass.

    Every observed rehearsal context gets equal total mass regardless of its
    episode count. The caller authenticates labels and defines the loss targets;
    these weights alone cannot establish retention or failure avoidance.
    """
    source, selected, contexts = map(np.asarray, (primary_weights, rehearsal_episodes, context_ids))
    if (
        source.ndim != 2
        or source.dtype.kind not in "fiu"
        or not 4 <= source.size <= 200000
        or not np.isfinite(source).all()
        or np.any(source < 0)
        or np.max(source) > 1e6
        or not np.any(source > 0)
        or selected.shape != (len(source),)
        or selected.dtype.kind != "b"
        or contexts.shape != selected.shape
        or contexts.dtype.kind not in "iu"
        or np.any(contexts < 0)
        or np.any(contexts >= 200000)
        or not np.any(selected)
        or np.any(selected & np.any(source > 0, axis=1))
        or type(rehearsal_mass_ratio) is not float
        or not np.isfinite(rehearsal_mass_ratio)
        or not 0 < rehearsal_mass_ratio <= 1
    ):
        raise ValueError("finite disjoint context-bound rehearsal sequence weights required")
    result = np.array(source, dtype=np.float64, copy=True)
    unique, inverse, counts = np.unique(contexts[selected], return_inverse=True, return_counts=True)
    mass = float(source.sum()) * rehearsal_mass_ratio
    result[selected] = (mass / (len(unique) * counts[inverse] * source.shape[1]))[:, None]
    if not np.isfinite(result).all() or np.any(result[selected] <= 0):
        raise ValueError("positive finite rehearsal mass required")
    return result
