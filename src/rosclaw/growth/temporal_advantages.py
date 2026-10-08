"""Offline GAE targets from caller-bound transitions, never motor commands.

Rewards and frozen value predictions must come from an independently verified
rollout. This numeric helper does not authenticate provenance, predict events,
change rewards, select a policy, or authorize execution. See Schulman et al.,
https://arxiv.org/abs/1506.02438. Targets are not actor observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TemporalAdvantageTargets:
    """Owned, read-only unnormalized targets; no training or safety claim."""

    advantages: np.ndarray[Any, Any]
    value_targets: np.ndarray[Any, Any]
    td_residuals: np.ndarray[Any, Any]


def generalized_advantage_targets(
    rewards: Any,
    values: Any,
    next_values: Any,
    episode_ends: Any,
    terminated: Any,
    *,
    discount: float,
    trace_decay: float,
) -> TemporalAdvantageTargets:
    """Estimate GAE for [streams, consecutive transitions] without resets leaking.

    True termination removes next-state bootstrapping. Episode ends (including
    time-limit truncation) stop the advantage recursion, but truncation keeps
    the bootstrap. ``next_values`` must describe each actual next observation,
    not an auto-reset observation. The last recorded transition bootstraps but
    has no unrecorded future advantages. No rows are dropped or normalized.
    """
    if any(
        type(v) not in (int, float) or not np.isfinite(v) or not 0 <= v <= 1
        for v in (discount, trace_decay)
    ):
        raise ValueError("explicit finite discount and trace decay in [0, 1] required")
    numeric = [np.asarray(v) for v in (rewards, values, next_values)]
    ends, terminal = [np.asarray(v) for v in (episode_ends, terminated)]
    shape = numeric[0].shape
    if (
        len(shape) != 2
        or not 1 <= shape[0] <= 65536
        or not 1 <= shape[1] <= 4096
        or shape[0] * shape[1] > 2_000_000
        or any(v.shape != shape or v.dtype.kind not in "fiu" for v in numeric)
        or ends.shape != shape
        or terminal.shape != shape
        or ends.dtype != np.bool_
        or terminal.dtype != np.bool_
        or np.any(terminal & ~ends)
        or any(not np.isfinite(v).all() for v in numeric)
    ):
        raise ValueError(
            "complete bounded finite transitions and explicit boolean boundaries required"
        )
    reward, value, next_value = [np.array(v, dtype=np.float64, copy=True) for v in numeric]
    ends, terminal = np.array(ends, copy=True), np.array(terminal, copy=True)
    advantage = np.empty(shape, dtype=np.float64)
    carry = np.zeros(shape[0], dtype=np.float64)
    try:
        with np.errstate(over="raise", invalid="raise"):
            residual = reward + discount * np.where(terminal, 0.0, next_value) - value
            for frame in range(shape[1] - 1, -1, -1):
                carry = residual[:, frame] + discount * trace_decay * np.where(
                    ends[:, frame], 0.0, carry
                )
                advantage[:, frame] = carry
            targets = advantage + value
    except FloatingPointError as error:
        raise ValueError(
            "finite temporal targets required; arithmetic overflow rejected"
        ) from error
    if any(not np.isfinite(v).all() for v in (advantage, targets, residual)):
        raise ValueError("finite temporal targets required")
    for array in (advantage, targets, residual):
        array.flags.writeable = False
    return TemporalAdvantageTargets(advantage, targets, residual)
