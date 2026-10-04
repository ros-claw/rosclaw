"""Explicit whole-context cross-fit MC critic, without execution authority.

Repeated trajectories from one context stay in the same held-out fold. Context
IDs are offline labels, never actor observations. This is not online TD learning.
The historical whole-rollout critic and its default behavior remain untouched.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def context_crossfit_advantages(
    features: Any,
    phases: Any,
    trajectory_ids: Any,
    returns: Any,
    *,
    trajectory_context_ids: Any,
) -> dict[str, Any]:
    """Four deterministic context-disjoint folds, with complete frozen returns.

    Callers must authenticate physical labels and declare this different critic
    objective before fitting. Numeric validation grants no evidence or authority.
    """
    phi, phase, group, reward, contexts = [
        np.asarray(v) for v in (features, phases, trajectory_ids, returns, trajectory_context_ids)
    ]
    if phi.ndim != 2 or group.ndim != 1 or not len(group):
        raise ValueError("aligned complete whole-context critic data required")
    n, d = phi.shape
    if (
        not 200 <= n <= 200000
        or not 1 <= d <= 512
        or any(v.shape != (n,) for v in (phase, group, reward))
        or phase.dtype.kind not in "iu"
        or group.dtype.kind not in "iu"
        or phi.dtype.kind not in "fiu"
        or reward.dtype.kind not in "fiu"
        or contexts.ndim != 1
        or contexts.dtype.kind not in "iu"
        or not np.isfinite(phi).all()
        or not np.isfinite(reward).all()
        or np.any(np.diff(group) < 0)
        or group[0] != 0
        or not 3 <= int(group[-1]) < n
        or not np.array_equal(np.unique(group), np.arange(int(group[-1]) + 1))
        or contexts.shape != (int(group[-1]) + 1,)
        or np.any(contexts < 0)
        or not 4 <= len(np.unique(contexts)) <= len(contexts)
        or not 0 <= int(phase.min()) <= int(phase.max()) < 16
        or not np.array_equal(np.unique(phase), np.arange(int(phase.max()) + 1))
    ):
        raise ValueError("aligned complete whole-context critic data required")
    if any(not np.all(reward[group == g] == reward[group == g][0]) for g in np.unique(group)):
        raise ValueError("one authenticated terminal return per trajectory required")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            result = _fit_context_advantages(
                np.asarray(phi, dtype=np.float64),
                phase,
                group,
                np.asarray(reward, dtype=np.float64),
                contexts,
            )
    except (FloatingPointError, np.linalg.LinAlgError) as error:
        raise ValueError("derived context critic must remain finite and solvable") from error
    if not all(
        np.isfinite(result[k]).all()
        for k in (
            "advantages",
            "critic_readout",
            "crossfit_predictions",
            "target_mean",
            "target_scale",
        )
    ):
        raise ValueError("derived context critic must remain finite and solvable")
    return result


def _fit_context_advantages(phi: Any, phase: Any, group: Any, reward: Any, contexts: Any) -> Any:
    n, d = phi.shape
    labels, inverse = np.unique(contexts, return_inverse=True)
    folds = inverse % 4
    sample_folds = folds[group]
    target_mean = float(reward.mean())
    target_scale = max(float(reward.std()), 1.0)
    values = np.zeros(n)
    critic = np.zeros((int(phase.max()) + 1, d))

    def regression(mask: Any) -> Any:
        design = phi[mask]
        if len(design) < 50:
            raise ValueError("whole-context cross-fit support too small")
        # Fit in raw return units: held-out return normalization must not leak
        # into training through the ridge-regularized intercept.
        return np.linalg.solve(design.T @ design + 0.01 * np.eye(d), design.T @ reward[mask])

    for p in range(len(critic)):
        for fold in range(4):
            test = (phase == p) & (sample_folds == fold)
            if not np.any(test):
                raise ValueError("every phase must have held-out support in every context fold")
            values[test] = phi[test] @ regression((phase == p) & (sample_folds != fold))
        critic[p] = regression(phase == p)
    advantage = (reward - values) / target_scale
    advantage = (advantage - advantage.mean()) / max(float(advantage.std()), 1e-6)
    return {
        "advantages": advantage,
        "critic_readout": critic,
        "critic_readout_unit": "raw_terminal_return",
        "crossfit_predictions": values,
        "target_mean": target_mean,
        "target_scale": target_scale,
        "crossfit_unit": "whole_context",
        "crossfit_folds": 4,
        "trajectory_fold_ids": folds,
        "context_ids": labels,
        "context_fold_ids": np.arange(len(labels)) % 4,
        "ridge": 0.01,
    }
