"""Offline same-context episode credit with a leave-one-episode-out baseline.

An episode's baseline contains only OTHER declared trials of its context.
Physical provenance, equal task conditions, behavior identity and independent
random draws remain caller obligations. No critic fitting, actor update,
environment, TD/PPO, execution, promotion or robot authorization occurs here.
"""

from typing import Any

import numpy as np


def grouped_terminal_credit(
    returns: Any, trajectory_ids: Any, *, trajectory_context_ids: Any
) -> dict[str, Any]:
    reward, groups, contexts = [
        np.asarray(v) for v in (returns, trajectory_ids, trajectory_context_ids)
    ]
    if (
        reward.ndim != 1
        or not 4 <= len(reward) <= 200000
        or reward.dtype.kind not in "fiu"
        or groups.shape != reward.shape
        or groups.dtype.kind not in "iu"
        or contexts.ndim != 1
        or contexts.dtype.kind not in "iu"
        or not 4 <= len(contexts) <= 4096
        or not np.isfinite(reward).all()
        or np.max(np.abs(reward)) > 1e6
        or np.any(contexts < 0)
        or not np.array_equal(np.unique(groups), np.arange(len(contexts)))
        or not 2 <= len(np.unique(contexts)) <= 128
    ):
        raise ValueError("complete finite terminal returns and whole-episode contexts required")
    labels, counts = np.unique(contexts, return_counts=True)
    if np.any((counts < 2) | (counts > 32)):
        raise ValueError("two to thirty-two separately declared trials per context required")
    episode_returns = np.empty(len(contexts), dtype=np.float64)
    for group in range(len(contexts)):
        observed = reward[groups == group]
        if not np.all(observed == observed[0]):
            raise ValueError("terminal episode return must be constant across its rows")
        episode_returns[group] = observed[0]
    baselines = np.empty(len(contexts), dtype=np.float64)
    for context in labels:
        indices = np.flatnonzero(contexts == context)
        # Do not sum including the target and then subtract: excluding the
        # target explicitly also prevents its extreme value rounding away
        # meaningful differences between the other trials.
        for group in indices:
            baselines[group] = episode_returns[indices[indices != group]].mean()
    difference = episode_returns - baselines
    scale = max(float(difference.std()), 1e-6)
    advantage = (difference - difference.mean()) / scale
    if not np.isfinite(advantage).all():
        raise ValueError("finite whole-episode relative credit required")
    return {
        "algorithm": "GROUPED_TERMINAL_LEAVE_ONE_EPISODE_OUT_CREDIT_V1",
        "advantages": advantage[groups],
        "episode_advantages": advantage,
        "episode_baselines": baselines,
        "episode_returns": episode_returns,
        "declared_episode_count": len(contexts),
        "declared_context_count": len(labels),
        "trials_per_context": {int(k): int(v) for k, v in zip(labels, counts, strict=True)},
        "baseline_contains_target_episode": False,
        "all_declared_episodes_retained": True,
        "context_is_actor_observation": False,
        "future_return_is_actor_observation": False,
        "actual_critic_fits": 0,
        "independent_physical_draws_verified": False,
        "physical_batch_verified": False,
        "actor_updated": False,
        "online_ppo_claimed": False,
        "runtime_execution_authorized": False,
        "promotion_authorized": False,
        "hardware_authorized": False,
    }
