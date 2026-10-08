"""Task-neutral, proposal-only AR(1) residual policy gradient.

Inputs are frozen numeric batches, not bodies or executors. Previous candidate
means remain differentiable: correlated exploration is not IID PPO. A bounded
KL update is not a distributional retention or physical safety guarantee.
Torch is imported only when optimization is explicitly requested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ResidualGradientConfig:
    residual_cap: float = 0.05
    learning_rate: float = 1e-4
    rho: float = 0.9
    steps: int = 160
    seed: int = 202610348

    def validate(self) -> None:
        for name, lower, upper in (
            ("residual_cap", 0.005, 0.2),
            ("learning_rate", 1e-6, 4e-4),
            ("rho", 0.0, 0.95),
        ):
            value = getattr(self, name)
            if (
                type(value) not in (float, int)
                or not np.isfinite(value)
                or not lower <= value <= upper
            ):
                raise ValueError("finite bounded residual gradient configuration required")
        if (
            type(self.steps) is not int
            or not 1 <= self.steps <= 160
            or type(self.seed) is not int
            or not 0 <= self.seed < 2**32
        ):
            raise ValueError("bounded deterministic optimizer budget required")


def conditional_means(mean: Any, action: Any, first: Any, rho: float) -> Any:
    """Reconstruct history-conditioned means, resetting at each trajectory."""
    return np.where(
        first[:, None], mean, mean + rho * (np.roll(action, 1, axis=0) - np.roll(mean, 1, axis=0))
    )


def terminal_crossfit_advantages(
    features: Any,
    phases: Any,
    groups: Any,
    returns: Any,
) -> dict[str, Any]:
    """Whole-trajectory four-fold terminal critic; never split correlated frames.

    Group IDs must be consecutive sealed trajectory IDs starting at zero.
    One terminal return per trajectory is required. This is an MC critic, not
    a claimed bootstrapped online actor-critic or a physical outcome verifier.
    """
    phi, phase, group, reward = [np.asarray(v) for v in (features, phases, groups, returns)]
    if phi.ndim != 2:
        raise ValueError("aligned finite whole-trajectory critic data required")
    n, d = phi.shape
    if (
        not 200 <= n <= 200000
        or not 1 <= d <= 512
        or any(v.shape != (n,) for v in (phase, group, reward))
        or phase.dtype.kind not in "iu"
        or group.dtype.kind not in "iu"
        or phi.dtype.kind not in "fiu"
        or reward.dtype.kind not in "fiu"
        or not np.isfinite(phi).all()
        or not np.isfinite(reward).all()
        or np.any(np.diff(group) < 0)
        or group[0] != 0
        or not 3 <= int(group[-1]) < n
        or not np.array_equal(np.unique(group), np.arange(int(group[-1]) + 1))
        or not 0 <= int(phase.min()) <= int(phase.max()) < 16
        or not np.array_equal(np.unique(phase), np.arange(int(phase.max()) + 1))
        or not 1 <= int(phase.max()) + 1 <= 16
    ):
        raise ValueError("aligned finite whole-trajectory critic data required")
    if any(not np.all(reward[group == g] == reward[group == g][0]) for g in np.unique(group)):
        raise ValueError("one actual terminal return per trajectory required")
    target_mean, target_scale = float(reward.mean()), max(float(reward.std()), 1.0)
    targets = (reward - target_mean) / target_scale
    values = np.zeros(n)
    critic = np.zeros((int(phase.max()) + 1, d))

    def regression(ids: Any) -> Any:
        design = phi[ids]
        if len(design) < 50:
            raise ValueError("whole-trajectory critic cross-fit support too small")
        return np.linalg.solve(design.T @ design + 0.01 * np.eye(d), design.T @ targets[ids])

    for p in range(len(critic)):
        for fold in range(4):
            test = (phase == p) & (group % 4 == fold)
            values[test] = phi[test] @ regression((phase == p) & (group % 4 != fold))
        critic[p] = regression(phase == p)
    advantage = targets - values
    advantage = (advantage - advantage.mean()) / max(float(advantage.std()), 1e-6)
    return {
        "advantages": advantage,
        "critic_readout": critic,
        "target_mean": target_mean,
        "target_scale": target_scale,
        "crossfit_unit": "whole_rollout",
        "crossfit_folds": 4,
        "ridge": 0.01,
    }


def fit_correlated_residual(
    *,
    layers: Any,
    context: Any,
    baseline: Any,
    gates: Any,
    actions: Any,
    marginal_std: Any,
    first: Any,
    advantages: Any,
    old_log_probability: Any,
    config: ResidualGradientConfig | None = None,
) -> dict[str, Any]:
    """Optimize a frozen-data residual; no simulator, checkpoint or authority IO.

    Caller must independently establish batch provenance and physical outcome.
    This engine verifies the actual conditional behavior density before fitting.
    It never changes the base policy, gate values or protected observations.
    """
    if config is None:
        config = ResidualGradientConfig()
    if not isinstance(config, ResidualGradientConfig):
        raise ValueError("typed bounded residual configuration required")
    config.validate()
    x, base, gate, action, marginal, resets, importance, logp_old = [
        np.asarray(v)
        for v in (
            context,
            baseline,
            gates,
            actions,
            marginal_std,
            first,
            advantages,
            old_log_probability,
        )
    ]
    if x.ndim != 2 or any(
        v.dtype.kind not in "fiu" for v in (x, base, gate, action, marginal, importance, logp_old)
    ):
        raise ValueError("finite aligned frozen multi-trajectory numeric batch required")
    n = len(x)
    if (
        x.ndim != 2
        or not 4 <= n <= 200000
        or not 1 <= x.shape[1] <= 512
        or base.ndim != 2
        or base.shape[0] != n
        or not 1 <= base.shape[1] <= 64
        or action.shape != base.shape
        or any(v.shape != (n,) for v in (gate, marginal, resets, importance, logp_old))
        or resets.dtype.kind != "b"
        or not resets[0]
        or np.count_nonzero(resets) < 2
        or np.any((gate < 0) | (gate > 1))
        or np.any((marginal < 0.01) | (marginal > 0.15))
        or not all(
            np.isfinite(v).all() for v in (x, base, gate, action, marginal, importance, logp_old)
        )
    ):
        raise ValueError("finite aligned frozen multi-trajectory numeric batch required")
    if not isinstance(layers, (tuple, list)) or not 1 <= len(layers) <= 4:
        raise ValueError("bounded residual network required")
    original_layers = []
    width = x.shape[1]
    for layer in layers:
        if not isinstance(layer, (tuple, list)) or len(layer) != 2:
            raise ValueError("aligned finite residual layer required")
        w, b = (np.asarray(v, dtype=np.float64) for v in layer)
        if (
            w.ndim != 2
            or w.shape[1] != width
            or not 1 <= w.shape[0] <= 512
            or b.shape != (w.shape[0],)
            or not np.isfinite(w).all()
            or not np.isfinite(b).all()
        ):
            raise ValueError("aligned finite residual layer required")
        width = w.shape[0]
        original_layers.append((w, b))
    if width != base.shape[1]:
        raise ValueError("residual output must match frozen action dimension")
    std = marginal * np.where(resets, 1.0, np.sqrt(1 - config.rho**2))
    if np.any(std < 0.01):
        raise ValueError("bounded conditional innovation scale required")
    hidden = x
    for w, b in original_layers:
        hidden = np.tanh(hidden @ w.T + b)
    initial_mean = base + config.residual_cap * gate[:, None] * hidden
    conditioned = conditional_means(initial_mean, action, resets, config.rho)
    density = np.sum(
        -0.5 * ((action - conditioned) / std[:, None]) ** 2
        - np.log(std[:, None])
        - 0.5 * np.log(2 * np.pi),
        axis=1,
    )
    if not np.allclose(density, logp_old, atol=1e-8, rtol=0):
        raise ValueError("batch is not the declared history-conditioned behavior policy")

    import torch

    # CPU float64 is the reference. Restore process-wide Torch settings even
    # when fitting fails; a library call must not silently reconfigure its host.
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.set_num_threads(4)
            torch.random.default_generator.manual_seed(config.seed)
            torch.use_deterministic_algorithms(True)
            return _fit_torch(
                torch,
                original_layers,
                x,
                base,
                gate,
                action,
                std,
                marginal,
                resets,
                importance,
                config,
            )
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def _fit_torch(
    torch: Any,
    original_layers: Any,
    x: Any,
    base: Any,
    gate: Any,
    action: Any,
    std: Any,
    marginal: Any,
    resets: Any,
    importance: Any,
    config: ResidualGradientConfig,
) -> dict[str, Any]:
    weights = [torch.nn.Parameter(torch.tensor(w, dtype=torch.float64)) for w, _ in original_layers]
    biases = [torch.nn.Parameter(torch.tensor(b, dtype=torch.float64)) for _, b in original_layers]
    parameters = weights + biases
    optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
    inp, frozen_base, frozen_gate, desired, noise, marginal_noise, advantage = [
        torch.tensor(v, dtype=torch.float64)
        for v in (x, base, gate[:, None], action, std[:, None], marginal[:, None], importance)
    ]
    first_mask = torch.tensor(resets[:, None])
    previous_action = torch.tensor(np.roll(action, 1, axis=0), dtype=torch.float64)

    def mean() -> Any:
        hidden = inp
        for w, b in zip(weights, biases, strict=True):
            hidden = torch.tanh(hidden @ w.T + b)
        return frozen_base + config.residual_cap * frozen_gate * hidden

    def condition(mu: Any) -> Any:
        # DO NOT detach previous candidate means or substitute behavior means.
        return torch.where(first_mask, mu, mu + config.rho * (previous_action - mu.roll(1, dims=0)))

    def probability(mu: Any) -> Any:
        return (
            -0.5 * ((desired - mu) / noise).square() - noise.log() - 0.5 * np.log(2 * np.pi)
        ).sum(1)

    with torch.no_grad():
        original = mean().clone()
        original_conditional = condition(original).clone()
        logp0 = probability(original_conditional).clone()

    def objective() -> tuple[Any, Any, Any]:
        mu = mean()
        conditional = condition(mu)
        ratio = torch.exp(probability(conditional) - logp0)
        kl = ((conditional - original_conditional).square() / (2 * noise.square())).sum(1).mean()
        marginal_kl = ((mu - original).square() / (2 * marginal_noise.square())).sum(1).mean()
        loss = -torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage).mean()
        return loss + 10 * kl, kl, marginal_kl

    history = [float(objective()[0].detach())]
    for _ in range(config.steps):
        optimizer.zero_grad()
        loss, _, _ = objective()
        if not torch.isfinite(loss):
            raise ValueError("nonfinite correlated residual gradient")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1)
        previous = [p.detach().clone() for p in parameters]
        optimizer.step()
        directions = [p.detach().clone() - old for p, old in zip(parameters, previous, strict=True)]
        accepted = False
        for reduction in range(13):
            with torch.no_grad():
                for parameter, old, direction in zip(parameters, previous, directions, strict=True):
                    parameter.copy_(old + (0.5**reduction) * direction)
                trial, kl, marginal_kl = objective()
                if (
                    torch.isfinite(trial)
                    and max(float(kl), float(marginal_kl)) <= 0.0049
                    and float(trial) < history[-1] - 1e-9
                ):
                    history.append(float(trial))
                    accepted = True
                    break
        if not accepted:
            with torch.no_grad():
                for parameter, old in zip(parameters, previous, strict=True):
                    parameter.copy_(old)
            break
    if len(history) == 1:
        raise ValueError("no actual correlated residual learning")
    final = mean().detach().numpy()
    old = original.numpy()
    conditional_kl = float(
        np.mean(
            np.sum(
                (
                    conditional_means(final, action, resets, config.rho)
                    - conditional_means(old, action, resets, config.rho)
                )
                ** 2
                / (2 * std[:, None] ** 2),
                axis=1,
            )
        )
    )
    marginal_kl = float(np.mean(np.sum((final - old) ** 2 / (2 * marginal[:, None] ** 2), axis=1)))
    if not all(np.isfinite(v) and 0 <= v <= 0.005 for v in (conditional_kl, marginal_kl)):
        raise ValueError("independent numerical KL exceeded declared budget")
    return {
        "layers": [
            {"weight": w.detach().numpy().tolist(), "bias": b.detach().numpy().tolist()}
            for w, b in zip(weights, biases, strict=True)
        ],
        "completed_optimizer_steps": len(history) - 1,
        "full_batch_loss_history": history,
        "exact_mean_conditional_kl": conditional_kl,
        "exact_mean_marginal_kl": marginal_kl,
        "algorithm": "BOUNDED_CORRELATED_RESIDUAL_PPO_V1",
        "residual_cap": float(config.residual_cap),
        "learning_rate": float(config.learning_rate),
        "rho": float(config.rho),
        "optimizer_device": "cpu_float64_reference",
        "frozen_baseline": True,
        "frozen_guard": True,
        "distributional_retention_guaranteed": False,
        "physical_batch_verified": False,
        "promotion_authorized": False,
        "hardware_authorized": False,
    }
