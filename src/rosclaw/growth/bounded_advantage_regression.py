"""Task-neutral advantage-weighted action regression, proposal-only.

AWR-inspired marginal action likelihood; NOT iid PPO, on-policy collection,
TD-lambda, or a physical safety guarantee. Conditional AR behavior likelihood
is checked for input identity and both conditional/marginal KL bound updates.
No simulator, checkpoint, robot, activation, or executor is accessed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rosclaw.growth.correlated_residual_gradient import (
    ResidualGradientConfig,
    conditional_means,
)


@dataclass(frozen=True)
class AdvantageRegressionConfig(ResidualGradientConfig):
    residual_cap: float = 0.2
    learning_rate: float = 4e-4
    seed: int = 202610391
    temperature: float = 0.5
    maximum_weight: float = 20.0

    def validate(self) -> None:
        super().validate()
        for name, lower, upper in (("temperature", 0.1, 5.0), ("maximum_weight", 1.0, 20.0)):
            value = getattr(self, name)
            if (
                type(value) not in (float, int)
                or not np.isfinite(value)
                or not lower <= value <= upper
            ):
                raise ValueError("finite bounded advantage-regression configuration required")


def advantage_weights(advantages: Any, config: AdvantageRegressionConfig) -> Any:
    config.validate()
    values = np.asarray(advantages, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("finite vector of frozen advantages required")
    scaled = np.clip(values / config.temperature, -64.0, np.log(config.maximum_weight))
    weights = np.exp(scaled)
    return weights / weights.mean()


def fit_advantage_residual(
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
    config: AdvantageRegressionConfig | None = None,
) -> dict[str, Any]:
    config = AdvantageRegressionConfig() if config is None else config
    if not isinstance(config, AdvantageRegressionConfig):
        raise ValueError("typed advantage-regression configuration required")
    config.validate()
    x, base, gate, action, marginal, resets, importance, old_logp = [
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
    if x.ndim != 2 or base.ndim != 2:
        raise ValueError("finite aligned numeric regression batch required")
    n = len(x)
    if (
        not 4 <= n <= 200000
        or not 1 <= x.shape[1] <= 512
        or base.shape[0] != n
        or not 1 <= base.shape[1] <= 64
        or action.shape != base.shape
        or any(v.shape != (n,) for v in (gate, marginal, resets, importance, old_logp))
        or resets.dtype.kind != "b"
        or not resets[0]
        or np.count_nonzero(resets) < 2
        or any(
            v.dtype.kind not in "fiu"
            for v in (x, base, gate, action, marginal, importance, old_logp)
        )
        or not all(
            np.isfinite(v).all() for v in (x, base, gate, action, marginal, importance, old_logp)
        )
        or np.any((gate < 0) | (gate > 1))
        or np.any((marginal < 0.01) | (marginal > 0.15))
    ):
        raise ValueError("finite aligned numeric regression batch required")
    if not isinstance(layers, (list, tuple)) or not 1 <= len(layers) <= 4:
        raise ValueError("bounded finite residual layers required")
    width, original_layers = x.shape[1], []
    for layer in layers:
        if not isinstance(layer, (list, tuple)) or len(layer) != 2:
            raise ValueError("bounded finite residual layers required")
        w, b = (np.asarray(v, dtype=np.float64) for v in layer)
        if (
            w.ndim != 2
            or w.shape[1] != width
            or not 1 <= w.shape[0] <= 512
            or b.shape != (w.shape[0],)
            or not np.isfinite(w).all()
            or not np.isfinite(b).all()
        ):
            raise ValueError("bounded finite residual layers required")
        width = w.shape[0]
        original_layers.append((w, b))
    if width != base.shape[1]:
        raise ValueError("aligned residual action output required")
    hidden = x
    for w, b in original_layers:
        hidden = np.tanh(hidden @ w.T + b)
    initial = base + config.residual_cap * gate[:, None] * hidden
    innovation = marginal * np.where(resets, 1.0, np.sqrt(1 - config.rho**2))
    if np.any(innovation < 0.01):
        raise ValueError("bounded conditional innovation scale required")
    old_conditional = conditional_means(initial, action, resets, config.rho)
    density = np.sum(
        -0.5 * ((action - old_conditional) / innovation[:, None]) ** 2
        - np.log(innovation[:, None])
        - 0.5 * np.log(2 * np.pi),
        axis=1,
    )
    if not np.allclose(density, old_logp, atol=1e-8, rtol=0):
        raise ValueError("declared actual conditional behavior likelihood required")
    weights = advantage_weights(importance, config)
    import torch

    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.set_num_threads(4)
            torch.random.default_generator.manual_seed(config.seed)
            torch.use_deterministic_algorithms(True)
            return _fit(
                torch,
                original_layers,
                x,
                base,
                gate,
                action,
                marginal,
                innovation,
                resets,
                weights,
                config,
            )
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def _fit(
    torch: Any,
    layers: Any,
    x: Any,
    base: Any,
    gate: Any,
    action: Any,
    marginal: Any,
    innovation: Any,
    resets: Any,
    weights: Any,
    config: AdvantageRegressionConfig,
) -> dict[str, Any]:
    w = [torch.nn.Parameter(torch.tensor(v[0], dtype=torch.float64, device="cpu")) for v in layers]
    b = [torch.nn.Parameter(torch.tensor(v[1], dtype=torch.float64, device="cpu")) for v in layers]
    parameters = w + b
    optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
    inp, frozen_base, frozen_gate, desired, noise, marginal_noise, importance = [
        torch.tensor(v, dtype=torch.float64, device="cpu")
        for v in (x, base, gate[:, None], action, innovation[:, None], marginal[:, None], weights)
    ]
    first_mask = torch.tensor(resets[:, None], device="cpu")
    prev_action = torch.tensor(np.roll(action, 1, axis=0), dtype=torch.float64, device="cpu")

    def mean() -> Any:
        hidden = inp
        for weight, bias in zip(w, b, strict=True):
            hidden = torch.tanh(hidden @ weight.T + bias)
        return frozen_base + config.residual_cap * frozen_gate * hidden

    def condition(mu: Any) -> Any:
        return torch.where(first_mask, mu, mu + config.rho * (prev_action - mu.roll(1, dims=0)))

    with torch.no_grad():
        original = mean().clone()
        original_conditional = condition(original).clone()

    def objective() -> tuple[Any, Any, Any]:
        mu = mean()
        conditional_kl = (
            ((condition(mu) - original_conditional).square() / (2 * noise.square())).sum(1).mean()
        )
        marginal_kl = ((mu - original).square() / (2 * marginal_noise.square())).sum(1).mean()
        # Marginal weighted Gaussian NLL: do not substitute PPO ratios or
        # claim that correlated actions were generated by an IID behavior.
        squared_error = 0.5 * ((desired - mu) / marginal_noise).square().sum(1)
        loss = (importance * squared_error).mean() + 10 * conditional_kl
        return loss, conditional_kl, marginal_kl

    history = [float(objective()[0].detach())]
    for _ in range(config.steps):
        optimizer.zero_grad()
        loss, _, _ = objective()
        if not torch.isfinite(loss):
            raise ValueError("nonfinite advantage regression")
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
                trial, ckl, mkl = objective()
                if (
                    torch.isfinite(trial)
                    and max(float(ckl), float(mkl)) <= 0.0049
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
        raise ValueError("no actual advantage-regression update")
    final = mean().detach().numpy()
    old = original.numpy()
    ckl = float(
        np.mean(
            np.sum(
                (
                    conditional_means(final, action, resets, config.rho)
                    - conditional_means(old, action, resets, config.rho)
                )
                ** 2
                / (2 * innovation[:, None] ** 2),
                axis=1,
            )
        )
    )
    mkl = float(np.mean(np.sum((final - old) ** 2 / (2 * marginal[:, None] ** 2), axis=1)))
    if not all(np.isfinite(v) and 0 <= v <= 0.005 for v in (ckl, mkl)):
        raise ValueError("independent numerical regression KL budget exceeded")
    return {
        "algorithm": "BOUNDED_ADVANTAGE_WEIGHTED_RESIDUAL_REGRESSION_V1",
        "layers": [
            {"weight": weight.detach().numpy().tolist(), "bias": bias.detach().numpy().tolist()}
            for weight, bias in zip(w, b, strict=True)
        ],
        "residual_cap": config.residual_cap,
        "learning_rate": config.learning_rate,
        "rho": config.rho,
        "temperature": config.temperature,
        "maximum_weight": config.maximum_weight,
        "normalized_weight_mean": float(weights.mean()),
        "normalized_weight_min": float(weights.min()),
        "normalized_weight_max": float(weights.max()),
        "completed_optimizer_steps": len(history) - 1,
        "full_batch_loss_history": history,
        "exact_mean_conditional_kl": ckl,
        "exact_mean_marginal_kl": mkl,
        "frozen_baseline": True,
        "frozen_guard": True,
        "physical_batch_verified": False,
        "distributional_retention_guaranteed": False,
        "promotion_authorized": False,
        "hardware_authorized": False,
    }
