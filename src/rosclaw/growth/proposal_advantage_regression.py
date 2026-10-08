"""Task-neutral explicitly budgeted proposal regression; no runtime authority.

AWR-inspired explicitly selected action likelihood; NOT iid PPO, on-policy collection,
TD-lambda, or a physical safety guarantee. Conditional AR behavior likelihood
is checked for input identity and both conditional/marginal KL bound updates.
No simulator, checkpoint, robot, activation, or executor is accessed here.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from rosclaw.growth.correlated_residual_gradient import (
    ResidualGradientConfig,
    conditional_means,
)
from rosclaw.growth.sample_weighting import sample_weight_receipt, validate_sample_weights


@dataclass(frozen=True)
class ProposalAdvantageRegressionConfig(ResidualGradientConfig):
    residual_cap: float = 0.2
    learning_rate: float = 4e-4
    seed: int = 202610391
    temperature: float = 0.5
    maximum_weight: float = 20.0
    maximum_mean_kl: float = 0.005
    execution_ceiling: str = "PROPOSAL_ONLY_NO_RUNTIME"
    compute_device: str = "cpu"
    likelihood_profile: str = "marginal"

    def validate(self) -> None:
        super().validate()
        if type(self.likelihood_profile) is not str or self.likelihood_profile not in (
            "marginal",
            "conditional-ar1",
        ):
            raise ValueError("explicit supported regression likelihood required")
        if (
            type(self.compute_device) is not str
            or re.fullmatch(r"cpu|cuda:[0-9]{1,2}", self.compute_device) is None
        ):
            raise ValueError("explicit bounded numeric compute device required")
        for name, lower, upper in (("temperature", 0.1, 5.0), ("maximum_weight", 1.0, 20.0)):
            value = getattr(self, name)
            if (
                type(value) not in (float, int)
                or not np.isfinite(value)
                or not lower <= value <= upper
            ):
                raise ValueError("finite bounded advantage-regression configuration required")

        if (
            type(self.maximum_mean_kl) not in (float, int)
            or not np.isfinite(self.maximum_mean_kl)
            or not 0.005 <= self.maximum_mean_kl <= 0.5
            or self.execution_ceiling != "PROPOSAL_ONLY_NO_RUNTIME"
        ):
            raise ValueError("explicit bounded proposal-only trust region required")


def advantage_weights(advantages: Any, config: ProposalAdvantageRegressionConfig) -> Any:
    config.validate()
    values = np.asarray(advantages, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("finite vector of frozen advantages required")
    scaled = np.clip(values / config.temperature, -64.0, np.log(config.maximum_weight))
    weights = np.exp(scaled)
    return weights / weights.mean()


def fit_proposal_advantage_residual(
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
    config: ProposalAdvantageRegressionConfig | None = None,
    sample_weights: Any | None = None,
) -> dict[str, Any]:
    config = ProposalAdvantageRegressionConfig() if config is None else config
    if not isinstance(config, ProposalAdvantageRegressionConfig):
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
    weighting_receipt = None
    if sample_weights is not None:
        multipliers = validate_sample_weights(sample_weights, n)
        weighting_receipt = sample_weight_receipt(multipliers)
        if not np.all(multipliers == 1):
            weights = weights * multipliers
            weights = weights / weights.mean()
    import torch

    devices = []
    if config.compute_device != "cpu":
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
            raise ValueError(
                "explicit deterministic cuBLAS configuration required before GPU fitting"
            )
        index = int(config.compute_device.split(":")[1])
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise ValueError("requested numeric CUDA device unavailable")
        devices = [index]
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        with torch.random.fork_rng(devices=devices):
            torch.set_num_threads(4)
            torch.random.default_generator.manual_seed(config.seed)
            if devices:
                with torch.cuda.device(devices[0]):
                    torch.cuda.manual_seed(config.seed)
            torch.use_deterministic_algorithms(True)
            result = _fit(
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
            if weighting_receipt is not None:
                result["sample_weighting"] = weighting_receipt
            if devices:
                result["compute_device"] = config.compute_device
                result["cross_device_bit_identity_claimed"] = False
            if config.likelihood_profile != "marginal":
                result["likelihood_profile"] = config.likelihood_profile
            return result
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
    config: ProposalAdvantageRegressionConfig,
) -> dict[str, Any]:
    device = config.compute_device
    w = [torch.nn.Parameter(torch.tensor(v[0], dtype=torch.float64, device=device)) for v in layers]
    b = [torch.nn.Parameter(torch.tensor(v[1], dtype=torch.float64, device=device)) for v in layers]
    parameters = w + b
    optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
    inp, frozen_base, frozen_gate, desired, noise, marginal_noise, importance = [
        torch.tensor(v, dtype=torch.float64, device=device)
        for v in (x, base, gate[:, None], action, innovation[:, None], marginal[:, None], weights)
    ]
    first_mask = torch.tensor(resets[:, None], device=device)
    prev_action = torch.tensor(np.roll(action, 1, axis=0), dtype=torch.float64, device=device)

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
        conditional = condition(mu)
        conditional_kl = (
            ((conditional - original_conditional).square() / (2 * noise.square())).sum(1).mean()
        )
        marginal_kl = ((mu - original).square() / (2 * marginal_noise.square())).sum(1).mean()
        # These are two explicit supervised proposal objectives, not PPO
        # ratios. Conditional likelihood uses the actual recorded previous
        # action and the innovation scale; reset rows never cross episodes.
        center = mu if config.likelihood_profile == "marginal" else conditional
        scale = marginal_noise if config.likelihood_profile == "marginal" else noise
        squared_error = 0.5 * ((desired - center) / scale).square().sum(1)
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
                    and max(float(ckl), float(mkl))
                    <= (
                        0.0049 if config.maximum_mean_kl == 0.005 else 0.98 * config.maximum_mean_kl
                    )
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
    final = mean().detach().cpu().numpy()
    old = original.cpu().numpy()
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
    if not all(np.isfinite(v) and 0 <= v <= config.maximum_mean_kl for v in (ckl, mkl)):
        raise ValueError("independent numerical regression KL budget exceeded")
    return {
        "algorithm": "PROPOSAL_TRUST_REGION_ADVANTAGE_REGRESSION_V1",
        "layers": [
            {
                "weight": weight.detach().cpu().numpy().tolist(),
                "bias": bias.detach().cpu().numpy().tolist(),
            }
            for weight, bias in zip(w, b, strict=True)
        ],
        "execution_ceiling": config.execution_ceiling,
        "maximum_mean_kl": config.maximum_mean_kl,
        "runtime_execution_authorized": False,
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
