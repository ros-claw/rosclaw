"""Offline numeric recurrent clipped-policy updates and MC value fitting.

The caller supplies authenticated rollouts and frozen advantages. This module
verifies the numeric behavior density, not physical provenance. It operates on
latent Gaussian proposals, never actuators or projected-action likelihoods.
"""

import copy
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.growth.causal_residual_memory import CausalResidualMemory


@dataclass(frozen=True)
class RecurrentClippedActorCriticConfig:
    steps: int = 128
    batch_episodes: int = 4
    learning_rate: float = 0.0002
    residual_cap: float = 0.2
    std: float = 0.1
    rho: float = 0.9
    clip_epsilon: float = 0.2
    maximum_mean_kl: float = 0.02
    value_loss_weight: float = 0.5
    seed: int = 0
    compute_device: str = "cpu"

    def validate(self) -> None:
        integers = ((self.steps, 1, 2048), (self.batch_episodes, 1, 64), (self.seed, 0, 2**32 - 1))
        floats = (
            (self.learning_rate, 1e-6, 0.01),
            (self.residual_cap, 0.001, 1.0),
            (self.std, 0.001, 1.0),
            (self.rho, 0.0, 0.99),
            (self.clip_epsilon, 0.01, 0.4),
            (self.maximum_mean_kl, 0.0001, 0.1),
            (self.value_loss_weight, 0.001, 10.0),
        )
        if (
            any(type(v) is not int or not low <= v <= high for v, low, high in integers)
            or any(
                type(v) is not float or not np.isfinite(v) or not low <= v <= high
                for v, low, high in floats
            )
            or type(self.compute_device) is not str
            or re.fullmatch(r"cpu|cuda:[0-9]{1,2}", self.compute_device) is None
        ):
            raise ValueError("bounded explicit recurrent clipped-update configuration required")


def initial_value_parameters(input_dimension: int, *, seed: int = 0) -> dict[str, Any]:
    if (
        type(input_dimension) is not int
        or not 1 <= input_dimension <= 1024
        or type(seed) is not int
        or not 0 <= seed < 2**32
    ):
        raise ValueError("bounded explicit value-network initialization required")
    rng = np.random.default_rng(seed)
    return {
        "weight_0": (rng.normal(size=(64, input_dimension)) / np.sqrt(input_dimension)).tolist(),
        "bias_0": np.zeros(64).tolist(),
        "weight_1": np.zeros((1, 64)).tolist(),
        "bias_1": [0.0],
    }


def _value_parameters(value: Any, dimension: int) -> dict[str, Any]:
    shapes = {"weight_0": (64, dimension), "bias_0": (64,), "weight_1": (1, 64), "bias_1": (1,)}
    if type(value) is not dict or set(value) != set(shapes):
        raise ValueError("complete owned value-network parameters required")
    owned = {}
    for key, shape in shapes.items():
        array = np.asarray(value[key])
        if (
            array.shape != shape
            or array.dtype.kind not in "fiu"
            or not np.isfinite(array).all()
            or np.max(np.abs(array)) > 1e6
        ):
            raise ValueError("finite bounded value-network parameters required")
        owned[key] = np.array(array, dtype=np.float64, copy=True)
    return owned


def _json_hash(value: Any) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    )


def fit_recurrent_clipped_actor_critic(
    *,
    actor_parameters: Any,
    critic_parameters: Any,
    context: Any,
    baseline: Any,
    gates: Any,
    latent_actions: Any,
    behavior_log_probabilities: Any,
    advantages: Any,
    returns: Any,
    config: RecurrentClippedActorCriticConfig,
) -> dict[str, Any]:
    """Fit a bounded proposal actor and persistent MC critic, without execution.

    Advantages are frozen upstream labels, not actor inputs. Episode boundaries
    reset both GRU state and AR conditioning. Every proposed joint optimizer
    update is checked on ALL supplied rows for conditional and marginal mean
    KL; a violating update is rolled back, including critic and optimizer state.
    """
    if type(config) is not RecurrentClippedActorCriticConfig:
        raise ValueError("explicit recurrent clipped-update configuration required")
    config.validate()
    memory = CausalResidualMemory(actor_parameters)
    values = [
        np.asarray(v)
        for v in (
            context,
            baseline,
            gates,
            latent_actions,
            behavior_log_probabilities,
            advantages,
            returns,
        )
    ]
    x, b, g, a, old_log, adv, ret = values
    if x.ndim != 3 or not 1 <= x.shape[0] <= 1024 or not 1 <= x.shape[1] <= 512:
        raise ValueError("complete bounded episode sequences required")
    n, t, d = x.shape
    k = memory.output_dimension
    if (
        d != memory.input_dimension
        or d > 1024
        or k > 128
        or x.size > 25000000
        or b.shape != (n, t, k)
        or a.shape != b.shape
        or any(v.shape != (n, t) for v in (g, old_log, adv, ret))
        or any(
            v.dtype.kind not in "fiu" or not np.isfinite(v).all() or np.max(np.abs(v)) > 1e6
            for v in values
        )
        or np.any((g < 0) | (g > 1))
    ):
        raise ValueError("complete finite aligned latent-policy learning rows required")
    x, b, g, a, old_log, adv, ret = [np.array(v, dtype=np.float64, copy=True) for v in values]
    original_actor = memory.parameters()
    original_critic = _value_parameters(critic_parameters, d)
    original_mean = np.empty_like(b)
    for episode in range(n):
        stepper = memory.new_episode()
        for frame in range(t):
            original_mean[episode, frame] = b[episode, frame] + (
                config.residual_cap
                * g[episode, frame]
                * stepper.step(x[episode, frame], index=frame)
            )
    scales = np.full((n, t), config.std * np.sqrt(1 - config.rho**2))
    scales[:, 0] = config.std

    def condition(mu: Any) -> Any:
        result = mu.copy()
        result[:, 1:] += config.rho * (a[:, :-1] - mu[:, :-1])
        return result

    original_conditional = condition(original_mean)
    density = np.sum(
        -0.5 * ((a - original_conditional) / scales[:, :, None]) ** 2
        - np.log(scales[:, :, None])
        - 0.5 * np.log(2 * np.pi),
        axis=2,
    )
    density_error = float(np.max(np.abs(density - old_log)))
    if not np.allclose(density, old_log, atol=1e-8, rtol=0):
        raise ValueError("actual conditional behavior density must reconstruct on every row")
    arrays = {
        "context": x,
        "baseline": b,
        "gates": g,
        "latent_actions": a,
        "behavior_log_probabilities": old_log,
        "advantages": adv,
        "returns": ret,
    }
    data_hash = _json_hash(
        {
            key: {
                "shape": list(v.shape),
                "dtype": str(v.dtype),
                "bytes_hash": "sha256:" + hashlib.sha256(v.tobytes(order="C")).hexdigest(),
            }
            for key, v in arrays.items()
        }
    )
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as error:
        raise RuntimeError(
            "recurrent clipped actor-critic fitting requires optional torch"
        ) from error
    device = torch.device(config.compute_device)
    if device.type == "cuda" and (
        not torch.cuda.is_available()
        or device.index is None
        or device.index >= torch.cuda.device_count()
    ):
        raise ValueError("explicit available fitting device required")
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    threads = torch.get_num_threads()
    history = []
    rejected_updates = 0
    accepted_updates = 0
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[] if device.type == "cpu" else [device.index]):
            tensors = {
                key: torch.as_tensor(v, dtype=torch.float64, device=device)
                for key, v in arrays.items()
            }
            actor = {
                key: torch.tensor(v, dtype=torch.float64, device=device, requires_grad=True)
                for key, v in original_actor.items()
            }
            critic = {
                key: torch.tensor(v, dtype=torch.float64, device=device, requires_grad=True)
                for key, v in original_critic.items()
            }
            parameters = [*actor.values(), *critic.values()]
            optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
            reference_mean = torch.as_tensor(original_mean, dtype=torch.float64, device=device)
            reference_conditional = torch.as_tensor(
                original_conditional, dtype=torch.float64, device=device
            )
            sigma = torch.as_tensor(scales, dtype=torch.float64, device=device)

            def actor_mean(batch: Any) -> Any:
                features = tensors["context"][batch]
                state = torch.zeros(
                    (len(batch), memory.hidden_dimension), dtype=torch.float64, device=device
                )
                outputs = []
                for frame in range(t):
                    xi = functional.linear(
                        features[:, frame], actor["weight_ih"], actor["bias_ih"]
                    ).chunk(3, dim=1)
                    hh = functional.linear(state, actor["weight_hh"], actor["bias_hh"]).chunk(
                        3, dim=1
                    )
                    reset, update = torch.sigmoid(xi[0] + hh[0]), torch.sigmoid(xi[1] + hh[1])
                    candidate = torch.tanh(xi[2] + reset * hh[2])
                    state = torch.clamp((1 - update) * candidate + update * state, -1, 1)
                    outputs.append(
                        torch.tanh(
                            functional.linear(state, actor["head_weight"], actor["head_bias"])
                        )
                    )
                residual = torch.stack(outputs, dim=1)
                return (
                    tensors["baseline"][batch]
                    + config.residual_cap * tensors["gates"][batch, :, None] * residual
                )

            def conditional(mu: Any, batch: Any) -> Any:
                previous = tensors["latent_actions"][batch, :-1]
                return torch.cat(
                    (mu[:, :1], mu[:, 1:] + config.rho * (previous - mu[:, :-1])), dim=1
                )

            def value_prediction(batch: Any) -> Any:
                hidden = torch.tanh(
                    functional.linear(
                        tensors["context"][batch], critic["weight_0"], critic["bias_0"]
                    )
                )
                return functional.linear(hidden, critic["weight_1"], critic["bias_1"]).squeeze(-1)

            all_rows = torch.arange(n, device=device)

            def measured_kl() -> tuple[float, float]:
                # Chunk episodes, not frames: each chunk keeps complete causal
                # histories and includes every row in the exact mean bounds.
                marginal_total = conditional_total = 0.0
                for start in range(0, n, 32):
                    batch = all_rows[start : start + 32]
                    mu = actor_mean(batch)
                    marginal_total += float(
                        ((mu - reference_mean[batch]) ** 2 / (2 * config.std**2)).sum().cpu()
                    )
                    conditional_total += float(
                        (
                            (conditional(mu, batch) - reference_conditional[batch]) ** 2
                            / (2 * sigma[batch, :, None] ** 2)
                        )
                        .sum()
                        .cpu()
                    )
                return marginal_total / (n * t), conditional_total / (n * t)

            generator = torch.Generator(device="cpu").manual_seed(config.seed)
            order, cursor = torch.empty(0, dtype=torch.int64), 0
            for _ in range(config.steps):
                if cursor >= len(order):
                    order, cursor = torch.randperm(n, generator=generator), 0
                batch = order[cursor : cursor + config.batch_episodes].to(device)
                cursor += len(batch)
                mu = actor_mean(batch)
                center = conditional(mu, batch)
                logp = (
                    -0.5
                    * ((tensors["latent_actions"][batch] - center) / sigma[batch, :, None]) ** 2
                    - torch.log(sigma[batch, :, None])
                    - 0.5 * np.log(2 * np.pi)
                ).sum(dim=2)
                log_ratio = logp - tensors["behavior_log_probabilities"][batch]
                if not bool(torch.isfinite(log_ratio).all()) or bool(
                    (torch.abs(log_ratio) > 60).any()
                ):
                    raise ValueError("finite bounded latent likelihood ratios required")
                ratio = torch.exp(log_ratio)
                frozen_advantage = tensors["advantages"][batch]
                surrogate = torch.minimum(
                    ratio * frozen_advantage,
                    torch.clamp(ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon)
                    * frozen_advantage,
                )
                actor_loss = -surrogate.mean()
                value_loss = ((value_prediction(batch) - tensors["returns"][batch]) ** 2).mean()
                loss = actor_loss + config.value_loss_weight * value_loss
                if not bool(torch.isfinite(loss)):
                    raise ValueError("nonfinite recurrent joint objective")
                optimizer.zero_grad()
                loss.backward()
                if not all(
                    p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters
                ):
                    raise ValueError("nonfinite recurrent joint gradient")
                torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                saved = [p.detach().clone() for p in parameters]
                optimizer_state = copy.deepcopy(optimizer.state_dict())
                optimizer.step()
                with torch.no_grad():
                    marginal_kl, conditional_kl = measured_kl()
                if (
                    not np.isfinite(marginal_kl + conditional_kl)
                    or max(marginal_kl, conditional_kl) > config.maximum_mean_kl
                ):
                    with torch.no_grad():
                        for parameter, old in zip(parameters, saved, strict=True):
                            parameter.copy_(old)
                    optimizer.load_state_dict(optimizer_state)
                    rejected_updates += 1
                    break
                accepted_updates += 1
                history.append(
                    {
                        "actor_loss": float(actor_loss.detach().cpu()),
                        "value_loss": float(value_loss.detach().cpu()),
                        "marginal_mean_kl": marginal_kl,
                        "conditional_mean_kl": conditional_kl,
                    }
                )
            with torch.no_grad():
                final_marginal_kl, final_conditional_kl = measured_kl()
                final_value_mse = float(
                    ((value_prediction(all_rows) - tensors["returns"]) ** 2).mean().cpu()
                )
            fitted_actor = {
                key: value.detach().cpu().numpy().tolist() for key, value in actor.items()
            }
            fitted_critic = {
                key: value.detach().cpu().numpy().tolist() for key, value in critic.items()
            }
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    CausalResidualMemory(fitted_actor)
    _value_parameters(fitted_critic, d)
    if (
        not np.isfinite(final_value_mse)
        or max(final_marginal_kl, final_conditional_kl) > config.maximum_mean_kl
    ):
        raise ValueError("finite fitted critic and original full-batch KL bounds required")
    return {
        "algorithm": "RECURRENT_CLIPPED_LATENT_POLICY_WITH_MC_VALUE_FITTING_V1",
        "actor_parameters": fitted_actor,
        "critic_parameters": fitted_critic,
        "config": asdict(config),
        "source_hash": "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "memory_source_hash": "sha256:"
        + hashlib.sha256(
            Path(__file__).with_name("causal_residual_memory.py").read_bytes()
        ).hexdigest(),
        "input_numeric_hash": data_hash,
        "original_actor_parameters_hash": _json_hash(original_actor),
        "fitted_actor_parameters_hash": _json_hash(fitted_actor),
        "original_critic_parameters_hash": _json_hash(
            {k: v.tolist() for k, v in original_critic.items()}
        ),
        "fitted_critic_parameters_hash": _json_hash(fitted_critic),
        "episode_count": n,
        "episode_horizon": t,
        "all_rows_behavior_density_validated": True,
        "positive_advantage_rows": int(np.sum(adv > 0)),
        "negative_advantage_rows": int(np.sum(adv < 0)),
        "zero_advantage_rows": int(np.sum(adv == 0)),
        "behavior_density_max_abs_error": density_error,
        "accepted_joint_optimizer_steps": accepted_updates,
        "rejected_and_rolled_back_updates": rejected_updates,
        "actor_parameters_changed": _json_hash(original_actor) != _json_hash(fitted_actor),
        "full_batch_marginal_mean_kl": final_marginal_kl,
        "full_batch_conditional_mean_kl": final_conditional_kl,
        "all_rows_in_kl_checks": True,
        "final_mc_value_mse": final_value_mse,
        "accepted_update_history": history,
        "state_reset_at_each_episode": True,
        "previous_candidate_mean_in_ar_conditioning": True,
        "frozen_advantages_are_actor_inputs": False,
        "returns_are_actor_inputs": False,
        "latent_likelihood_not_projected_action_likelihood": True,
        "td_bootstrapping": False,
        "on_policy_collection_provenance_verified": False,
        "physical_batch_verified": False,
        "promotion_authorized": False,
        "runtime_execution_authorized": False,
        "hardware_authorized": False,
    }
