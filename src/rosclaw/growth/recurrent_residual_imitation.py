"""Bounded sequence imitation with causal GRU state; not RL or robot control."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.growth.causal_residual_memory import CausalResidualMemory
from rosclaw.growth.staged_action_projection import staged_action_projection


@dataclass(frozen=True)
class RecurrentResidualImitationConfig:
    steps: int = 512
    batch_episodes: int = 4
    learning_rate: float = 0.0004
    residual_cap: float = 0.2
    seed: int = 0
    compute_device: str = "cpu"

    def validate(self) -> None:
        if (
            type(self.steps) is not int
            or not 1 <= self.steps <= 2048
            or type(self.batch_episodes) is not int
            or not 1 <= self.batch_episodes <= 32
            or type(self.seed) is not int
            or not 0 <= self.seed < 2**32
            or type(self.learning_rate) not in (float, int)
            or not np.isfinite(self.learning_rate)
            or not 1e-6 <= self.learning_rate <= 0.001
            or type(self.residual_cap) not in (float, int)
            or not np.isfinite(self.residual_cap)
            or not 0.01 <= self.residual_cap <= 0.2
            or type(self.compute_device) is not str
            or re.fullmatch(r"cpu|cuda:[0-9]{1,2}", self.compute_device) is None
        ):
            raise ValueError("bounded complete recurrent imitation configuration required")


def fit_recurrent_residual_imitation(
    *,
    parameters: Any,
    context: Any,
    baseline: Any,
    gates: Any,
    targets: Any,
    training_weights: Any,
    config: RecurrentResidualImitationConfig,
    action_projection: Any = None,
) -> dict[str, Any]:
    config.validate()
    memory = CausalResidualMemory(parameters)
    source = [np.asarray(v) for v in (context, baseline, gates, targets, training_weights)]
    if any(v.dtype.kind not in "fiu" for v in source):
        raise ValueError("finite numeric recurrent training rows required")
    x, base, gate, teacher, weight = [np.array(v, dtype=np.float64, copy=True) for v in source]
    if (
        x.ndim != 3
        or not 1 <= x.shape[0] <= 740
        or not 1 <= x.shape[1] <= 512
        or x.shape[0] * x.shape[1] > 200000
        or x.shape[2] != memory.input_dimension
        or base.shape != (*x.shape[:2], memory.output_dimension)
        or teacher.shape != base.shape
        or gate.shape != x.shape[:2]
        or weight.shape != gate.shape
        or any(
            not np.isfinite(v).all() or np.max(np.abs(v)) > 1e6
            for v in (x, base, gate, teacher, weight)
        )
        or np.any((gate < 0) | (gate > 1))
        or np.any(weight < 0)
        or np.count_nonzero(weight > 0) < 32
    ):
        raise ValueError("complete bounded aligned recurrent episode rows required")
    original_rows, positive_rows = weight.size, int(np.count_nonzero(weight > 0))
    original_episodes, horizon = x.shape[:2]
    projection_arrays = None
    projection_receipt = None
    if action_projection is not None:
        if type(action_projection) is not dict or set(action_projection) != {
            "previous",
            "lower",
            "upper",
            "executed_targets",
            "cap",
            "slew",
            "raw_loss_weight",
        }:
            raise ValueError("complete explicit offline action projection required")
        projection_arrays = [
            np.asarray(action_projection[k])
            for k in ("previous", "lower", "upper", "executed_targets")
        ]
        raw_loss_weight = action_projection["raw_loss_weight"]
        if (
            any(
                v.shape != base.shape or v.dtype.kind not in "fiu" or not np.isfinite(v).all()
                for v in projection_arrays
            )
            or type(raw_loss_weight) is not float
            or not np.isfinite(raw_loss_weight)
            or not 0.001 <= raw_loss_weight <= 1.0
        ):
            raise ValueError(
                "aligned recorded action labels and positive latent auxiliary weight required"
            )
        projection_arrays = [np.array(v, dtype=np.float64, copy=True) for v in projection_arrays]
        previous, lower, upper, executed_targets = projection_arrays
        projected_teacher = staged_action_projection(
            teacher,
            previous,
            lower,
            upper,
            cap=action_projection["cap"],
            slew=action_projection["slew"],
        )
        if not np.allclose(projected_teacher, executed_targets, atol=1e-12, rtol=0):
            raise ValueError(
                "original latent teacher must reconstruct every recorded executed action"
            )
        encoded_projection = json.dumps(
            {
                **{
                    k: v.tolist()
                    for k, v in zip(
                        ("previous", "lower", "upper", "executed_targets"),
                        projection_arrays,
                        strict=True,
                    )
                },
                **{k: action_projection[k] for k in ("cap", "slew", "raw_loss_weight")},
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        projection_receipt = {
            "schema": "rosclaw.growth.recurrent_executed_action_objective.v1",
            "data_hash": "sha256:" + hashlib.sha256(encoded_projection).hexdigest(),
            "projection_source_hash": "sha256:"
            + hashlib.sha256(
                Path(__file__).with_name("staged_action_projection.py").read_bytes()
            ).hexdigest(),
            "cap": action_projection["cap"],
            "slew": action_projection["slew"],
            "raw_loss_weight": raw_loss_weight,
            "order": "CAP_TANH_THEN_SLEW_THEN_FINAL_BOX",
            "all_original_rows_validated": original_rows,
            "teacher_actions_reconstructed": True,
            "teacher_forced_previous_actions_not_closed_loop_rollout": True,
            "projection_labels_are_recurrent_inputs": False,
            "physical_batch_verified": False,
            "promotion_authorized": False,
            "hardware_authorized": False,
        }
    selected = weight.sum(axis=1) > 0
    x, base, gate, teacher, weight = [v[selected] for v in (x, base, gate, teacher, weight)]
    if projection_arrays is not None:
        projection_arrays = [v[selected] for v in projection_arrays]
    positive = weight > 0
    weight /= weight[weight > 0].mean()
    if not np.isfinite(weight).all() or np.any(weight[positive] <= 0):
        raise ValueError("positive recurrent training weights lost numeric representation")
    import torch

    devices: list[int] = []
    if config.compute_device != "cpu":
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
            raise ValueError("deterministic cuBLAS configuration required before recurrent fitting")
        index = int(config.compute_device.split(":")[1])
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise ValueError("requested recurrent CUDA device unavailable")
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
            p = {
                k: torch.nn.Parameter(
                    torch.tensor(v, dtype=torch.float64, device=config.compute_device)
                )
                for k, v in memory.parameters().items()
            }
            tx, tb, tg, tt, tw = [
                torch.tensor(v, dtype=torch.float64, device=config.compute_device)
                for v in (x, base, gate, teacher, weight)
            ]
            projected_tensors = None
            if projection_arrays is not None:
                projected_tensors = [
                    torch.tensor(v, dtype=torch.float64, device=config.compute_device)
                    for v in projection_arrays
                ]

            def weighted_error(indices):
                # Only current x and prior h enter each prediction. Teacher
                # targets enter the offline loss, never the recurrent state.
                hidden = torch.zeros(
                    (len(indices), memory.hidden_dimension),
                    dtype=torch.float64,
                    device=config.compute_device,
                )
                error = torch.zeros((), dtype=torch.float64, device=config.compute_device)
                for tick in range(horizon):
                    xi = torch.chunk(tx[indices, tick] @ p["weight_ih"].T + p["bias_ih"], 3, dim=1)
                    hh = torch.chunk(hidden @ p["weight_hh"].T + p["bias_hh"], 3, dim=1)
                    reset, update = torch.sigmoid(xi[0] + hh[0]), torch.sigmoid(xi[1] + hh[1])
                    candidate = torch.tanh(xi[2] + reset * hh[2])
                    hidden = torch.clamp((1 - update) * candidate + update * hidden, -1, 1)
                    head = torch.tanh(hidden @ p["head_weight"].T + p["head_bias"])
                    prediction = (
                        tb[indices, tick] + config.residual_cap * tg[indices, tick, None] * head
                    )
                    row_error = torch.mean((prediction - tt[indices, tick]) ** 2, dim=1)
                    if projected_tensors is not None:
                        prior, low, high, actual = [v[indices, tick] for v in projected_tensors]
                        cap, slew = action_projection["cap"], action_projection["slew"]
                        proposed = prior + torch.clamp(
                            cap * torch.tanh(prediction) - prior, -slew, slew
                        )
                        projected = torch.minimum(torch.maximum(proposed, low), high)
                        row_error = (
                            torch.mean(((projected - actual) / cap) ** 2, dim=1)
                            + action_projection["raw_loss_weight"] * row_error
                        )
                    error = error + torch.sum(row_error * tw[indices, tick])
                return error, torch.sum(tw[indices])

            def full_loss():
                numerator = torch.zeros((), dtype=torch.float64, device=config.compute_device)
                denominator = torch.zeros_like(numerator)
                for begin in range(0, len(x), config.batch_episodes):
                    indices = torch.arange(
                        begin,
                        min(begin + config.batch_episodes, len(x)),
                        device=config.compute_device,
                    )
                    error, mass = weighted_error(indices)
                    numerator, denominator = numerator + error, denominator + mass
                return numerator / denominator

            with torch.no_grad():
                initial_loss = float(full_loss().cpu())
            optimizer = torch.optim.Adam(list(p.values()), lr=config.learning_rate)
            history = []
            order = torch.empty(0, dtype=torch.int64, device=config.compute_device)
            cursor = 0
            for _ in range(config.steps):
                if cursor >= len(order):
                    order = torch.randperm(len(x), device=config.compute_device)
                    cursor = 0
                indices = order[cursor : cursor + config.batch_episodes]
                cursor += len(indices)
                error, mass = weighted_error(indices)
                loss = error / mass
                if not bool(torch.isfinite(loss)):
                    raise ValueError("nonfinite recurrent imitation loss")
                optimizer.zero_grad()
                loss.backward()
                if not all(
                    v.grad is not None and bool(torch.isfinite(v.grad).all()) for v in p.values()
                ):
                    raise ValueError("nonfinite recurrent imitation gradient")
                torch.nn.utils.clip_grad_norm_(list(p.values()), 1.0, error_if_nonfinite=True)
                optimizer.step()
                history.append(float(loss.detach().cpu()))
            with torch.no_grad():
                final_loss = float(full_loss().cpu())
            fitted = {k: v.detach().cpu().numpy().tolist() for k, v in p.items()}
            CausalResidualMemory(fitted)
            if not np.isfinite(initial_loss) or not np.isfinite(final_loss):
                raise ValueError("nonfinite recurrent fitted loss")
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    canonical = json.dumps(fitted, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    result = {
        "algorithm": "DATASET_WEIGHTED_CAUSAL_RECURRENT_RESIDUAL_IMITATION_V1",
        "parameters": fitted,
        "fitted_parameters_hash": "sha256:" + hashlib.sha256(canonical).hexdigest(),
        "source_hash": "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "memory_source_hash": "sha256:"
        + hashlib.sha256(
            Path(__file__).with_name("causal_residual_memory.py").read_bytes()
        ).hexdigest(),
        "config": {
            "steps": config.steps,
            "batch_episodes": config.batch_episodes,
            "learning_rate": config.learning_rate,
            "residual_cap": config.residual_cap,
            "seed": config.seed,
            "compute_device": config.compute_device,
        },
        "original_input_rows": original_rows,
        "original_episode_count": original_episodes,
        "episode_horizon": horizon,
        "positive_weight_rows": positive_rows,
        "zero_weight_rows": original_rows - positive_rows,
        "completed_optimizer_steps": config.steps,
        "minibatch_loss_history": history,
        "initial_full_positive_weight_loss": initial_loss,
        "final_full_positive_weight_loss": final_loss,
        "training_loss_improved": final_loss < initial_loss,
        "state_reset_at_each_episode": True,
        "teacher_targets_are_recurrent_inputs": False,
        "causal_architecture_not_input_feature_provenance": True,
        "input_feature_causality_verified": False,
        "physical_batch_verified": False,
        "zero_weight_rows_are_not_negative_gradient_examples": True,
        "online_rl_claimed": False,
        "on_policy_ppo": False,
        "kl_trust_region_applied": False,
        "activation_ceiling": "SIM_ONLY",
        "runtime_execution_authorized": False,
        "promotion_authorized": False,
        "hardware_authorized": False,
    }
    if projection_receipt is not None:
        result["executed_action_objective"] = projection_receipt
    return result
