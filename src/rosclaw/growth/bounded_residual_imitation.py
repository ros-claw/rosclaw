"""Optional supervised bounded residual fit, not RL or runtime authority.

The downstream adapter authenticates teacher data and chooses training weights.
Zero-weight rows remain declared input, not new episodes or negative examples.
No simulator, transport, checkpoint activation or policy promotion is accessed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BoundedResidualImitationConfig:
    steps: int = 1200
    batch_size: int = 1024
    learning_rate: float = 0.0004
    residual_cap: float = 0.2
    seed: int = 202610674
    compute_device: str = "cpu"

    def validate(self) -> None:
        if (
            type(self.steps) is not int
            or not 1 <= self.steps <= 4096
            or type(self.batch_size) is not int
            or not 32 <= self.batch_size <= 4096
            or type(self.seed) is not int
            or not 0 <= self.seed < 2**32
            or type(self.compute_device) is not str
            or re.fullmatch(r"cpu|cuda:[0-9]{1,2}", self.compute_device) is None
            or type(self.learning_rate) is not float
            or not np.isfinite(self.learning_rate)
            or not 1e-6 <= self.learning_rate <= 0.001
            or type(self.residual_cap) is not float
            or not np.isfinite(self.residual_cap)
            or not 0.01 <= self.residual_cap <= 0.2
        ):
            raise ValueError("explicit bounded supervised residual configuration required")


def fit_bounded_residual_imitation(
    *,
    layers: Any,
    context: Any,
    baseline: Any,
    gates: Any,
    targets: Any,
    training_weights: Any,
    config: BoundedResidualImitationConfig | None = None,
) -> dict[str, Any]:
    config = BoundedResidualImitationConfig() if config is None else config
    if type(config) is not BoundedResidualImitationConfig:
        raise ValueError("typed supervised residual configuration required")
    config.validate()
    raw = [np.asarray(v) for v in (context, baseline, gates, targets, training_weights)]
    if any(v.dtype.kind not in "fiu" for v in raw):
        raise ValueError("finite aligned numeric supervised rows required")
    x, base, gate, teacher, weight = [np.array(v, dtype=np.float64, copy=True) for v in raw]
    if (
        x.ndim != 2
        or not 32 <= len(x) <= 200000
        or not 1 <= x.shape[1] <= 512
        or base.ndim != 2
        or base.shape[0] != len(x)
        or not 1 <= base.shape[1] <= 64
        or teacher.shape != base.shape
        or gate.shape != (len(x),)
        or weight.shape != (len(x),)
        or not all(
            np.isfinite(v).all() and np.max(np.abs(v)) <= 1e6
            for v in (x, base, gate, teacher, weight)
        )
        or np.any((gate < 0) | (gate > 1))
        or np.any(weight < 0)
        or np.count_nonzero(weight > 0) < 32
    ):
        raise ValueError("finite aligned complete supervised rows required")
    if type(layers) not in (list, tuple) or not 1 <= len(layers) <= 4:
        raise ValueError("complete bounded residual layers required")
    owned = []
    width = x.shape[1]
    for layer in layers:
        if type(layer) not in (list, tuple) or len(layer) != 2:
            raise ValueError("complete bounded residual layers required")
        values = [np.asarray(v) for v in layer]
        if any(v.dtype.kind not in "fiu" for v in values):
            raise ValueError("finite aligned residual layers required")
        w, b = [np.array(v, dtype=np.float64, copy=True) for v in values]
        if (
            w.ndim != 2
            or w.shape[1] != width
            or not 1 <= w.shape[0] <= 512
            or b.shape != (w.shape[0],)
            or not all(np.isfinite(v).all() and np.max(np.abs(v)) <= 1e6 for v in (w, b))
        ):
            raise ValueError("finite aligned residual layers required")
        width = w.shape[0]
        owned.append((w, b))
    if width != base.shape[1]:
        raise ValueError("residual output must match teacher actions")
    positive = weight > 0
    input_rows, selected_rows = len(x), int(positive.sum())
    x, base, gate, teacher, weight = [v[positive] for v in (x, base, gate, teacher, weight)]
    weight /= weight.mean()
    import torch

    devices = []
    if config.compute_device != "cpu":
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
            raise ValueError("deterministic cuBLAS configuration required before GPU fitting")
        index = int(config.compute_device.split(":")[1])
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise ValueError("requested supervised CUDA device unavailable")
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
            parameters = [
                (
                    torch.nn.Parameter(
                        torch.tensor(w, dtype=torch.float64, device=config.compute_device)
                    ),
                    torch.nn.Parameter(
                        torch.tensor(b, dtype=torch.float64, device=config.compute_device)
                    ),
                )
                for w, b in owned
            ]
            tx, tb, tg, tt, tw = [
                torch.tensor(v, dtype=torch.float64, device=config.compute_device)
                for v in (x, base, gate, teacher, weight)
            ]

            def prediction(indices):
                hidden = tx[indices]
                for w, b in parameters:
                    hidden = torch.tanh(hidden @ w.T + b)
                return tb[indices] + config.residual_cap * tg[indices, None] * hidden

            def loss_at(indices):
                error = torch.mean((prediction(indices) - tt[indices]) ** 2, dim=1)
                return torch.sum(error * tw[indices]) / torch.sum(tw[indices])

            all_rows = torch.arange(selected_rows, device=config.compute_device)
            with torch.no_grad():
                initial_loss = float(loss_at(all_rows).cpu())
            optimizer = torch.optim.Adam(
                [p for layer in parameters for p in layer], lr=config.learning_rate
            )
            history = []
            order = torch.empty(0, dtype=torch.int64, device=config.compute_device)
            cursor = 0
            for _ in range(config.steps):
                if cursor >= len(order):
                    order = torch.randperm(selected_rows, device=config.compute_device)
                    cursor = 0
                selected = order[cursor : cursor + config.batch_size]
                cursor += len(selected)
                loss = loss_at(selected)
                if not bool(torch.isfinite(loss)):
                    raise ValueError("nonfinite supervised residual loss")
                optimizer.zero_grad()
                loss.backward()
                if not all(
                    p.grad is not None and bool(torch.isfinite(p.grad).all())
                    for layer in parameters
                    for p in layer
                ):
                    raise ValueError("nonfinite supervised residual gradient")
                torch.nn.utils.clip_grad_norm_(
                    [p for layer in parameters for p in layer], 1.0, error_if_nonfinite=True
                )
                optimizer.step()
                history.append(float(loss.detach().cpu()))
            with torch.no_grad():
                final_loss = float(loss_at(all_rows).cpu())
            fitted = [
                {
                    "weight": w.detach().cpu().numpy().tolist(),
                    "bias": b.detach().cpu().numpy().tolist(),
                }
                for w, b in parameters
            ]
            if not np.isfinite(final_loss) or any(
                not np.isfinite(np.asarray(v)).all() for layer in fitted for v in layer.values()
            ):
                raise ValueError("nonfinite fitted supervised residual")
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    return {
        "algorithm": "DATASET_WEIGHTED_BOUNDED_RESIDUAL_IMITATION_V1",
        "source_hash": "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "layers": fitted,
        "fitted_layers_hash": "sha256:"
        + hashlib.sha256(
            json.dumps(fitted, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest(),
        "config": {
            "steps": config.steps,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "residual_cap": config.residual_cap,
            "seed": config.seed,
            "compute_device": config.compute_device,
        },
        "original_input_rows": input_rows,
        "positive_weight_rows": selected_rows,
        "zero_weight_rows": input_rows - selected_rows,
        "initial_full_positive_weight_loss": initial_loss,
        "final_full_positive_weight_loss": final_loss,
        "minibatch_loss_history": history,
        "completed_optimizer_steps": config.steps,
        "training_loss_improved": final_loss < initial_loss,
        "teacher_rows_are_not_new_physical_episodes": True,
        "zero_weight_rows_are_not_negative_gradient_examples": True,
        "temporal_teacher_causality_verified": False,
        "physical_batch_verified": False,
        "online_rl_claimed": False,
        "on_policy_ppo": False,
        "kl_trust_region_applied": False,
        "activation_ceiling": "SIM_ONLY",
        "runtime_execution_authorized": False,
        "promotion_authorized": False,
        "hardware_authorized": False,
    }
