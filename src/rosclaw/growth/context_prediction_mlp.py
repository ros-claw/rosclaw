"""Optional offline context-disjoint neural prediction, never a motor policy.

Features/targets and their physical meaning belong to the downstream adapter.
Held-out contexts never enter normalization, optimization, or model selection.
No simulator, robot transport, policy-gradient density, or activation is used.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


def _hash(value: dict[str, Any]) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    )


def _matrix(value: Any) -> np.ndarray[Any, Any]:
    raw = np.asarray(value)
    if (
        raw.ndim != 2
        or raw.dtype.kind not in "fiu"
        or not 1 <= raw.shape[0] <= 200000
        or not 1 <= raw.shape[1] <= 8192
        or raw.size > 200000000
    ):
        raise ValueError("bounded numeric prediction matrix required")
    result = raw.astype(np.float64, copy=True)
    if not np.isfinite(result).all() or np.max(np.abs(result)) > 1e6:
        raise ValueError("finite bounded prediction matrix required")
    return result


def predict(model: dict[str, Any], features: Any) -> np.ndarray[Any, Any]:
    if (
        type(model) is not dict
        or model.get("schema") != "rosclaw.growth.context_prediction_mlp.v1"
        or model.get("source_hash")
        != "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        or model.get("model_hash") != _hash({k: v for k, v in model.items() if k != "model_hash"})
        or model.get("prediction_only") is not True
        or model.get("activation_ceiling") != "SIM_ONLY"
        or any(
            model.get(k) is not False
            for k in ("motor_policy", "promotion_authorized", "hardware_authorized")
        )
    ):
        raise ValueError("sealed source-bound prediction-only network required")
    x = _matrix(features)
    try:
        input_mean, input_scale, target_mean, target_scale = [
            np.asarray(model[k], dtype=np.float64)
            for k in ("input_mean", "input_scale", "target_mean", "target_scale")
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("finite aligned prediction normalization required") from exc
    if target_mean.ndim != 1:
        raise ValueError("finite aligned prediction normalization required")
    width, output = x.shape[1], len(target_mean)
    if (
        input_mean.shape != (width,)
        or input_scale.shape != (width,)
        or not 1 <= output <= 512
        or target_scale.shape != (output,)
        or not all(
            np.isfinite(v).all() for v in (input_mean, input_scale, target_mean, target_scale)
        )
        or np.any(input_scale <= 0)
        or np.any(target_scale <= 0)
    ):
        raise ValueError("finite aligned prediction normalization required")
    hidden = ((x - input_mean) / input_scale).astype(np.float32)
    if not np.isfinite(hidden).all() or np.max(np.abs(hidden)) > 1e6:
        raise ValueError("prediction features exceed normalized numeric envelope")
    layers = model.get("layers")
    if type(layers) is not list or len(layers) != 3:
        raise ValueError("complete finite prediction network required")
    for i, (layer, next_width) in enumerate(zip(layers, (128, 64, output), strict=True)):
        if type(layer) is not dict or set(layer) != {"weight", "bias"}:
            raise ValueError("finite aligned prediction network required")
        weight, bias = (
            np.asarray(layer["weight"], dtype=np.float32),
            np.asarray(layer["bias"], dtype=np.float32),
        )
        if (
            weight.shape != (next_width, width)
            or bias.shape != (next_width,)
            or not np.isfinite(weight).all()
            or not np.isfinite(bias).all()
        ):
            raise ValueError("finite aligned prediction network required")
        hidden = hidden @ weight.T + bias
        if not np.isfinite(hidden).all():
            raise ValueError("nonfinite intermediate neural prediction")
        if i < 2:
            hidden = np.tanh(hidden)
        width = next_width
    result = hidden.astype(np.float64) * target_scale + target_mean
    if not np.isfinite(result).all():
        raise ValueError("nonfinite neural prediction")
    return result


def fit_context_predictor(
    features: Any,
    targets: Any,
    context_ids: Any,
    *,
    held_out_contexts: tuple[int, ...],
    seed: int = 0,
    epochs: int = 200,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    device: str = "cpu",
) -> dict[str, Any]:
    """Fixed-config supervised fit; learning never updates a policy or teacher.

    Run in an isolated training process. Torch is optional until this call;
    pure NumPy prediction works without it. Thread/RNG state is restored.
    """
    x, y = _matrix(features), _matrix(targets)
    groups = np.asarray(context_ids)
    if (
        y.shape[0] != len(x)
        or y.shape[1] > 512
        or groups.shape != (len(x),)
        or groups.dtype.kind not in "iu"
        or type(held_out_contexts) is not tuple
        or not held_out_contexts
        or any(type(v) is not int for v in held_out_contexts)
        or len(set(held_out_contexts)) != len(held_out_contexts)
        or not set(held_out_contexts) <= set(groups.tolist())
    ):
        raise ValueError("explicit disjoint observed context split required")
    train = ~np.isin(groups, held_out_contexts)
    if int(train.sum()) < 32 or int((~train).sum()) < 16:
        raise ValueError("at least 32 training and 16 held-out rows required")
    if (
        type(seed) is not int
        or not 0 <= seed < 2**32
        or type(epochs) is not int
        or not 1 <= epochs <= 2000
        or type(batch_size) is not int
        or not 16 <= batch_size <= 4096
        or type(learning_rate) is not float
        or not 1e-6 <= learning_rate <= 0.05
        or type(device) is not str
        or not re.fullmatch(r"cpu|cuda:[0-9]{1,2}", device)
    ):
        raise ValueError("bounded explicit neural prediction configuration required")
    xm, ym = x[train].mean(axis=0), y[train].mean(axis=0)
    xs, ys = x[train].std(axis=0), y[train].std(axis=0)
    xs, ys = np.where(xs >= 1e-8, xs, 1.0), np.where(ys >= 1e-8, ys, 1.0)
    normalized_x, normalized_y = (x[train] - xm) / xs, (y[train] - ym) / ys
    if not np.isfinite(normalized_x).all() or not np.isfinite(normalized_y).all():
        raise ValueError("nonfinite prediction training normalization")
    import torch

    gpu = [] if device == "cpu" else [int(device.split(":")[1])]
    if gpu and (not torch.cuda.is_available() or gpu[0] >= torch.cuda.device_count()):
        raise ValueError("requested prediction GPU is unavailable")
    threads = torch.get_num_threads()
    try:
        with torch.random.fork_rng(devices=gpu):
            torch.set_num_threads(4)
            torch.random.default_generator.manual_seed(seed)
            if gpu:
                with torch.cuda.device(gpu[0]):
                    torch.cuda.manual_seed(seed)
            network = torch.nn.Sequential(
                torch.nn.Linear(x.shape[1], 128),
                torch.nn.Tanh(),
                torch.nn.Linear(128, 64),
                torch.nn.Tanh(),
                torch.nn.Linear(64, y.shape[1]),
            ).to(device)
            tx, ty = [
                torch.tensor(v, dtype=torch.float32, device=device)
                for v in (normalized_x, normalized_y)
            ]
            optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
            updates = 0
            for _ in range(epochs):
                indices = torch.randperm(len(tx))
                for start in range(0, len(tx), batch_size):
                    selected = indices[start : start + batch_size].to(device)
                    loss = torch.mean((network(tx[selected]) - ty[selected]) ** 2)
                    if not bool(torch.isfinite(loss)):
                        raise ValueError("nonfinite neural prediction loss")
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    updates += 1
            layers = [
                {
                    "weight": layer.weight.detach().cpu().tolist(),
                    "bias": layer.bias.detach().cpu().tolist(),
                }
                for layer in network
                if isinstance(layer, torch.nn.Linear)
            ]
    finally:
        torch.set_num_threads(threads)
    model = {
        "schema": "rosclaw.growth.context_prediction_mlp.v1",
        "source_hash": "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_mean": xm.tolist(),
        "input_scale": xs.tolist(),
        "target_mean": ym.tolist(),
        "target_scale": ys.tolist(),
        "layers": layers,
        "prediction_only": True,
        "motor_policy": False,
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "hardware_authorized": False,
    }
    model["model_hash"] = _hash(model)
    prediction = predict(model, x[~train])
    per_dimension = np.mean(((prediction - y[~train]) / ys) ** 2, axis=0)
    mean_baseline = np.mean(((ym - y[~train]) / ys) ** 2, axis=0)
    zero_baseline = np.mean((y[~train] / ys) ** 2, axis=0)
    if not all(np.isfinite(v).all() for v in (per_dimension, mean_baseline, zero_baseline)):
        raise ValueError("nonfinite held-out prediction metrics")
    return {
        "model": model,
        "train_contexts": sorted(set(groups[train].tolist())),
        "held_out_contexts": list(held_out_contexts),
        "train_rows": int(train.sum()),
        "held_out_rows": int((~train).sum()),
        "optimizer_updates": updates,
        "algorithm": "CONTEXT_DISJOINT_SUPERVISED_MLP_NOT_RL",
        "held_out_standardized_mse": float(per_dimension.mean()),
        "held_out_mse_per_dimension": per_dimension.tolist(),
        "train_mean_baseline_standardized_mse": float(mean_baseline.mean()),
        "zero_baseline_standardized_mse": float(zero_baseline.mean()),
        "input_normalization_training_contexts_only": True,
        "hyperparameters_chosen_on_holdout": False,
        "motor_policy_updates": 0,
        "promotion_authorized": False,
        "hardware_authorized": False,
    }
