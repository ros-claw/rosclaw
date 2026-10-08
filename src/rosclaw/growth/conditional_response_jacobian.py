"""Optional offline local-response learning: effect = J(state) @ intervention.

Paired response targets and physical provenance belong to the adapter. This
predictor is not a policy, a differentiable simulator, a verified causal model
outside its measured domain, online RL, or execution authority.
"""

import hashlib
import re
from pathlib import Path
from typing import Any

import numpy as np

import rosclaw.growth.context_prediction_mlp as prediction_support
from rosclaw.growth.context_prediction_mlp import _hash, _matrix


def _source() -> str:
    return "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _support_source() -> str:
    return "sha256:" + hashlib.sha256(Path(prediction_support.__file__).read_bytes()).hexdigest()


def predict(model: dict[str, Any], states: Any, interventions: Any) -> np.ndarray[Any, Any]:
    if (
        type(model) is not dict
        or model.get("schema") != "rosclaw.growth.conditional_response_jacobian.v1"
        or model.get("source_hash") != _source()
        or model.get("dependency_source_hash") != _support_source()
        or model.get("model_hash") != _hash({k: v for k, v in model.items() if k != "model_hash"})
        or model.get("prediction_only") is not True
        or model.get("local_linear_intervention_model") is not True
        or model.get("activation_ceiling") != "SIM_ONLY"
        or any(
            model.get(k) is not False
            for k in ("motor_policy", "promotion_authorized", "hardware_authorized")
        )
    ):
        raise ValueError("sealed source-bound prediction-only response model required")
    x, a = _matrix(states), _matrix(interventions)
    if len(x) != len(a):
        raise ValueError("aligned states and interventions required")
    try:
        mean, scale, action_scale, effect_scale = [
            np.asarray(model[k], dtype=np.float64)
            for k in ("state_mean", "state_scale", "intervention_scale", "effect_scale")
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("finite aligned response normalization required") from exc
    width, action_width = x.shape[1], a.shape[1]
    if (
        mean.shape != (width,)
        or scale.shape != (width,)
        or action_scale.shape != (action_width,)
        or effect_scale.ndim != 1
        or not 1 <= action_width <= 64
        or not 1 <= len(effect_scale) <= 64
        or not all(np.isfinite(v).all() for v in (mean, scale, action_scale, effect_scale))
        or any(np.any(v <= 0) for v in (scale, action_scale, effect_scale))
    ):
        raise ValueError("positive finite aligned response normalization required")
    hidden = ((x - mean) / scale).astype(np.float32)
    normalized_action = a / action_scale
    if not all(
        np.isfinite(v).all() and np.max(np.abs(v)) <= 1e6 for v in (hidden, normalized_action)
    ):
        raise ValueError("normalized response inputs exceed finite envelope")
    layers = model.get("layers")
    if type(layers) is not list or len(layers) != 3:
        raise ValueError("complete response network required")
    for index, (layer, next_width) in enumerate(
        zip(layers, (128, 64, len(effect_scale) * action_width), strict=True)
    ):
        if type(layer) is not dict or set(layer) != {"weight", "bias"}:
            raise ValueError("complete finite response layer required")
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
            raise ValueError("finite aligned response layer required")
        hidden = hidden @ weight.T + bias
        if not np.isfinite(hidden).all():
            raise ValueError("nonfinite response network output")
        if index < 2:
            hidden = np.tanh(hidden)
        width = next_width
    jacobian = hidden.astype(np.float64).reshape(len(x), len(effect_scale), action_width)
    result = np.einsum("noi,ni->no", jacobian, normalized_action) * effect_scale
    if not np.isfinite(result).all():
        raise ValueError("nonfinite predicted local response")
    return result


def fit_conditional_response_jacobian(
    states: Any,
    interventions: Any,
    effects: Any,
    context_ids: Any,
    *,
    held_out_contexts: tuple[int, ...],
    seed: int = 0,
    epochs: int = 100,
    batch_size: int = 1024,
    learning_rate: float = 0.001,
    device: str = "cpu",
) -> dict[str, Any]:
    """Train J on measured effects; training contexts alone define scales/prior.

    The fixed global least-squares Jacobian is both the initial final-layer
    bias and an explicit baseline. Action/effect normalization does not subtract
    means, preserving the exact zero-response reference throughout training.
    """
    x, a, y = _matrix(states), _matrix(interventions), _matrix(effects)
    groups = np.asarray(context_ids)
    if (
        len(a) != len(x)
        or len(y) != len(x)
        or not 1 <= a.shape[1] <= 64
        or not 1 <= y.shape[1] <= 64
        or groups.shape != (len(x),)
        or groups.dtype.kind not in "iu"
        or type(held_out_contexts) is not tuple
        or not held_out_contexts
        or any(type(v) is not int for v in held_out_contexts)
        or len(set(held_out_contexts)) != len(held_out_contexts)
        or not set(held_out_contexts) <= set(groups.tolist())
    ):
        raise ValueError("aligned paired effects and explicit disjoint contexts required")
    train = ~np.isin(groups, held_out_contexts)
    if int(train.sum()) < 32 or int((~train).sum()) < 16:
        raise ValueError("at least 32 training and 16 held rows required")
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
        or re.fullmatch(r"cpu|cuda:[0-9]{1,2}", device) is None
    ):
        raise ValueError("bounded explicit response fitting configuration required")
    mean, scale = x[train].mean(axis=0), x[train].std(axis=0)
    scale = np.where(scale >= 1e-8, scale, 1.0)
    action_scale = np.sqrt(np.mean(a[train] ** 2, axis=0))
    effect_scale = np.sqrt(np.mean(y[train] ** 2, axis=0))
    action_scale, effect_scale = [
        np.where(v >= 1e-12, v, 1.0) for v in (action_scale, effect_scale)
    ]
    nx, na, ny = (x[train] - mean) / scale, a[train] / action_scale, y[train] / effect_scale
    if not all(np.isfinite(v).all() and np.max(np.abs(v)) <= 1e6 for v in (nx, na, ny)):
        raise ValueError("finite bounded normalized response training required")
    global_jacobian = np.linalg.lstsq(na, ny, rcond=1e-8)[0].T
    import torch

    gpu = [] if device == "cpu" else [int(device.split(":")[1])]
    if gpu and (not torch.cuda.is_available() or gpu[0] >= torch.cuda.device_count()):
        raise ValueError("requested response GPU unavailable")
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
                torch.nn.Linear(64, y.shape[1] * a.shape[1]),
            ).to(device)
            with torch.no_grad():
                network[-1].weight.zero_()
                network[-1].bias.copy_(
                    torch.tensor(global_jacobian.reshape(-1), dtype=torch.float32, device=device)
                )
            tx, ta, ty = [torch.tensor(v, dtype=torch.float32, device=device) for v in (nx, na, ny)]
            optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
            updates = 0
            for _ in range(epochs):
                indices = torch.randperm(len(tx))
                for start in range(0, len(tx), batch_size):
                    selected = indices[start : start + batch_size].to(device)
                    jacobian = network(tx[selected]).reshape(len(selected), y.shape[1], a.shape[1])
                    response = torch.einsum("noi,ni->no", jacobian, ta[selected])
                    loss = torch.mean((response - ty[selected]) ** 2)
                    if not bool(torch.isfinite(loss)):
                        raise ValueError("nonfinite response fitting loss")
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
        "schema": "rosclaw.growth.conditional_response_jacobian.v1",
        "source_hash": _source(),
        "dependency_source_hash": _support_source(),
        "state_mean": mean.tolist(),
        "state_scale": scale.tolist(),
        "intervention_scale": action_scale.tolist(),
        "effect_scale": effect_scale.tolist(),
        "layers": layers,
        "prediction_only": True,
        "local_linear_intervention_model": True,
        "activation_ceiling": "SIM_ONLY",
        "motor_policy": False,
        "promotion_authorized": False,
        "hardware_authorized": False,
    }
    model["model_hash"] = _hash(model)
    predicted = predict(model, x[~train], a[~train])
    per = np.mean(((predicted - y[~train]) / effect_scale) ** 2, axis=0)
    global_prediction = (a[~train] / action_scale) @ global_jacobian.T * effect_scale
    global_per = np.mean(((global_prediction - y[~train]) / effect_scale) ** 2, axis=0)
    zero_per = np.mean((y[~train] / effect_scale) ** 2, axis=0)
    if not all(np.isfinite(v).all() for v in (per, global_per, zero_per)):
        raise ValueError("nonfinite held response metrics")
    return {
        "model": model,
        "optimizer_updates": updates,
        "algorithm": "CONTEXT_DISJOINT_LOCAL_LINEAR_RESPONSE_NOT_RL",
        "train_contexts": sorted(set(groups[train].tolist())),
        "held_out_contexts": list(held_out_contexts),
        "train_rows": int(train.sum()),
        "held_out_rows": int((~train).sum()),
        "held_out_standardized_mse": float(per.mean()),
        "held_out_mse_per_dimension": per.tolist(),
        "global_jacobian_standardized_mse": float(global_per.mean()),
        "global_jacobian_mse_per_dimension": global_per.tolist(),
        "zero_effect_standardized_mse": float(zero_per.mean()),
        "zero_effect_mse_per_dimension": zero_per.tolist(),
        "global_jacobian_training_contexts_only": global_jacobian.tolist(),
        "all_normalization_training_contexts_only": True,
        "zero_intervention_exact_zero_response": True,
        "hyperparameters_chosen_on_holdout": False,
        "motor_policy_updates": 0,
        "promotion_authorized": False,
        "hardware_authorized": False,
    }
