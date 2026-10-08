"""Independent offline critic regression, never policy learning or execution.

Fixed full-dataset target normalization preserves initial predictions in the
original units. This borrows output-preserving parametrization from PopArt
(https://arxiv.org/abs/1602.07714), but is not adaptive online PopArt: statistics
are frozen for the fit. Rewards/targets are not clipped or relabelled. Callers
must authenticate data and evaluate any resulting policy separately.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.growth.recurrent_clipped_actor_critic import _json_hash, _value_parameters


@dataclass(frozen=True)
class NormalizedValueRegressionConfig:
    steps: int = 512
    batch_episodes: int = 8
    learning_rate: float = 0.001
    minimum_target_scale: float = 0.0001
    maximum_gradient_norm: float = 1.0
    seed: int = 0
    compute_device: str = "cpu"

    def validate(self) -> None:
        integers = ((self.steps, 1, 8192), (self.batch_episodes, 1, 64), (self.seed, 0, 2**32 - 1))
        floats = (
            (self.learning_rate, 1e-6, 0.01),
            (self.minimum_target_scale, 1e-6, 1e6),
            (self.maximum_gradient_norm, 0.001, 100.0),
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
            raise ValueError("bounded explicit independent value regression configuration required")


def fit_normalized_value_regression(
    *,
    context: Any,
    value_targets: Any,
    critic_parameters: Any,
    config: NormalizedValueRegressionConfig,
) -> dict[str, Any]:
    """Fit only a 64-hidden-unit value predictor with its own optimizer budget.

    All episodes and frames enter the target statistics and final measurements.
    Minibatches contain whole episodes. Returned parameters predict ORIGINAL
    units directly; no runtime normalization wrapper is required. There are no
    actor inputs, likelihood claims, motor commands, gates, or promotion rights.
    Training error is not held-out calibration or evidence of physical gain.
    """
    if type(config) is not NormalizedValueRegressionConfig:
        raise ValueError("explicit normalized value regression configuration required")
    config.validate()
    x, y = np.asarray(context), np.asarray(value_targets)
    if (
        x.ndim != 3
        or not 1 <= x.shape[0] <= 65536
        or not 1 <= x.shape[1] <= 4096
        or not 1 <= x.shape[2] <= 1024
        or x.size > 20_000_000
        or y.shape != x.shape[:2]
        or x.dtype.kind not in "fiu"
        or y.dtype.kind not in "fiu"
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or np.max(np.abs(x.astype(np.float64))) > 1e6
        or np.max(np.abs(y.astype(np.float64))) > 1e6
    ):
        raise ValueError("complete bounded finite episode contexts and value targets required")
    x, y = np.array(x, dtype=np.float64, copy=True), np.array(y, dtype=np.float64, copy=True)
    original = _value_parameters(critic_parameters, x.shape[2])
    mean = float(y.mean())
    scale = max(float(y.std()), config.minimum_target_scale)
    normalized_target = (y - mean) / scale
    if not np.isfinite(normalized_target).all():
        raise ValueError("finite normalized value targets required")
    normalized = {k: v.copy() for k, v in original.items()}
    normalized["weight_1"] /= scale
    normalized["bias_1"] = (normalized["bias_1"] - mean) / scale

    def predict(parameters: Any) -> Any:
        hidden = np.tanh(x @ parameters["weight_0"].T + parameters["bias_0"])
        return (hidden @ parameters["weight_1"].T + parameters["bias_1"]).squeeze(-1)

    initial = predict(original)
    restored = predict(normalized) * scale + mean
    preservation_error = float(np.max(np.abs(initial - restored)))
    if not np.isfinite(initial).all() or not np.allclose(initial, restored, rtol=1e-12, atol=1e-8):
        raise ValueError("initial original-unit value predictions must be preserved")
    data_hash = _json_hash(
        {
            name: {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "bytes_hash": "sha256:" + hashlib.sha256(array.tobytes(order="C")).hexdigest(),
            }
            for name, array in (("context", x), ("value_targets", y))
        }
    )
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as error:
        raise RuntimeError("independent value regression requires optional torch") from error
    device = torch.device(config.compute_device)
    if device.type == "cuda" and (
        not torch.cuda.is_available()
        or device.index is None
        or device.index >= torch.cuda.device_count()
    ):
        raise ValueError("explicit available value fitting device required")
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    threads = torch.get_num_threads()
    history = []
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[] if device.type == "cpu" else [device.index]):
            features = torch.as_tensor(x, dtype=torch.float64, device=device)
            targets = torch.as_tensor(normalized_target, dtype=torch.float64, device=device)
            parameters = {
                k: torch.tensor(v, dtype=torch.float64, device=device, requires_grad=True)
                for k, v in normalized.items()
            }
            optimizer = torch.optim.Adam(parameters.values(), lr=config.learning_rate)
            generator = torch.Generator(device="cpu").manual_seed(config.seed)
            order, cursor = torch.empty(0, dtype=torch.int64), 0
            for _ in range(config.steps):
                if cursor >= len(order):
                    order, cursor = torch.randperm(len(x), generator=generator), 0
                batch = order[cursor : cursor + config.batch_episodes].to(device)
                cursor += len(batch)
                hidden = torch.tanh(
                    functional.linear(features[batch], parameters["weight_0"], parameters["bias_0"])
                )
                prediction = functional.linear(
                    hidden, parameters["weight_1"], parameters["bias_1"]
                ).squeeze(-1)
                loss = ((prediction - targets[batch]) ** 2).mean()
                if not bool(torch.isfinite(loss)):
                    raise ValueError("finite independent value regression objective required")
                optimizer.zero_grad()
                loss.backward()
                if not all(
                    v.grad is not None and bool(torch.isfinite(v.grad).all())
                    for v in parameters.values()
                ):
                    raise ValueError("finite independent value regression gradients required")
                torch.nn.utils.clip_grad_norm_(
                    list(parameters.values()), config.maximum_gradient_norm, error_if_nonfinite=True
                )
                optimizer.step()
                if not all(bool(torch.isfinite(v).all()) for v in parameters.values()):
                    raise ValueError("finite fitted value regression parameters required")
                history.append(float(loss.detach().cpu()))
            fitted = {k: v.detach().cpu().numpy().copy() for k, v in parameters.items()}
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.set_num_threads(threads)
    fitted["weight_1"] *= scale
    fitted["bias_1"] = fitted["bias_1"] * scale + mean
    exported = {k: v.tolist() for k, v in fitted.items()}
    _value_parameters(exported, x.shape[2])
    final = predict(fitted)
    mse = float(np.mean((final - y) ** 2))
    if not np.isfinite(final).all() or not np.isfinite(mse):
        raise ValueError("finite original-unit final value predictions required")
    return {
        "algorithm": "INDEPENDENT_FIXED_NORMALIZED_VALUE_REGRESSION_V1",
        "critic_parameters": exported,
        "config": asdict(config),
        "source_hash": "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "parameter_contract_source_hash": "sha256:"
        + hashlib.sha256(
            Path(__file__).with_name("recurrent_clipped_actor_critic.py").read_bytes()
        ).hexdigest(),
        "input_numeric_hash": data_hash,
        "original_critic_parameters_hash": _json_hash({k: v.tolist() for k, v in original.items()}),
        "fitted_critic_parameters_hash": _json_hash(exported),
        "episode_count": len(x),
        "episode_horizon": x.shape[1],
        "frame_rows": y.size,
        "target_mean": mean,
        "target_scale": scale,
        "maximum_initial_output_preservation_error": preservation_error,
        "initial_value_mse": float(np.mean((initial - y) ** 2)),
        "final_value_mse": mse,
        "training_mean_baseline_mse": float(np.mean((y - mean) ** 2)),
        "accepted_value_optimizer_steps": len(history),
        "normalized_training_loss_history": history,
        "returned_predictions_in_original_units": True,
        "all_rows_in_target_statistics_and_final_measurements": True,
        "adaptive_online_popart": False,
        "actor_optimizer_updates": 0,
        "physical_batch_verified": False,
        "held_out_calibration_verified": False,
        "physical_gain_verified": False,
        "promotion_authorized": False,
        "runtime_execution_authorized": False,
        "hardware_authorized": False,
    }
