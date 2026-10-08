"""Offline MC target credit from context-excluded normalized value fits.

This validates numeric fit bindings and recomputes predictions, not optimizer
execution or physical provenance. Callers authenticate complete trajectories,
declare folds before fitting, and retain the physical promotion gates. Context
IDs are offline split labels, never actor observations. Initial critic training
history is not established here; these folds are not a private Fresh exam.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.growth.normalized_value_regression import NormalizedValueRegressionConfig
from rosclaw.growth.recurrent_clipped_actor_critic import _json_hash, _value_parameters


def _numeric_hash(context: Any, targets: Any) -> str:
    return _json_hash(
        {
            name: {
                "shape": list(a.shape),
                "dtype": str(a.dtype),
                "bytes_hash": "sha256:" + hashlib.sha256(a.tobytes(order="C")).hexdigest(),
            }
            for name, a in (("context", context), ("value_targets", targets))
        }
    )


_VALUE_PREDICTION_FRAME_BATCH = 65536


def _predict(context: Any, parameters: Any) -> Any:
    # Bound hidden activations independently of input feature width. Splitting
    # only the episode axis preserves each chronological matrix multiplication;
    # never reshape, reorder or discard frames to reduce memory.
    episodes_per_batch = max(1, _VALUE_PREDICTION_FRAME_BATCH // context.shape[1])
    predictions = np.empty(context.shape[:2], dtype=np.float64)
    for start in range(0, len(context), episodes_per_batch):
        end = start + episodes_per_batch
        hidden = np.tanh(context[start:end] @ parameters["weight_0"].T + parameters["bias_0"])
        predictions[start:end] = (hidden @ parameters["weight_1"].T + parameters["bias_1"]).squeeze(
            -1
        )
    return predictions


def cross_fitted_value_advantages(
    *,
    context: Any,
    value_targets: Any,
    episode_context_ids: Any,
    initial_critic_parameters: Any,
    fold_fits: Any,
) -> dict[str, Any]:
    """Subtract held-out state values without fitting or modifying any actor.

    Every repeated episode of a context stays in its context_id modulo K fold.
    The caller supplies exactly K predeclared fits (2..10), each with fold_index,
    sorted train_context_ids/held_out_context_ids, and a value_fit receipt from
    fit_normalized_value_regression. Full-data fits cannot supply the baseline.
    Targets may vary chronologically inside a complete finite-task episode.
    They are not clipped, rescaled or regenerated; only advantages are globally
    centered and scaled with the fixed 1e-6 standard-deviation floor.
    """
    x, y, ids = (np.asarray(v) for v in (context, value_targets, episode_context_ids))
    if (
        x.ndim != 3
        or not 1 <= x.shape[0] <= 65536
        or not 1 <= x.shape[1] <= 4096
        or not 1 <= x.shape[2] <= 1024
        or x.size > 20_000_000
        or y.shape != x.shape[:2]
        or ids.shape != (x.shape[0],)
        or ids.dtype.kind not in "iu"
        or np.any(ids < 0)
        or np.any(ids > 2**31 - 1)
        or any(a.dtype.kind not in "fiu" or not np.isfinite(a).all() for a in (x, y))
        or np.max(np.abs(x.astype(np.float64))) > 1e6
        or np.max(np.abs(y.astype(np.float64))) > 1e6
        or type(fold_fits) is not list
        or not 2 <= len(fold_fits) <= 10
    ):
        raise ValueError("complete bounded finite contexts, targets and predeclared folds required")
    x, y, ids = (
        np.array(a, dtype=d, copy=True)
        for a, d in ((x, np.float64), (y, np.float64), (ids, np.int64))
    )
    labels = np.unique(ids)
    folds = ids % len(fold_fits)
    if len(labels) < len(fold_fits) or set(folds.tolist()) != set(range(len(fold_fits))):
        raise ValueError("every fixed context fold needs nonempty held-out and training support")
    initial = _value_parameters(initial_critic_parameters, x.shape[2])
    initial_hash = _json_hash({k: a.tolist() for k, a in initial.items()})
    value_source = (
        "sha256:"
        + hashlib.sha256(
            Path(__file__).with_name("normalized_value_regression.py").read_bytes()
        ).hexdigest()
    )
    contract_source = (
        "sha256:"
        + hashlib.sha256(
            Path(__file__).with_name("recurrent_clipped_actor_critic.py").read_bytes()
        ).hexdigest()
    )
    predictions, mean_baseline = np.empty_like(y), np.empty_like(y)
    bindings = []
    for fold, fitted in enumerate(fold_fits):
        test = folds == fold
        train = ~test
        held_ids = labels[labels % len(fold_fits) == fold].tolist()
        train_ids = labels[labels % len(fold_fits) != fold].tolist()
        if (
            type(fitted) is not dict
            or type(fitted.get("fold_index")) is not int
            or fitted["fold_index"] != fold
            or any(
                type(fitted.get(k)) is not list or any(type(v) is not int for v in fitted[k])
                for k in ("train_context_ids", "held_out_context_ids")
            )
            or fitted["train_context_ids"] != train_ids
            or fitted["held_out_context_ids"] != held_ids
            or type(fitted.get("value_fit")) is not dict
        ):
            raise ValueError("exact ordered whole-context excluded fit declarations required")
        receipt = fitted["value_fit"]
        if type(receipt.get("config")) is not dict or set(receipt["config"]) != set(
            asdict(NormalizedValueRegressionConfig())
        ):
            raise ValueError("complete original value fit configuration required")
        try:
            config = NormalizedValueRegressionConfig(**receipt["config"])
            config.validate()
        except (KeyError, TypeError) as error:
            raise ValueError("complete original value fit configuration required") from error
        tx, ty = x[train], y[train]
        mean = float(ty.mean())
        scale = max(float(ty.std()), config.minimum_target_scale)
        if (
            receipt.get("algorithm") != "INDEPENDENT_FIXED_NORMALIZED_VALUE_REGRESSION_V1"
            or receipt.get("source_hash") != value_source
            or receipt.get("parameter_contract_source_hash") != contract_source
            or receipt.get("input_numeric_hash") != _numeric_hash(tx, ty)
            or receipt.get("original_critic_parameters_hash") != initial_hash
            or type(receipt.get("target_mean")) is not float
            or type(receipt.get("target_scale")) is not float
            or receipt.get("target_mean") != mean
            or receipt.get("target_scale") != scale
            or any(
                type(receipt.get(k)) is not int or receipt[k] != v
                for k, v in (
                    ("episode_count", len(tx)),
                    ("episode_horizon", x.shape[1]),
                    ("frame_rows", ty.size),
                    ("accepted_value_optimizer_steps", config.steps),
                    ("actor_optimizer_updates", 0),
                )
            )
            or receipt.get("returned_predictions_in_original_units") is not True
            or receipt.get("all_rows_in_target_statistics_and_final_measurements") is not True
            or any(
                receipt.get(k) is not False
                for k in (
                    "adaptive_online_popart",
                    "physical_batch_verified",
                    "held_out_calibration_verified",
                    "physical_gain_verified",
                    "promotion_authorized",
                    "runtime_execution_authorized",
                    "hardware_authorized",
                )
            )
        ):
            raise ValueError("value fit must bind exactly its excluded-context training subset")
        history = receipt.get("normalized_training_loss_history")
        if (
            type(history) is not list
            or len(history) != config.steps
            or any(type(v) not in (int, float) or not np.isfinite(v) or v < 0 for v in history)
        ):
            raise ValueError("complete finite value optimizer loss receipt required")
        parameters = _value_parameters(receipt.get("critic_parameters"), x.shape[2])
        parameter_hash = _json_hash({k: a.tolist() for k, a in parameters.items()})
        if parameter_hash != receipt.get("fitted_critic_parameters_hash"):
            raise ValueError("complete actual fitted value parameters must match receipt")
        for key, measured in (
            ("initial_value_mse", float(np.mean((_predict(tx, initial) - ty) ** 2))),
            ("final_value_mse", float(np.mean((_predict(tx, parameters) - ty) ** 2))),
            ("training_mean_baseline_mse", float(np.mean((ty - mean) ** 2))),
        ):
            reported = receipt.get(key)
            if (
                type(reported) not in (int, float)
                or not np.isfinite(reported)
                or not np.isclose(reported, measured, rtol=1e-10, atol=1e-8)
            ):
                raise ValueError("original-unit training measurements must match actual parameters")
        predictions[test] = _predict(x[test], parameters)
        mean_baseline[test] = mean
        bindings.append(
            {
                "fold_index": fold,
                "train_context_ids": train_ids,
                "held_out_context_ids": held_ids,
                "training_episode_indices": np.flatnonzero(train).tolist(),
                "predicted_episode_indices": np.flatnonzero(test).tolist(),
                "value_fit_numeric_hash": receipt["input_numeric_hash"],
                "value_fit_receipt_hash": _json_hash(receipt),
                "critic_parameter_hash": parameter_hash,
            }
        )
    raw = y - predictions
    advantages = (raw - float(raw.mean())) / max(float(raw.std()), 1e-6)
    if not np.isfinite(predictions).all() or not np.isfinite(advantages).all():
        raise ValueError("finite original-unit held-out credit required")
    for a in (predictions, raw, advantages, folds):
        a.flags.writeable = False
    return {
        "algorithm": "CONTEXT_EXCLUDED_FIXED_NORMALIZED_VALUE_MC_CREDIT_V1",
        "predictions": predictions,
        "raw_advantages": raw,
        "advantages": advantages,
        "episode_fold_ids": folds,
        "fold_bindings": bindings,
        "input_numeric_hash": _numeric_hash(x, y),
        "episode_context_ids_hash": "sha256:" + hashlib.sha256(ids.tobytes()).hexdigest(),
        "initial_critic_parameters_hash": initial_hash,
        "source_hash": "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "value_fit_source_hash": value_source,
        "parameter_contract_source_hash": contract_source,
        "held_out_mse": float(np.mean(raw**2)),
        "held_out_training_mean_baseline_mse": float(np.mean((y - mean_baseline) ** 2)),
        "raw_advantage_mean": float(raw.mean()),
        "raw_advantage_std": float(raw.std()),
        "all_supplied_rows_preserved": True,
        "context_ids_are_actor_observations": False,
        "critic_fits_performed_here": 0,
        "actor_optimizer_updates": 0,
        "optimizer_execution_independently_verified": False,
        "physical_batch_verified": False,
        "private_fresh_verified": False,
        "physical_gain_verified": False,
        "runtime_execution_authorized": False,
        "promotion_authorized": False,
        "hardware_authorized": False,
    }
