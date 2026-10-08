"""Numeric neural MC credit from whole-context held-out predictions.

No fitting, simulator, actor, activation, or robot IO occurs here. The caller
authenticates physical returns and actual fit receipts. Normalization and fold
identities are checked independently; this is not online TD or GAE.
"""

from typing import Any

import numpy as np

from rosclaw.growth.context_crossfit import context_crossfit_advantages
from rosclaw.growth.context_prediction_mlp import predict


def neural_context_advantages(
    features: Any,
    phases: Any,
    trajectory_ids: Any,
    returns: Any,
    *,
    trajectory_context_ids: Any,
    fold_fit_results: Any,
) -> dict[str, Any]:
    # Retain the original validator and a clearly named linear diagnostic,
    # rather than presenting a linear readout as the actual neural critic.
    linear = context_crossfit_advantages(
        features,
        phases,
        trajectory_ids,
        returns,
        trajectory_context_ids=trajectory_context_ids,
    )
    phi, phase, groups, reward, contexts = [
        np.asarray(v) for v in (features, phases, trajectory_ids, returns, trajectory_context_ids)
    ]
    x = np.column_stack((phi, phase)).astype(np.float64)
    labels = np.unique(contexts)
    sample_contexts = contexts[groups]
    if type(fold_fit_results) is not list or len(fold_fit_results) != 4:
        raise ValueError("four complete context-disjoint neural fit results required")
    predictions = np.zeros(len(x), dtype=np.float64)
    model_hashes = []
    for fold, fitted in enumerate(fold_fit_results):
        held = labels[fold::4].tolist()
        test = np.isin(sample_contexts, held)
        train = ~test
        if (
            type(fitted) is not dict
            or any(
                type(fitted.get(k)) is not list or any(type(v) is not int for v in fitted[k])
                for k in ("held_out_contexts", "train_contexts")
            )
            or fitted.get("held_out_contexts") != held
            or fitted.get("train_contexts") != labels[~np.isin(labels, held)].tolist()
            or fitted.get("train_rows") != int(train.sum())
            or fitted.get("held_out_rows") != int(test.sum())
            or fitted.get("algorithm") != "CONTEXT_DISJOINT_SUPERVISED_MLP_NOT_RL"
            or type(fitted.get("optimizer_updates")) is not int
            or not 1 <= fitted["optimizer_updates"] <= 25000000
            or fitted.get("input_normalization_training_contexts_only") is not True
            or fitted.get("hyperparameters_chosen_on_holdout") is not False
            or fitted.get("motor_policy_updates") != 0
            or type(fitted.get("motor_policy_updates")) is not int
            or any(
                fitted.get(k) is not False for k in ("promotion_authorized", "hardware_authorized")
            )
        ):
            raise ValueError("complete non-authorizing whole-context neural fit required")
        model = fitted.get("model")
        if type(model) is not dict:
            raise ValueError("complete source-bound neural critic required")
        xm, xs = x[train].mean(axis=0), x[train].std(axis=0)
        ym, ys = reward[train].mean(), reward[train].std()
        xs = np.where(xs >= 1e-8, xs, 1.0)
        ys = ys if ys >= 1e-8 else 1.0
        expected = {
            "input_mean": xm,
            "input_scale": xs,
            "target_mean": np.array([ym]),
            "target_scale": np.array([ys]),
        }
        if any(not np.array_equal(np.asarray(model.get(k)), v) for k, v in expected.items()):
            raise ValueError("critic normalization must use exactly its training contexts")
        values = predict(model, x[test])
        if values.shape != (int(test.sum()), 1):
            raise ValueError("scalar raw-return neural critic required")
        predictions[test] = values[:, 0]
        model_hashes.append(model["model_hash"])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            residual = (reward - predictions) / linear["target_scale"]
            advantage = (residual - residual.mean()) / max(float(residual.std()), 1e-6)
            mse = float(np.mean((reward - predictions) ** 2))
        except FloatingPointError as exc:
            raise ValueError("finite neural MC credit required") from exc
    if not np.isfinite(advantage).all() or not np.isfinite(mse):
        raise ValueError("finite neural MC credit required")
    return {
        "advantages": advantage,
        "crossfit_predictions": predictions,
        "target_mean": linear["target_mean"],
        "target_scale": linear["target_scale"],
        "linear_diagnostic_critic_readout": linear["critic_readout"],
        "raw_return_prediction_mse": mse,
        "critic_kind": "WHOLE_CONTEXT_NEURAL_CROSSFIT_MC_NOT_TD_OR_GAE",
        "critic_model_hashes": model_hashes,
        "overlapping_context_count": 0,
        "context_is_actor_observation": False,
        "physical_batch_verified": False,
        "actor_updated": False,
        "runtime_execution_authorized": False,
        "promotion_authorized": False,
        "hardware_authorized": False,
    }
