"""Offline prediction differences with an exact zero-intervention reference.

This imposes an algebraic reference, not accuracy, causal identification,
calibration, model training, a motor policy or execution authority. Callers
authenticate paired physical targets and declare the intervention columns.
"""

from typing import Any

import numpy as np

from rosclaw.growth.context_prediction_mlp import _matrix, predict


def predict_zero_anchored_effect(
    model: dict[str, Any], features: Any, *, intervention_columns: tuple[int, ...]
) -> np.ndarray[Any, Any]:
    """Return f(state, intervention) - f(state, zero), preserving inputs.

    Zero is in original feature units, before the model's own normalization.
    This cannot convert an absolute-state predictor into a verified effect
    model; its adapter must establish the paired-target semantics and evidence.
    """
    actual = _matrix(features)
    if (
        type(intervention_columns) is not tuple
        or not intervention_columns
        or any(type(i) is not int or not 0 <= i < actual.shape[1] for i in intervention_columns)
        or len(set(intervention_columns)) != len(intervention_columns)
    ):
        raise ValueError("explicit unique intervention feature columns required")
    reference = actual.copy()
    reference[:, intervention_columns] = 0.0
    result = predict(model, actual) - predict(model, reference)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite anchored effect prediction")
    return result
