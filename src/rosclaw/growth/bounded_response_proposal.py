"""Finite local quadratic proposals, never physical validity or motor authority."""

from typing import Any

import numpy as np


def bounded_response_proposal(
    response_matrix: Any,
    predicted_error: Any,
    objective_weights: Any,
    *,
    maximum_increment: float,
    regularization: float,
) -> dict[str, Any]:
    """Solve a small regularized local model and reject predicted regressions.

    State/action meanings and independent model validity belong downstream.
    Bounds constrain numeric increments only, not physical actuator safety.
    """
    raw = [np.asarray(v) for v in (response_matrix, predicted_error, objective_weights)]
    if any(v.dtype.kind not in "fiu" for v in raw):
        raise ValueError("finite numeric local response proposal inputs required")
    matrix, error, weights = [v.astype(np.float64, copy=True) for v in raw]
    if (
        matrix.ndim != 2
        or not 1 <= matrix.shape[0] <= 512
        or not 1 <= matrix.shape[1] <= 128
        or error.shape != (matrix.shape[0],)
        or weights.shape != error.shape
        or not all(np.isfinite(v).all() for v in (matrix, error, weights))
        or any(np.max(np.abs(v)) > 1e6 for v in (matrix, error, weights))
        or np.any(weights < 0)
        or not np.any(weights > 0)
        or type(maximum_increment) is not float
        or not 0 < maximum_increment <= 0.1
        or type(regularization) is not float
        or not np.isfinite(regularization)
        or not 1e-8 <= regularization <= 1e6
    ):
        raise ValueError("bounded finite aligned local response proposal required")
    weighted = matrix * weights[:, None]
    gram = matrix.T @ weighted + regularization * np.eye(matrix.shape[1])
    gradient = weighted.T @ error
    try:
        delta = -np.linalg.solve(gram, gradient)
    except np.linalg.LinAlgError as exc:
        raise ValueError("finite solvable local response system required") from exc
    delta = np.clip(delta, -maximum_increment, maximum_increment)
    baseline = float(np.sum(weights * error * error))
    candidate = float(
        np.sum(weights * (error + matrix @ delta) ** 2) + regularization * np.sum(delta * delta)
    )
    if not np.isfinite(delta).all() or not np.isfinite(candidate):
        raise ValueError("finite local response proposal result required")
    rejected = candidate > baseline
    if rejected:
        delta = np.zeros(matrix.shape[1])
        candidate = baseline
    return {
        "schema": "rosclaw.growth.bounded_response_proposal.v1",
        "target_increment": delta.tolist(),
        "predicted_baseline_cost": baseline,
        "predicted_candidate_cost": candidate,
        "predicted_regression_rejected": rejected,
        "maximum_increment": maximum_increment,
        "physical_validity_verified": False,
        "runtime_execution_authorized": False,
        "promotion_authorized": False,
        "hardware_authorized": False,
    }
