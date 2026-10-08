"""Task-neutral positive loss weights; never select or drop training rows."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def validate_sample_weights(value: Any, row_count: int) -> np.ndarray[Any, Any]:
    weights = np.asarray(value)
    if (
        type(row_count) is not int
        or not 4 <= row_count <= 200000
        or weights.shape != (row_count,)
        or weights.dtype.kind not in "fiu"
        or not np.isfinite(weights).all()
        or np.any(weights < 1 / 16)
        or np.any(weights > 16)
        or not np.isclose(weights.mean(), 1.0, atol=1e-12, rtol=0)
    ):
        raise ValueError("aligned positive bounded unit-mean sample weights required")
    owned = np.array(weights, dtype=np.float64, copy=True)
    owned.flags.writeable = False
    return owned


def balanced_partition_weights(partitions: Any) -> np.ndarray[Any, Any]:
    """Give every observed partition equal total loss mass, retaining all rows.

    Partitions are caller-defined integer labels, not rewards or robot roles.
    Reject excessive imbalance rather than silently clipping or dropping rows.
    """
    labels = np.asarray(partitions)
    if (
        labels.ndim != 1
        or not 4 <= len(labels) <= 200000
        or labels.dtype.kind not in "iu"
        or np.any(labels < 0)
        or np.any(labels >= 16)
    ):
        raise ValueError("bounded integer partition labels required")
    unique, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    weights = len(labels) / (len(unique) * counts[inverse])
    return validate_sample_weights(weights, len(labels))


def sample_weight_receipt(weights: Any) -> dict[str, Any]:
    values = np.asarray(weights)
    if values.ndim != 1:
        raise ValueError("aligned sample-weight receipt required")
    values = validate_sample_weights(values, len(values))
    payload = json.dumps(values.tolist(), separators=(",", ":"), allow_nan=False).encode()
    return {
        "schema": "rosclaw.growth.positive_sample_weighting.v1",
        "row_count": len(values),
        "sample_weight_hash": "sha256:" + hashlib.sha256(payload).hexdigest(),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "all_numeric_rows_retained": True,
        "physical_batch_verified": False,
        "runtime_execution_authorized": False,
        "promotion_authorized": False,
        "hardware_authorized": False,
    }
