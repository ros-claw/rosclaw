"""Task-neutral protection of a frozen encoder's linear readout anchors.

Training mathematics only: this neither activates a policy nor proves robot
safety, retention on unseen observations, or promotion eligibility. The encoder
and observation contract must remain frozen. Physics gates are still mandatory.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, cast

import numpy as np


def _hash(value: dict[str, Any]) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    )


class AnchorProtectionPlane:
    """Null-space residuals leave protected latent vectors unchanged.

    A small norm threshold zeros floating-point projection noise, rather than
    allowing a large plastic readout to amplify it at a protected observation.
    This is a local algebraic guarantee, NOT a distributional safety guarantee.
    """

    def __init__(self, anchors: Any) -> None:
        values = np.asarray(anchors, dtype=np.float64)
        if (
            values.ndim != 2
            or not 1 <= values.shape[0] <= 4096
            or not 1 <= values.shape[1] <= 512
            or not np.isfinite(values).all()
        ):
            raise ValueError("finite bounded two-dimensional latent anchor bank required")
        self._anchors = values.copy()
        _, singular, right = np.linalg.svd(values, full_matrices=False)
        if not np.isfinite(singular).all() or not np.isfinite(right).all():
            raise ValueError("nonfinite anchor factorization")
        threshold = float(singular[0]) * 1e-10 if singular.size else 0.0
        self.rank = int(np.count_nonzero(singular > threshold))
        self.dimension = values.shape[1]
        if self.rank == self.dimension:
            self._projection = np.zeros((self.dimension, self.dimension))
        else:
            basis = right[: self.rank]
            self._projection = np.eye(self.dimension) - basis.T @ basis

    @property
    def plastic_dimensions(self) -> int:
        return self.dimension - self.rank

    def project(self, latent: Any) -> np.ndarray[Any, Any]:
        values = np.asarray(latent, dtype=np.float64)
        if values.shape != (self.dimension,) or not np.isfinite(values).all():
            raise ValueError("finite aligned latent vector required")
        result = self._projection @ values
        if not np.isfinite(result).all():
            raise ValueError("nonfinite latent projection")
        if np.linalg.norm(result) <= 1e-10:
            return np.zeros(self.dimension)
        return cast(np.ndarray[Any, Any], result)

    def restrict_update(self, readout_delta: Any) -> np.ndarray[Any, Any]:
        values = np.asarray(readout_delta, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.dimension or not np.isfinite(values).all():
            raise ValueError("finite aligned linear readout update required")
        result = values @ self._projection
        if not np.isfinite(result).all():
            raise ValueError("nonfinite projected readout update")
        return cast(np.ndarray[Any, Any], result)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema": "rosclaw.growth.anchor_protection_plane.v1",
            "anchors": self._anchors.tolist(),
            "dimension": self.dimension,
            "rank": self.rank,
            "plastic_dimensions": self.plastic_dimensions,
            "rank_relative_tolerance": 1e-10,
            "zero_norm_tolerance": 1e-10,
            "requires_frozen_encoder": True,
            "training_only": True,
            "hardware_authorized": False,
            "promotion_authorized": False,
        }
        result["plane_hash"] = _hash(result)
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> AnchorProtectionPlane:
        if value.get("plane_hash") != _hash({k: v for k, v in value.items() if k != "plane_hash"}):
            raise ValueError("unsealed anchor protection plane")
        plane = cls(value.get("anchors"))
        if value != plane.to_dict():
            raise ValueError("anchor plane contract drift")
        return plane


def fit_protected_readout(
    plane: AnchorProtectionPlane,
    *,
    current_readout: Any,
    latents: Any,
    residual_targets: Any,
    sample_weights: Any,
    ridge: float = 1e-8,
) -> np.ndarray[Any, Any]:
    """Weighted residual regression confined to the plastic latent subspace.

    Targets and weights are learner data, not authorizations. A learner must
    independently evaluate any resulting policy, including all retention gates.
    """
    current = np.asarray(current_readout, dtype=np.float64)
    inputs = np.asarray(latents, dtype=np.float64)
    targets = np.asarray(residual_targets, dtype=np.float64)
    weights = np.asarray(sample_weights, dtype=np.float64)
    if (
        inputs.ndim != 2
        or not 1 <= inputs.shape[0] <= 4096
        or inputs.shape[1] != plane.dimension
        or current.ndim != 2
        or not 1 <= current.shape[0] <= 512
        or current.shape[1] != plane.dimension
        or targets.shape != (inputs.shape[0], current.shape[0])
        or weights.shape != (inputs.shape[0],)
        or not all(np.isfinite(v).all() for v in (current, inputs, targets, weights))
        or np.any(weights <= 0)
        or type(ridge) not in (float, int)
        or not np.isfinite(ridge)
        or not 0 < ridge <= 1
    ):
        raise ValueError("finite aligned weighted protected readout problem required")
    design = np.stack([plane.project(row) for row in inputs])
    sqrt_weights = np.sqrt(weights / np.max(weights))
    weighted_design = design * sqrt_weights[:, None]
    error = (targets - design @ current.T) * sqrt_weights[:, None]
    # Primal solve caps factorization at the declared latent dimension, not the
    # number of feedback records. No pseudoinverse of hardware or robot state.
    gram = weighted_design.T @ weighted_design + float(ridge) * np.eye(plane.dimension)
    if not np.isfinite(gram).all() or not np.isfinite(error).all():
        raise ValueError("nonfinite protected regression system")
    delta = np.linalg.solve(gram, weighted_design.T @ error).T
    result = current + plane.restrict_update(delta)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite protected readout result")
    return cast(np.ndarray[Any, Any], result)
