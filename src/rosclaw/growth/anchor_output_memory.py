"""Task-neutral memory of frozen observations and their parent predictions.

Unlike reverting a new head to the first warm start, this preserves predictions
actually learned by later parents. This is a numeric training primitive, never
an execution permit. Exact-state retention does not prove trajectory safety.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from rosclaw.growth.anchor_kernel import AnchorKernelGuard, _hash


class AnchorOutputMemory:
    """Blend a finite proposal with recorded predictions in a frozen metric.

    Nearest-reference selection can be discontinuous between distinct anchors.
    Neither smooth motion nor unseen-state retention follows from this contract.
    State must include all causal context that changes the parent prediction.
    """

    def __init__(
        self,
        observations: Any,
        predictions: Any,
        *,
        bandwidth: float,
        encoder_hash: str,
        parent_policy_hash: str,
        evidence_hash: str,
        predecessor_memory_hash: str | None = None,
        accelerated: bool = True,
    ) -> None:
        if any(
            not isinstance(v, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", v)
            for v in (encoder_hash, parent_policy_hash, evidence_hash)
        ):
            raise ValueError("explicit frozen encoder, parent and evidence identities required")
        if predecessor_memory_hash is not None and (
            not isinstance(predecessor_memory_hash, str)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", predecessor_memory_hash)
        ):
            raise ValueError("sealed predecessor memory identity required")
        guard = AnchorKernelGuard(observations, bandwidth=bandwidth, accelerated=accelerated)
        x = np.asarray(observations, dtype=np.float64).copy()
        y = np.asarray(predictions, dtype=np.float64)
        if (
            y.ndim != 2
            or y.shape[0] != len(x)
            or not 1 <= y.shape[1] <= 512
            or not np.isfinite(y).all()
            or np.max(np.abs(y)) > 1e6
        ):
            raise ValueError("finite aligned bounded parent predictions required")
        x[x == 0] = 0.0  # Normalize signed zero before detecting identical states.
        seen: dict[bytes, int] = {}
        for i, row in enumerate(x):
            key = row.tobytes()
            if key in seen and not np.array_equal(y[i], y[seen[key]]):
                raise ValueError(
                    "identical observations have conflicting parent predictions; causal context is missing"
                )
            seen[key] = i
        self._observations = x
        self._predictions = y.copy()
        self._observations.flags.writeable = False
        self._predictions.flags.writeable = False
        self._guard = guard
        self._encoder_hash = encoder_hash
        self._parent_policy_hash = parent_policy_hash
        self._evidence_hash = evidence_hash
        self._predecessor_memory_hash = predecessor_memory_hash
        self._tree: Any = None
        if accelerated:
            try:
                from scipy.spatial import cKDTree
            except ImportError:
                pass
            else:
                self._tree = cKDTree(x, copy_data=True)

    @property
    def encoder_hash(self) -> str:
        return self._encoder_hash

    @property
    def parent_policy_hash(self) -> str:
        return self._parent_policy_hash

    @property
    def evidence_hash(self) -> str:
        return self._evidence_hash

    @property
    def output_dimension(self) -> int:
        return int(self._predictions.shape[1])

    def _nearest(self, observation: Any) -> int:
        if self._tree is not None:
            if len(self._observations) == 1:
                return 0
            distances, indices = self._tree.query(observation, k=2)
            if abs(distances[0] - distances[1]) > 1e-12 * max(distances[0], distances[1]):
                return int(indices[0])
        # NumPy reference path also supplies deterministic lowest-index ties.
        delta = self._observations - observation
        return int(np.argmin(np.einsum("ni,ni->n", delta, delta)))

    def blend(self, observation: Any, proposal: Any, *, encoder_hash: str) -> np.ndarray[Any, Any]:
        if encoder_hash != self.encoder_hash:
            raise ValueError("frozen observation encoder identity changed")
        gate = self._guard.gate(observation)
        predicted = np.asarray(proposal, dtype=np.float64)
        if (
            predicted.shape != (self.output_dimension,)
            or not np.isfinite(predicted).all()
            or np.max(np.abs(predicted)) > 1e6
        ):
            raise ValueError("finite aligned bounded numeric proposal required")
        reference = self._predictions[self._nearest(observation)]
        if gate == 0:
            return reference.copy()
        return np.asarray(gate * predicted + (1 - gate) * reference, dtype=np.float64)

    def extend(
        self, observations: Any, predictions: Any, *, parent_policy_hash: str, evidence_hash: str
    ) -> AnchorOutputMemory:
        """Return a new memory; preserve the old states/outputs without mutation.

        Conflicting duplicates and total-capacity overflow are rejected. No
        anchors are silently dropped to fit an unbounded learning lifetime.
        """
        addition = AnchorOutputMemory(
            observations,
            predictions,
            bandwidth=self._guard.bandwidth,
            encoder_hash=self.encoder_hash,
            parent_policy_hash=parent_policy_hash,
            evidence_hash=evidence_hash,
            accelerated=self._tree is not None,
        )
        return AnchorOutputMemory(
            np.concatenate((self._observations, addition._observations)),
            np.concatenate((self._predictions, addition._predictions)),
            bandwidth=self._guard.bandwidth,
            encoder_hash=self.encoder_hash,
            parent_policy_hash=parent_policy_hash,
            evidence_hash=evidence_hash,
            accelerated=self._tree is not None,
            predecessor_memory_hash=self.to_dict()["memory_hash"],
        )

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": "rosclaw.growth.anchor_output_memory.v1",
            "observations": self._observations.tolist(),
            "predictions": self._predictions.tolist(),
            "bandwidth": self._guard.bandwidth,
            "encoder_hash": self.encoder_hash,
            "parent_policy_hash": self.parent_policy_hash,
            "evidence_hash": self.evidence_hash,
            "predecessor_memory_hash": self._predecessor_memory_hash,
            "local_guarantee_only": True,
            "requires_frozen_encoder": True,
            "distributional_retention_guaranteed": False,
            "training_only": True,
            "promotion_authorized": False,
            "hardware_authorized": False,
        }
        value["memory_hash"] = _hash(value)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any], *, accelerated: bool = True) -> AnchorOutputMemory:
        if value.get("memory_hash") != _hash(
            {k: v for k, v in value.items() if k != "memory_hash"}
        ):
            raise ValueError("unsealed prediction memory")
        memory = cls(
            value.get("observations"),
            value.get("predictions"),
            bandwidth=value.get("bandwidth"),
            encoder_hash=value.get("encoder_hash"),
            parent_policy_hash=value.get("parent_policy_hash"),
            evidence_hash=value.get("evidence_hash"),
            predecessor_memory_hash=value.get("predecessor_memory_hash"),
            accelerated=accelerated,
        )
        if value != memory.to_dict():
            raise ValueError("prediction memory schema or authority drift")
        return memory
