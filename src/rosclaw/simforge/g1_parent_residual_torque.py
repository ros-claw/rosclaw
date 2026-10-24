"""Explicit parent-conditioned correction head over a verified G1 GRU encoder.

New, SIM-only artifact semantics; never reinterpret an old absolute-torque head
as a residual. A zero head is exactly the parent, even when candidate-only power
or slew constraints would otherwise reshape that parent. Nonzero proposals use
the existing independent torque projector. No transport or execution authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.control.parent_residual import compose_parent_residual, parent_context
from rosclaw.simforge.g1_neural_torque import (
    G1_NEURAL_TORQUE_OBSERVATIONS,
    G1NeuralTorqueArtifact,
    G1NeuralTorquePolicy,
    G1TorqueControlFrame,
    G1TorqueProjection,
    G1TorqueSafetyProjector,
)
from rosclaw.simforge.tasks.g1_goalforge.concepts import (
    G1_DDS_JOINT_NAMES,
    G1_HARD_TORQUE_LIMITS,
)

_SCHEMA = "rosclaw.simforge.g1_parent_residual_head.v1"
_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_BYTES = 4 * 1024 * 1024
G1_PARENT_RESIDUAL_OBSERVATIONS = (
    *G1_NEURAL_TORQUE_OBSERVATIONS,
    *("current_parent_torque_ratio/" + name for name in G1_DDS_JOINT_NAMES),
)


@dataclass(frozen=True)
class G1ParentResidualArtifact:
    artifact_hash: str
    encoder_artifact_hash: str
    body_hash: str
    parent_policy_hash: str
    dataset_hash: str
    residual_limit_ratio: float
    update_index: int
    head_weight: np.ndarray
    head_bias: np.ndarray

    def __post_init__(self) -> None:
        if (
            any(
                not isinstance(h, str) or not _HASH.fullmatch(h)
                for h in (
                    self.artifact_hash,
                    self.encoder_artifact_hash,
                    self.body_hash,
                    self.parent_policy_hash,
                    self.dataset_hash,
                )
            )
            or type(self.residual_limit_ratio) not in (int, float)
            or not math.isfinite(self.residual_limit_ratio)
            or not 1e-4 <= self.residual_limit_ratio <= 0.02
            or type(self.update_index) is not int
            or not 0 <= self.update_index <= 10**9
        ):
            raise ValueError("invalid residual artifact commitments or authority")
        # Ownership, not a read-only view of caller-owned mutable storage.
        for key in ("head_weight", "head_bias"):
            value = np.array(getattr(self, key), dtype=np.float32, copy=True)
            if not np.isfinite(value).all() or np.any(np.abs(value) > 1e4):
                raise ValueError("finite bounded residual tensors required")
            if (key == "head_bias" and value.shape != (29,)) or (
                key == "head_weight"
                and (value.ndim != 2 or value.shape[0] != 29 or not 30 <= value.shape[1] <= 1053)
            ):
                raise ValueError("exact residual head dimensions required")
            value.flags.writeable = False
            object.__setattr__(self, key, value)


def serialize_parent_residual_head(
    *,
    encoder: G1NeuralTorqueArtifact,
    parent_policy_hash: str,
    dataset_hash: str,
    residual_limit_ratio: float = 0.01,
    update_index: int = 0,
    head_weight: np.ndarray | None = None,
    head_bias: np.ndarray | None = None,
) -> bytes:
    weight = (
        np.zeros((29, encoder.hidden_dim + 29), np.float32) if head_weight is None else head_weight
    )
    bias = np.zeros(29, np.float32) if head_bias is None else head_bias
    metadata = {
        "schema_version": _SCHEMA,
        "encoder_artifact_hash": encoder.artifact_hash,
        "body_hash": encoder.body_hash,
        "parent_policy_hash": parent_policy_hash,
        "dataset_hash": dataset_hash,
        "residual_limit_ratio": residual_limit_ratio,
        "update_index": update_index,
        "activation_ceiling": "SIM_ONLY",
        "action_semantics": "PARENT_PLUS_BOUNDED_RESIDUAL",
        "observation_names": list(G1_PARENT_RESIDUAL_OBSERVATIONS),
        "head_weight": np.asarray(weight).tolist(),
        "head_bias": np.asarray(bias).tolist(),
    }
    payload = json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    _decode(payload, encoder=encoder)
    return payload


def _decode(payload: bytes, *, encoder: G1NeuralTorqueArtifact) -> G1ParentResidualArtifact:
    if not 0 < len(payload) <= _MAX_BYTES:
        raise ValueError("bounded parent residual head required")
    try:
        m = json.loads(payload)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("invalid residual head JSON") from exc
    keys = {
        "schema_version",
        "encoder_artifact_hash",
        "body_hash",
        "parent_policy_hash",
        "dataset_hash",
        "residual_limit_ratio",
        "update_index",
        "activation_ceiling",
        "action_semantics",
        "observation_names",
        "head_weight",
        "head_bias",
    }
    if (
        not isinstance(m, dict)
        or set(m) != keys
        or m["schema_version"] != _SCHEMA
        or m["activation_ceiling"] != "SIM_ONLY"
        or m["action_semantics"] != "PARENT_PLUS_BOUNDED_RESIDUAL"
        or m["observation_names"] != list(G1_PARENT_RESIDUAL_OBSERVATIONS)
    ):
        raise ValueError("explicit SIM residual schema and131-column observation contract required")
    for key in ("encoder_artifact_hash", "body_hash", "parent_policy_hash", "dataset_hash"):
        if not isinstance(m[key], str) or not _HASH.fullmatch(m[key]):
            raise ValueError("residual head hashes must be SHA256 commitments")
    if (
        m["encoder_artifact_hash"] != encoder.artifact_hash
        or m["body_hash"] != encoder.body_hash
        or encoder.observation_names != G1_NEURAL_TORQUE_OBSERVATIONS
    ):
        raise ValueError("exact verified102-column encoder and body required")
    ratio = m["residual_limit_ratio"]
    if (
        type(ratio) not in (int, float)
        or not math.isfinite(ratio)
        or not 1e-4 <= ratio <= 0.02
        or type(m["update_index"]) is not int
        or not 0 <= m["update_index"] <= 10**9
    ):
        raise ValueError("bounded explicit residual authority and update index required")
    values = {}
    for key, shape in (("head_weight", (29, encoder.hidden_dim + 29)), ("head_bias", (29,))):
        value = np.asarray(m[key])
        if (
            value.shape != shape
            or value.dtype.kind not in "fi"
            or not np.isfinite(value).all()
            or np.any(np.abs(value) > 1e4)
        ):
            raise ValueError("finite exact residual head tensor required")
        values[key] = value.astype(np.float32)
    return G1ParentResidualArtifact(
        "sha256:" + hashlib.sha256(payload).hexdigest(),
        encoder.artifact_hash,
        encoder.body_hash,
        m["parent_policy_hash"],
        m["dataset_hash"],
        float(ratio),
        m["update_index"],
        values["head_weight"],
        values["head_bias"],
    )


def load_parent_residual_head(
    path: Path,
    *,
    encoder: G1NeuralTorqueArtifact,
    expected_hash: str,
    expected_body_hash: str,
    expected_parent_policy_hash: str,
) -> G1ParentResidualArtifact:
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= _MAX_BYTES:
        raise ValueError("ordinary bounded parent residual head required")
    artifact = _decode(path.read_bytes(), encoder=encoder)
    if (
        artifact.artifact_hash != expected_hash
        or artifact.body_hash != expected_body_hash
        or artifact.parent_policy_hash != expected_parent_policy_hash
    ):
        raise ValueError("residual artifact/body/parent commitment differs")
    return artifact


class _ResidualProjector(G1TorqueSafetyProjector):
    def project(
        self,
        proposed: np.ndarray,
        *,
        parent: np.ndarray,
        previous: np.ndarray,
        frame: G1TorqueControlFrame,
    ) -> G1TorqueProjection:
        if np.array_equal(proposed, parent):
            return self.fallback(parent=parent, frame=frame, reason="zero_residual_parent")
        return super().project(proposed, parent=parent, previous=previous, frame=frame)


class G1ParentResidualTorquePolicy(G1NeuralTorquePolicy):
    """Existing physical guards/history; a separately bound context-conditioned head."""

    def __init__(
        self,
        artifact: G1ParentResidualArtifact,
        *,
        encoder: G1NeuralTorqueArtifact,
        expected_body_hash: str,
        expected_parent_policy_hash: str,
    ) -> None:
        if (
            type(artifact) is not G1ParentResidualArtifact
            or artifact.encoder_artifact_hash != encoder.artifact_hash
            or artifact.body_hash != expected_body_hash
            or encoder.body_hash != expected_body_hash
            or artifact.parent_policy_hash != expected_parent_policy_hash
            or artifact.head_weight.shape != (29, encoder.hidden_dim + 29)
        ):
            raise ValueError(
                "separately qualified encoder, residual head and actual parent required"
            )
        self.residual_artifact = artifact
        # Internal encoder ABI stays102. It is NOT serialized as an old policy,
        # and the old absolute head is never evaluated. New head hash binds both.
        internal = replace(
            encoder,
            artifact_hash=artifact.artifact_hash,
            parent_policy_hash=artifact.parent_policy_hash,
            dataset_hash=artifact.dataset_hash,
            update_index=artifact.update_index,
        )
        super().__init__(
            internal,
            expected_body_hash=expected_body_hash,
            expected_parent_policy_hash=expected_parent_policy_hash,
        )
        self.projector = _ResidualProjector(encoder.safety)
        self._parent: np.ndarray = np.zeros(29, np.float64)
        self._parent_ratio: np.ndarray = np.zeros(29, np.float32)
        self._residual: np.ndarray | None = None
        self._residuals: list[np.ndarray] = []
        self._residual_valid: list[bool] = []
        self._conditioned_observations: list[np.ndarray] = []

    def reset(self) -> None:
        super().reset()
        self._parent.fill(0)
        self._parent_ratio.fill(0)
        self._residual = None
        self._residuals.clear()
        self._residual_valid.clear()
        self._conditioned_observations.clear()

    def command(self, frame: G1TorqueControlFrame, parent_torque: np.ndarray) -> np.ndarray:
        if self._pending is not None:
            raise RuntimeError("residual policy awaits actual applied torque")
        parent = np.asarray(parent_torque, dtype=np.float64)
        hard = np.asarray(G1_HARD_TORQUE_LIMITS, np.float64)
        if (
            parent.shape != (29,)
            or not np.isfinite(parent).all()
            or np.any(np.abs(parent) > hard * 0.85)
        ):
            raise ValueError("finite actual parent inside immutable total effort ceiling required")
        self._parent = parent.copy()
        self._parent_ratio = (parent / hard).astype(np.float32)
        self._residual = None
        return super().command(frame, parent)

    def _infer_normalized(self, normalized: np.ndarray) -> np.ndarray:
        self._advance_hidden(normalized)
        inputs = np.concatenate((self._hidden, self._parent_ratio))
        head = self.residual_artifact
        latent = head.head_weight @ inputs + head.head_bias
        if not np.isfinite(latent).all():
            raise FloatingPointError("residual head produced nonfinite output")
        hard = np.asarray(G1_HARD_TORQUE_LIMITS, np.float64)
        self._residual = np.tanh(latent.astype(np.float64)) * hard * head.residual_limit_ratio
        return compose_parent_residual(
            self._parent, self._residual, hard, maximum_ratio=head.residual_limit_ratio
        )

    def note_applied(self, torque: np.ndarray) -> None:
        super().note_applied(torque)
        self._conditioned_observations.append(
            parent_context(
                self._observations[-1], self._parent, np.asarray(G1_HARD_TORQUE_LIMITS, np.float64)
            )
        )
        self._residuals.append(
            np.zeros(29, np.float32)
            if self._residual is None
            else self._residual.astype(np.float32)
        )
        self._residual_valid.append(self._residual is not None)
        self._residual = None

    def residual_episode(self) -> dict[str, np.ndarray]:
        episode = super().episode()
        return {
            "observations": np.asarray(self._conditioned_observations, np.float32).copy(),
            "actor_residuals": np.asarray(self._residuals, np.float32),
            "actor_residual_valid": np.asarray(self._residual_valid, np.bool_),
            "applied_actions": episode.actions.copy(),
            "parent_actions": episode.parent_actions.copy(),
        }

    def residual_receipt(self) -> dict[str, Any]:
        return {
            **self.build_receipt().to_dict(),
            "schema_version": "rosclaw.simforge.g1_parent_residual_receipt.v1",
            "residual_schema": _SCHEMA,
            "encoder_artifact_hash": self.residual_artifact.encoder_artifact_hash,
            "residual_limit_ratio": self.residual_artifact.residual_limit_ratio,
            "residual_observation_names": G1_PARENT_RESIDUAL_OBSERVATIONS,
            "effective_head_input": "pretrained_GRU_hidden_plus_current_parent_ratio",
            "residual_metadata_action_semantics": "PARENT_PLUS_BOUNDED_RESIDUAL",
            "hardware_authorized": False,
        }
