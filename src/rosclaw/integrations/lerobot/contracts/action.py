"""Action Proposal Contract v2 for the ROSClaw × LeRobot bridge.

This module must not import torch or lerobot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


ACTION_PROPOSAL_SCHEMA_VERSION = "rosclaw.action_proposal.v2"

# Action representations supported by P4.
ACTION_REPRESENTATIONS = (
    "joint_position",
    "joint_delta",
    "joint_velocity",
    "joint_effort",
    "cartesian_pose",
    "cartesian_delta",
    "gripper_position",
    "hybrid",
    "unknown",
)

# Physical units supported by P4.
ACTION_UNITS = (
    "radian",
    "degree",
    "meter",
    "meter_per_second",
    "newton",
    "newton_meter",
    "normalized",
    "raw_device_unit",
    "unknown",
)


@dataclass
class ActionChunkMetadata:
    """Metadata for chunked action policies."""

    is_chunk: bool = False
    length: int = 1
    control_hz: float | None = None
    duration_sec: float | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"is_chunk": self.is_chunk, "length": self.length}
        if self.control_hz is not None:
            out["control_hz"] = self.control_hz
        if self.duration_sec is not None:
            out["duration_sec"] = self.duration_sec
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionChunkMetadata":
        return cls(
            is_chunk=bool(data.get("is_chunk", False)),
            length=int(data.get("length", 1)),
            control_hz=data.get("control_hz"),
            duration_sec=data.get("duration_sec"),
        )


@dataclass
class ActionProposalV2:
    """A semantically typed action proposal emitted by the LeRobot bridge."""

    proposal_id: str
    session_id: str | None
    step_index: int
    policy_path: str
    policy_hash: str | None
    runtime_id: str | None
    processor_hash: str | None
    representation: str
    reference_frame: str | None
    values: list[float]
    shape: list[int]
    dtype: str
    names: list[str]
    units: str
    chunk: ActionChunkMetadata
    timing: dict[str, Any] = field(default_factory=dict)
    safety: dict[str, Any] = field(default_factory=dict)
    raw_model_output: list[float] | None = None
    schema_version: str = ACTION_PROPOSAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if isinstance(self.chunk, dict):
            self.chunk = ActionChunkMetadata.from_dict(self.chunk)
        if not isinstance(self.timing, dict):
            self.timing = dict(self.timing or {})
        if not isinstance(self.safety, dict):
            self.safety = dict(self.safety or {})

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "schema_version": self.schema_version,
            "proposal_id": self.proposal_id,
            "session_id": self.session_id,
            "step_index": self.step_index,
            "policy": {
                "path": self.policy_path,
                "policy_hash": self.policy_hash,
                "runtime_id": self.runtime_id,
                "processor_hash": self.processor_hash,
            },
            "representation": self.representation,
            "reference_frame": self.reference_frame,
            "action": {
                "values": self.values,
                "shape": self.shape,
                "dtype": self.dtype,
                "names": self.names,
                "units": self.units,
            },
            "chunk": self.chunk.to_dict(),
            "timing": self.timing,
            "safety": self.safety,
        }
        if self.raw_model_output is not None:
            out["raw_model_output"] = self.raw_model_output
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionProposalV2":
        policy = data.get("policy", {})
        action = data.get("action", {})
        chunk = data.get("chunk", {})
        return cls(
            proposal_id=str(data.get("proposal_id", "")),
            session_id=data.get("session_id"),
            step_index=int(data.get("step_index", 0)),
            policy_path=str(policy.get("path", "")),
            policy_hash=policy.get("policy_hash"),
            runtime_id=policy.get("runtime_id"),
            processor_hash=policy.get("processor_hash"),
            representation=str(data.get("representation", "unknown")),
            reference_frame=data.get("reference_frame"),
            values=list(action.get("values", [])),
            shape=list(action.get("shape", [])),
            dtype=str(action.get("dtype", "float32")),
            names=list(action.get("names", [])),
            units=str(action.get("units", "unknown")),
            chunk=ActionChunkMetadata.from_dict(chunk),
            timing=dict(data.get("timing", {})),
            safety=dict(data.get("safety", {})),
            raw_model_output=data.get("raw_model_output"),
            schema_version=str(data.get("schema_version", ACTION_PROPOSAL_SCHEMA_VERSION)),
        )


def infer_action_representation(metadata: dict[str, Any]) -> str:
    """Infer action representation from policy metadata.

    Priority:
      1. output_features["action"].get("representation")
      2. extra["action_representation"]
      3. "unknown"
    """
    output_features = metadata.get("output_features", {})
    action_feature = output_features.get("action", {})
    representation = action_feature.get("representation")
    if representation is not None:
        return str(representation)
    extra = metadata.get("extra", {})
    representation = extra.get("action_representation")
    if representation is not None:
        return str(representation)
    return "unknown"


def infer_action_names(
    metadata: dict[str, Any],
    manifest_action_space: list[str] | None = None,
    action_dim: int | None = None,
) -> list[str]:
    """Infer per-action joint/axis names.

    Priority:
      1. output_features["action"]["names"]
      2. extra["action_names"]
      3. manifest embodiment.action_space
      4. fallback ["action_0", ..., "action_{N-1}"] when action_dim is known
    """
    output_features = metadata.get("output_features", {})
    action_feature = output_features.get("action", {})
    names = action_feature.get("names")
    if isinstance(names, list) and names:
        return [str(n) for n in names]

    extra = metadata.get("extra", {})
    names = extra.get("action_names")
    if isinstance(names, list) and names:
        return [str(n) for n in names]

    if manifest_action_space:
        return [str(n) for n in manifest_action_space]

    if action_dim is not None and action_dim > 0:
        return [f"action_{i}" for i in range(action_dim)]

    return []


def infer_action_units(metadata: dict[str, Any]) -> str:
    """Infer action units from policy metadata.

    Priority:
      1. output_features["action"]["unit"]
      2. extra["action_unit"]
      3. "unknown"
    """
    output_features = metadata.get("output_features", {})
    action_feature = output_features.get("action", {})
    unit = action_feature.get("unit")
    if unit is not None:
        return str(unit)
    extra = metadata.get("extra", {})
    unit = extra.get("action_unit")
    if unit is not None:
        return str(unit)
    return "unknown"


def infer_action_shape(metadata: dict[str, Any]) -> list[int] | None:
    """Return the policy action shape from output_features if available."""
    output_features = metadata.get("output_features", {})
    action_feature = output_features.get("action", {})
    shape = action_feature.get("shape")
    if isinstance(shape, (list, tuple)):
        return [int(d) for d in shape]
    return None


def build_action_proposal_v2(
    *,
    proposal_id: str,
    session_id: str | None,
    step_index: int,
    policy_path: str,
    policy_metadata: dict[str, Any],
    manifest_action_space: list[str] | None = None,
    processed_action: dict[str, Any],
    raw_action: dict[str, Any] | None = None,
    runtime_id: str | None = None,
    processor_hash: str | None = None,
    timing: dict[str, Any] | None = None,
    safety: dict[str, Any] | None = None,
) -> ActionProposalV2:
    """Build a v2 action proposal from a processed worker action."""
    values = list(processed_action.get("values", []))
    shape = list(processed_action.get("shape", []))
    dtype = str(processed_action.get("dtype", "float32"))
    action_dim = shape[-1] if shape else len(values)

    representation = infer_action_representation(policy_metadata)
    names = infer_action_names(policy_metadata, manifest_action_space, action_dim)
    units = infer_action_units(policy_metadata)

    is_chunk = len(shape) == 2 and shape[0] > 1
    chunk = ActionChunkMetadata(
        is_chunk=is_chunk,
        length=shape[0] if is_chunk else 1,
    )

    return ActionProposalV2(
        proposal_id=proposal_id,
        session_id=session_id,
        step_index=step_index,
        policy_path=policy_path,
        policy_hash=policy_metadata.get("policy_hash"),
        runtime_id=runtime_id,
        processor_hash=processor_hash,
        representation=representation,
        reference_frame=None,
        values=values,
        shape=shape,
        dtype=dtype,
        names=names,
        units=units,
        chunk=chunk,
        timing=timing or {},
        safety=safety or {},
        raw_model_output=raw_action.get("values") if raw_action else None,
    )


def validate_action_values(values: list[float]) -> tuple[bool, str | None]:
    """Check for NaN/Inf in action values."""
    import math

    for i, v in enumerate(values):
        if not isinstance(v, (int, float)):
            return False, f"action value at index {i} is not numeric: {type(v).__name__}"
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return False, f"action value at index {i} is NaN/Inf"
    return True, None
