"""Body action mapping schemas.

This module defines the policy action space, the body action space, and the
mapping between them.  It is free of torch/lerobot imports so it can be used
from the ROSClaw core Python runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MappingCompatibility(str, Enum):
    """Compatibility level between a policy action space and a body."""

    EXACT = "exact"
    CONVERTIBLE = "convertible"
    PARTIAL = "partial"
    INCOMPATIBLE = "incompatible"
    UNKNOWN = "unknown"


@dataclass
class ActionSpace:
    """Describes the action space emitted by a policy proposal."""

    representation: str
    names: list[str] = field(default_factory=list)
    units: list[str] = field(default_factory=list)
    reference_frame: str = ""
    shape: list[int] = field(default_factory=list)
    dtype: str = "float32"
    is_chunked: bool = False
    chunk_size: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "representation": self.representation,
            "names": self.names,
            "units": self.units,
            "reference_frame": self.reference_frame,
            "shape": self.shape,
            "dtype": self.dtype,
            "is_chunked": self.is_chunked,
            "chunk_size": self.chunk_size,
            "metadata": self.metadata,
        }

    @classmethod
    def from_proposal_v2(cls, proposal: dict[str, Any]) -> "ActionSpace":
        """Infer an action space from a ``rosclaw.action_proposal.v2`` dict."""
        action = proposal.get("action", {})
        names = action.get("names", [])
        units = action.get("units", [])
        representation = action.get("representation", "unknown")
        reference_frame = action.get("reference_frame", "")
        shape = action.get("shape", [])
        dtype = action.get("dtype", "float32")
        chunk = proposal.get("chunk", {})
        chunk_size = chunk.get("size") if isinstance(chunk, dict) else None
        is_chunked = bool(chunk_size)
        return cls(
            representation=representation,
            names=names,
            units=units,
            reference_frame=reference_frame,
            shape=shape,
            dtype=dtype,
            is_chunked=is_chunked,
            chunk_size=chunk_size,
        )


@dataclass
class BodyActionSpace:
    """Describes the action space the current effective body can accept."""

    body_id: str
    representation: str = "joint_position"
    joint_names: list[str] = field(default_factory=list)
    units: list[str] = field(default_factory=list)
    joint_limits: dict[str, dict[str, float]] = field(default_factory=dict)
    reference_frame: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "body_id": self.body_id,
            "representation": self.representation,
            "joint_names": self.joint_names,
            "units": self.units,
            "joint_limits": self.joint_limits,
            "reference_frame": self.reference_frame,
            "metadata": self.metadata,
        }


@dataclass
class JointMapping:
    """One entry mapping a policy action dimension to a body joint."""

    policy_name: str
    body_name: str
    policy_index: int
    body_index: int
    scale: float = 1.0
    offset: float = 0.0
    sign: int = 1
    unit_conversion: str | None = None
    compatible: bool = True
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "body_name": self.body_name,
            "policy_index": self.policy_index,
            "body_index": self.body_index,
            "scale": self.scale,
            "offset": self.offset,
            "sign": self.sign,
            "unit_conversion": self.unit_conversion,
            "compatible": self.compatible,
            "reason": self.reason,
        }


@dataclass
class ActionMapping:
    """Complete mapping from a policy action space to a body action space."""

    policy_space: ActionSpace
    body_space: BodyActionSpace
    joints: list[JointMapping] = field(default_factory=list)
    compatibility: MappingCompatibility = MappingCompatibility.UNKNOWN
    block_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    allow_partial: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_space": self.policy_space.to_dict(),
            "body_space": self.body_space.to_dict(),
            "joints": [j.to_dict() for j in self.joints],
            "compatibility": self.compatibility.value,
            "block_reasons": self.block_reasons,
            "warnings": self.warnings,
            "allow_partial": self.allow_partial,
        }

    def is_blocked(self) -> bool:
        """Return True if this mapping must not be used to command the body."""
        if self.compatibility in (MappingCompatibility.INCOMPATIBLE, MappingCompatibility.UNKNOWN):
            return True
        if self.compatibility == MappingCompatibility.PARTIAL and not self.allow_partial:
            return True
        if self.block_reasons:
            return True
        return False


@dataclass
class MappedAction:
    """Result of applying an action mapping to a policy action vector."""

    body_action_values: list[float]
    body_joint_names: list[str]
    compatibility: MappingCompatibility
    blocked: bool
    block_reasons: list[str]
    warnings: list[str]
    reference_frame: str = ""
    chunk_size: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "body_action_values": self.body_action_values,
            "body_joint_names": self.body_joint_names,
            "compatibility": self.compatibility.value,
            "blocked": self.blocked,
            "block_reasons": self.block_reasons,
            "warnings": self.warnings,
            "reference_frame": self.reference_frame,
            "chunk_size": self.chunk_size,
        }
