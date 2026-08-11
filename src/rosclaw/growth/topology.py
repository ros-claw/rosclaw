"""Generic support-topology contracts for interaction and boundary coverage."""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from rosclaw.feedback.contracts import canonical_hash
from rosclaw.growth._validation import require_identifier, unique_identifiers


@dataclass(frozen=True)
class SupportAxis:
    axis_id: str
    level_ids: tuple[str, ...]
    schema_version: str = "rosclaw.growth.support_axis.v1"

    def __post_init__(self) -> None:
        require_identifier("axis_id", self.axis_id)
        object.__setattr__(
            self,
            "level_ids",
            unique_identifiers(self.level_ids, label="level_ids"),
        )
        if len(self.level_ids) > 64:
            raise ValueError("a support axis cannot exceed 64 levels")


@dataclass(frozen=True)
class SupportPoint:
    coordinates: tuple[tuple[str, str], ...]
    schema_version: str = "rosclaw.growth.support_point.v1"

    def __post_init__(self) -> None:
        coordinates = tuple(self.coordinates)
        if not coordinates or len({axis_id for axis_id, _ in coordinates}) != len(coordinates):
            raise ValueError("support-point axes must be non-empty and unique")
        for axis_id, level_id in coordinates:
            require_identifier("support point axis", axis_id)
            require_identifier("support point level", level_id)
        object.__setattr__(self, "coordinates", tuple(sorted(coordinates)))

    @property
    def point_id(self) -> str:
        return "/".join(f"{axis_id}={level_id}" for axis_id, level_id in self.coordinates)


@dataclass(frozen=True)
class SupportTopologyDecision:
    complete: bool
    required_point_count: int
    observed_point_count: int
    missing_point_ids: tuple[str, ...]
    invalid_point_ids: tuple[str, ...]
    schema_version: str = "rosclaw.growth.support_topology_decision.v1"

    def __post_init__(self) -> None:
        for label in ("required_point_count", "observed_point_count"):
            value = getattr(self, label)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{label} must be a non-negative integer")
        for label in ("missing_point_ids", "invalid_point_ids"):
            values = tuple(getattr(self, label))
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must be unique")
            object.__setattr__(self, label, values)
        if self.complete != (not self.missing_point_ids and not self.invalid_point_ids):
            raise ValueError("topology completeness must match missing and invalid points")

    @property
    def decision_hash(self) -> str:
        return canonical_hash(self.to_dict())

    @property
    def hardware_authorized(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "complete": self.complete,
            "required_point_count": self.required_point_count,
            "observed_point_count": self.observed_point_count,
            "missing_point_ids": list(self.missing_point_ids),
            "invalid_point_ids": list(self.invalid_point_ids),
            "hardware_authorized": self.hardware_authorized,
        }


@dataclass(frozen=True)
class SupportTopologyContract:
    """Require the full Cartesian support grid, including interactions."""

    axes: tuple[SupportAxis, ...]
    schema_version: str = "rosclaw.growth.support_topology_contract.v1"

    def __post_init__(self) -> None:
        axes = tuple(self.axes)
        if not 1 <= len(axes) <= 8 or any(not isinstance(axis, SupportAxis) for axis in axes):
            raise ValueError("support topology requires 1..8 SupportAxis records")
        if len({axis.axis_id for axis in axes}) != len(axes):
            raise ValueError("support topology axis ids must be unique")
        required_count = _product(len(axis.level_ids) for axis in axes)
        if required_count > 4096:
            raise ValueError("support topology cannot exceed 4096 required points")
        object.__setattr__(self, "axes", tuple(sorted(axes, key=lambda axis: axis.axis_id)))

    @property
    def required_points(self) -> tuple[SupportPoint, ...]:
        return tuple(
            SupportPoint(
                coordinates=tuple(zip((axis.axis_id for axis in self.axes), levels, strict=True))
            )
            for levels in itertools.product(*(axis.level_ids for axis in self.axes))
        )

    def evaluate(self, observed: tuple[SupportPoint, ...]) -> SupportTopologyDecision:
        required = {point.point_id for point in self.required_points}
        observed_ids: set[str] = set()
        invalid_ids: set[str] = set()
        for point in observed:
            if not isinstance(point, SupportPoint):
                invalid_ids.add("invalid_record")
                continue
            if point.point_id in required:
                observed_ids.add(point.point_id)
            else:
                invalid_ids.add(point.point_id)
        missing = tuple(sorted(required - observed_ids))
        invalid = tuple(sorted(invalid_ids))
        return SupportTopologyDecision(
            complete=not missing and not invalid,
            required_point_count=len(required),
            observed_point_count=len(observed_ids),
            missing_point_ids=missing,
            invalid_point_ids=invalid,
        )


def _product(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


__all__ = [
    "SupportAxis",
    "SupportPoint",
    "SupportTopologyContract",
    "SupportTopologyDecision",
]
