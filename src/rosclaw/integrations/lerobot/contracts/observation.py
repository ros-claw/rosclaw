"""Observation Contract v1 for the ROSClaw × LeRobot bridge.

This module must not import torch or lerobot.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal


OBSERVATION_CONTRACT_SCHEMA_VERSION = "rosclaw.observation_contract.v1"
OBSERVATION_SNAPSHOT_SCHEMA_VERSION = "rosclaw.observation_snapshot.v1"


@dataclass
class ObservationFeature:
    """A single feature expected by the policy."""

    required: bool = True
    source: dict[str, Any] = field(default_factory=dict)
    dtype: str = "float32"
    shape: list[int] = field(default_factory=list)
    unit: str | None = None
    max_age_ms: int = 1000
    transport: str = "artifact_ref"
    color_space: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "required": self.required,
            "source": self.source,
            "dtype": self.dtype,
            "shape": self.shape,
            "max_age_ms": self.max_age_ms,
            "transport": self.transport,
        }
        if self.unit is not None:
            out["unit"] = self.unit
        if self.color_space is not None:
            out["color_space"] = self.color_space
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObservationFeature":
        return cls(
            required=bool(data.get("required", True)),
            source=dict(data.get("source", {})),
            dtype=str(data.get("dtype", "float32")),
            shape=list(data.get("shape", [])),
            unit=data.get("unit"),
            max_age_ms=int(data.get("max_age_ms", 1000)),
            transport=str(data.get("transport", "artifact_ref")),
            color_space=data.get("color_space"),
        )


@dataclass
class ObservationContract:
    """Describes how ROSClaw observations map to a LeRobot policy."""

    id: str
    policy: dict[str, Any]
    body: dict[str, Any]
    task: dict[str, Any] = field(default_factory=lambda: {"required": False, "target_key": "task"})
    features: dict[str, ObservationFeature] = field(default_factory=dict)
    schema_version: str = OBSERVATION_CONTRACT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "policy": self.policy,
            "body": self.body,
            "task": self.task,
            "features": {k: v.to_dict() for k, v in self.features.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObservationContract":
        features = data.get("features", {})
        return cls(
            id=str(data.get("id", "")),
            policy=dict(data.get("policy", {})),
            body=dict(data.get("body", {})),
            task=dict(data.get("task", {"required": False, "target_key": "task"})),
            features={k: ObservationFeature.from_dict(v) for k, v in features.items()},
            schema_version=str(data.get("schema_version", OBSERVATION_CONTRACT_SCHEMA_VERSION)),
        )

    def get_state_feature(self) -> ObservationFeature | None:
        return self.features.get("observation.state")

    def get_state_names(self) -> list[str] | None:
        feat = self.get_state_feature()
        if feat is None:
            return None
        source = feat.source or {}
        names = source.get("names")
        if isinstance(names, list):
            return [str(n) for n in names]
        return None


@dataclass
class ObservationFeatureSnapshot:
    """A single feature value inside a snapshot."""

    valid: bool = True
    values: list[float] | None = None
    names: list[str] | None = None
    unit: str | None = None
    artifact_path: str | None = None
    sha256: str | None = None
    captured_at_monotonic_ns: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"valid": self.valid}
        for key in (
            "values",
            "names",
            "unit",
            "artifact_path",
            "sha256",
            "captured_at_monotonic_ns",
            "error",
        ):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObservationFeatureSnapshot":
        return cls(
            valid=bool(data.get("valid", True)),
            values=data.get("values"),
            names=data.get("names"),
            unit=data.get("unit"),
            artifact_path=data.get("artifact_path"),
            sha256=data.get("sha256"),
            captured_at_monotonic_ns=data.get("captured_at_monotonic_ns"),
            error=data.get("error"),
        )


@dataclass
class ObservationSnapshot:
    """A concrete observation tied to a session."""

    snapshot_id: str
    session_id: str | None = None
    body_id: str | None = None
    captured_at_monotonic_ns: int = 0
    task: str = ""
    features: dict[str, ObservationFeatureSnapshot] = field(default_factory=dict)
    schema_version: str = OBSERVATION_SNAPSHOT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "snapshot_id": self.snapshot_id,
            "session_id": self.session_id,
            "body_id": self.body_id,
            "captured_at_monotonic_ns": self.captured_at_monotonic_ns,
            "task": self.task,
            "features": {k: v.to_dict() for k, v in self.features.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObservationSnapshot":
        features = data.get("features", {})
        return cls(
            snapshot_id=str(data.get("snapshot_id", "")),
            session_id=data.get("session_id"),
            body_id=data.get("body_id"),
            captured_at_monotonic_ns=int(data.get("captured_at_monotonic_ns", 0)),
            task=str(data.get("task", "")),
            features={k: ObservationFeatureSnapshot.from_dict(v) for k, v in features.items()},
            schema_version=str(data.get("schema_version", OBSERVATION_SNAPSHOT_SCHEMA_VERSION)),
        )


@dataclass
class ObservationValidationResult:
    """Result of validating a snapshot against a contract."""

    status: Literal["ok", "blocked"] = "ok"
    errors: list[dict[str, Any]] = field(default_factory=list)
    missing_optional_features: list[str] = field(default_factory=list)
    age_ms_by_feature: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "errors": self.errors,
            "missing_optional_features": self.missing_optional_features,
            "age_ms_by_feature": self.age_ms_by_feature,
        }


def validate_observation_snapshot(
    contract: ObservationContract,
    snapshot: ObservationSnapshot,
    *,
    now_monotonic_ns: int | None = None,
) -> ObservationValidationResult:
    """Validate a snapshot against a contract.

    Required features must be present, fresh, and valid. Optional features that
    are missing or stale are recorded but do not block unless the contract says
    otherwise.
    """
    if now_monotonic_ns is None:
        now_monotonic_ns = int(time.monotonic() * 1e9)

    errors: list[dict[str, Any]] = []
    missing_optional: list[str] = []
    age_ms_by_feature: dict[str, float] = {}

    for key, expected in contract.features.items():
        actual = snapshot.features.get(key)
        if actual is None or (actual.values is None and actual.artifact_path is None):
            if expected.required:
                errors.append(
                    {
                        "feature": key,
                        "code": "observation_required_missing",
                        "message": f"Required observation feature '{key}' is missing.",
                    }
                )
            else:
                missing_optional.append(key)
            continue

        if not actual.valid:
            if expected.required:
                errors.append(
                    {
                        "feature": key,
                        "code": "observation_feature_invalid",
                        "message": f"Observation feature '{key}' is marked invalid.",
                    }
                )
            else:
                missing_optional.append(key)
            continue

        captured = actual.captured_at_monotonic_ns
        if captured is None:
            captured = snapshot.captured_at_monotonic_ns
        age_ms = (now_monotonic_ns - captured) / 1e6 if captured is not None else 0.0
        age_ms_by_feature[key] = age_ms
        if age_ms > expected.max_age_ms:
            if expected.required:
                errors.append(
                    {
                        "feature": key,
                        "code": "observation_stale",
                        "message": (
                            f"Observation feature '{key}' is stale: "
                            f"age={age_ms:.1f}ms > max_age_ms={expected.max_age_ms}."
                        ),
                    }
                )
            else:
                missing_optional.append(key)

    return ObservationValidationResult(
        status="blocked" if errors else "ok",
        errors=errors,
        missing_optional_features=missing_optional,
        age_ms_by_feature=age_ms_by_feature,
    )
