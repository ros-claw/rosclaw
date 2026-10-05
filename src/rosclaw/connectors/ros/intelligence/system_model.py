"""Versioned observations. Missing data means UNKNOWN, never healthy."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rosclaw.contracts.common import ContractModel, content_hash


class Observation(BaseModel):
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)
    source: str
    captured_at: datetime


class Signal(Observation):
    freshness_policy: Literal["stream", "latched"] = "stream"
    max_age_ms: float = Field(1000, gt=0, le=60000)
    topic: str
    rate_hz: float | None = Field(None, ge=0)
    jitter_ms: float | None = Field(None, ge=0)
    last_message_age_ms: float | None = Field(None, ge=0)
    publisher_count: int | None = Field(None, ge=0)
    subscriber_count: int | None = Field(None, ge=0)


class Transform(Observation):
    parent: str
    child: str
    static: bool = False
    age_ms: float | None = None  # Negative means future-dated in ROS clock domain.
    future_tolerance_ms: float = Field(100, ge=100, le=5100)


class Lifecycle(Observation):
    name: str
    state: Literal["UNCONFIGURED", "INACTIVE", "ACTIVE", "FINALIZED", "UNKNOWN"]


class RosSystemModel(ContractModel):
    SCHEMA = "rosclaw.ros_system_model.v1"
    schema_version: Literal["rosclaw.ros_system_model.v1"] = "rosclaw.ros_system_model.v1"
    robot_id: str
    snapshot_id: str
    snapshot_hash: str = ""
    captured_at: datetime
    environment: dict[str, Any] = Field(default_factory=dict)
    body: dict[str, Any] = Field(default_factory=dict)
    graph: dict[str, Any] = Field(default_factory=dict)
    signals: list[Signal] = Field(default_factory=list)
    transforms: list[Transform] = Field(default_factory=list)
    lifecycle: list[Lifecycle] = Field(default_factory=list)
    qos: dict[str, Any] = Field(default_factory=dict)
    navigation: dict[str, Any] = Field(default_factory=dict)
    observations: dict[str, Any] = Field(default_factory=dict)
    inference: dict[str, Any] = Field(default_factory=dict)
    completeness: dict[str, bool] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def timestamps_are_aware(self) -> RosSystemModel:
        timestamps = [self.captured_at]
        timestamps.extend(x.captured_at for x in [*self.signals, *self.transforms, *self.lifecycle])
        if any(x.tzinfo is None for x in timestamps):
            raise ValueError("observations require timezone-aware timestamps")
        return self

    def compute_snapshot_hash(self) -> str:
        payload = self.to_dict()
        payload.pop("snapshot_hash", None)
        payload.pop("snapshot_id", None)
        return content_hash("rossnap", payload)

    def seal(self) -> RosSystemModel:
        self.snapshot_hash = self.compute_snapshot_hash()
        self.snapshot_id = self.snapshot_hash.replace("rossnap_", "rossys_")
        return self

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RosSystemModel:
        model = cls.model_validate_contract(data)
        if not model.snapshot_hash or model.snapshot_hash != model.compute_snapshot_hash():
            raise ValueError("ROS snapshot hash mismatch")
        return model

    def age_ms(self, now: datetime) -> float:
        return (now - self.captured_at).total_seconds() * 1000

    def to_dict(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        # Optional v1 additions preserve previously sealed snapshots when the
        # original stream/future-timestamp defaults apply.
        for signal in payload["signals"]:
            for key, default in {"freshness_policy": "stream", "max_age_ms": 1000}.items():
                if signal[key] == default:
                    signal.pop(key)
        for edge in payload["transforms"]:
            if edge["future_tolerance_ms"] == 100:
                edge.pop("future_tolerance_ms")
        return payload
