"""Strict, capability-only App manifest contract."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import yaml
from packaging.version import InvalidVersion, Version
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

APP_API_VERSION = "rosclaw.io/v1"
APP_KIND = "App"
_CAPABILITY_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)+$")
_FORBIDDEN_KEYS = {
    "adapter",
    "device",
    "device_path",
    "driver",
    "mcp_server",
    "port",
    "register",
    "ros_topic",
    "server_name",
    "topic",
}
_FORBIDDEN_VALUE_FRAGMENTS = (
    "/dev/",
    "/proc/",
    "/sys/",
    "/cmd_vel",
    "ttyacm",
    "ttyusb",
    "modbus register",
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class AppMetadata(_StrictModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9-]{1,63}$")
    version: str = "0.1.0"
    title: str | None = None
    summary: str | None = None

    @field_validator("version")
    @classmethod
    def _valid_version(cls, value: str) -> str:
        try:
            Version(value)
        except InvalidVersion as exc:
            raise ValueError("App version must be PEP 440 compatible") from exc
        return value


class AppRequirements(_StrictModel):
    capabilities: list[str] = Field(min_length=1)

    @field_validator("capabilities")
    @classmethod
    def _capabilities_are_abstract(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("App capability requirements must be unique")
        for value in values:
            if _CAPABILITY_RE.fullmatch(value) is None:
                raise ValueError(f"App capability is not a canonical capability id: {value!r}")
        return values


class WorkflowStep(_StrictModel):
    call: str
    save_as: str | None = None
    input: dict[str, Any] = Field(default_factory=dict)
    timeout_sec: float = Field(default=30.0, gt=0.0, le=3600.0)

    @field_validator("call")
    @classmethod
    def _call_is_capability(cls, value: str) -> str:
        if _CAPABILITY_RE.fullmatch(value) is None:
            raise ValueError(f"App call is not a canonical capability id: {value!r}")
        return value

    @field_validator("save_as")
    @classmethod
    def _save_name_is_safe(cls, value: str | None) -> str | None:
        if value is not None and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", value) is None:
            raise ValueError("save_as must be a lowercase context identifier")
        return value

    @field_validator("input")
    @classmethod
    def _input_has_no_southbound_binding(cls, value: dict[str, Any]) -> dict[str, Any]:
        _reject_southbound_details(value)
        return value


class AppVerification(_StrictModel):
    require: list[str] = Field(default_factory=list)


class AppManifest(_StrictModel):
    api_version: Literal["rosclaw.io/v1"] = Field(alias="apiVersion")
    kind: Literal["App"]
    metadata: AppMetadata
    requires: AppRequirements
    workflow: list[WorkflowStep] = Field(min_length=1)
    verification: AppVerification = Field(default_factory=AppVerification)

    @model_validator(mode="after")
    def _workflow_uses_declared_capabilities(self) -> AppManifest:
        required = set(self.requires.capabilities)
        undeclared = sorted({step.call for step in self.workflow} - required)
        if undeclared:
            raise ValueError("App workflow calls undeclared capabilities: " + ", ".join(undeclared))
        names = [step.save_as for step in self.workflow if step.save_as is not None]
        if len(names) != len(set(names)):
            raise ValueError("App workflow save_as identifiers must be unique")
        return self

    @classmethod
    def from_path(cls, path: str | Path) -> AppManifest:
        candidate = Path(path)
        if candidate.is_dir():
            candidate = candidate / "app.yaml"
        raw = yaml.safe_load(candidate.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"App manifest must be a mapping: {candidate}")
        return cls.model_validate(raw)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True, mode="json", exclude_none=True)


def _reject_southbound_details(value: Any, *, path: str = "input") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).casefold()
            if normalized in _FORBIDDEN_KEYS or normalized.endswith("_register"):
                raise ValueError(f"App {path}.{key} binds a forbidden southbound detail")
            _reject_southbound_details(child, path=f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _reject_southbound_details(child, path=f"{path}[{index}]")
        return
    if isinstance(value, str):
        normalized = value.casefold()
        if any(fragment in normalized for fragment in _FORBIDDEN_VALUE_FRAGMENTS):
            raise ValueError(f"App {path} contains a forbidden southbound detail")


__all__ = [
    "APP_API_VERSION",
    "APP_KIND",
    "AppManifest",
    "AppMetadata",
    "AppRequirements",
    "AppVerification",
    "WorkflowStep",
]
