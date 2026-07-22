"""Strict schema for the Robot Pack hardware onboarding unit."""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

import yaml
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ROBOT_PACK_SCHEMA_VERSION = "rosclaw.robot_pack.v1"
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class SupportTier(StrEnum):
    """Evidence-backed hardware support levels shared with product status."""

    H0_INDEXED = "H0_INDEXED"
    H1_CONTRACT_VERIFIED = "H1_CONTRACT_VERIFIED"
    H2_SIMULATION_VERIFIED = "H2_SIMULATION_VERIFIED"
    H3_HARDWARE_READ_VERIFIED = "H3_HARDWARE_READ_VERIFIED"
    H4_HARDWARE_ACTUATION_VERIFIED = "H4_HARDWARE_ACTUATION_VERIFIED"
    H5_AGENT_BLACKBOX_VERIFIED = "H5_AGENT_BLACKBOX_VERIFIED"
    H6_REFERENCE_SUPPORTED = "H6_REFERENCE_SUPPORTED"

    @property
    def rank(self) -> int:
        return list(type(self)).index(self)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class PackIdentity(_StrictModel):
    namespace: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{1,63}$")
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]{1,127}$")
    version: str
    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    vendor: str = Field(min_length=1)
    kind: Literal["sensor", "manipulator", "mobile_base", "humanoid", "composite"]
    aliases: list[str] = Field(default_factory=list)

    @field_validator("version")
    @classmethod
    def _valid_version(cls, value: str) -> str:
        try:
            Version(value)
        except InvalidVersion as exc:
            raise ValueError("version must be PEP 440 compatible") from exc
        return value

    @field_validator("aliases")
    @classmethod
    def _valid_aliases(cls, values: list[str]) -> list[str]:
        if any(re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,127}", value) is None for value in values):
            raise ValueError("aliases must be lowercase Robot Pack identifiers")
        if len(values) != len(set(values)):
            raise ValueError("aliases must be unique")
        return values

    @property
    def canonical_ref(self) -> str:
        return f"rosclaw://robot_pack/{self.namespace}/{self.name}@{self.version}"


class PlatformCompatibility(_StrictModel):
    os: list[str] = Field(min_length=1)
    arch: list[str] = Field(min_length=1)
    python: str = ">=3.11,<3.14"
    ros_distributions: list[str] = Field(default_factory=list)

    @field_validator("python")
    @classmethod
    def _valid_python_specifier(cls, value: str) -> str:
        try:
            SpecifierSet(value)
        except InvalidSpecifier as exc:
            raise ValueError("python must be a valid version specifier") from exc
        return value


class PackComponent(_StrictModel):
    kind: Literal[
        "body",
        "hardware_adapter",
        "capability",
        "policy",
        "calibration",
        "simulation",
        "verification",
        "example_app",
        "documentation",
    ]
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]{1,127}$")
    ref: str = Field(min_length=1)
    version: str | None = None
    path: str | None = None
    digest: Annotated[str | None, Field(pattern=r"^sha256:[0-9a-f]{64}$")] = None
    required: bool = True

    @model_validator(mode="after")
    def _local_component_is_safe_and_locked(self) -> PackComponent:
        if self.path is None:
            if self.kind == "hardware_adapter" and self.required and not self.version:
                raise ValueError(
                    f"required hardware adapter {self.id!r} must lock an exact version/revision"
                )
            return self
        candidate = PurePosixPath(self.path)
        if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
            raise ValueError(f"component path must remain inside the pack: {self.path!r}")
        if self.digest is None:
            raise ValueError(f"local component {self.id!r} requires a sha256 digest")
        return self


class DeviceVariant(_StrictModel):
    model: str = Field(min_length=1)
    model_patterns: list[str] = Field(min_length=1)
    product_ids: list[str] = Field(min_length=1)
    body_profile: str = Field(min_length=1)

    @field_validator("product_ids")
    @classmethod
    def _normalize_product_ids(cls, values: list[str]) -> list[str]:
        normalized = [value.lower().removeprefix("0x") for value in values]
        if any(not re.fullmatch(r"[0-9a-f]{4}", value) for value in normalized):
            raise ValueError("product_ids must contain four hexadecimal digits")
        return normalized


class DeviceContract(_StrictModel):
    type: Literal["camera", "hand", "arm", "mobile_base", "composite"]
    vendor_ids: list[str] = Field(min_length=1)
    variants: list[DeviceVariant] = Field(min_length=1)

    @field_validator("vendor_ids")
    @classmethod
    def _normalize_vendor_ids(cls, values: list[str]) -> list[str]:
        normalized = [value.lower().removeprefix("0x") for value in values]
        if any(not re.fullmatch(r"[0-9a-f]{4}", value) for value in normalized):
            raise ValueError("vendor_ids must contain four hexadecimal digits")
        return normalized


class DiscoveryContract(_StrictModel):
    backend: Literal["realsense", "serial_modbus", "ros2", "manual"]
    fallback_backends: list[Literal["linux_sysfs", "manual"]] = Field(default_factory=list)
    required_identity_fields: list[str] = Field(min_length=1)


class ToolRequirement(_StrictModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]{1,127}$")
    any_of: list[str] = Field(min_length=1)


class AdapterContract(_StrictModel):
    component_id: str
    transport: Literal["mcp_stdio", "ros2", "serial_modbus"]
    server_name_patterns: list[str] = Field(default_factory=list)
    tools: list[ToolRequirement] = Field(default_factory=list)
    direct_driver_access: Literal["forbidden", "operator_only"] = "forbidden"


class CapabilityContract(_StrictModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]{1,127}$")
    title: str = Field(min_length=1)
    safety_class: Literal["read_only", "guarded_motion", "actuation"]
    adapter_component_id: str
    adapter_tools_any_of: list[str] = Field(min_length=1)
    execution_modes: list[Literal["REAL", "SHADOW", "SIMULATION", "FIXTURE"]] = Field(min_length=1)
    required_evidence: Literal[
        "REQUESTED",
        "DISPATCH_CONFIRMED",
        "DRIVER_CONFIRMED",
        "PHYSICALLY_OBSERVED",
        "TASK_VERIFIED",
    ]


class SafetyContract(_StrictModel):
    perception_only: bool = False
    actuation: Literal["forbidden", "guarded", "allowed"]
    direct_driver_access: Literal["forbidden", "operator_only"]
    agent_southbound_access: Literal["daemon_only"] = "daemon_only"


class VerificationStage(_StrictModel):
    id: Literal["contract", "simulation", "read-only", "shadow", "actuation", "agent-blackbox"]
    target_tier: SupportTier
    checks: list[str] = Field(min_length=1)
    requires_hardware: bool
    requires_independent_observer: bool = False


class SupportContract(_StrictModel):
    baseline_tier: SupportTier
    candidate_tier: SupportTier
    evidence_ids: list[str] = Field(default_factory=list)


class SignatureContract(_StrictModel):
    required: Literal[True] = True
    scheme: Literal["ed25519"] = "ed25519"
    key_id: str = Field(min_length=1)
    file: str = "signatures/manifest.ed25519"
    payload: Literal["manifest-and-checksums-v1"] = "manifest-and-checksums-v1"

    @field_validator("file")
    @classmethod
    def _safe_signature_path(cls, value: str) -> str:
        candidate = PurePosixPath(value)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("signature file must remain inside the pack")
        return value


class IntegrityContract(_StrictModel):
    checksums_file: str = "checksums.txt"
    signature: SignatureContract

    @field_validator("checksums_file")
    @classmethod
    def _safe_checksums_path(cls, value: str) -> str:
        candidate = PurePosixPath(value)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("checksums file must remain inside the pack")
        return value


class RobotPackManifest(_StrictModel):
    """A locked aggregate of body, adapter, policy, and verification assets."""

    schema_version: Literal["rosclaw.robot_pack.v1"]
    pack: PackIdentity
    compatibility: PlatformCompatibility
    components: list[PackComponent] = Field(min_length=1)
    device: DeviceContract
    discovery: DiscoveryContract
    adapter: AdapterContract
    capabilities: list[CapabilityContract] = Field(min_length=1)
    safety: SafetyContract
    verification: list[VerificationStage] = Field(min_length=1)
    support: SupportContract
    integrity: IntegrityContract

    @model_validator(mode="after")
    def _validate_cross_references_and_claims(self) -> RobotPackManifest:
        component_ids = [component.id for component in self.components]
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("component ids must be unique")
        capability_ids = [capability.id for capability in self.capabilities]
        if len(capability_ids) != len(set(capability_ids)):
            raise ValueError("capability ids must be unique")
        stage_ids = [stage.id for stage in self.verification]
        if len(stage_ids) != len(set(stage_ids)):
            raise ValueError("verification stage ids must be unique")

        component_map = {component.id: component for component in self.components}
        adapter_component = component_map.get(self.adapter.component_id)
        if adapter_component is None or adapter_component.kind != "hardware_adapter":
            raise ValueError("adapter.component_id must reference a hardware_adapter component")
        for capability in self.capabilities:
            component = component_map.get(capability.adapter_component_id)
            if component is None or component.kind != "hardware_adapter":
                raise ValueError(
                    f"capability {capability.id!r} must reference a hardware_adapter component"
                )

        if self.support.baseline_tier.rank > self.support.candidate_tier.rank:
            raise ValueError("support baseline_tier cannot exceed candidate_tier")
        if (
            self.support.baseline_tier.rank >= SupportTier.H1_CONTRACT_VERIFIED.rank
            and not self.support.evidence_ids
        ):
            raise ValueError("H1 or higher baseline requires at least one Evidence ID")
        if any(
            stage.target_tier.rank > self.support.candidate_tier.rank for stage in self.verification
        ):
            raise ValueError("verification stage target cannot exceed support candidate_tier")
        if any(
            stage.target_tier
            in {
                SupportTier.H3_HARDWARE_READ_VERIFIED,
                SupportTier.H4_HARDWARE_ACTUATION_VERIFIED,
            }
            and (not stage.requires_hardware or not stage.requires_independent_observer)
            for stage in self.verification
        ):
            raise ValueError(
                "H3/H4 verification requires hardware and independent observation"
            )

        if self.safety.actuation == "forbidden" and any(
            capability.safety_class != "read_only" for capability in self.capabilities
        ):
            raise ValueError("actuation-forbidden packs may expose only read_only capabilities")
        if self.safety.perception_only and self.safety.actuation != "forbidden":
            raise ValueError("perception-only packs must forbid actuation")
        return self

    @property
    def canonical_ref(self) -> str:
        return self.pack.canonical_ref

    def component(self, component_id: str) -> PackComponent:
        for component in self.components:
            if component.id == component_id:
                return component
        raise KeyError(component_id)

    @classmethod
    def from_path(cls, path: str | Path) -> RobotPackManifest:
        source = Path(path)
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Robot Pack manifest must be a mapping: {source}")
        return cls.model_validate(raw)


def validate_component_digest(value: str) -> bool:
    """Return whether a digest uses the only supported lock algorithm."""

    return bool(_SHA256_RE.fullmatch(value))


__all__ = [
    "ROBOT_PACK_SCHEMA_VERSION",
    "CapabilityContract",
    "DeviceVariant",
    "RobotPackManifest",
    "SupportTier",
    "VerificationStage",
]
