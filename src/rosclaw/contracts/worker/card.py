"""WorkerCardV1 (总纲 §9.3) — a *declaration*, not a fact.

Probe results and task outcomes provide the runtime evidence; the card only
states what a worker claims to be. Registration validates schema version,
adapter compatibility, digest, license metadata, capability schemas, data
scopes, health probe and forbidden permissions.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field

from rosclaw.contracts.common import ContractModel


class WorkerKind(StrEnum):
    NATIVE = "native"
    TOOL = "tool"
    HARNESS = "harness"
    HUMAN = "human"
    ROBOT = "robot"


class CapabilityDecl(ContractModel):
    SCHEMA = "rosclaw.worker_capability.v1"

    schema_version: Literal["rosclaw.worker_capability.v1"] = "rosclaw.worker_capability.v1"
    name: str = Field(..., description="e.g. code.repository_edit")
    input_schema: str | None = None
    output_schema: str | None = None
    side_effect_class: str = Field(
        default="none",
        description="none | sandbox_process | workspace_write | network_write | physical",
    )


class WorkerImplementation(ContractModel):
    SCHEMA = "rosclaw.worker_implementation.v1"

    schema_version: Literal["rosclaw.worker_implementation.v1"] = "rosclaw.worker_implementation.v1"
    product: str
    version: str = ""
    executable_ref: str | None = None


class WorkerConstraints(ContractModel):
    SCHEMA = "rosclaw.worker_constraints.v1"

    schema_version: Literal["rosclaw.worker_constraints.v1"] = "rosclaw.worker_constraints.v1"
    supported_platforms: list[str] = Field(default_factory=list)
    requires_network: bool = False
    max_concurrency: int = 1
    max_context_tokens: int = 0


class WorkerSecurity(ContractModel):
    SCHEMA = "rosclaw.worker_security.v1"

    schema_version: Literal["rosclaw.worker_security.v1"] = "rosclaw.worker_security.v1"
    isolation: Literal["process", "container", "service", "remote", "human"] = "process"
    credential_profile_ref: str | None = None
    default_data_scopes: list[str] = Field(default_factory=list)
    forbidden_scopes: list[str] = Field(
        default_factory=lambda: [
            "daemon_private_ledger",
            "physical_permits",
            "raw_secrets",
            "direct_hardware",
        ]
    )


class WorkerHealth(ContractModel):
    SCHEMA = "rosclaw.worker_health.v1"

    schema_version: Literal["rosclaw.worker_health.v1"] = "rosclaw.worker_health.v1"
    probe: str = "adapter:ping"
    heartbeat_interval_sec: int = 15
    lease_ttl_sec: int = 45


class WorkerProvenance(ContractModel):
    SCHEMA = "rosclaw.worker_provenance.v1"

    schema_version: Literal["rosclaw.worker_provenance.v1"] = "rosclaw.worker_provenance.v1"
    source: str = "unknown"
    package_digest: str | None = None
    license: str = "UNVERIFIED"


class WorkerTrust(ContractModel):
    SCHEMA = "rosclaw.worker_trust.v1"

    schema_version: Literal["rosclaw.worker_trust.v1"] = "rosclaw.worker_trust.v1"
    initial_level: Literal["UNVERIFIED", "T0", "T1", "T2", "T3"] = "UNVERIFIED"
    evidence_count: int = 0


class WorkerCardV1(ContractModel):
    SCHEMA = "rosclaw.worker_card.v1"
    HASH_PREFIX = "wcard"

    schema_version: Literal["rosclaw.worker_card.v1"] = "rosclaw.worker_card.v1"
    worker_id: str
    display_name: str = ""
    kind: WorkerKind = WorkerKind.TOOL
    adapter_type: str = "process_stdio"
    adapter_version: str = "1.0.0"
    implementation: WorkerImplementation
    capabilities: list[CapabilityDecl] = Field(default_factory=list)
    constraints: WorkerConstraints = Field(default_factory=WorkerConstraints)
    security: WorkerSecurity = Field(default_factory=WorkerSecurity)
    health: WorkerHealth = Field(default_factory=WorkerHealth)
    provenance: WorkerProvenance = Field(default_factory=WorkerProvenance)
    trust: WorkerTrust = Field(default_factory=WorkerTrust)
